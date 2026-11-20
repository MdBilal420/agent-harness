import json
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from harness.tools import registry
from harness.config import (
    STEP_BUDGET,
    COMPACTION_THRESHOLD,
    COMPACTION_KEEP_RECENT,
    PLAN_REMINDER_INTERVAL,
    PLAN_GATED_TOOLS,
    RALPH_MAX_CONTINUATIONS,
    RALPH_CONTINUATION_PROMPT,
    RALPH_STUCK_THRESHOLD,
    SESSION_SPEND_LIMIT_USD,
)
from harness.sandbox import Sandbox
from harness.tools.bash import set_sandbox

from harness.memory import (
    load_agents_md,
    save_agents_md,
    validate_agents_md_structure,
)

from harness.context import Compactor

from harness.planning import (
    load_plan_for_injection,
    has_open_items,
    plan_exists,
    build_reminder_message,
    build_create_plan_reminder,
)

from harness.ralph import (
    should_fire as ralph_should_fire,
    is_goal_met,
    snapshot_plan_tasks,
    open_tasks_fingerprint,
    build_carry_forward,
    build_continuation_notice,
    build_cap_hit_notice,
    build_stuck_notice,
    build_no_plan_refusal,
    build_no_goal_refusal,
)

from harness.spend import (
    init_session_spend,
    get_session_spend,
    record_session_spend,
    SpendLimitExceeded,
)


load_dotenv()

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "8192"))

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


# Tool budget message.
BUDGET_HIT_MESSAGE = f"""
You've reached the step budget for this turn
({STEP_BUDGET} tool-calling rounds).

Do not make any more tool calls.

Respond directly to the user with:

1. What you accomplished in this turn.
2. What remains to be done.
3. What the user should ask next to continue the work.

Your response will be the final message for this turn.
The user will reply to it and you can continue from there.
"""


SYSTEM_PROMPT = (
    Path(__file__).parent / "prompts" / "system_prompt.txt"
).read_text()

SESSION_END_MEMORY_PROMPT = (
    Path(__file__).parent / "prompts" / "session_end_memory.txt"
).read_text()


# ---------------------------------------------------------
# Tool schema conversion
# ---------------------------------------------------------

def get_anthropic_tools() -> list[dict]:
    """Convert registry schemas into Anthropic tool schemas."""

    tools = []

    for schema in registry.get_schemas():

        # Convert OpenAI function schema.
        if schema.get("type") == "function":
            function = schema["function"]

            tools.append({
                "name": function["name"],
                "description": function.get("description", ""),
                "input_schema": function.get(
                    "parameters",
                    {
                        "type": "object",
                        "properties": {},
                    },
                ),
            })

        else:
            # Already in Anthropic format.
            tools.append(schema)

    return tools


# ---------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------

def build_system_prompt(
    agents_md: str,
    plan_content: str | None = None,
    extra_context: str = "",
) -> str:
    """Build the system prompt for the current model call."""

    parts = [
        SYSTEM_PROMPT,
        f"# Project Memory (AGENTS.md)\n\n{agents_md}",
    ]

    if plan_content:
        parts.append(
            f"# Current Plan (workspace/plan.md)\n\n{plan_content}"
        )

    if extra_context:
        parts.append(extra_context)

    return "\n\n".join(parts)


# ---------------------------------------------------------
# Anthropic API
# ---------------------------------------------------------

def generate_response(
    messages: list,
    system_prompt: str,
    tools: list,
    force_no_tools: bool = False,
):
    """Generate a response using Anthropic's Messages API.

    Records estimated spend. Raises SpendLimitExceeded if the session
    limit was already hit before this call (does not raise after the
    call — the caller closes out tool_use cleanly, then checks).
    """

    tracker = get_session_spend()
    if tracker is not None:
        tracker.raise_if_exceeded()

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=messages,
        tools=tools,
        tool_choice=(
            {"type": "none"}
            if force_no_tools
            else {"type": "auto"}
        ),
    )
    record_session_spend(response, model=MODEL)
    return response


def extract_text(response) -> str:
    """Extract text blocks from an Anthropic response."""

    return "\n".join(
        block.text
        for block in response.content
        if block.type == "text"
    )


def get_tool_use_blocks(response) -> list:
    """Extract tool_use blocks from an Anthropic response."""

    return [
        block
        for block in response.content
        if block.type == "tool_use"
    ]


# ---------------------------------------------------------
# Session-end memory consolidation
# ---------------------------------------------------------

def _consolidate_memory(messages: list, client) -> None:
    """Consolidate session history into AGENTS.md."""

    print("Consolidating memory...")

    try:
        current_agents_md = load_agents_md()

        consolidation_system = (
            f"{SESSION_END_MEMORY_PROMPT}\n\n"
            f"=== Current AGENTS.md ===\n"
            f"{current_agents_md}"
        )

        # Conversation history contains only Anthropic-compatible
        # user and assistant messages.
        # System instructions are passed separately.
        tracker = get_session_spend()
        if tracker is not None and tracker.exceeded:
            print(
                "Skipping memory consolidation — spend limit already reached."
            )
            return

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=consolidation_system,
            messages=messages,
        )
        record_session_spend(response, model=MODEL)

        proposed = extract_text(response).strip()

        if not proposed:
            print(
                "Memory consolidation returned empty content. "
                "Keeping current AGENTS.md."
            )
            return

        if not validate_agents_md_structure(proposed):
            print(
                "Memory consolidation returned malformed structure "
                "(missing section headers). Keeping current AGENTS.md."
            )
            return

        save_agents_md(proposed)
        print("Memory updated.")

    except Exception as e:
        print(
            f"Memory consolidation failed: {e}. "
            "Keeping current AGENTS.md."
        )


# ---------------------------------------------------------
# Context compaction
# ---------------------------------------------------------

def _run_compaction(
    compactor: Compactor,
    messages: list,
) -> None:
    """Run compaction and update the conversation history."""

    before = compactor.get_last_token_count()

    new_messages = compactor.compact(messages)

    if new_messages is None:
        print(
            "[Compaction skipped: not enough older history to summarize "
            f"(need more than {COMPACTION_KEEP_RECENT} user turns).]"
        )
        return

    messages.clear()
    messages.extend(new_messages)

    approximate_after = compactor.approximate_char_count(messages)

    print(
        f"[Context compacted at {before:,} tokens "
        f"→ ~{approximate_after:,} tokens. "
        f"Full history: .harness/context_log.jsonl]"
    )


# ---------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------

def _tool_call_key(name: str, arguments) -> str:
    """Canonical key for deduping identical tool calls in one batch."""
    try:
        payload = json.dumps(arguments, sort_keys=True, default=str)
    except TypeError:
        payload = str(arguments)
    return f"{name}:{payload}"


def dispatch_tools(tool_blocks: list) -> list[dict]:
    """Execute Anthropic tool calls and return structured results.

    Identical (name, input) pairs in the same batch are executed once; later
    duplicates reuse the first result to avoid wasted work and tokens.
    """

    tool_results = []
    needs_plan = not plan_exists()
    seen: dict[str, str] = {}

    for block in tool_blocks:
        # Soft prompts were ignored; refuse mutating work until plan.md exists.
        if needs_plan and block.name in PLAN_GATED_TOOLS:
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": (
                    f"[planning] Refused `{block.name}` because plan.md does "
                    "not exist yet. Call update_plan(task, status=\"todo\") "
                    "once per subtask to create the plan, then retry this tool."
                ),
                "is_error": True,
            })
            print(f"[planning] gated {block.name} — no plan.md yet")
            continue

        key = _tool_call_key(block.name, block.input)
        if key in seen:
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": (
                    f"{seen[key]}\n"
                    "[deduped: identical tool call already executed "
                    "in this batch]"
                ),
            })
            print(f"Tool: {block.name} (deduped)")
            continue

        try:
            result = registry.dispatch(
                block.name,
                block.input,
            )
            content = str(result)
            seen[key] = content

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content,
            })

            if block.name == "update_plan":
                needs_plan = not plan_exists()

        except Exception as e:
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": f"Tool execution failed: {e}",
                "is_error": True,
            })

    return tool_results


# ---------------------------------------------------------
# Session context
# ---------------------------------------------------------

def _build_session_context() -> tuple[list, str]:
    """Load AGENTS.md + plan.md and return empty history + system prompt."""

    agents_md = load_agents_md()
    plan_content = load_plan_for_injection()

    if plan_content is not None:
        print(
            f"[Loaded plan.md into context: "
            f"{len(plan_content):,} chars]"
        )

    system_prompt = build_system_prompt(
        agents_md=agents_md,
        plan_content=plan_content,
    )

    return [], system_prompt


# ---------------------------------------------------------
# ReAct loop
# ---------------------------------------------------------

def _run_react_loop(
    messages: list,
    system_prompt: str,
    tools: list,
    compactor: Compactor,
):
    """Run tool-calling rounds until the model stops or hits the budget."""

    step_count = 0
    budget_hit = False

    while True:

        response = generate_response(
            messages=messages,
            system_prompt=(
                system_prompt
                if not budget_hit
                else system_prompt + "\n\n" + BUDGET_HIT_MESSAGE
            ),
            tools=tools,
            force_no_tools=budget_hit,
        )

        # Record Anthropic token usage for compaction.
        compactor.record_response_usage(response)

        # Preserve the complete assistant response.
        # This includes text and tool_use blocks.
        messages.append({
            "role": "assistant",
            "content": response.content,
        })

        tool_blocks = get_tool_use_blocks(response)
        tracker = get_session_spend()
        spend_hit = tracker is not None and tracker.exceeded

        # Model has finished its turn.
        if response.stop_reason != "tool_use":
            if spend_hit:
                # Surface the reply the user already paid for, then stop.
                text = extract_text(response)
                if text:
                    print(f"\nagent > {text}\n")
                raise SpendLimitExceeded(tracker)
            return response

        # Spend or step budget exhausted: do not execute tools.
        # Return error results so the Anthropic conversation stays valid.
        if spend_hit or step_count >= STEP_BUDGET:
            if spend_hit:
                err = (
                    "Session spend limit reached. "
                    "This tool was not executed."
                )
            else:
                err = (
                    "Tool execution budget exhausted. "
                    "This tool was not executed."
                )
                budget_hit = True

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": err,
                        "is_error": True,
                    }
                    for block in tool_blocks
                ],
            })

            if spend_hit:
                raise SpendLimitExceeded(tracker)

            continue

        if not tool_blocks:
            raise RuntimeError(
                "Anthropic returned stop_reason='tool_use' "
                "without any tool_use blocks."
            )

        # Execute all requested tools.
        tool_results = dispatch_tools(tool_blocks)

        # Append structured results as a user message.
        messages.append({
            "role": "user",
            "content": tool_results,
        })

        step_count += 1


# ---------------------------------------------------------
# Ralph loop
# ---------------------------------------------------------

def _run_ralph(
    messages: list,
    system_prompt: str,
    tools: list,
    compactor: Compactor,
    goal: str,
    continuation_prompt: str,
) -> str:
    """Drive the agent until the goal is met or a stop condition hits.

    Mutates ``messages`` in place. Returns the (possibly refreshed)
    system prompt so the outer REPL stays in sync after context resets.

    Continuations clear chat history (to bound cost) but inject a compact
    carry-forward summary so the agent does not re-explore from scratch.
    Mid-run follow-up plan items do not block the original goal, and
    unchanged open-task sets trip an early stuck stop.
    """

    continuation_count = 0
    baseline_tasks = snapshot_plan_tasks()
    next_user_content = continuation_prompt
    last_fingerprint: frozenset[str] | None = None
    stuck_streak = 0

    while True:
        messages.append({
            "role": "user",
            "content": next_user_content,
        })

        # Refresh plan into the system prompt each Ralph iteration.
        plan_content = load_plan_for_injection()
        agents_md = load_agents_md()
        system_prompt = build_system_prompt(
            agents_md=agents_md,
            plan_content=plan_content,
        )

        response = _run_react_loop(
            messages,
            system_prompt,
            tools,
            compactor,
        )

        assistant_text = extract_text(response)

        if assistant_text:
            print(f"\nagent > {assistant_text}\n")
        else:
            print(
                "\n[Ralph] Agent returned no text "
                "(stop_reason="
                f"{getattr(response, 'stop_reason', None)}). "
                "Treating as incomplete.\n"
            )

        is_done, reason = is_goal_met(
            client,
            goal,
            messages,
            baseline_tasks=baseline_tasks,
        )

        if is_done:
            print(f"[Ralph] Goal met: {reason}")
            return system_prompt

        fingerprint = open_tasks_fingerprint()
        if fingerprint == last_fingerprint:
            stuck_streak += 1
        else:
            stuck_streak = 0
            last_fingerprint = fingerprint

        if stuck_streak >= RALPH_STUCK_THRESHOLD:
            print(
                build_stuck_notice(
                    RALPH_STUCK_THRESHOLD,
                    sorted(fingerprint),
                )
            )
            return system_prompt

        continuation_count += 1

        if continuation_count > RALPH_MAX_CONTINUATIONS:
            print(build_cap_hit_notice(RALPH_MAX_CONTINUATIONS))
            return system_prompt

        print(build_continuation_notice(continuation_count, reason))

        carry = build_carry_forward(goal, reason, assistant_text)

        # Bound cost with a history reset, but keep the same list object
        # the REPL holds and carry forward a compact resume brief.
        messages.clear()
        fresh_messages, system_prompt = _build_session_context()
        messages.extend(fresh_messages)
        next_user_content = f"{continuation_prompt}\n\n{carry}"


# ---------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------

def run():
    """Run the agent's conversation loop until the user quits."""

    # Step 1: Start sandbox and connect it to the bash tool.
    sandbox = Sandbox()
    sandbox.start()
    set_sandbox(sandbox)

    # Create one Compactor for the entire session.
    compactor = Compactor(client)
    spend = init_session_spend(SESSION_SPEND_LIMIT_USD, MODEL)

    try:
        messages, system_prompt = _build_session_context()
        agents_md = load_agents_md()

        # Convert tool schemas once per session.
        tools = get_anthropic_tools()

        # Track planning reminder interval across turns.
        turns_since_reminder = 0

        # Last real user task — used as Ralph's goal.
        last_substantive_input = None

        print(
            "Agent ready. Type 'quit' or 'exit' to leave.\n"
            "Type /compact to force compaction.\n"
            "Type /ralph [instruction] to start Ralph.\n"
            f"Spend limit: ${SESSION_SPEND_LIMIT_USD:.2f} "
            f"(estimated from token usage).\n"
        )

        while True:

            # -------------------------------------------------
            # Automatic compaction
            # -------------------------------------------------

            try:
                if compactor.should_compact():
                    _run_compaction(compactor, messages)
            except SpendLimitExceeded as e:
                print(str(e))
                continue

            current_tokens = compactor.get_last_token_count()
            spend_line = f" | Spend: {spend.format_status()}"

            if current_tokens > 0 or spend.total_usd > 0:
                print(
                    f"[Context: {current_tokens:,} / "
                    f"{COMPACTION_THRESHOLD:,} tokens{spend_line}]"
                )

            if spend.exceeded:
                print(spend.format_limit_notice())
                print(
                    "Spend limit reached — further model calls are blocked. "
                    "Type 'quit' to exit."
                )

            # -------------------------------------------------
            # User input
            # -------------------------------------------------

            user_input = input("you > ").strip()

            if user_input.lower() in {"quit", "exit"}:
                _consolidate_memory(messages, client)
                print(f"Session spend: {spend.format_status()}")
                print("Goodbye.")
                break

            if not user_input:
                continue

            # Manual compaction.
            if user_input == "/compact":
                try:
                    _run_compaction(compactor, messages)
                except SpendLimitExceeded as e:
                    print(str(e))
                continue

            # Ralph command.
            if user_input == "/ralph" or user_input.startswith("/ralph "):

                if spend.exceeded:
                    print(spend.format_limit_notice())
                    continue

                if not plan_exists():
                    print(build_no_plan_refusal())
                    continue

                if not ralph_should_fire(last_substantive_input):
                    print(build_no_goal_refusal())
                    continue

                if user_input == "/ralph":
                    continuation_prompt = RALPH_CONTINUATION_PROMPT
                else:
                    continuation_prompt = user_input[len("/ralph "):].strip()
                    if not continuation_prompt:
                        continuation_prompt = RALPH_CONTINUATION_PROMPT

                try:
                    system_prompt = _run_ralph(
                        messages,
                        system_prompt,
                        tools,
                        compactor,
                        goal=last_substantive_input,
                        continuation_prompt=continuation_prompt,
                    )
                except SpendLimitExceeded as e:
                    print(str(e))

                # Reload memory in case Ralph's work updated AGENTS.md
                # via tools; plan is refreshed each normal turn anyway.
                agents_md = load_agents_md()
                continue

            if spend.exceeded:
                print(spend.format_limit_notice())
                continue

            # Regular user input.
            last_substantive_input = user_input

            messages.append({
                "role": "user",
                "content": user_input,
            })

            # -------------------------------------------------
            # Planning reminder
            # -------------------------------------------------

            turns_since_reminder += 1

            # Always refresh plan content from disk so updates stick mid-session.
            plan_content = load_plan_for_injection()

            reminder = None
            if not plan_exists():
                # Inject on every turn until a plan exists (interval alone
                # skipped turn 1, which is when most scaffolding happens).
                reminder = build_create_plan_reminder()
            elif (
                turns_since_reminder >= PLAN_REMINDER_INTERVAL
                and has_open_items()
            ):
                reminder = build_reminder_message()
                turns_since_reminder = 0

            turn_system_prompt = build_system_prompt(
                agents_md=agents_md,
                plan_content=plan_content,
                extra_context=reminder or "",
            )

            if reminder:
                print(
                    f"[Planning reminder injected — "
                    f"{turns_since_reminder} turns since last]"
                )

            # -------------------------------------------------
            # ReAct dispatch loop
            # -------------------------------------------------

            try:
                response = _run_react_loop(
                    messages,
                    turn_system_prompt,
                    tools,
                    compactor,
                )
            except SpendLimitExceeded as e:
                print(str(e))
                continue

            # Keep the outer system prompt aligned with the latest plan.
            system_prompt = build_system_prompt(
                agents_md=agents_md,
                plan_content=plan_content,
            )

            # -------------------------------------------------
            # Final assistant response
            # -------------------------------------------------

            assistant_text = extract_text(response)

            if not assistant_text:
                raise RuntimeError(
                    "Agent loop terminated without a text response. "
                    "Check the API response and termination logic."
                )

            print(f"\nagent > {assistant_text}\n")

    finally:
        # Always stop the sandbox, even if an exception occurs.
        tracker = get_session_spend()
        if tracker is not None:
            print(f"Session spend: {tracker.format_status()}")
        print("Stopping sandbox...")
        sandbox.stop()
        print("Sandbox stopped.")


if __name__ == "__main__":
    run()
