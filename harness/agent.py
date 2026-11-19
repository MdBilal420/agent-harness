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
    PLAN_FILENAME,
    PLAN_GATED_TOOLS,
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
    """Generate a response using Anthropic's Messages API."""

    return client.messages.create(
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
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=consolidation_system,
            messages=messages,
        )

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

def dispatch_tools(tool_blocks: list) -> list[dict]:
    """Execute Anthropic tool calls and return structured results."""

    tool_results = []
    needs_plan = not plan_exists()

    for block in tool_blocks:
        print(f"Tool: {block.name}")

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

        try:
            result = registry.dispatch(
                block.name,
                block.input,
            )

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(result),
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

    try:
        # Step 2: Load persistent project memory.
        agents_md = load_agents_md()

        # Anthropic keeps system instructions outside conversation history.
        plan_content = load_plan_for_injection()

        print(f"Plan content: {plan_content}")
        print(f"Agents MD: {agents_md}")

        system_prompt = build_system_prompt(
            agents_md=agents_md,
            plan_content=plan_content,
        )

        if plan_content is not None:
            print(
                f"[Loaded plan.md into context: "
                f"{len(plan_content):,} chars]"
            )

        # Conversation history contains only user and assistant messages.
        messages = []

        # Convert tool schemas once per session.
        tools = get_anthropic_tools()

        # Track planning reminder interval across turns.
        turns_since_reminder = 0

        print(
            "Agent ready. Type 'quit' or 'exit' to leave. "
            "Type /compact to force compaction.\n"
        )

        while True:

            # -------------------------------------------------
            # Automatic compaction
            # -------------------------------------------------

            if compactor.should_compact():
                _run_compaction(compactor, messages)

            current_tokens = compactor.get_last_token_count()

            if current_tokens > 0:
                print(
                    f"[Context: {current_tokens:,} / "
                    f"{COMPACTION_THRESHOLD:,} tokens]"
                )

            # -------------------------------------------------
            # User input
            # -------------------------------------------------

            user_input = input("you > ").strip()

            if user_input.lower() in {"quit", "exit"}:
                _consolidate_memory(messages, client)
                print("Goodbye.")
                break

            if not user_input:
                continue

            # Manual compaction.
            if user_input == "/compact":
                _run_compaction(compactor, messages)
                continue

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

            step_count = 0
            budget_hit = False

            while True:

                response = generate_response(
                    messages=messages,
                    system_prompt=(
                        turn_system_prompt
                        if not budget_hit
                        else turn_system_prompt + "\n\n" + BUDGET_HIT_MESSAGE
                    ),
                    tools=tools,
                    force_no_tools=budget_hit,
                )

                # Record Anthropic token usage.
                compactor.record_response_usage(response)

                # Preserve the complete assistant response.
                # This includes text and tool_use blocks.
                messages.append({
                    "role": "assistant",
                    "content": response.content,
                })

                tool_blocks = get_tool_use_blocks(response)

                # Model has finished its turn.
                if response.stop_reason != "tool_use":
                    break

                # Budget exhausted: do not execute the requested tools.
                # Instead, return error results for every pending tool_use
                # so the Anthropic conversation remains valid.
                if step_count >= STEP_BUDGET:
                    budget_hit = True

                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": (
                                    "Tool execution budget exhausted. "
                                    "This tool was not executed."
                                ),
                                "is_error": True,
                            }
                            for block in tool_blocks
                        ],
                    })

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

            # -------------------------------------------------
            # Final assistant response
            # -------------------------------------------------

            assistant_text = extract_text(response)

            if not assistant_text:
                raise RuntimeError(
                    "Agent loop terminated without a text response. "
                    "Check the API response and termination logic."
                )

            # The complete assistant response was already appended
            # to history, so do not append it a second time.

            print(f"\nagent > {assistant_text}\n")

    finally:
        # Always stop the sandbox, even if an exception occurs.
        print("Stopping sandbox...")
        sandbox.stop()
        print("Sandbox stopped.")


if __name__ == "__main__":
    run()