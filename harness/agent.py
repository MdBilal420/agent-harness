import anthropic
import os
from dotenv import load_dotenv
from harness.tools import registry
from harness.memory import (
    CATEGORY_HEADERS,
    load_agents_md,
    validate_agents_md_structure,
    save_agents_md,
)
from harness.sandbox import Sandbox
from pathlib import Path
from harness.tools.bash import set_sandbox

from harness.config import MODEL, STEP_BUDGET, MAX_TOKENS, COMPACTION_THRESHOLD, COMPACTION_KEEP_RECENT
from harness.context import Compactor


load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


MAX_TOOL_CALLS = 100
MAX_TOOL_ROUNDS = STEP_BUDGET

BUDGET_HIT_MESSAGE = """
The tool execution budget for this turn has been exhausted.
Do not request or execute any more tools.

Respond directly to the user with:
1. What you accomplished.
2. What remains to be done.
3. What the user should ask next to continue the work.
"""

SYSTEM_PROMPT = open("harness/prompts/system_prompt.txt", encoding="utf-8").read()
SESSION_END_MEMORY_PROMPT = open("harness/prompts/session_end_memory.txt", encoding="utf-8").read()


conversation_history = []

def generate_response(conversation_history, agents_md, budget_hit=False):
    """Generate a response from the agent"""

    system_prompt = SYSTEM_PROMPT + "\n\n" + agents_md

    if budget_hit:
        system_prompt += "\n\n" + BUDGET_HIT_MESSAGE

    return client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS, 
        system=system_prompt,
        messages=conversation_history,
        tools=registry.get_schemas(),
        tool_choice={"type" : "none"} if budget_hit else {"type" : "auto"},
    )
def _text_from_message(message):
    return "\n".join(block.text for block in message.content if block.type == "text")


def _assistant_content(message) -> list[dict]:
    """Serialize assistant content blocks for the next API request."""
    blocks = []
    for block in message.content:
        if block.type == "text":
            blocks.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            blocks.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
    return blocks


def _consolidate_memory(conversation_history) -> None:
    """Consolidate the memory of the session"""

    print("Consolidating memory...")
    try:
        current_agents_md = load_agents_md()

        consolidation_context = (
            f"{SESSION_END_MEMORY_PROMPT}\n\n"
            f"Current AGENTS.md:\n{current_agents_md}\n\n"
        )

        consolidation_messages = conversation_history + [
            {"role": "user", "content": consolidation_context},
        ]
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=consolidation_messages,
        )
        assistant_text = _text_from_message(response)

        if not assistant_text.strip():
            print("No memory consolidation needed")
            return

        print("New AGENTS.md:\n" + assistant_text)
        if not validate_agents_md_structure(assistant_text):
            missing = [
                header
                for header in CATEGORY_HEADERS.values()
                if header not in assistant_text
            ]
            print(f"Invalid AGENTS.md structure; missing headers: {missing}")
            return

        save_agents_md(assistant_text)
        print("Memory consolidated successfully")
    except Exception as e:
        print(f"Error consolidating memory: {e}")
        raise e

def _run_compact(compactor, conversation_history):
    "Run the compaction"
    before = compactor.get_last_token_count()
    new_messages = compactor.compact(conversation_history)

    if new_messages is None:
        print("Skipping compaction: no older messages to summarise")
        return

    conversation_history.clear()
    conversation_history.extend(new_messages)
    print(f"Context compacted from {before} to {compactor.get_last_token_count()} tokens")

    approx_after = compactor.approximate_char_count(conversation_history)
    print(f"Approximate context size after compaction: {approx_after} characters")

    print("Compaction complete")

def run():
    """Run the agent's conversation loop until the user exits"""

    print("Welcome to the agent! Type 'exit' to end the conversation.")
    print("--------------------------------")

    sandbox = Sandbox()
    sandbox.start()
    set_sandbox(sandbox)

    compactor = Compactor(client)
    
    try:
        agents_md = load_agents_md()
        while True:

            if compactor.should_compact():
                _run_compact(compactor, conversation_history)

            current_tokens = compactor.get_last_token_count()
            if current_tokens > 0:
                print(f"Current token count: {current_tokens} / {COMPACTION_THRESHOLD}")

            user_input = input("You: ")
            if user_input.lower() == "exit":
                _consolidate_memory(conversation_history)
                print("Exiting...")
                break
            
            if not user_input.strip():
                continue
            
            if user_input.lower() == "/compact":
                _run_compact(compactor, conversation_history)
                continue

            conversation_history.append({"role": "user", "content": user_input})

            tool_rounds = 0
            tool_calls_used = 0
            budget_hit = False

            while True:
                response = generate_response(conversation_history, agents_md, budget_hit)
                conversation_history.append({
                    "role": "assistant",
                    "content": _assistant_content(response),
                })

                if response.usage:
                    compactor.record_token_usage(response.usage.input_tokens)

                if response.stop_reason != "tool_use":
                    break

                tool_use_blocks = [
                    block
                    for block in response.content
                    if block.type == "tool_use"
                ]

                # Every tool_use must be paired with a tool_result before the
                # next API call. If the budget is exhausted, return error
                # results and force a text-only summary on the next round.
                if (
                    tool_rounds >= MAX_TOOL_ROUNDS
                    or tool_calls_used + len(tool_use_blocks) > MAX_TOOL_CALLS
                ):
                    conversation_history.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": "Tool budget exhausted; result unavailable.",
                                "is_error": True,
                            }
                            for block in tool_use_blocks
                        ],
                    })
                    budget_hit = True
                    continue

                tool_results = []
                for block in tool_use_blocks:
                    print(f"Tool: {block.name}")
                    try:
                        result = registry.dispatch(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result),
                        })
                    except Exception as e:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Error executing tool: {str(e)}",
                            "is_error": True,
                        })
                conversation_history.append({
                    "role": "user",
                    "content": tool_results,
                })
                tool_calls_used += len(tool_use_blocks)
                tool_rounds += 1

            assistant_text = _text_from_message(response) or "(no text response)"

            print("\n \n Agent: " + assistant_text)
            print("--------------------------------")
    finally:
        print("Shutting down sandbox...")
        sandbox.stop()
        print("Sandbox stopped")

if __name__ == "__main__":
    run()