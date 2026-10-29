import anthropic
import os
from dotenv import load_dotenv
from harness.tools import registry
from harness.memory import load_agents_md
from harness.sandbox import Sandbox
from pathlib import Path
from harness.tools.bash import set_sandbox

from harness.config import MODEL, STEP_BUDGET, MAX_TOKENS

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


MAX_TOOL_CALLS = 100
MAX_TOOL_ROUNDS = STEP_BUDGET

SYSTEM_PROMPT = open("harness/prompts/system_prompt.txt", encoding="utf-8").read()
BUDGET_HIT_MESSAGE = """
The tool execution budget for this turn has been exhausted.
Do not request or execute any more tools.

Respond directly to the user with:
1. What you accomplished.
2. What remains to be done.
3. What the user should ask next to continue the work.
"""


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


def run():
    """Run the agent's conversation loop until the user exits"""

    print("Welcome to the agent! Type 'exit' to end the conversation.")
    print("--------------------------------")

    sandbox = Sandbox()
    sandbox.start()
    set_sandbox(sandbox)
    
    try:
        agents_md = load_agents_md()
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("Exiting...")
                break
            
            if not user_input.strip():
                continue

            conversation_history.append({"role": "user", "content": user_input})

            tool_rounds = 0
            tool_calls_used = 0
            budget_hit = False

            while True:
                response = generate_response(conversation_history, agents_md, budget_hit)
                text = _text_from_message(response)
                conversation_history.append({"role": "assistant", "content": text})
                if response.stop_reason != "tool_use":
                    break

                if tool_rounds >= MAX_TOOL_ROUNDS:
                    budget_hit = True
                    continue

                tool_use_blocks = [
                    block
                    for block in response.content
                    if block.type == "tool_use"
                ]

                if tool_calls_used + len(tool_use_blocks) > MAX_TOOL_CALLS:
                    budget_hit = True
                    continue

                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
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
                if tool_results:
                    conversation_history.append({
                        "role": "user",
                        "content": f"Tool results: {tool_results}",
                    })
                    tool_calls_used += len(tool_use_blocks)
                    tool_rounds += 1

            assistant_text = "\n".join(
                block.text for block in response.content if block.type == "text"
            ) or "(no text response)"

            print("\n \n Agent: " + assistant_text)
            print("--------------------------------")
    finally:
        print("Shutting down sandbox...")
        sandbox.stop()
        print("Sandbox stopped")

if __name__ == "__main__":
    run()