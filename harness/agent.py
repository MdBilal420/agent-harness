import anthropic
import os
from dotenv import load_dotenv
from harness.tools import registry
from harness.memory import load_agents_md

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 8192

SYSTEM_PROMPT = open("harness/prompts/system_prompt.txt").read()


conversation_history = []

def generate_response(conversation_history, agents_md):
    """Generate a response from the agent"""

    system_prompt = SYSTEM_PROMPT + "\n\n" + agents_md

    return client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS, 
        system=system_prompt,
        messages=conversation_history,
        tools=registry.get_schemas(),
    )
def _text_from_message(message):
    return "".join(block.text for block in message.content if block.type == "text")


def run():
    """Run the agent's conversation loop until the user exits"""

    print("Welcome to the agent! Type 'exit' to end the conversation.")
    print("--------------------------------")
    agents_md = load_agents_md()
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting...")
            break
        
        if not user_input.strip():
            continue

        conversation_history.append({"role": "user", "content": user_input})

        while True:
            response = generate_response(conversation_history, agents_md)
            text = _text_from_message(response)
            conversation_history.append({"role": "assistant", "content": text})
            if response.stop_reason != "tool_use":
                break

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
        assistant_text = "\n".join(
            block.text for block in response.content if block.type == "text"
        ) or "(no text response)"
        conversation_history.append({"role": "assistant", "content": assistant_text})
        print("\n \n Agent: " + assistant_text)
        print("--------------------------------")

if __name__ == "__main__":
    run()