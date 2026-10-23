import anthropic
import os
from dotenv import load_dotenv
from harness.tools import registry

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1000

SYSTEM_PROMPT = open("harness/prompts/system_prompt.txt").read()


conversation_history = []

def generate_response(conversation_history):
    """Generate a response from the agent"""
    return client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=conversation_history,
        tools=registry.get_schemas(),
    )


def _text_from_message(message):
    return "".join(block.text for block in message.content if block.type == "text")


def run():
    """Run the agent's conversation loop until the user exits"""

    print("Welcome to the agent! Type 'exit' to end the conversation.")
    print("--------------------------------")
 
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting...")
            break
        
        if not user_input.strip():
            continue

        conversation_history.append({"role": "user", "content": user_input})

        while True:
            response = generate_response(conversation_history)
            assistant_message = response.content or "no text response - used tools only"
            conversation_history.append({"role": "assistant", "content": response.content})

            text = _text_from_message(response)
            if text:
                print(f"Agent: {text}")

            if response.stop_reason != "tool_use":
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                print(f"Tool: {block.name}({dict(block.input)})")
                result = registry.dispatch(block.name, dict(block.input))
                print(f"Result: {result}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

            conversation_history.append({"role": "user", "content": tool_results})

        print("--------------------------------")
    

if __name__ == "__main__":
    run()