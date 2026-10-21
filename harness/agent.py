import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1000

SYSTEM_PROMPT = """
You are a coding agent in a terminal. Help the user write, debug, and run code.

Rules:
- Be concise. Prefer working code over long explanation.
- State assumptions only when they change the solution.
- When changing code, show only the relevant snippet or exact commands.
- Run or suggest the smallest command that verifies the change.
- Do not destroy data, push, or change git history unless the user asks.
- If something is blocked (missing file, secret, or permission), say so and stop.
"""

conversation_history = []

def generate_response(conversation_history):
    """Generate a response from the agent"""
    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=conversation_history,
    )
    return message.content[0].text

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

        # Append the user input to the conversation history
        conversation_history.append({"role": "user", "content": user_input})

        # Generate a response from the agent
        response = generate_response(conversation_history)

        # Append the agent's response to the conversation history
        conversation_history.append({"role": "assistant", "content": response})

        # Display the response
        print(f"Agent: {response}")
        print("--------------------------------")
    

if __name__ == "__main__":
    run()