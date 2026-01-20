import asyncio
import sys
import uuid

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

# Load env before importing settings
load_dotenv()

from app.agent.llm_engine import AgentEngine
from app.utils.logger import logger


async def run_cli_session():
    """
    Runs a CLI-based interactive session with the AI Agent.
    """

    # 1. Generate a unique Session/Plan ID
    session_id = str(uuid.uuid4())[:8]
    print(f"\nStarting AI Agent Session [ID: {session_id}]")
    print("Type 'exit', 'quit', or 'q' to end the session.\n")

    # 2. Initialize the Engine
    engine = AgentEngine(plan_id=session_id)

    try:
        await engine.initialize()

        chat_history = []

        while True:
            try:
                user_input = input(f"\nUser ({session_id}): ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting session...")
                break

            # Process the input
            print(f"Agent ({session_id}): ", end="", flush=True)

            final_answer = ""

            # Stream events
            async for event in engine.chat(user_input, chat_history=chat_history):
                if event.type == "result":
                    # This is the final text answer
                    final_answer = event.details
                    print(f"\n{final_answer}")
                elif event.type == "tool":
                    # Visual feedback for tool usage
                    print(f"\n   [TOOL] {event.details}", end="")
                elif event.type == "observation":
                    print(" OK", end="")
                elif event.type == "error":
                    print(f"\n   [ERROR] {event.details}")

            # Update history
            chat_history.append(HumanMessage(content=user_input))
            if final_answer:
                chat_history.append(AIMessage(content=final_answer))

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        logger.error(f"Critical error in main loop: {e}", exc_info=True)
    finally:
        # Ensure resources are cleaned up (Sandbox killed)
        await engine.cleanup()
        print("Session closed.")


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
        if policy is not None:
            asyncio.set_event_loop_policy(policy())

    try:
        asyncio.run(run_cli_session())
    except KeyboardInterrupt:
        pass
