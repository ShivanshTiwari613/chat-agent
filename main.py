# filepath: main.py

import asyncio
import os
import sys
import uuid
import signal
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

from app.agent.llm_engine import AgentEngine
from app.utils.logger import logger

async def run_cli_session():
    session_id = str(uuid.uuid4())[:8]
    print(f"\n" + "="*50)
    print(f"Starting AI Agent Session [ID: {session_id}]")
    print(f"Environment: Persistent Python Sandbox (E2B) + Namespaced Index")
    print("="*50)
    print("Commands:")
    print(" - /upload [path]  : Upload a file (PDF, CSV, Code, etc.)")
    print(" - exit / quit     : End the session")
    print(" - Ctrl+C          : Graceful shutdown\n")

    engine = AgentEngine(plan_id=session_id)

    try:
        await engine.initialize()
        chat_history = []

        while True:
            try:
                user_input = input(f"User: ").strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\n\n[SYSTEM] Interrupt received. Shutting down...")
                break

            if not user_input: continue

            # Handle File Upload Command
            if user_input.startswith("/upload "):
                file_path = user_input.replace("/upload ", "").strip()
                if os.path.exists(file_path):
                    print(f"   [SYSTEM] Processing {file_path}...")
                    await engine.add_files([file_path])
                    print(f"   [SYSTEM] File indexed and ready.")
                else:
                    print(f"   [ERROR] File not found: {file_path}")
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                break

            print(f"Agent: ", end="", flush=True)
            final_answer = ""

            try:
                async for event in engine.chat(user_input, chat_history=chat_history):
                    if event.type == "result":
                        final_answer = event.details
                        print(f"\n{final_answer}")
                    elif event.type == "tool":
                        print(f"\n   [TOOL] {event.details}", end="")
                    elif event.type == "observation":
                        # Optional: could print tool results here if needed
                        pass
                    elif event.type == "error":
                        print(f"\n   [ERROR] {event.details}")
            except Exception as e:
                print(f"\n   [CRITICAL] Chat execution error: {e}")

            chat_history.append(HumanMessage(content=user_input))
            if final_answer:
                chat_history.append(AIMessage(content=final_answer))

    except Exception as e:
        logger.error(f"Critical session error: {e}", exc_info=True)
    finally:
        print("\n[SYSTEM] Cleaning up resources...")
        await engine.cleanup()
        print("Session closed.")

if __name__ == "__main__":
    # Ensure stdout uses UTF-8 to prevent encoding errors with special characters
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    try:
        asyncio.run(run_cli_session())
    except KeyboardInterrupt:
        pass # Already handled inside run_cli_session