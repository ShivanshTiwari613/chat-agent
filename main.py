# filepath: main.py

import asyncio
import os
import sys
import uuid
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

from app.agent.llm_engine import AgentEngine
from app.utils.logger import logger

async def run_cli_session():
    session_id = str(uuid.uuid4())[:8]
    print(f"\nStarting AI Agent Session [ID: {session_id}]")
    print("Commands:")
    print(" - /upload [path]  : Upload a file (PDF, CSV, Code, etc.)")
    print(" - exit / quit     : End the session\n")

    engine = AgentEngine(plan_id=session_id)

    try:
        await engine.initialize()
        chat_history = []

        while True:
            user_input = input(f"\nUser: ").strip()
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

            async for event in engine.chat(user_input, chat_history=chat_history):
                if event.type == "result":
                    final_answer = event.details
                    print(f"\n{final_answer}")
                elif event.type == "tool":
                    print(f"\n   [TOOL] {event.details}", end="")
                elif event.type == "error":
                    print(f"\n   [ERROR] {event.details}")

            chat_history.append(HumanMessage(content=user_input))
            if final_answer:
                chat_history.append(AIMessage(content=final_answer))

    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
    finally:
        await engine.cleanup()
        print("Session closed.")

if __name__ == "__main__":
    asyncio.run(run_cli_session())