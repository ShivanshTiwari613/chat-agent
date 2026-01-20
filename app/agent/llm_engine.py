# filepath: app/agent/llm_engine.py

import asyncio
import re
from typing import Any, AsyncGenerator, List, Dict, Union

# Using explic_classicit paths to ensure compatibility with your environment
from langchain_classic.agents.agent import AgentExecutor
from langchain_classic.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain_core.messages import BaseMessage
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory

from app.agent.prompt import get_agent_prompt
from app.api.schema import AgentEvent
from app.sandbox.e2b_handler import E2BSandbox
from app.tool.coding_tool import CodingTool
from app.tool.searching_tool import SearchingTool
from app.utils.logger import logger
from config.settings import settings


class AgentEngine:
    """
    The main orchestrator for a single agent session.
    Manages the LLM, the Tools, and the E2B Sandbox lifecycle.
    """

    def __init__(self, plan_id: str):
        self.plan_id = plan_id

        # 1. Initialize the Sandbox Handler
        self.sandbox_handler = E2BSandbox(plan_id=plan_id, timeout=1800)

        # 2. Initialize the LLM (Gemini 2.0)
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL_NAME,
            temperature=0,
            google_api_key=settings.GOOGLE_API_KEY,
            max_output_tokens=4096,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )

        self.agent_executor: AgentExecutor | None = None

    async def initialize(self):
        """
        Async initialization: Starts sandbox and builds the agent chain.
        """
        logger.info(f"[{self.plan_id}] Initializing Agent Engine...")

        # Start the E2B Sandbox
        await self.sandbox_handler.start()

        # Instantiate our custom tool logic with the shared sandbox
        search_logic = SearchingTool(sandbox_handler=self.sandbox_handler)
        coding_logic = CodingTool(sandbox_handler=self.sandbox_handler)

        # Wrap tools as StructuredTools
        tools = [
            StructuredTool.from_function(
                coroutine=search_logic.execute,
                name=search_logic.name,
                description=search_logic.description,
                args_schema=search_logic.args_schema,
            ),
            StructuredTool.from_function(
                coroutine=coding_logic.execute,
                name=coding_logic.name,
                description=coding_logic.description,
                args_schema=coding_logic.args_schema,
            ),
        ]

        # Get your detailed prompt
        prompt = get_agent_prompt()

        # Construct the Agent
        agent = create_tool_calling_agent(self.llm, tools, prompt)

        # Create the Executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            max_iterations=15,
        )
        logger.info(f"[{self.plan_id}] Agent Engine initialized successfully.")

    def _normalize_output(self, value: Any) -> str:
        """
        Robustly cleans Gemini 2.0 responses. 
        Strips 'extras', 'signature', and raw dictionary leaks.
        """
        if not value:
            return ""

        # If the LLM returns a list of dictionaries (Common in Gemini 2.0)
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, dict):
                    # Grab only the text content, ignore 'extras', 'signature', etc.
                    parts.append(item.get("text", ""))
                else:
                    parts.append(str(item))
            final_text = "".join(parts)
        elif isinstance(value, dict):
            final_text = value.get("text", str(value))
        else:
            final_text = str(value)

        # NUCLEAR OPTION: If the string STILL looks like a dictionary leak, 
        # use regex to strip out anything that looks like 'extras': {...}
        import re
        # Remove 'extras': { ... }
        final_text = re.sub(r"'extras':\s*\{.*?\}(,\s*)?", "", final_text, flags=re.DOTALL)
        # Remove 'signature': '...'
        final_text = re.sub(r"'signature':\s*'.*?'(,\s*)?", "", final_text, flags=re.DOTALL)
        # Remove 'index': 0, etc
        final_text = re.sub(r"'index':\s*\d+(,\s*)?", "", final_text, flags=re.DOTALL)
        # Clean up any leftover curly braces or brackets from the leak
        final_text = final_text.replace("{'type': 'text', 'text':", "").replace("}]", "").replace("[{", "")
        
        return final_text.strip()

    async def chat(
        self, user_input: str, chat_history: List[BaseMessage] = []
    ) -> AsyncGenerator[AgentEvent, None]:
        """
        Process a user message and stream back status events and the final answer.
        """
        if not self.agent_executor:
            raise RuntimeError("Agent not initialized. Call .initialize() first.")

        yield AgentEvent(
            type="status", label="START", details="Agent started processing..."
        )

        final_output_captured: str = ""
        saw_tool_call = False

        try:
            # We use astream to capture the sequence of tool calls and answers
            async for chunk in self.agent_executor.astream(
                {"input": user_input, "chat_history": chat_history}
            ):
                # 1. Catching tool usage (Action)
                if "actions" in chunk:
                    for action in chunk["actions"]:
                        saw_tool_call = True
                        tool_name = getattr(action, "tool", "tool")
                        yield AgentEvent(
                            type="tool",
                            label="TOOL_CALL",
                            details=f"Agent using {tool_name}...",
                        )

                # 2. Catching tool output (Observation)
                elif "steps" in chunk:
                    for action_obj, _ in chunk["steps"]:
                        tool_name = getattr(action_obj, "tool", "tool")
                        yield AgentEvent(
                            type="observation",
                            label="TOOL_RESULT",
                            details=f"Finished {tool_name}.",
                        )
                
                # 3. Catching the final answer (Output)
                elif "output" in chunk:
                    final_output_captured = self._normalize_output(chunk["output"])
                    if final_output_captured:
                        yield AgentEvent(
                            type="result", 
                            label="ANSWER", 
                            details=final_output_captured
                        )

            # Fallback for Gemini 2.0 if astream fails to yield the 'output' key cleanly
            if not final_output_captured and not saw_tool_call:
                result = await self.agent_executor.ainvoke(
                    {"input": user_input, "chat_history": chat_history}
                )
                final_output_captured = self._normalize_output(result.get("output", ""))
                if final_output_captured:
                    yield AgentEvent(type="result", label="ANSWER", details=final_output_captured)

        except Exception as e:
            logger.error(f"Error during agent execution: {e}", exc_info=True)
            yield AgentEvent(type="error", label="ERROR", details=str(e))

    async def cleanup(self):
        """
        Clean up resources (close sandbox).
        """
        logger.info(f"[{self.plan_id}] Cleaning up Agent Engine...")
        if self.sandbox_handler:
            self.sandbox_handler.close()