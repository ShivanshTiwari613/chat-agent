# filepath: app/agent/llm_engine.py

import asyncio
import os
import re
import json
from typing import Any, AsyncGenerator, List, Dict, Union

# Standard LangChain and Google GenAI imports
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
from app.tool.file_tool import FileIntelligenceTool
from app.utils.file_processor import EphemeralFileIndex, FileProcessor
from app.utils.logger import logger
from config.settings import settings


class AgentEngine:
    """
    The main orchestrator for a single agent session.
    """

    def __init__(self, plan_id: str):
        self.plan_id = plan_id
        self.sandbox_handler = E2BSandbox(plan_id=plan_id, timeout=1800)
        self.file_index = EphemeralFileIndex()

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

    async def add_files(self, file_paths: List[str]):
        logger.info(f"[{self.plan_id}] Ingesting {len(file_paths)} files...")
        
        for path in file_paths:
            if not os.path.exists(path):
                logger.warning(f"File not found: {path}")
                continue

            filename = os.path.basename(path)
            base_name, ext = os.path.splitext(filename)
            ext = ext.lower()

            # Path 1: Data Files (CSV/Excel)
            if ext in [".csv", ".xlsx", ".xls", ".json"]:
                await self.sandbox_handler.ensure_running()
                try:
                    with open(path, "rb") as f:
                        content = f.read()
                        sb = self.sandbox_handler.sandbox
                        if sb:
                            await asyncio.to_thread(sb.files.write, filename, content)
                            logger.info(f"Uploaded {filename} to Sandbox.")
                except Exception as e:
                    logger.error(f"Failed to upload {filename} to sandbox: {e}")

            # Path 2 & 3: RAG & Precision Search Support
            try:
                text_content = await asyncio.to_thread(FileProcessor.extract_content, path)
                if text_content:
                    if ext in [".py", ".js", ".ts", ".go", ".java", ".cpp", ".c"]:
                        self.file_index.add_code_structure(filename, text_content)

                    self.file_index.add_text(text_content, filename)

                    # Save extracted text to Sandbox for "Precision Search Protocol"
                    await self.sandbox_handler.ensure_running()
                    sb = self.sandbox_handler.sandbox
                    if sb:
                        sandbox_txt_name = f"{base_name}.txt"
                        await asyncio.to_thread(sb.files.write, sandbox_txt_name, text_content)
                        logger.info(f"Saved {sandbox_txt_name} for precision searching.")

            except Exception as e:
                logger.error(f"Failed to process {filename}: {e}")

        if self.file_index.chunks:
            await asyncio.to_thread(self.file_index.finalize)

    async def initialize(self):
        logger.info(f"[{self.plan_id}] Initializing Agent Engine Chain...")
        await self.sandbox_handler.start()

        search_logic = SearchingTool(sandbox_handler=self.sandbox_handler)
        coding_logic = CodingTool(sandbox_handler=self.sandbox_handler)
        file_logic = FileIntelligenceTool(index=self.file_index)

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
            StructuredTool.from_function(
                coroutine=file_logic.execute,
                name=file_logic.name,
                description=file_logic.description,
                args_schema=file_logic.args_schema,
            ),
        ]

        prompt = get_agent_prompt()
        agent = create_tool_calling_agent(self.llm, tools, prompt)

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            max_iterations=15,
        )

    def _normalize_output(self, value: Any) -> str:
        """
        Extracts clean text from Gemini 2.0 output, handling potential dict/list structures
        and stripping internal metadata markers.
        """
        if not value:
            return ""

        # If it's already a clean string, just strip minor metadata leftovers
        if isinstance(value, str):
            # Attempt to parse as JSON if it looks like a Gemini dict-string
            if value.startswith("{") and "text" in value:
                try:
                    data = json.loads(value.replace("'", '"')) # Simple fix for single quotes
                    if isinstance(data, dict):
                        return data.get("text", value)
                except:
                    pass
            
            # Fallback to regex cleaning if JSON parsing fails
            clean = re.sub(r"'extras':\s*\{.*?\}(,\s*)?", "", value, flags=re.DOTALL)
            clean = re.sub(r"'signature':\s*'.*?'(,\s*)?", "", clean, flags=re.DOTALL)
            clean = re.sub(r"'index':\s*\d+(,\s*)?", "", clean, flags=re.DOTALL)
            return clean.strip("{}[]' \n")

        # Handle List of parts (often seen in Gemini streaming)
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, dict):
                    parts.append(item.get("text", ""))
                else:
                    parts.append(str(item))
            return "".join(parts).strip()

        # Handle single dictionary
        if isinstance(value, dict):
            return value.get("text", str(value)).strip()

        return str(value).strip()

    async def chat(
        self, user_input: str, chat_history: List[BaseMessage] = []
    ) -> AsyncGenerator[AgentEvent, None]:
        if not self.agent_executor:
            raise RuntimeError("Agent not initialized.")

        # --- SESSION STATE INJECTION ---
        # Get unique filenames from the index
        indexed_files = list(set(m['source'] for m in self.file_index.chunk_metadata))
        file_context = ""
        if indexed_files:
            file_context = f"\n\n[SESSION CONTEXT: The following files are currently indexed and available in your sandbox as .txt files: {', '.join(indexed_files)}]"

        # Combine user input with context to ensure the agent never "forgets" what it has
        contextual_input = f"{user_input}{file_context}"

        yield AgentEvent(type="status", label="START", details="Agent thinking...")

        final_output_captured: str = ""
        saw_tool_call = False

        try:
            async for chunk in self.agent_executor.astream(
                {"input": contextual_input, "chat_history": chat_history}
            ):
                if "actions" in chunk:
                    for action in chunk["actions"]:
                        saw_tool_call = True
                        tool_name = getattr(action, "tool", "tool")
                        yield AgentEvent(type="tool", label="TOOL_CALL", details=f"Agent using {tool_name}...")

                elif "steps" in chunk:
                    for action_obj, _ in chunk["steps"]:
                        tool_name = getattr(action_obj, "tool", "tool")
                        yield AgentEvent(type="observation", label="TOOL_RESULT", details=f"Finished {tool_name}.")
                
                elif "output" in chunk:
                    final_output_captured = self._normalize_output(chunk["output"])
                    if final_output_captured:
                        yield AgentEvent(type="result", label="ANSWER", details=final_output_captured)

            if not final_output_captured and not saw_tool_call:
                result = await self.agent_executor.ainvoke(
                    {"input": contextual_input, "chat_history": chat_history}
                )
                final_output_captured = self._normalize_output(result.get("output", ""))
                if final_output_captured:
                    yield AgentEvent(type="result", label="ANSWER", details=final_output_captured)

        except Exception as e:
            logger.error(f"Error during agent chat execution: {e}", exc_info=True)
            yield AgentEvent(type="error", label="ERROR", details=str(e))

    async def cleanup(self):
        if self.sandbox_handler:
            self.sandbox_handler.close()