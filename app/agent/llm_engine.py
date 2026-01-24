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
from app.tool.terminal_tool import TerminalTool
from app.utils.file_processor import EphemeralFileIndex, FileProcessor
from app.utils.logger import logger
from config.settings import settings


class AgentEngine:
    """
    The main orchestrator for a single agent session using a Namespaced Intelligence Engine.
    """

    def __init__(self, plan_id: str):
        self.plan_id = plan_id
        self.sandbox_handler = E2BSandbox(plan_id=plan_id, timeout=1800)
        self.file_index = EphemeralFileIndex()

        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL_NAME,
            temperature=0,
            google_api_key=settings.GOOGLE_API_KEY,
            max_output_tokens=8192,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )

        self.agent_executor: AgentExecutor | None = None

    async def add_files(self, file_paths: List[str]):
        """
        Ingests files and ZIPs into the Namespaced Intelligence Engine.
        """
        logger.info(f"[{self.plan_id}] Categorizing and indexing {len(file_paths)} uploads...")
        
        all_items_to_index = []

        for path in file_paths:
            if not os.path.exists(path):
                logger.warning(f"File not found: {path}")
                continue

            filename = os.path.basename(path)
            ext = os.path.splitext(filename)[1].lower()

            if ext == ".zip":
                extracted_items = await asyncio.to_thread(FileProcessor.process_zip, path)
                all_items_to_index.extend(extracted_items)
            elif ext in [".csv", ".xlsx", ".xls", ".json"]:
                await self.sandbox_handler.ensure_running()
                try:
                    with open(path, "rb") as f:
                        content = f.read()
                        sb = self.sandbox_handler.sandbox
                        if sb:
                            await asyncio.to_thread(sb.files.write, filename, content)
                except Exception as e:
                    logger.error(f"Failed to upload {filename} to sandbox: {e}")
            else:
                content = await asyncio.to_thread(FileProcessor.extract_content, path)
                if content:
                    ns = "blueprint" if ext in [".py", ".js", ".ts", ".go", ".java", ".cpp", ".c", ".h"] else "vault"
                    all_items_to_index.append({"name": filename, "content": content, "namespace": ns})

        for item in all_items_to_index:
            self.file_index.add_text(item["content"], item["name"], namespace=item["namespace"])
            if item["namespace"] == "blueprint":
                self.file_index.add_code_structure(item["name"], item["content"])
            
            await self.sandbox_handler.ensure_running()
            sb = self.sandbox_handler.sandbox
            if sb:
                # IMPORTANT: Map all uploads to .txt for precision searching
                safe_name = item["name"].replace("/", "_").replace("\\", "_")
                ext_check = os.path.splitext(safe_name)[1].lower()
                if ext_check != ".txt":
                    safe_name = f"{os.path.splitext(safe_name)[0]}.txt"
                await asyncio.to_thread(sb.files.write, safe_name, item["content"])

        if self.file_index.chunks:
            await asyncio.to_thread(self.file_index.finalize)

    async def initialize(self):
        await self.sandbox_handler.start()

        search_logic = SearchingTool(sandbox_handler=self.sandbox_handler, file_index=self.file_index)
        coding_logic = CodingTool(sandbox_handler=self.sandbox_handler)
        file_logic = FileIntelligenceTool(index=self.file_index)
        terminal_logic = TerminalTool(sandbox_handler=self.sandbox_handler)

        tools = [
            StructuredTool.from_function(coroutine=search_logic.execute, name=search_logic.name, description=search_logic.description, args_schema=search_logic.args_schema),
            StructuredTool.from_function(coroutine=coding_logic.execute, name=coding_logic.name, description=coding_logic.description, args_schema=coding_logic.args_schema),
            StructuredTool.from_function(coroutine=file_logic.execute, name=file_logic.name, description=file_logic.description, args_schema=file_logic.args_schema),
            StructuredTool.from_function(coroutine=terminal_logic.execute, name=terminal_logic.name, description=terminal_logic.description, args_schema=terminal_logic.args_schema),
        ]

        self.agent_executor = AgentExecutor(
            agent=create_tool_calling_agent(self.llm, tools, get_agent_prompt()),
            tools=tools,
            verbose=True,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            max_iterations=20,
        )

    def _normalize_output(self, value: Any) -> str:
        if not value: return ""
        if isinstance(value, str):
            if value.strip().startswith("{") and "text" in value:
                try:
                    data = json.loads(value.replace("'", '"'))
                    if isinstance(data, dict): value = data.get("text", value)
                except: pass
            clean = re.sub(r"'extras':\s*\{.*?\}(,\s*)?", "", value, flags=re.DOTALL)
            clean = re.sub(r"'signature':\s*'.*?'(,\s*)?", "", clean, flags=re.DOTALL)
            return clean.strip()
        if isinstance(value, list):
            return "".join([item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in value]).strip()
        if isinstance(value, dict):
            return value.get("text", str(value)).strip()
        return str(value).strip()

    async def chat(
        self, user_input: str, chat_history: List[BaseMessage] = []
    ) -> AsyncGenerator[AgentEvent, None]:
        if not self.agent_executor:
            raise RuntimeError("Agent not initialized.")

        namespaces = {}
        for m in self.file_index.chunk_metadata:
            ns = m['namespace']
            source = m['source']
            if ns not in namespaces: namespaces[ns] = set()
            namespaces[ns].add(source)

        context_parts = ["\n\n[SESSION CONTEXT - NAMESPACED FILES]:"]
        if not namespaces:
            context_parts.append("- No files currently indexed.")
        else:
            for ns, files in namespaces.items():
                context_parts.append(f"- {ns.upper()}: {', '.join(files)}")
        
        # Explicitly tell the agent that file extensions are changed in the sandbox
        context_parts.append("\nNote: Code files are mirrored as .txt in the sandbox for safety.")
        contextual_input = f"{user_input}{''.join(context_parts)}"

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
                        yield AgentEvent(type="tool", label="TOOL_CALL", details=f"Agent using {getattr(action, 'tool', 'tool')}...")
                elif "steps" in chunk:
                    for action_obj, _ in chunk["steps"]:
                        yield AgentEvent(type="observation", label="TOOL_RESULT", details=f"Finished {getattr(action_obj, 'tool', 'tool')}.")
                elif "output" in chunk:
                    final_output_captured = self._normalize_output(chunk["output"])
                    # FIXED: Yield even if the answer is short/empty to acknowledge completion
                    yield AgentEvent(type="result", label="ANSWER", details=final_output_captured or "[No output generated by agent]")

            if not final_output_captured and not saw_tool_call:
                result = await self.agent_executor.ainvoke({"input": contextual_input, "chat_history": chat_history})
                final_output_captured = self._normalize_output(result.get("output", ""))
                yield AgentEvent(type="result", label="ANSWER", details=final_output_captured or "[Task complete, but no specific message provided]")

        except Exception as e:
            logger.error(f"Chat error: {e}", exc_info=True)
            yield AgentEvent(type="error", label="ERROR", details=str(e))

    async def cleanup(self):
        if self.sandbox_handler:
            self.sandbox_handler.close()