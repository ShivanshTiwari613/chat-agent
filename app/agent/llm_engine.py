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
        """
        Ingests files and ZIPs into the Namespaced Intelligence Engine.
        Categories: vault (Docs), blueprint (Code), lab (Research).
        """
        logger.info(f"[{self.plan_id}] Categorizing and indexing {len(file_paths)} uploads...")
        
        all_items_to_index = []

        for path in file_paths:
            if not os.path.exists(path):
                logger.warning(f"File not found: {path}")
                continue

            filename = os.path.basename(path)
            ext = os.path.splitext(filename)[1].lower()

            # Handle ZIP files (Blueprint/Vault mix)
            if ext == ".zip":
                logger.info(f"Processing ZIP archive: {filename}")
                extracted_items = await asyncio.to_thread(FileProcessor.process_zip, path)
                all_items_to_index.extend(extracted_items)
            
            # Handle Data files (CSV/Excel -> Direct Sandbox)
            elif ext in [".csv", ".xlsx", ".xls", ".json"]:
                await self.sandbox_handler.ensure_running()
                try:
                    with open(path, "rb") as f:
                        content = f.read()
                        sb = self.sandbox_handler.sandbox
                        if sb:
                            await asyncio.to_thread(sb.files.write, filename, content)
                            logger.info(f"Uploaded data file {filename} to Sandbox.")
                except Exception as e:
                    logger.error(f"Failed to upload {filename} to sandbox: {e}")
            
            # Handle individual documents or code files
            else:
                content = await asyncio.to_thread(FileProcessor.extract_content, path)
                if content:
                    ns = "blueprint" if ext in [".py", ".js", ".ts", ".go", ".java", ".cpp", ".c", ".h"] else "vault"
                    all_items_to_index.append({
                        "name": filename,
                        "content": content,
                        "namespace": ns
                    })

        # Process all collected items into the Namespaced Index
        for item in all_items_to_index:
            name = item["name"]
            content = item["content"]
            ns = item["namespace"]

            # 1. Add to RAG Namespace
            self.file_index.add_text(content, name, namespace=ns)

            # 2. Map structure if code
            if ns == "blueprint":
                self.file_index.add_code_structure(name, content)

            # 3. Mirror as .txt in Sandbox for Precision Grep Fallback
            await self.sandbox_handler.ensure_running()
            sb = self.sandbox_handler.sandbox
            if sb:
                # Sanitize name for sandbox (replace path separators)
                safe_name = name.replace("/", "_").replace("\\", "_")
                base_name = os.path.splitext(safe_name)[0]
                await asyncio.to_thread(sb.files.write, f"{base_name}.txt", content)

        if self.file_index.chunks:
            logger.info(f"Finalizing indices for {len(self.file_index.chunks)} chunks...")
            await asyncio.to_thread(self.file_index.finalize)

    async def initialize(self):
        logger.info(f"[{self.plan_id}] Initializing Agent Engine Chain...")
        await self.sandbox_handler.start()

        search_logic = SearchingTool(sandbox_handler=self.sandbox_handler)
        coding_logic = CodingTool(sandbox_handler=self.sandbox_handler)
        file_logic = FileIntelligenceTool(index=self.file_index)
        terminal_logic = TerminalTool(sandbox_handler=self.sandbox_handler)

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
            StructuredTool.from_function(
                coroutine=terminal_logic.execute,
                name=terminal_logic.name,
                description=terminal_logic.description,
                args_schema=terminal_logic.args_schema,
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
        if not value: return ""
        if isinstance(value, str):
            if value.startswith("{") and "text" in value:
                try:
                    data = json.loads(value.replace("'", '"'))
                    if isinstance(data, dict): return data.get("text", value)
                except: pass
            clean = re.sub(r"'extras':\s*\{.*?\}(,\s*)?", "", value, flags=re.DOTALL)
            clean = re.sub(r"'signature':\s*'.*?'(,\s*)?", "", clean, flags=re.DOTALL)
            clean = re.sub(r"'index':\s*\d+(,\s*)?", "", clean, flags=re.DOTALL)
            return clean.strip("{}[]' \n")
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

        # --- NAMESPACED SESSION CONTEXT INJECTION ---
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
        
        context_parts.append("\nNote: All indexed text is also available in your sandbox as .txt files for precision searching.")
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
