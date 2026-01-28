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
    Orchestrator for Staged Hybrid Filtering.
    Features: Ultra-robust output normalization and execution safety.
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
        """Ingests files into the Staged Intelligence Engine."""
        all_items_to_index = []
        for path in file_paths:
            if not os.path.exists(path): continue
            filename = os.path.basename(path)
            ext = os.path.splitext(filename)[1].lower()

            if ext == ".zip":
                extracted = await asyncio.to_thread(FileProcessor.process_zip, path)
                all_items_to_index.extend(extracted)
            elif ext in [".csv", ".xlsx", ".xls", ".json"]:
                await self.sandbox_handler.ensure_running()
                with open(path, "rb") as f:
                    content = f.read()
                    sb = self.sandbox_handler.sandbox
                    if sb: await asyncio.to_thread(sb.files.write, filename, content)
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
                safe_name = item["name"].replace("/", "_").replace("\\", "_")
                if not safe_name.lower().endswith(".txt"):
                    safe_name = f"{os.path.splitext(safe_name)[0]}.txt"
                await asyncio.to_thread(sb.files.write, safe_name, item["content"])

        if self.file_index.chunks:
            await asyncio.to_thread(self.file_index.finalize)

    async def initialize(self):
        """Correctly initializes tools by instantiating them first."""
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
            handle_parsing_errors=True,
            max_iterations=15,
        )

    def _normalize_output(self, value: Any) -> str:
        """
        Hyper-robust normalization.
        Extracts clean text from complex JSON objects, lists, and signature blocks.
        """
        if not value: return ""
        
        # 1. Handle tool result objects
        if hasattr(value, "output_text"):
            return str(value.output_text).strip()
        
        # 2. Handle Dictionary results
        if isinstance(value, dict):
            # Check for standard content keys first
            for key in ["text", "output_text", "output", "content"]:
                if key in value and value[key]:
                    return self._normalize_output(value[key])
            # If no clear text key, check if it's a list under 'content'
            if "content" in value and isinstance(value["content"], list):
                return self._normalize_output(value["content"])
            return json.dumps(value)

        # 3. Handle Lists
        if isinstance(value, list):
            return "\n".join([self._normalize_output(v) for v in value if v]).strip()

        # 4. Handle Strings & JSON-Strings
        if isinstance(value, str):
            val = value.strip()
            
            # If string looks like JSON, try to parse it
            if (val.startswith("{") and val.endswith("}")) or (val.startswith("[") and val.endswith("]")):
                try:
                    # Cleanup common encoding errors before parse
                    clean_json = val.replace('\\"', '"').replace('\\n', '\n')
                    data = json.loads(clean_json)
                    return self._normalize_output(data)
                except: pass
            
            # Aggressively strip technical noise via Regex
            val = re.sub(r'["\']extras["\']:\s*\{.*?\}', "", val, flags=re.DOTALL)
            val = re.sub(r'["\']signature["\']:\s*["\'].*?["\']', "", val, flags=re.DOTALL)
            val = re.sub(r'["\']type["\']:\s*["\']text["\']', "", val)
            
            # Final bracket/key cleanup
            val = val.lstrip('{').rstrip('}').strip()
            if '"text":' in val:
                val = val.split('"text":', 1)[1].strip().lstrip('"').rsplit('"', 1)[0]
            
            return val.strip()

        return str(value).strip()

    async def chat(
        self, user_input: str, chat_history: List[BaseMessage] = []
    ) -> AsyncGenerator[AgentEvent, None]:
        if not self.agent_executor: raise RuntimeError("Agent not initialized.")

        namespaces = {}
        for m in self.file_index.chunk_metadata:
            ns = m['namespace']
            if ns not in namespaces: namespaces[ns] = set()
            namespaces[ns].add(m['source'])

        context_parts = ["\n\n[INTELLIGENCE POOLS]:"]
        for ns, files in namespaces.items():
            context_parts.append(f"- {ns.upper()}: {', '.join(files)}")
        
        contextual_input = f"{user_input}{''.join(context_parts)}"
        yield AgentEvent(type="status", label="START", details="Thinking...")

        final_output_captured: str = ""
        try:
            async for chunk in self.agent_executor.astream(
                {"input": contextual_input, "chat_history": chat_history}
            ):
                if "actions" in chunk:
                    for action in chunk["actions"]:
                        yield AgentEvent(type="tool", label="TOOL", details=f"Searching {action.tool}...")
                elif "output" in chunk:
                    final_output_captured = self._normalize_output(chunk["output"])
                    if final_output_captured:
                        yield AgentEvent(type="result", label="ANSWER", details=final_output_captured)

            # CRITICAL FALLBACK: If stream provided no result or malformed JSON
            if not final_output_captured:
                result = await self.agent_executor.ainvoke({"input": contextual_input, "chat_history": chat_history})
                final_output_captured = self._normalize_output(result.get("output", ""))
                yield AgentEvent(type="result", label="ANSWER", details=final_output_captured or "Analysis complete.")

        except Exception as e:
            logger.error(f"Chat error: {e}", exc_info=True)
            yield AgentEvent(type="error", label="ERROR", details=str(e))

    async def cleanup(self):
        if self.sandbox_handler: self.sandbox_handler.close()