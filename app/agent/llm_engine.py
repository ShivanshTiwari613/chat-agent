# filepath: app/agent/llm_engine.py

import asyncio
import os
import re
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
    Manages the LLM, the Tools (Web, Code, RAG, Code-Mapping), 
    and the E2B Sandbox lifecycle.
    """

    def __init__(self, plan_id: str):
        self.plan_id = plan_id

        # 1. Initialize the Sandbox Handler
        self.sandbox_handler = E2BSandbox(plan_id=plan_id, timeout=1800)

        # 2. Initialize the Ephemeral RAG Index (with Tree-Sitter support)
        self.file_index = EphemeralFileIndex()

        # 3. Initialize the LLM (Gemini 2.0)
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
        Differentiates and ingests files into three possible paths:
        1. Data files (CSV/XLSX) -> Uploaded to E2B Sandbox for Python execution.
        2. Code files (PY/JS/TS) -> Processed via Tree-Sitter for structural mapping.
        3. All text files (PDF/DOCX/PY) -> Indexed in RAM via Hybrid RAG (BM25 + FAISS).
        """
        logger.info(f"[{self.plan_id}] Ingesting {len(file_paths)} files into the intelligence engine...")
        
        for path in file_paths:
            if not os.path.exists(path):
                logger.warning(f"File not found: {path}")
                continue

            filename = os.path.basename(path)
            ext = os.path.splitext(path)[1].lower()

            # --- PATH 1: DATA ANALYSIS (Sandbox) ---
            if ext in [".csv", ".xlsx", ".xls", ".json"]:
                await self.sandbox_handler.ensure_running()
                try:
                    with open(path, "rb") as f:
                        content = f.read()
                        sb = self.sandbox_handler.sandbox
                        if not sb:
                            logger.error("Sandbox not available; skipping upload.")
                            continue
                        await asyncio.to_thread(sb.files.write, filename, content)
                    logger.info(f"Uploaded {filename} to Sandbox for data analysis.")
                except Exception as e:
                    logger.error(f"Failed to upload {filename} to sandbox: {e}")

            # --- PATH 2 & 3: RAG & CODE MAPPING ---
            try:
                # Extract text content
                text_content = await asyncio.to_thread(FileProcessor.extract_content, path)
                
                if text_content:
                    # If it's code, build the structural map (Skeleton)
                    if ext in [".py", ".js", ".ts", ".go", ".java", ".cpp", ".c"]:
                        self.file_index.add_code_structure(filename, text_content)
                        logger.info(f"Created structural map for {filename} using Tree-Sitter.")

                    # Add to standard RAG chunks
                    self.file_index.add_text(text_content, filename)
                    logger.info(f"Indexed {filename} into Hybrid RAG.")
            except Exception as e:
                logger.error(f"Failed to process {filename}: {e}")

        # Finalize indices (Compute Embeddings and BM25 Scores)
        if self.file_index.chunks:
            logger.info(f"[{self.plan_id}] Finalizing Hybrid RAG search indices...")
            await asyncio.to_thread(self.file_index.finalize)

    async def initialize(self):
        """
        Async initialization: Starts sandbox and builds the agent chain.
        """
        logger.info(f"[{self.plan_id}] Initializing Agent Engine Chain...")

        # Ensure E2B Sandbox is alive
        await self.sandbox_handler.start()

        # Instantiate logic for tools
        search_logic = SearchingTool(sandbox_handler=self.sandbox_handler)
        coding_logic = CodingTool(sandbox_handler=self.sandbox_handler)
        file_logic = FileIntelligenceTool(index=self.file_index)

        # Build StructuredTools
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

        # Construct Agent with Contextual Prompt
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
        logger.info(f"[{self.plan_id}] Agent Engine fully initialized and tools registered.")

    def _normalize_output(self, value: Any) -> str:
        """
        Cleans Gemini 2.0 responses. 
        Strips internal JSON metadata ('extras', 'signature', 'index') 
        that often leaks during streaming.
        """
        if not value:
            return ""

        # Handle list of parts
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, dict):
                    parts.append(item.get("text", ""))
                else:
                    parts.append(str(item))
            final_text = "".join(parts)
        elif isinstance(value, dict):
            final_text = value.get("text", str(value))
        else:
            final_text = str(value)

        # Regex to strip common Gemini 2.0 metadata leaks
        final_text = re.sub(r"'extras':\s*\{.*?\}(,\s*)?", "", final_text, flags=re.DOTALL)
        final_text = re.sub(r"'signature':\s*'.*?'(,\s*)?", "", final_text, flags=re.DOTALL)
        final_text = re.sub(r"'index':\s*\d+(,\s*)?", "", final_text, flags=re.DOTALL)
        
        # Clean up leftover curly braces or list wrappers
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

        yield AgentEvent(type="status", label="START", details="Agent thinking...")

        final_output_captured: str = ""
        saw_tool_call = False

        try:
            async for chunk in self.agent_executor.astream(
                {"input": user_input, "chat_history": chat_history}
            ):
                # 1. Action Identification
                if "actions" in chunk:
                    for action in chunk["actions"]:
                        saw_tool_call = True
                        tool_name = getattr(action, "tool", "tool")
                        yield AgentEvent(type="tool", label="TOOL_CALL", details=f"Agent using {tool_name}...")

                # 2. Observation Capture
                elif "steps" in chunk:
                    for action_obj, _ in chunk["steps"]:
                        tool_name = getattr(action_obj, "tool", "tool")
                        yield AgentEvent(type="observation", label="TOOL_RESULT", details=f"Finished {tool_name}.")
                
                # 3. Final Answer Processing
                elif "output" in chunk:
                    final_output_captured = self._normalize_output(chunk["output"])
                    if final_output_captured:
                        yield AgentEvent(type="result", label="ANSWER", details=final_output_captured)

            # Fallback for complex reasoning chains where output isn't in final chunk
            if not final_output_captured and not saw_tool_call:
                result = await self.agent_executor.ainvoke(
                    {"input": user_input, "chat_history": chat_history}
                )
                final_output_captured = self._normalize_output(result.get("output", ""))
                if final_output_captured:
                    yield AgentEvent(type="result", label="ANSWER", details=final_output_captured)

        except Exception as e:
            logger.error(f"Error during agent chat execution: {e}", exc_info=True)
            yield AgentEvent(type="error", label="ERROR", details=str(e))

    async def cleanup(self):
        """
        Clean up resources: shuts down the sandbox.
        """
        logger.info(f"[{self.plan_id}] Cleaning up Agent Engine session...")
        if self.sandbox_handler:
            self.sandbox_handler.close()
