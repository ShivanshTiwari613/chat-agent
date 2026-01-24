# filepath: app/tool/terminal_tool.py

import asyncio
from typing import Any, Dict, List, Type
from pydantic import BaseModel, Field, PrivateAttr
from app.sandbox.e2b_handler import E2BSandbox
from app.tool.base import BaseTool, ToolResult
from app.utils.logger import logger

class TerminalArguments(BaseModel):
    command: str = Field(description="The shell command to execute (e.g., 'ls -lh', 'cat file.txt', 'grep -C 5 \"keyword\" file.txt').")

class TerminalTool(BaseTool):
    """
    Tool to execute shell commands directly in the E2B Sandbox terminal.
    Use this for filesystem exploration and fast text searching.
    """
    name: str = "execute_terminal_command"
    description: str = (
        "Executes a Linux shell command in the sandbox. "
        "Use this to: "
        "1. List files (ls) "
        "2. Read file contents (cat, head, tail) "
        "3. Search for exact strings across files very fast (grep) "
        "4. Check current directory (pwd)"
    )
    args_schema: Type[BaseModel] = TerminalArguments
    _sandbox_handler: E2BSandbox = PrivateAttr()

    def __init__(self, sandbox_handler: E2BSandbox, **data):
        super().__init__(**data)
        self._sandbox_handler = sandbox_handler

    async def execute(self, command: str, **kwargs) -> ToolResult:
        await self._sandbox_handler.ensure_running()
        sb = self._sandbox_handler.sandbox
        if not sb:
            return ToolResult(success=False, output_text="Sandbox not available.")

        logger.info(f"TerminalTool: Executing '{command}'")

        try:
            # E2B commands.run is synchronous in the SDK, so we use to_thread
            # Note: Using the modern E2B 'commands' namespace
            execution = await asyncio.to_thread(sb.commands.run, command)
            
            output_parts = []
            if execution.stdout:
                output_parts.append(f"STDOUT:\n{execution.stdout}")
            if execution.stderr:
                output_parts.append(f"STDERR:\n{execution.stderr}")

            final_output = "\n\n".join(output_parts) if output_parts else "Command executed successfully (no output)."

            if execution.exit_code != 0:
                return ToolResult(
                    success=False,
                    output_text=final_output,
                    error_message=f"Command failed with exit code {execution.exit_code}"
                )

            return ToolResult(success=True, output_text=final_output)

        except Exception as e:
            logger.error(f"TerminalTool Failure: {e}")
            return ToolResult(success=False, output_text="Terminal execution error.", error_message=str(e))