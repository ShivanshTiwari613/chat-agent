# filepath: app/tool/coding_tool.py

import asyncio
from typing import Any, Dict, List, Type
from pydantic import BaseModel, Field, PrivateAttr

from app.sandbox.e2b_handler import E2BSandbox
from app.tool.base import BaseTool, ToolResult
from app.utils.logger import logger

class CodingArguments(BaseModel):
    """Schema for the run_python_code tool."""
    code: str = Field(description="The Python code snippet to execute.")

class CodingTool(BaseTool):
    """
    Tool to execute Python code in a persistent E2B Code Interpreter sandbox.
    Variables, files, and imports are preserved between executions.
    """

    name: str = "run_python_code"
    description: str = (
        "Executes Python code in a stateful sandbox environment. "
        "Use this for calculations, data analysis, or reading files saved in the sandbox. "
        "Variables defined in previous turns are preserved. "
        "Always print the output you want to see."
    )
    
    args_schema: Type[BaseModel] = CodingArguments

    # Private attribute to hold the sandbox reference
    _sandbox_handler: E2BSandbox = PrivateAttr()

    def __init__(self, sandbox_handler: E2BSandbox, **data):
        super().__init__(**data)
        self._sandbox_handler = sandbox_handler

    async def execute(self, code: str, **kwargs) -> ToolResult:
        """
        Executes the provided Python code in the Code Interpreter sandbox.
        """

        # Ensure sandbox and its internal Jupyter server are ready
        await self._sandbox_handler.ensure_running()

        sb = self._sandbox_handler.sandbox
        if not sb:
            return ToolResult(
                success=False,
                output_text="Failed to access the sandbox.",
                error_message="Sandbox instance is None after ensure_running().",
            )

        logger.info(f"CodingTool: Executing snippet (length: {len(code)})")

        try:
            # Run the code inside the E2B sandbox
            # Using asyncio.to_thread because the CodeInterpreter sync methods block
            execution = None
            for attempt in range(5):
                try:
                    execution = await asyncio.to_thread(sb.run_code, code)
                    break
                except Exception as e:
                    error_str = str(e)
                    if "port is not open" in error_str or "502" in error_str:
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    raise
            if execution is None:
                raise RuntimeError("Sandbox did not become ready to run code.")

            # Process output logs
            stdout = "\n".join(execution.logs.stdout)
            stderr = "\n".join(execution.logs.stderr)

            # Process execution results (e.g. the last expression or charts)
            results_text = []
            for result in execution.results:
                if hasattr(result, "text") and result.text:
                    results_text.append(result.text)
                elif hasattr(result, "formats"):
                    # For images/dataframes, we note their presence
                    formats = result.formats
                    if callable(formats):
                        formats = formats()
                    if isinstance(formats, dict):
                        keys = list(formats.keys())
                        results_text.append(f"[Rich Output: {keys}]")
                    else:
                        results_text.append("[Rich Output]")
                else:
                    results_text.append(str(result))

            # Build a cohesive output string
            output_parts = []
            if stdout:
                output_parts.append(f"STDOUT:\n{stdout}")
            if stderr:
                output_parts.append(f"STDERR:\n{stderr}")
            if results_text:
                output_parts.append("RESULTS:\n" + "\n".join(results_text))

            final_output = "\n\n".join(output_parts) if output_parts else "Code executed successfully with no output."

            # Handle runtime errors (exceptions in the user's code)
            if execution.error:
                error_msg = (
                    f"{execution.error.name}: {execution.error.value}\n"
                    f"{execution.error.traceback}"
                )
                logger.warning(f"CodingTool: Runtime error in sandbox.")
                return ToolResult(
                    success=False,
                    output_text=final_output,
                    error_message=error_msg,
                )

            return ToolResult(success=True, output_text=final_output)

        except Exception as e:
            logger.error(f"CodingTool: Critical execution failure: {e}", exc_info=True)
            return ToolResult(
                success=False, 
                output_text="Critical error during code execution.", 
                error_message=str(e)
            )
