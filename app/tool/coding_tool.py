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
    """

    name: str = "run_python_code"
    description: str = (
        "Executes Python code in a stateful sandbox. Use this for calculations, "
        "complex data analysis (pandas/matplotlib), and PRECISION SEARCH within files. "
        "All uploaded documents are available as .txt files in the current directory. "
        "If you generate a plot or a file, inform the user it is available in the environment. "
        "Always print() the final result or variables you want to see."
    )
    
    args_schema: Type[BaseModel] = CodingArguments
    _sandbox_handler: E2BSandbox = PrivateAttr()

    def __init__(self, sandbox_handler: E2BSandbox, **data):
        super().__init__(**data)
        self._sandbox_handler = sandbox_handler

    async def execute(self, code: str, **kwargs) -> ToolResult:
        """
        Executes the provided Python code in the Code Interpreter sandbox.
        """
        await self._sandbox_handler.ensure_running()

        sb = self._sandbox_handler.sandbox
        if not sb:
            return ToolResult(
                success=False,
                output_text="Failed to access the sandbox.",
                error_message="Sandbox instance is None.",
            )

        logger.info(f"CodingTool: Executing snippet (length: {len(code)})")

        try:
            execution = None
            # Robustness: Retry logic for transient sandbox connection issues
            for attempt in range(3):
                try:
                    execution = await asyncio.to_thread(sb.run_code, code)
                    break
                except Exception as e:
                    if "port" in str(e).lower() or "502" in str(e):
                        logger.warning(f"Sandbox connection attempt {attempt+1} failed, retrying...")
                        await asyncio.sleep(1 * (attempt + 1))
                        continue
                    raise

            if execution is None:
                raise RuntimeError("Sandbox execution failed to initialize.")

            # 1. Process standard output and errors
            stdout = "\n".join(execution.logs.stdout)
            stderr = "\n".join(execution.logs.stderr)

            # 2. Process rich results (dataframes, plots, etc.)
            results_text = []
            for result in execution.results:
                # Handle text-based results
                if hasattr(result, "text") and result.text:
                    results_text.append(result.text)
                
                # Handle visual/binary results (like Matplotlib charts)
                if hasattr(result, "formats"):
                    formats = result.formats() if callable(result.formats) else result.formats
                    if isinstance(formats, dict) and formats:
                        found_formats = [fmt for fmt in formats.keys() if fmt != 'text']
                        if found_formats:
                            results_text.append(f"[Visual Output Generated: {', '.join(found_formats)}]")

            # 3. Construct final output string
            output_parts = []
            if stdout: 
                output_parts.append(f"--- STDOUT ---\n{stdout}")
            if stderr: 
                output_parts.append(f"--- STDERR ---\n{stderr}")
            if results_text: 
                output_parts.append("--- RESULTS ---\n" + "\n".join(results_text))

            final_output = "\n\n".join(output_parts) if output_parts else "Executed successfully (no output)."

            # Handle execution errors reported by the sandbox
            if execution.error:
                return ToolResult(
                    success=False,
                    output_text=final_output,
                    error_message=f"{execution.error.name}: {execution.error.value}"
                )

            return ToolResult(success=True, output_text=final_output)

        except Exception as e:
            logger.error(f"CodingTool Failure: {e}")
            return ToolResult(success=False, output_text="Execution error occurred.", error_message=str(e))