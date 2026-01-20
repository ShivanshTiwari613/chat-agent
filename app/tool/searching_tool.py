# filepath: app/tool/searching_tool.py

import asyncio
from functools import partial
from typing import Any, Dict, List, Optional, Union, Type

from pydantic import BaseModel, Field, PrivateAttr
from tavily import TavilyClient

from app.tool.base import BaseTool, ToolResult
from app.sandbox.e2b_handler import E2BSandbox
from app.utils.logger import logger
from config.settings import settings

# ---------------------------------------------------------------------------
# External service clients
# ---------------------------------------------------------------------------
tavily_client: Optional[TavilyClient] = None

if settings.TAVILY_API_KEY:
    try:
        tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Tavily client: {e}")
else:
    logger.warning("TAVILY_API_KEY not found. Tavily search will be disabled.")

# ---------------------------------------------------------------------------
# Tool Schema
# ---------------------------------------------------------------------------

class SearchArguments(BaseModel):
    """Schema for the search_web tool."""
    query: Union[str, List[str]] = Field(
        description="Topic or question to research. Can be a single string or a list of queries."
    )
    num_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Search results per query."
    )

# ---------------------------------------------------------------------------
# Main Tool
# ---------------------------------------------------------------------------

class SearchingTool(BaseTool):
    """
    Web search tool that gathers data and saves it to a file in the sandbox.
    This prevents metadata overflow and allows the AI to process large amounts of data.
    """

    name: str = "search_web"
    description: str = (
        "Performs a web search to gather information on a topic. "
        "The tool will automatically save the raw research data into a file named 'research_notes.txt' "
        "in your sandbox. You should then read this file using the 'run_python_code' tool "
        "(e.g., print(open('research_notes.txt').read())) to synthesize your final answer or table."
    )

    args_schema: Type[BaseModel] = SearchArguments

    # Private attribute to hold the sandbox reference
    _sandbox_handler: E2BSandbox = PrivateAttr()

    def __init__(self, sandbox_handler: E2BSandbox, **data):
        super().__init__(**data)
        self._sandbox_handler = sandbox_handler

    async def execute(
        self,
        query: Union[str, List[str]],
        num_results: int = 5,
        **kwargs: Any,
    ) -> ToolResult:
        """Run the search and save the content to the sandbox filesystem."""

        if not tavily_client:
            return ToolResult(
                success=False,
                output_text="Tavily client not initialised.",
                error_message="Tavily client not initialised.",
            )

        # 1. Normalize query to a list
        search_queries = query if isinstance(query, list) else [query]

        # 2. Execute Tavily searches and accumulate content
        all_research_data = []
        
        loop = asyncio.get_event_loop()
        for q in search_queries:
            logger.info(f"SearchingTool: Researching '{q}'")
            try:
                # Use standard search to get content snippets
                call = partial(
                    tavily_client.search,
                    query=q,
                    search_depth="advanced",
                    max_results=num_results,
                )
                resp = await loop.run_in_executor(None, call)
                
                results = resp.get("results", [])
                for r in results:
                    entry = (
                        f"SOURCE: {r.get('url')}\n"
                        f"TITLE: {r.get('title')}\n"
                        f"CONTENT: {r.get('content')}\n"
                        "---"
                    )
                    all_research_data.append(entry)
            except Exception as e:
                logger.error(f"Tavily error for '{q}': {e}")

        if not all_research_data:
            return ToolResult(success=False, output_text="No research data found on the web.")

        # 3. Save the data to the sandbox via the CodeInterpreter interface
        combined_data = "\n\n".join(all_research_data)
        try:
            await self._sandbox_handler.ensure_running()
            
            # Access the underlying CodeInterpreter instance
            sb = self._sandbox_handler.sandbox
            if not sb:
                raise RuntimeError("Sandbox is not initialized.")

            # Correct method for CodeInterpreter to write a file
            await asyncio.to_thread(sb.files.write, "research_notes.txt", combined_data)
            
            summary_text = (
                "Research complete. All raw data has been saved to 'research_notes.txt'. "
                "Now, use 'run_python_code' to read this file and extract the data you need for your response."
            )
            return ToolResult(success=True, output_text=summary_text)

        except Exception as e:
            logger.error(f"Failed to write research file to sandbox: {e}")
            return ToolResult(
                success=False, 
                output_text="Research completed but failed to save file to sandbox.",
                error_message=str(e)
            )