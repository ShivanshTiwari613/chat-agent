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
# Tool Schema
# ---------------------------------------------------------------------------

class SearchArguments(BaseModel):
    """Schema for the search_web tool with extensive crawling."""
    query: Union[str, List[str]] = Field(
        description="Topic or question to research. Can be a single string or a list of queries."
    )
    num_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of top search results to crawl deeply per query."
    )
    crawl_depth: bool = Field(
        default=True,
        description="If True, extracts the full content of the found web pages instead of just snippets."
    )

# ---------------------------------------------------------------------------
# Main Tool
# ---------------------------------------------------------------------------

class SearchingTool(BaseTool):
    """
    Extensive Web Crawler tool. Gathers deep content from the web and 
    stores it in the sandbox for analysis.
    """

    name: str = "search_and_crawl_web"
    description: str = (
        "Performs deep research by searching the web and crawling full page content. "
        "The raw results are saved to 'research_notes.txt' in your sandbox. "
        "IMPORTANT: After this tool finishes, you MUST use 'run_python_code' to read "
        "'research_notes.txt' and find the specific answers requested."
    )

    args_schema: Type[BaseModel] = SearchArguments
    _sandbox_handler: E2BSandbox = PrivateAttr()
    _tavily: Optional[TavilyClient] = PrivateAttr(default=None)

    def __init__(self, sandbox_handler: E2BSandbox, **data):
        super().__init__(**data)
        self._sandbox_handler = sandbox_handler
        self._tavily = None
        if settings.TAVILY_API_KEY:
            self._tavily = TavilyClient(api_key=settings.TAVILY_API_KEY)

    async def _perform_single_search(self, q: str, num_results: int) -> List[Dict]:
        loop = asyncio.get_event_loop()
        try:
            if not self._tavily: return []
            call = partial(
                self._tavily.search,
                query=q,
                search_depth="advanced",
                max_results=num_results,
                include_raw_content=True
            )
            resp = await loop.run_in_executor(None, call)
            return resp.get("results", [])
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return []

    async def _extract_full_content(self, urls: List[str]) -> List[Dict]:
        if not urls or not self._tavily: return []
        loop = asyncio.get_event_loop()
        try:
            call = partial(self._tavily.extract, urls=urls)
            resp = await loop.run_in_executor(None, call)
            return resp.get("results", [])
        except Exception as e:
            logger.error(f"Tavily extraction error: {e}")
            return []

    async def execute(
        self,
        query: Union[str, List[str]],
        num_results: int = 5,
        crawl_depth: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        if not self._tavily:
            return ToolResult(success=False, output_text="Tavily API key missing.")

        search_queries = [query] if isinstance(query, str) else query
        
        # 1. Breadth Search
        search_tasks = [self._perform_single_search(q, num_results) for q in search_queries]
        search_results_lists = await asyncio.gather(*search_tasks)
        
        all_results = [item for sublist in search_results_lists for item in sublist]
        urls_to_crawl = list({url for url in (r.get("url") for r in all_results) if isinstance(url, str)})

        all_research_data = []
        crawled_content_map = {}

        # 2. Depth Crawling
        if crawl_depth and urls_to_crawl:
            chunk_size = 20
            for i in range(0, len(urls_to_crawl), chunk_size):
                chunk = urls_to_crawl[i : i + chunk_size]
                extracted = await self._extract_full_content(chunk)
                for item in extracted:
                    crawled_content_map[item.get("url")] = item.get("raw_content")

        # 3. Consolidation
        for r in all_results:
            url = r.get("url")
            content = crawled_content_map.get(url) or r.get("content")
            entry = (
                f"SOURCE: {url}\nTITLE: {r.get('title')}\n"
                f"CONTENT:\n{content}\n"
                f"{'='*50}\n"
            )
            all_research_data.append(entry)

        if not all_research_data:
            return ToolResult(success=False, output_text="No results found on the web.")

        # 4. Save to Sandbox
        combined_data = "\n\n".join(all_research_data)
        try:
            await self._sandbox_handler.ensure_running()
            sb = self._sandbox_handler.sandbox
            if sb:
                await asyncio.to_thread(sb.files.write, "research_notes.txt", combined_data)
                
                # REFINED OUTPUT: Commands the agent to continue autonomously
                summary = (
                    f"SUCCESS: Crawled {len(urls_to_crawl)} sources and saved full content to 'research_notes.txt'. "
                    "ACTION REQUIRED: You must now use 'run_python_code' to read 'research_notes.txt' "
                    "and extract the specific details to answer the user's question."
                )
                return ToolResult(success=True, output_text=summary)
        except Exception as e:
            return ToolResult(success=False, output_text="Sandbox write failed.", error_message=str(e))

        return ToolResult(success=False, output_text="Unknown error in search.")