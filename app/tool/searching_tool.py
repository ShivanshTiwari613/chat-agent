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
        "Performs deep research by searching the web and crawling the full content of top results. "
        "The tool saves comprehensive raw data (full page text) into 'research_notes.txt' "
        "in your sandbox. Use 'run_python_code' to read and analyze this data."
    )

    args_schema: Type[BaseModel] = SearchArguments
    _sandbox_handler: E2BSandbox = PrivateAttr()
    _tavily: Optional[TavilyClient] = PrivateAttr(default=None)

    def __init__(self, sandbox_handler: E2BSandbox, **data):
        super().__init__(**data)
        self._sandbox_handler = sandbox_handler
        
        # Initialize client here to ensure settings are loaded
        self._tavily = None
        if settings.TAVILY_API_KEY:
            self._tavily = TavilyClient(api_key=settings.TAVILY_API_KEY)

    def _get_tavily_client(self) -> TavilyClient:
        """Return a configured Tavily client or raise a clear error."""
        if not self._tavily:
            raise RuntimeError("Tavily API key missing.")
        return self._tavily

    async def _perform_single_search(self, q: str, num_results: int) -> List[Dict]:
        """Performs an advanced search for a single query."""
        loop = asyncio.get_event_loop()
        try:
            client = self._get_tavily_client()
            call = partial(
                client.search,
                query=q,
                search_depth="advanced",
                max_results=num_results,
                include_raw_content=True # Gets more data than a standard snippet
            )
            resp = await loop.run_in_executor(None, call)
            return resp.get("results", [])
        except Exception as e:
            logger.error(f"Tavily search error for '{q}': {e}")
            return []

    async def _extract_full_content(self, urls: List[str]) -> List[Dict]:
        """Crawl and extract the full text/markdown from a list of URLs."""
        if not urls:
            return []
        loop = asyncio.get_event_loop()
        try:
            # Tavily's extract endpoint pulls the full clean content of the page
            client = self._get_tavily_client()
            call = partial(client.extract, urls=urls)
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
            return ToolResult(
                success=False,
                output_text="Tavily API key missing.",
                error_message="Tavily API key missing.",
            )

        search_queries = [query] if isinstance(query, str) else query
        
        # 1. BREADTH: Perform all searches in parallel
        logger.info(f"SearchingTool: Starting search for {len(search_queries)} queries...")
        search_tasks = [self._perform_single_search(q, num_results) for q in search_queries]
        search_results_lists = await asyncio.gather(*search_tasks)
        
        # Flatten results and collect URLs
        all_results = [item for sublist in search_results_lists for item in sublist]
        urls_to_crawl = list({url for url in (r.get("url") for r in all_results) if isinstance(url, str)})

        all_research_data = []

        # 2. DEPTH: If crawl_depth is enabled, extract full content from URLs
        crawled_content_map = {}
        if crawl_depth and urls_to_crawl:
            logger.info(f"SearchingTool: Crawling {len(urls_to_crawl)} URLs for full content...")
            # Extract in chunks (Tavily supports up to 20 URLs per extract call)
            chunk_size = 20
            for i in range(0, len(urls_to_crawl), chunk_size):
                chunk = urls_to_crawl[i : i + chunk_size]
                extracted = await self._extract_full_content(chunk)
                for item in extracted:
                    crawled_content_map[item.get("url")] = item.get("raw_content")

        # 3. CONSOLIDATE: Format the data for the sandbox
        for r in all_results:
            url = r.get("url")
            title = r.get("title")
            # Use full crawled content if available, otherwise fallback to snippet
            content = crawled_content_map.get(url) or r.get("content")
            
            entry = (
                f"SOURCE: {url}\n"
                f"TITLE: {title}\n"
                f"FULL_CONTENT:\n{content}\n"
                f"{'='*50}\n"
            )
            all_research_data.append(entry)

        if not all_research_data:
            return ToolResult(success=False, output_text="No research data found.")

        # 4. SAVE: Write to E2B Sandbox
        combined_data = "\n\n".join(all_research_data)
        try:
            await self._sandbox_handler.ensure_running()
            sb = self._sandbox_handler.sandbox
            if not sb:
                return ToolResult(
                    success=False,
                    output_text="Sandbox not available.",
                    error_message="Sandbox not available.",
                )
            
            # Use to_thread for file I/O to avoid blocking the event loop
            await asyncio.to_thread(sb.files.write, "research_notes.txt", combined_data)
            
            summary = (
                f"Extensive research complete. Crawled {len(urls_to_crawl)} unique sources. "
                "The full-page content has been stored in 'research_notes.txt'. "
                "You can now analyze this file using Python code to find specific details."
            )
            return ToolResult(success=True, output_text=summary)

        except Exception as e:
            logger.error(f"Sandbox write error: {e}")
            return ToolResult(success=False, output_text="Sandbox write failed.", error_message=str(e))
