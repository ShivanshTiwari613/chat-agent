# filepath: app/tool/searching_tool.py

import asyncio
import uuid
import time
import re
import os
from functools import partial
from typing import Any, Dict, List, Optional, Union, Type

from pydantic import BaseModel, Field, PrivateAttr
from tavily import TavilyClient

from app.tool.base import BaseTool, ToolResult
from app.sandbox.e2b_handler import E2BSandbox
from app.utils.file_processor import EphemeralFileIndex
from app.utils.logger import logger
from app.utils.database import SessionLocal, FileRegistryRecord
from config.settings import settings

# Constant for persistent disk storage
STORAGE_BASE = "persistent_storage"

class SearchArguments(BaseModel):
    """Schema for the search_and_crawl_web tool."""
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
    populates the 'Lab' namespace in the Intelligence Engine.
    Returns structured UI-ready metadata for every source found.
    """

    name: str = "search_and_crawl_web"
    description: str = (
        "Performs deep research by searching the web and crawling full page content. "
        "Results are saved to unique research files in the sandbox and indexed in the 'lab' namespace. "
        "Returns a list of structured sources with titles, URLs, and descriptions for the UI."
    )

    args_schema: Type[BaseModel] = SearchArguments
    _sandbox_handler: E2BSandbox = PrivateAttr()
    _file_index: Optional[EphemeralFileIndex] = PrivateAttr()
    _tavily: Optional[TavilyClient] = PrivateAttr(default=None)

    def __init__(self, sandbox_handler: E2BSandbox, file_index: Optional[EphemeralFileIndex] = None, **data):
        super().__init__(**data)
        self._sandbox_handler = sandbox_handler
        self._file_index = file_index
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
            logger.warning(f"Tavily extraction not available or failed: {e}")
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

        # Identify the current plan_id (context passed via kwargs from AgentEngine)
        plan_id = getattr(self._sandbox_handler, "plan_id", "unknown")
        search_queries = [query] if isinstance(query, str) else query
        
        # 1. Generate a unique filename for this research session
        search_id = str(uuid.uuid4())[:8]
        base_query = search_queries[0] if isinstance(search_queries[0], str) else "research"
        clean_query_name = re.sub(r'[^\w\s-]', '', base_query[:20]).strip().replace(' ', '_')
        research_filename = f"research_{clean_query_name}_{search_id}.txt"

        # 2. Breadth Search
        search_tasks = [self._perform_single_search(q, num_results) for q in search_queries]
        search_results_lists = await asyncio.gather(*search_tasks)
        
        all_results = [item for sublist in search_results_lists for item in sublist]
        urls_to_crawl = list({url for url in (r.get("url") for r in all_results) if isinstance(url, str)})

        all_research_data = []
        ui_sources = [] 
        crawled_content_map = {}

        # 3. Depth Crawling
        if crawl_depth and urls_to_crawl:
            chunk_size = 20
            for i in range(0, len(urls_to_crawl), chunk_size):
                chunk = urls_to_crawl[i : i + chunk_size]
                extracted = await self._extract_full_content(chunk)
                for item in extracted:
                    crawled_content_map[item.get("url")] = item.get("raw_content")

        # 4. Consolidation & UI Metadata Generation
        for r in all_results:
            url = r.get("url")
            title = r.get("title", "Unknown Source")
            snippet = r.get("content", "No description available.")
            
            clean_snippet = snippet[:300].replace("\n", " ").strip() + "..."
            
            ui_sources.append({
                "title": title,
                "url": url,
                "description": clean_snippet
            })

            raw_content = crawled_content_map.get(url) or r.get("raw_content") or r.get("content")
            if raw_content:
                raw_content = "".join(char for char in raw_content if ord(char) >= 32 or char in "\n\r\t")
            
            entry = f"SOURCE: {url}\nTITLE: {title}\nCONTENT:\n{raw_content}\n{'='*50}\n"
            all_research_data.append(entry)

        if not all_research_data:
            return ToolResult(success=False, output_text="No research results found.")

        combined_data = "\n\n".join(all_research_data)

        # 5. INTEGRATION: Persistence to Disk and NeonDB
        try:
            # Save to local persistent disk
            session_storage_path = os.path.join(STORAGE_BASE, plan_id)
            os.makedirs(session_storage_path, exist_ok=True)
            full_local_path = os.path.join(session_storage_path, research_filename)
            
            with open(full_local_path, "w", encoding="utf-8") as f:
                f.write(combined_data)

            # Register in NeonDB
            db = SessionLocal()
            db.add(FileRegistryRecord(plan_id=plan_id, filename=research_filename, namespace="lab"))
            db.commit()
            db.close()
            
            # Update current in-memory index
            if self._file_index:
                self._file_index.add_text(combined_data, research_filename, namespace="lab")
                await asyncio.to_thread(self._file_index.finalize)
        except Exception as e:
            logger.warning(f"Failed to persist research results: {e}")

        # 6. Save to Sandbox Mirror
        try:
            await self._sandbox_handler.ensure_running()
            sb = self._sandbox_handler.sandbox
            if sb:
                await asyncio.to_thread(sb.files.write, research_filename, combined_data)
                
                return ToolResult(
                    success=True, 
                    output_text=f"Research complete. Data saved to '{research_filename}'. Scraped {len(ui_sources)} websites.",
                    output_data={"sources": ui_sources, "research_file": research_filename}
                )
        except Exception as e:
            logger.error(f"Sandbox write failed during search: {e}")
            return ToolResult(success=False, output_text="Research gathered but failed to save to sandbox.", error_message=str(e))

        return ToolResult(success=False, output_text="Unknown error occurred during web research.")