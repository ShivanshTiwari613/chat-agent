# filepath: app/tool/file_tool.py

from typing import Any, Dict, List, Type, Optional
from pydantic import BaseModel, Field, PrivateAttr
from app.tool.base import BaseTool, ToolResult
from app.utils.file_processor import EphemeralFileIndex

class FileAnalysisArgs(BaseModel):
    query: Optional[str] = Field(None, description="The specific question or topic to look up.")
    request_type: str = Field("search", description="Either 'search' for snippets or 'map' for the codebase structure.")

class FileIntelligenceTool(BaseTool):
    name: str = "analyze_documents_and_code"
    description: str = (
        "Search through documents or view the entire codebase structure. "
        "Use request_type='map' to see all classes/functions in the project. "
        "Use request_type='search' to find specific logic or text snippets."
    )
    args_schema: Type[BaseModel] = FileAnalysisArgs
    _index: EphemeralFileIndex = PrivateAttr()

    def __init__(self, index: EphemeralFileIndex, **data):
        super().__init__(**data)
        self._index = index

    async def execute(self, request_type: str = "search", query: Optional[str] = None, **kwargs) -> ToolResult:
        import asyncio
        
        if request_type == "map":
            map_text = self._index.get_full_code_map()
            return ToolResult(success=True, output_text=map_text)

        if not query:
            return ToolResult(
                success=False,
                output_text="Query is required for search.",
                error_message="Query is required for search.",
            )

        results = await asyncio.to_thread(self._index.search, query)
        if not results:
            return ToolResult(success=False, output_text="No relevant excerpts found.")
        
        return ToolResult(success=True, output_text="RELEVANT EXCERPTS:\n\n" + "\n---\n".join(results))
