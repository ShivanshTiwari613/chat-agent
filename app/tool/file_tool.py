# filepath: app/tool/file_tool.py

from typing import Any, Dict, List, Type, Optional
from pydantic import BaseModel, Field, PrivateAttr
from app.tool.base import BaseTool, ToolResult
from app.utils.file_processor import EphemeralFileIndex

class FileAnalysisArgs(BaseModel):
    query: Optional[str] = Field(None, description="The specific question or topic to look up.")
    request_type: str = Field(
        "search", 
        description="Either 'search' for semantic snippets, 'map' for codebase structure, or 'list' to see all indexed filenames."
    )

class FileIntelligenceTool(BaseTool):
    name: str = "analyze_documents_and_code"
    description: str = (
        "Search through indexed documents or explore codebase structures. "
        "1. Use 'search' for semantic/thematic queries. "
        "2. Use 'map' to see high-level code structure (classes/functions). "
        "3. Use 'list' only if you are unsure which files are indexed. "
        "CRITICAL: If you need an EXACT paragraph or a specific quote and semantic search fails, "
        "you MUST stop using this tool and use 'run_python_code' to search the raw .txt version "
        "of the file in your sandbox (Precision Search Protocol)."
    )
    args_schema: Type[BaseModel] = FileAnalysisArgs
    _index: EphemeralFileIndex = PrivateAttr()

    def __init__(self, index: EphemeralFileIndex, **data):
        super().__init__(**data)
        self._index = index

    async def execute(self, request_type: str = "search", query: Optional[str] = None, **kwargs) -> ToolResult:
        import asyncio
        
        # List indexed files
        if request_type == "list":
            files = list(set(m['source'] for m in self._index.chunk_metadata))
            if not files:
                return ToolResult(success=True, output_text="No documents are currently indexed.")
            return ToolResult(success=True, output_text=f"Indexed Files: {', '.join(files)}")

        # Codebase mapping
        if request_type == "map":
            map_text = self._index.get_full_code_map()
            return ToolResult(success=True, output_text=map_text)

        # Semantic Search
        if not query:
            return ToolResult(
                success=False,
                output_text="Error: 'query' parameter is required for search.",
                error_message="Missing query."
            )

        # Execute hybrid search (BM25 + Vector)
        results = await asyncio.to_thread(self._index.search, query)
        
        if not results:
            # Nudge the agent toward the Precision Search Protocol
            return ToolResult(
                success=False, 
                output_text=(
                    "No semantic matches found. IMPORTANT: The information may still exist. "
                    "Proceed to the Precision Search Protocol: use 'run_python_code' to "
                    "read the corresponding .txt file in your sandbox and search for exact keywords."
                )
            )
        
        # Provide formatted snippets to the LLM
        return ToolResult(
            success=True, 
            output_text="SEMANTIC SEARCH RESULTS (Top Matches):\n\n" + "\n---\n".join(results)
        )