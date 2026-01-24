# filepath: app/tool/file_tool.py

from typing import Any, Dict, List, Type, Optional
from pydantic import BaseModel, Field, PrivateAttr
from app.tool.base import BaseTool, ToolResult
from app.utils.file_processor import EphemeralFileIndex

class FileAnalysisArgs(BaseModel):
    query: Optional[str] = Field(None, description="The specific question or topic to look up.")
    namespace: Optional[str] = Field(
        None, 
        description="Filter search by category: 'vault' for documents/PDFs, 'blueprint' for code/ZIPs, or 'lab' for research notes."
    )
    request_type: str = Field(
        "search", 
        description="Either 'search' for snippets, 'map' for codebase structure, or 'list' to see all indexed filenames."
    )

class FileIntelligenceTool(BaseTool):
    name: str = "analyze_documents_and_code"
    description: str = (
        "Search through the Namespaced Intelligence Index. "
        "1. Use 'vault' namespace for documents, PDFs, and historical context. "
        "2. Use 'blueprint' namespace for codebases, ZIP archives, and structural logic. "
        "3. Use 'map' request_type to see the high-level skeleton of the Blueprint. "
        "4. Use 'search' request_type for semantic queries. "
        "Note: If semantic search misses an exact quote, switch to 'run_python_code' "
        "for a deterministic string search (Precision Search Protocol)."
    )
    args_schema: Type[BaseModel] = FileAnalysisArgs
    _index: EphemeralFileIndex = PrivateAttr()

    def __init__(self, index: EphemeralFileIndex, **data):
        super().__init__(**data)
        self._index = index

    async def execute(
        self, 
        request_type: str = "search", 
        query: Optional[str] = None, 
        namespace: Optional[str] = None, 
        **kwargs
    ) -> ToolResult:
        import asyncio
        
        # 1. Handle Metadata Listing
        if request_type == "list":
            files = list(set(m['source'] for m in self._index.chunk_metadata))
            if not files:
                return ToolResult(success=True, output_text="The Intelligence Index is currently empty.")
            return ToolResult(success=True, output_text=f"Indexed Files: {', '.join(files)}")

        # 2. Handle Blueprint Mapping
        if request_type == "map":
            map_text = self._index.get_full_code_map()
            return ToolResult(success=True, output_text=map_text)

        # 3. Handle Semantic/Keyword Search
        if not query:
            return ToolResult(
                success=False,
                output_text="Error: 'query' parameter is required for document searching.",
                error_message="Missing query."
            )

        # Use the Namespaced Hybrid Search
        results = await asyncio.to_thread(self._index.search, query, namespace=namespace)
        
        if not results:
            ns_msg = f" in the '{namespace}' namespace" if namespace else ""
            return ToolResult(
                success=False, 
                output_text=(
                    f"No semantic matches found{ns_msg}. "
                    "If the file exists, proceed to the Precision Search Protocol: "
                    "use 'run_python_code' to search the raw .txt file in your sandbox."
                )
            )
        
        header = f"RESULTS FROM {namespace.upper() if namespace else 'INDEX'}:\n\n"
        return ToolResult(
            success=True, 
            output_text=header + "\n---\n".join(results)
        )