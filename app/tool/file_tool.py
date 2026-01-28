# filepath: app/tool/file_tool.py

from typing import Any, Dict, List, Type, Optional
from pydantic import BaseModel, Field, PrivateAttr
from app.tool.base import BaseTool, ToolResult
from app.utils.file_processor import EphemeralFileIndex

class FileAnalysisArgs(BaseModel):
    query: Optional[str] = Field(None, description="The specific question or topic to look up.")
    namespace: Optional[str] = Field(
        None, 
        description="CRITICAL: Filter search by category: 'vault' (docs/PDFs), 'blueprint' (code), or 'lab' (web research). If omitted, searches all namespaces."
    )
    request_type: str = Field(
        "search", 
        description="Either 'search' for semantic snippets, 'map' for codebase structure, or 'list' to see all indexed filenames."
    )
    top_k: int = Field(
        8,
        description="Number of results to return. Increase for complex research, decrease for specific facts."
    )

class FileIntelligenceTool(BaseTool):
    """
    Namespaced Intelligence Engine: Implements Staged Hybrid Filtering.
    Stage 1: Pre-filters by Namespace (Vault/Blueprint/Lab) to eliminate noise.
    Stage 2: Executes Hybrid Vector + BM25 search on the filtered subset.
    """
    name: str = "analyze_documents_and_code"
    description: str = (
        "Advanced Namespaced Search Engine. Use this to find information in uploaded files. "
        "1. 'vault': Use for PDFs, documents, and historical text. "
        "2. 'blueprint': Use for code logic, signatures, and project structure. "
        "3. 'lab': Use for raw research data gathered from the web. "
        "4. 'map': Use with request_type='map' to see the high-level codebase skeleton. "
        "Note: If semantic search fails to find a specific string, use 'run_python_code' "
        "for a deterministic Precision Search (grep/regex) in the sandbox."
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
        top_k: int = 8,
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

        # 3. Handle Staged Hybrid Search
        if not query:
            return ToolResult(
                success=False,
                output_text="Error: 'query' parameter is required for document searching.",
                error_message="Missing query."
            )

        # Execute search through the updated EphemeralFileIndex
        # This now uses the Stage 1 (Pre-filtering) and Stage 2 (Hybrid Ranking) logic
        results = await asyncio.to_thread(
            self._index.search, 
            query, 
            namespace=namespace, 
            top_k=top_k
        )
        
        if not results:
            ns_msg = f" in the '{namespace}' pool" if namespace else ""
            return ToolResult(
                success=False, 
                output_text=(
                    f"No semantic matches found{ns_msg}. "
                    "RECOVERY PROTOCOL: If you are certain the file exists, switch to "
                    "'run_python_code' to perform a deterministic string search (grep/regex) "
                    "on the raw .txt files in your sandbox."
                )
            )
        
        header = f"STAGED SEARCH RESULTS ({namespace.upper() if namespace else 'GLOBAL INDEX'}):\n\n"
        return ToolResult(
            success=True, 
            output_text=header + "\n---\n".join(results)
        )