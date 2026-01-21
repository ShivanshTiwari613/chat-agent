# filepath: app/utils/file_processor.py

import os
import fitz  # PyMuPDF
from docx import Document
from typing import List, Dict, Any, Optional, cast
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tree_sitter
from tree_sitter_languages import get_language, get_parser
from app.utils.logger import logger

class EphemeralFileIndex:
    def __init__(self):
        # Local, lightweight encoder
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunks: List[str] = []
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.code_map: Dict[str, List[str]] = {} # Filename -> List of function/class signatures
        
        self.bm25: Optional[BM25Okapi] = None
        self.vector_index: Optional[faiss.IndexFlatL2] = None

    def add_code_structure(self, filename: str, content: str):
        """
        Uses Tree-Sitter to map out the 'skeleton' of code files.
        Updated to support tree-sitter 0.21.0+ API.
        """
        ext = os.path.splitext(filename)[1].lower()
        lang_map = {
            ".py": "python", 
            ".js": "javascript", 
            ".ts": "typescript", 
            ".go": "go", 
            ".cpp": "cpp", 
            ".c": "c",
            ".java": "java"
        }
        
        if ext not in lang_map:
            return

        try:
            language_name = lang_map[ext]
            language = get_language(language_name)
            parser = get_parser(language_name)
            tree = parser.parse(bytes(content, "utf8"))
            
            # Language-specific queries to avoid invalid node types.
            query_map = {
                "python": """
                    (function_definition name: (identifier) @func_name)
                    (class_definition name: (identifier) @class_name)
                """,
                "javascript": """
                    (function_declaration name: (identifier) @func_name)
                    (class_declaration name: (identifier) @class_name)
                    (method_definition name: (property_identifier) @method_name)
                """,
                "typescript": """
                    (function_declaration name: (identifier) @func_name)
                    (class_declaration name: (identifier) @class_name)
                    (method_definition name: (property_identifier) @method_name)
                """,
                "go": """
                    (function_declaration name: (identifier) @func_name)
                    (method_declaration name: (field_identifier) @method_name)
                    (type_spec name: (type_identifier) @type_name)
                """,
                "c": """
                    (function_definition declarator: (function_declarator declarator: (identifier) @func_name))
                    (struct_specifier name: (type_identifier) @struct_name)
                """,
                "cpp": """
                    (function_definition declarator: (function_declarator declarator: (identifier) @func_name))
                    (class_specifier name: (type_identifier) @class_name)
                    (struct_specifier name: (type_identifier) @struct_name)
                """,
                "java": """
                    (class_declaration name: (identifier) @class_name)
                    (method_declaration name: (identifier) @method_name)
                """,
            }
            query_str = query_map.get(language_name)
            if not query_str:
                return

            # Correct version-agnostic way to initialize a Query
            query = language.query(query_str)
            captures = query.captures(tree.root_node)
            
            signatures = []
            # In newer tree-sitter, captures can return a dict or a list of tuples
            # We handle the most common result format
            capture_items = captures.items() if isinstance(captures, dict) else captures
            for node, tag in capture_items:
                # Extract the source text for the specific node (e.g., the function name or header)
                start = node.start_byte
                end = node.end_byte
                node_text = content[start:end]
                
                # We want to capture the line of the definition for context
                line_content = node_text.split('\n')[0]
                signatures.append(f"{tag.replace('_', ' ').upper()}: {line_content}")
            
            if signatures:
                self.code_map[filename] = signatures
                logger.info(f"Successfully mapped {len(signatures)} symbols in {filename}")

        except Exception as e:
            # Fallback for version-specific constructor issues
            logger.warning(f"Tree-sitter parse error for {filename}: {e}")
            # If the complex query fails, we still have the file in the RAG index anyway

    def add_text(self, text: str, source_name: str):
        if not text: return
        
        # Standard cleaning
        text = text.replace('\x00', '') # Remove null bytes
        
        words = text.split()
        chunk_size = 400 
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            self.chunks.append(chunk)
            self.chunk_metadata.append({"source": source_name, "start_word": i})

    def finalize(self):
        """Builds the search indices."""
        if not self.chunks:
            return
        
        # 1. BM25
        tokenized_corpus = [doc.lower().split() for doc in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 2. Vector Index
        embeddings = self.encoder.encode(self.chunks, show_progress_bar=False)
        embeddings_np = np.array(embeddings).astype('float32')
        
        dimension = int(embeddings_np.shape[1])
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_np) # type: ignore
        self.vector_index = index

    def get_full_code_map(self) -> str:
        """Returns the high-level structure of all indexed code."""
        if not self.code_map:
            return "No structured code codebase found or Tree-Sitter mapping is empty."
        
        output = ["CODEBASE STRUCTURE MAP:"]
        for file, sigs in self.code_map.items():
            output.append(f"\nFILE: {file}")
            # Limit to top 20 symbols per file to avoid context bloat
            for s in sigs[:20]:
                output.append(f"  - {s}")
            if len(sigs) > 20:
                output.append(f"  ... and {len(sigs)-20} more symbols.")
        return "\n".join(output)

    def search(self, query: str, top_k: int = 5) -> List[str]:
        if not self.chunks or self.vector_index is None or self.bm25 is None:
            return []

        # 1. Vector Search
        query_vec = np.array(self.encoder.encode([query])).astype('float32')
        _, v_indices = self.vector_index.search(query_vec, top_k) # type: ignore
        
        # 2. BM25 Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        b_indices = np.argsort(bm25_scores)[-top_k:][::-1]

        # Hybrid Consensus
        combined_indices = list(set(v_indices[0].tolist()) | set(b_indices.tolist()))
        
        results: List[str] = []
        for idx in combined_indices:
            if idx != -1 and idx < len(self.chunks):
                results.append(f"FROM {self.chunk_metadata[idx]['source']}:\n{self.chunks[idx]}")
        
        return results[:top_k]

class FileProcessor:
    @staticmethod
    def extract_content(file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".pdf":
                text_parts: List[str] = []
                with fitz.open(file_path) as doc:
                    for page in doc:
                        text_parts.append(cast(str, page.get_text("text")))
                return "\n".join(text_parts)
            elif ext == ".docx":
                doc = Document(file_path)
                return "\n".join([p.text for p in doc.paragraphs])
            elif ext in [".py", ".txt", ".md", ".json", ".js", ".ts", ".go", ".java", ".c", ".cpp"]:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
        return ""
