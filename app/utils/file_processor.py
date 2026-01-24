# filepath: app/utils/file_processor.py

import os
import fitz  # PyMuPDF
import zipfile
import shutil
import tempfile
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
        # Professional-grade embedding model (768-dim)
        self.encoder = SentenceTransformer('all-mpnet-base-v2')
        self.chunks: List[str] = []
        
        # Enhanced metadata to support Namespacing
        # category: 'vault' (docs), 'blueprint' (code), 'lab' (research)
        self.chunk_metadata: List[Dict[str, Any]] = []
        
        # High-level structural maps for code (blueprint)
        self.code_map: Dict[str, List[str]] = {} 
        
        self.bm25_indices: Dict[str, BM25Okapi] = {} # Namespace -> BM25
        self.vector_index: Optional[faiss.IndexFlatL2] = None

    def add_code_structure(self, filename: str, content: str):
        """Uses Tree-Sitter to map out code symbols for the 'Blueprint' namespace."""
        ext = os.path.splitext(filename)[1].lower()
        lang_map = {
            ".py": "python", ".js": "javascript", ".ts": "typescript", 
            ".go": "go", ".cpp": "cpp", ".c": "c", ".java": "java"
        }
        if ext not in lang_map: return

        try:
            language_name = lang_map[ext]
            language = get_language(language_name)
            parser = get_parser(language_name)
            tree = parser.parse(bytes(content, "utf8"))
            
            # Focused queries for structural mapping
            query_map = {
                "python": "(function_definition name: (identifier) @f) (class_definition name: (identifier) @c)",
                "javascript": "(function_declaration name: (identifier) @f) (class_declaration name: (identifier) @c)",
                "typescript": "(function_declaration name: (identifier) @f) (class_declaration name: (identifier) @c)",
                "java": "(method_declaration name: (identifier) @f) (class_declaration name: (identifier) @c)",
            }
            query_str = query_map.get(language_name)
            if not query_str: return

            query = language.query(query_str)
            captures = query.captures(tree.root_node)
            
            signatures = []
            capture_items = captures.items() if isinstance(captures, dict) else captures
            for node, tag in capture_items:
                node_text = content[node.start_byte:node.end_byte]
                line = node_text.splitlines()[0]
                signatures.append(f"{tag.upper()}: {line}")
            
            if signatures:
                self.code_map[filename] = signatures
                logger.info(f"Blueprint: Mapped {len(signatures)} symbols in {filename}")
        except Exception as e:
            logger.warning(f"Tree-sitter error for {filename}: {e}")

    def add_text(self, text: str, source_name: str, namespace: str = "vault"):
        """Splits text into chunks and assigns to a specific Namespace."""
        if not text: return
        text = text.replace('\x00', '') 
        words = text.split()
        
        # Context-rich chunking
        chunk_size = 600  
        overlap = 150     
        
        if len(words) <= chunk_size:
            self.chunks.append(text)
            self.chunk_metadata.append({
                "source": source_name, 
                "namespace": namespace, 
                "chunk_idx": 0
            })
            return

        idx = 0
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            self.chunks.append(chunk)
            self.chunk_metadata.append({
                "source": source_name, 
                "namespace": namespace, 
                "chunk_idx": idx
            })
            idx += 1
            if i + chunk_size >= len(words): break

    def finalize(self):
        """Builds namespaced BM25 and a global Vector index."""
        if not self.chunks: return
        
        # 1. Build Namespaced BM25 (Deterministic Keyword Matching)
        namespaces = set(m['namespace'] for m in self.chunk_metadata)
        for ns in namespaces:
            ns_chunks = [
                self.chunks[i].lower().split() 
                for i, m in enumerate(self.chunk_metadata) if m['namespace'] == ns
            ]
            if ns_chunks:
                self.bm25_indices[ns] = BM25Okapi(ns_chunks)

        # 2. Global Vector Index (MPNet Semantic Matching)
        logger.info(f"Computing MPNet embeddings for {len(self.chunks)} chunks...")
        embeddings = self.encoder.encode(self.chunks, show_progress_bar=False)
        embeddings_np = np.array(embeddings).astype('float32')
        
        dimension = int(embeddings_np.shape[1])
        # Reset index to handle new data if finalize is called again
        self.vector_index = faiss.IndexFlatL2(dimension)
        self.vector_index.add(embeddings_np) # type: ignore
        logger.info("Intelligence indices finalized.")

    def get_full_code_map(self) -> str:
        """Returns the high-level structure of all indexed code."""
        if not self.code_map:
            return "No structured codebase found or Tree-Sitter mapping is empty."
        output = ["CODEBASE STRUCTURE MAP:"]
        for file, sigs in self.code_map.items():
            output.append(f"\nFILE: {file}")
            for s in sigs[:20]:
                output.append(f"  - {s}")
            if len(sigs) > 20:
                output.append(f"  ... and {len(sigs) - 20} more symbols.")
        return "\n".join(output)

    def search(self, query: str, namespace: Optional[str] = None, top_k: int = 8) -> List[str]:
        """Hybrid search with optional Namespace filtering."""
        if not self.chunks or self.vector_index is None:
            return []

        # 1. Vector Search
        query_vec = np.array(self.encoder.encode([query])).astype('float32')
        # We search more than top_k to allow for namespace filtering
        search_k = top_k * 5 if namespace else top_k * 2
        distances, v_indices = self.vector_index.search(query_vec, search_k) # type: ignore
        
        final_v_indices = []
        for idx in v_indices[0]:
            if idx == -1: continue
            if namespace and self.chunk_metadata[idx]['namespace'] != namespace:
                continue
            final_v_indices.append(idx)
            if len(final_v_indices) >= top_k: break

        # 2. Namespace BM25 Search (if applicable)
        final_b_indices = []
        if namespace and namespace in self.bm25_indices:
            tokenized_query = query.lower().split()
            bm25 = self.bm25_indices[namespace]
            scores = bm25.get_scores(tokenized_query)
            # Map top scores back to original global indices
            ns_global_map = [i for i, m in enumerate(self.chunk_metadata) if m['namespace'] == namespace]
            top_ns_indices = np.argsort(scores)[-top_k:][::-1]
            final_b_indices = [ns_global_map[i] for i in top_ns_indices if scores[i] > 0]

        # Consolidate results: Interleave BM25 and Vector to ensure variety
        combined_indices = []
        v_ptr, b_ptr = 0, 0
        seen = set()

        while len(combined_indices) < top_k and (v_ptr < len(final_v_indices) or b_ptr < len(final_b_indices)):
            if v_ptr < len(final_v_indices):
                idx = final_v_indices[v_ptr]
                if idx not in seen:
                    combined_indices.append(idx)
                    seen.add(idx)
                v_ptr += 1
            
            if len(combined_indices) < top_k and b_ptr < len(final_b_indices):
                idx = final_b_indices[b_ptr]
                if idx not in seen:
                    combined_indices.append(idx)
                    seen.add(idx)
                b_ptr += 1
        
        results: List[str] = []
        for idx in combined_indices:
            m = self.chunk_metadata[idx]
            results.append(f"[Namespace: {m['namespace']} | File: {m['source']}]:\n{self.chunks[idx]}")
        
        return results

class FileProcessor:
    @staticmethod
    def extract_content(file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".pdf":
                text_parts = []
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
            logger.error(f"Extraction error for {file_path}: {e}")
        return ""

    @staticmethod
    def process_zip(zip_path: str) -> List[Dict[str, str]]:
        """Extracts ZIP and returns list of {'name': str, 'content': str, 'namespace': str}"""
        results = []
        temp_dir = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, temp_dir)
                    ext = os.path.splitext(file)[1].lower()
                    
                    # Ignore hidden/system files
                    if file.startswith('.') or '__pycache__' in root:
                        continue

                    content = FileProcessor.extract_content(full_path)
                    if content:
                        # Auto-categorize
                        ns = "blueprint" if ext in [".py", ".js", ".ts", ".go", ".java", ".cpp", ".c", ".h"] else "vault"
                        results.append({
                            "name": rel_path,
                            "content": content,
                            "namespace": ns
                        })
            return results
        finally:
            shutil.rmtree(temp_dir)