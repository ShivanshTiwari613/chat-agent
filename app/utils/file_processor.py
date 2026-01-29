# filepath: app/utils/file_processor.py

import os
import fitz  # PyMuPDF
import zipfile
import shutil
import tempfile
import io
from docx import Document
from typing import List, Dict, Any, Optional, cast, Union
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tree_sitter_languages import get_language, get_parser
from app.utils.logger import logger

class EphemeralFileIndex:
    def __init__(self):
        # Professional-grade embedding model (768-dim)
        self.encoder = SentenceTransformer('all-mpnet-base-v2')
        self.chunks: List[str] = []
        
        # Enhanced metadata to support Namespacing and Source tracking
        self.chunk_metadata: List[Dict[str, Any]] = []
        
        # High-level structural maps for code (blueprint)
        self.code_map: Dict[str, List[str]] = {} 
        
        self.bm25_indices: Dict[str, BM25Okapi] = {} # Namespace -> BM25
        
        # STAGE 1 IMPLEMENTATION: Namespaced Vector Indices
        self.vector_indices: Dict[str, faiss.IndexFlatL2] = {}
        
        # Map to track global chunk indices for each namespace index
        self.ns_to_global_map: Dict[str, List[int]] = {}

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
            if isinstance(captures, list):
                for node, tag in captures:
                    node_text = content[node.start_byte:node.end_byte]
                    line = node_text.splitlines()[0]
                    signatures.append(f"{tag.upper()}: {line}")
            
            if signatures:
                self.code_map[filename] = signatures
                logger.info(f"Blueprint: Mapped {len(signatures)} symbols in {filename}")
        except Exception as e:
            logger.warning(f"Tree-sitter error for {filename}: {e}")

    def add_text(self, text: str, source_name: str, namespace: str = "vault"):
        """Splits text into chunks and assigns to a specific Namespace and Source."""
        if not text: return
        text = text.replace('\x00', '') 
        words = text.split()
        
        chunk_size = 600  
        overlap = 150     
        
        if len(words) <= chunk_size:
            self.chunks.append(text)
            self.chunk_metadata.append({
                "source": source_name, 
                "namespace": namespace, 
                "chunk_idx": 0
            })
        else:
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
        """Builds namespaced BM25 and separate Vector indices."""
        if not self.chunks: return
        
        namespaces = set(m['namespace'] for m in self.chunk_metadata)
        self.bm25_indices = {}
        self.vector_indices = {}
        self.ns_to_global_map = {}

        logger.info(f"Computing embeddings for {len(self.chunks)} chunks across {len(namespaces)} namespaces...")
        all_embeddings = self.encoder.encode(self.chunks, show_progress_bar=False)

        for ns in namespaces:
            ns_global_indices = [i for i, m in enumerate(self.chunk_metadata) if m['namespace'] == ns]
            self.ns_to_global_map[ns] = ns_global_indices

            ns_chunks_tokenized = [self.chunks[i].lower().split() for i in ns_global_indices]
            self.bm25_indices[ns] = BM25Okapi(ns_chunks_tokenized)

            ns_embeddings = np.array([all_embeddings[i] for i in ns_global_indices]).astype('float32')
            dimension = ns_embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            # type: ignore added to handle Faiss SWIG bindings
            index.add(ns_embeddings) # type: ignore
            self.vector_indices[ns] = index

        logger.info("Namespaced Intelligence indices finalized.")

    def get_full_code_map(self) -> str:
        if not self.code_map:
            return "No structured codebase found or Tree-Sitter mapping is empty."
        output = ["CODEBASE STRUCTURE MAP:"]
        for file, sigs in self.code_map.items():
            output.append(f"\nFILE: {file}")
            for s in sigs[:20]:
                output.append(f"  - {s}")
        return "\n".join(output)

    def search(self, query: str, namespace: Optional[str] = None, source_filter: Optional[str] = None, top_k: int = 8) -> List[str]:
        """
        Performs search with support for Namespace and Source (Filename) filtering.
        Corrected Pylance issue with FAISS search signature.
        """
        if not self.chunks: return []

        target_namespaces = [namespace] if namespace else list(self.vector_indices.keys())
        v_results_global_indices = []
        b_results_global_indices = []

        query_vec = np.array(self.encoder.encode([query])).astype('float32')
        tokenized_query = query.lower().split()

        for ns in target_namespaces:
            if ns not in self.vector_indices: continue
            ns_index = self.vector_indices[ns]
            ns_map = self.ns_to_global_map[ns]
            
            # Apply Vector Search
            # We cast ns_index to Any to resolve the Pylance/FAISS signature mismatch issue
            distances, local_indices = cast(Any, ns_index).search(query_vec, top_k * 2) 
            
            for loc_idx in local_indices[0]:
                if loc_idx != -1:
                    global_idx = ns_map[loc_idx]
                    # If source_filter is provided, skip chunks that don't match the filename
                    if source_filter and self.chunk_metadata[global_idx]['source'] != source_filter:
                        continue
                    v_results_global_indices.append(global_idx)

            # Apply Keyword Search
            if ns in self.bm25_indices:
                scores = self.bm25_indices[ns].get_scores(tokenized_query)
                top_loc_indices = np.argsort(scores)[-top_k*2:][::-1]
                for loc_idx in top_loc_indices:
                    if scores[loc_idx] > 0:
                        global_idx = ns_map[loc_idx]
                        if source_filter and self.chunk_metadata[global_idx]['source'] != source_filter:
                            continue
                        b_results_global_indices.append(global_idx)

        # Interleave results (Hybrid Reranking)
        combined_indices = []
        v_ptr, b_ptr = 0, 0
        seen = set()
        while len(combined_indices) < top_k and (v_ptr < len(v_results_global_indices) or b_ptr < len(b_results_global_indices)):
            if v_ptr < len(v_results_global_indices):
                idx = v_results_global_indices[v_ptr]
                if idx not in seen:
                    combined_indices.append(idx)
                    seen.add(idx)
                v_ptr += 1
            if len(combined_indices) < top_k and b_ptr < len(b_results_global_indices):
                idx = b_results_global_indices[b_ptr]
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
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

    @staticmethod
    def extract_content(file_path: str) -> Dict[str, Any]:
        """
        Extracts content from a file. 
        Returns a dictionary containing 'text' and 'images' (list of bytes).
        """
        ext = os.path.splitext(file_path)[1].lower()
        result = {"text": "", "images": []}
        
        try:
            if ext == ".pdf":
                text_parts = []
                with fitz.open(file_path) as doc:
                    for page_index in range(len(doc)):
                        page = doc[page_index]
                        text_parts.append(page.get_text("text"))
                        
                        # Extract Images from PDF
                        for img_index, img in enumerate(page.get_images(full=True)):
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            result["images"].append({
                                "name": f"pdf_img_p{page_index}_{img_index}.png",
                                "content": image_bytes
                            })
                result["text"] = "\n".join(text_parts)
                
            elif ext == ".docx":
                doc = Document(file_path)
                result["text"] = "\n".join([p.text for p in doc.paragraphs])
                
            elif ext in FileProcessor.IMAGE_EXTENSIONS:
                with open(file_path, "rb") as f:
                    result["images"].append({
                        "name": os.path.basename(file_path),
                        "content": f.read()
                    })
                    
            elif ext in [".py", ".txt", ".md", ".json", ".js", ".ts", ".go", ".java", ".c", ".cpp"]:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    result["text"] = f.read()
        except Exception as e:
            logger.error(f"Extraction error for {file_path}: {e}")
            
        return result

    @staticmethod
    def process_zip(zip_path: str) -> List[Dict[str, Any]]:
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
                    
                    if file.startswith('.') or '__pycache__' in root:
                        continue

                    extracted = FileProcessor.extract_content(full_path)
                    if extracted["text"] or extracted["images"]:
                        ns = "blueprint" if ext in [".py", ".js", ".ts", ".go", ".java", ".cpp", ".c", ".h"] else "vault"
                        results.append({
                            "name": rel_path,
                            "text": extracted["text"],
                            "images": extracted["images"],
                            "namespace": ns
                        })
            return results
        finally:
            shutil.rmtree(temp_dir)