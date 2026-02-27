import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
import logging
import hashlib
import gc
from typing import List, Dict, Tuple, Any
from datetime import datetime
import pandas as pd
from pathlib import Path
import torch
import pdfplumber
import re
# from rank_bm25 import BM25Okapi

# Unstructured imports
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# LangChain / Chroma imports
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Import your shared embedding service (user provided)
from embedding_service import embedding_service

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text_content(text: str) -> str:
    if not text:
        return ""

    # Remove script/style blocks
    text = re.sub(r'<(script|style).*?>.*?</\1>', '', text, flags=re.DOTALL|re.IGNORECASE)
    # Remove all HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove LaTeX inline math markers \( ... \) and \[ ... \]
    text = re.sub(r'\\\((.*?)\\\)', r'\1', text)
    text = re.sub(r'\\\[(.*?)\\\]', r'\1', text)
    # Remove extra escape characters
    text = text.replace("\\", "")
    # Collapse multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Preserve bullet points on new lines
    text = text.replace(" •", "\n•")
    text = text.replace(" -", "\n-")  # Handle dash bullets too
    text = text.replace(" *", "\n*")  # Handle asterisk bullets
    
    return text.strip()

# ----------------------------- Helpers ---------------------------------

def _get_element_text(el) -> str:
    """Return the text content of an Unstructured element robustly."""
    try:
        if hasattr(el, "get_text"):
            return el.get_text() or ""
        if hasattr(el, "text"):
            return el.text or ""
        # Sometimes partition_pdf returns plain strings
        return str(el) or ""
    except Exception:
        try:
            return str(el)
        except Exception:
            return ""


def _get_element_page(el) -> int:
    """Return a page number from element metadata (works for dict or object)."""
    try:
        meta = None
        if hasattr(el, "metadata"):
            meta = getattr(el, "metadata")
        elif hasattr(el, "meta"):
            meta = getattr(el, "meta")
        if isinstance(meta, dict):
            return int(meta.get("page_number") or meta.get("page") or 1)
        if meta and hasattr(meta, "page_number"):
            return int(getattr(meta, "page_number", 1))
    except Exception:
        pass
    return 1

def _get_element_title(el) -> str:
    """Best-effort fetch of a section/title from Unstructured metadata."""
    try:
        md = getattr(el, "metadata", None)
        # If metadata is an object with to_dict()
        if hasattr(md, "to_dict"):
            d = md.to_dict()
            return d.get("title") or d.get("section_title") or d.get("category") if d else ""
        # If metadata is a dict
        if isinstance(md, dict):
            return md.get("title") or md.get("section_title") or md.get("category") or ""
    except Exception:
        pass
    return ""

##### baai
# ---------------------- Embedding wrapper (robust) ----------------------  REAL EMBEDDING
class OfflineHuggingFaceEmbeddings:
    """Wrapper that adapts your embedding_service client to the interface expected by Chroma.
    Assumes embedding_service.get_model().encode(texts) exists.
    """
    def __init__(self):
        # if embedding_service._model is None:                                             # commented out for multiple loading
        #     logger.info("Initializing embedding for manual processing")
        self.client = embedding_service.get_model()

    def _ensure_list(self, arr: Any) -> Any:
        # Convert numpy arrays if present
        try:
            import numpy as np
            if isinstance(arr, np.ndarray):
                return arr.tolist()
        except Exception:
            pass
        return arr

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Encode in batches (client may support batching internally)
        out = self.client.encode(texts)
        out = self._ensure_list(out)
        return list(out)

    def embed_query(self, text: str) -> List[float]:
        out = self.client.encode([text])
        out = self._ensure_list(out)
        # return the first vector
        return list(out[0]) if isinstance(out, (list, tuple)) and len(out) > 0 else list(out)
##################
# class OfflineHuggingFaceEmbeddings:
#     def __init__(self):
#         self.client = embedding_service.get_model()

#     def _ensure_list(self, arr):
#         try:
#             import numpy as np
#             if isinstance(arr, np.ndarray):
#                 return arr.tolist()
#         except Exception:
#             pass
#         return arr

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         out = self.client.encode(texts)
#         out = self._ensure_list(out)
#         return list(out)

#     def embed_query(self, text: str) -> List[float]:
#         out = self.client.encode([text])
#         out = self._ensure_list(out)
#         return list(out[0]) if isinstance(out, (list, tuple)) and len(out) > 0 else list(out)
####### gemma #######
# class OfflineHuggingFaceEmbeddings:
#     """Wrapper that adapts EmbeddingGemma to the interface expected by Chroma."""
    
#     def __init__(self):
#         self.client = embedding_service.get_model()

#     def _ensure_list(self, arr: Any) -> Any:
#         # Convert numpy arrays if present
#         try:
#             import numpy as np
#             if isinstance(arr, np.ndarray):
#                 return arr.tolist()
#         except Exception:
#             pass
#         return arr

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         # Use encode_document for documents
#         out = self.client.encode_document(texts)
#         out = self._ensure_list(out)
#         return list(out)

#     def embed_query(self, text: str) -> List[float]:
#         # Use encode_query for queries
#         out = self.client.encode_query(text)
#         out = self._ensure_list(out)
#         return list(out) if isinstance(out, (list, tuple)) else list(out)
##########################
# class OfflineHuggingFaceEmbeddings:
#     def __init__(self):
#         self.client = embedding_service.get_model()

#     def _ensure_list(self, arr):
#         try:
#             import numpy as np
#             if isinstance(arr, np.ndarray):
#                 return arr.tolist()
#         except Exception:
#             pass
#         return arr

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         # BGE-M3 returns dict, extract only dense embeddings
#         result = self.client.encode(texts, max_length=8192)
#         embeddings = result['dense_vecs']
#         return self._ensure_list(embeddings)

#     def embed_query(self, text: str) -> List[float]:
#         # BGE-M3 encode single query, extract dense embedding
#         result = self.client.encode([text], max_length=8192)
#         embedding = result['dense_vecs'][0]
#         return self._ensure_list(embedding)


# ---------------------- GPU / Memory Manager ----------------------------
class GPUMemoryManager:
    @staticmethod
    def cleanup():
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            gc.collect()


# ---------------------- PDF Extraction ---------------------------------
class EnhancedPDFExtractor:
    """Extract text using Unstructured and tables using pdfplumber.
    Produces a list of langchain.schema.Document objects with consistent metadata.
    """
    def extract_from_pdf(self, pdf_path: str) -> List[Document]:
        documents: List[Document] = []

        # 1) Text extraction via unstructured.partition_pdf
        elements = []
        for strategy in ["fast", "hi_res", "auto"]:
            try:
                logger.info("Unstructured parsing (text) for %s", pdf_path)
                elements = partition_pdf(
                    filename=pdf_path,
                    strategy=strategy,        #### hi_res
                    infer_table_structure=False,
                    include_metadata=True,
                )
                if elements:  # Success!
                    logger.info("Successfully extracted with strategy: %s", strategy)
                    break
            except Exception as e:
                logger.warning("Strategy '%s' failed: %s", strategy, str(e))
                continue    

            # Filter out non-text categories defensively
        if elements:
                text_elements = []
                for el in elements:
                    try:
                        cat = getattr(el, 'category', '') or ''
                        if isinstance(cat, str) and cat.lower() in ("image", "figurecaption", "header", "footer", "pagebreak", "table"):
                            continue
                    except Exception:
                        pass
                    text_elements.append(el)

                # print(f"\n=== DIAGNOSTIC: Elements before chunking ===")
                # for el in text_elements:
                    # print(f"PAGE: {_get_element_page(el)} | CAT: {getattr(el, 'category', '')} | TEXT: {_get_element_text(el)[:80]}")
                # print(f"=== End diagnostic (total elements: {len(text_elements)}) ===\n")            

            # chunk by title with fallback
                try:
                    chunks = chunk_by_title(
                        text_elements,
                        max_characters=2000,
                        new_after_n_chars=1600,
                        combine_text_under_n_chars=200,
                        multipage_sections=True
                    )
                except Exception:
                    # fallback: treat each element as chunk
                    chunks = text_elements

                for chunk in chunks:
                    text = _get_element_text(chunk)
                    page = _get_element_page(chunk)
                    text = clean_text_content(text)
                    if text and text.strip():
                        metadata = {
                            'page': int(page),
                            'content_type': 'text',
                             'is_table': False,
                            'source': pdf_path
                            }
                        documents.append(Document(page_content=text, metadata=metadata))
                logger.info("Unstructured: extracted %d text chunks", len([d for d in documents if not d.metadata.get('is_table')]))
        else:
            logger.exception("Unstructured text extraction failed for %s", pdf_path)

        # 2) Table extraction via pdfplumber
        try:
            logger.info("pdfplumber parsing (tables) for %s", pdf_path)
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        tables = page.extract_tables() or []
                        if not tables:
                            continue
                        logger.info("Page %d: found %d tables", page_num, len(tables))
                        for table_idx, table in enumerate(tables):
                            try:
                                table_text = self._format_table_perfectly(table, page_num, table_idx)
                                if table_text:
                                    table_text = clean_text_content(table_text)
                                    doc_meta = {
                                        'page': page_num,
                                        'content_type': 'table',
                                        'is_table': True,
                                        'table_index': table_idx,
                                        'source': pdf_path
                                    }
                                    documents.append(Document(page_content=table_text, metadata=doc_meta))
                                    # Add rows
                                    self._create_table_row_documents(table, page_num, table_idx, pdf_path, documents)  ######uncomment
                            except Exception:
                                logger.exception("Failed formatting table %s on page %d", table_idx, page_num)
                    except Exception:
                        logger.exception("Table extraction error on page %d", page_num)
        except Exception:
            logger.exception("pdfplumber failed for %s", pdf_path)

        logger.info("Total extracted chunks from %s: %d", pdf_path, len(documents))
        return documents

    def _format_table_perfectly(self, table: List[List], page_num: int, table_idx: int) -> str:
        """Format table in perfect structure for LLM - keeping your existing logic"""
        if not table or len(table) == 0:
            return ""
        
        formatted_lines = []
        formatted_lines.append(f"TABLE {table_idx + 1} (Page {page_num}):")
        formatted_lines.append("=" * 50)
        
        headers = table[0] if table else []
        
        for row_idx, row in enumerate(table):
            if row_idx == 0:
                formatted_lines.append("HEADERS: " + " | ".join(str(cell or "").strip() for cell in row))
                formatted_lines.append("-" * 50)
            else:
                row_data = []
                for col_idx, cell in enumerate(row):
                    if col_idx < len(headers) and headers[col_idx]:
                        header = str(headers[col_idx]).strip()
                        value = str(cell or "").strip()
                        if header and value:
                            row_data.append(f"{header}: {value}")
                
                if row_data:
                    formatted_lines.append(" | ".join(row_data))
        
        return "\n".join(formatted_lines)
    def _create_table_row_documents(self, table: List[List], page_num: int, table_idx: int, 
                               pdf_path: str, documents: List[Document]):
        """Create individual searchable documents for each table row"""
        if not table or len(table) < 2:
            return
        
        headers = table[0]
        
        for row_idx, row in enumerate(table[1:], 1):
            row_pairs = []
            for col_idx, cell in enumerate(row):
                if col_idx < len(headers) and headers[col_idx]:
                    header = str(headers[col_idx]).strip()
                    value = str(cell or "").strip()
                    if header and value:
                        row_pairs.append(f"{header}: {value}")
            
            if row_pairs:
                row_text = f"Table Row (Page {page_num}): " + " | ".join(row_pairs)
                
                row_metadata = {
                    'page': page_num,
                    'content_type': 'table_row',
                    'is_table_row': True,
                    'table_index': table_idx,
                    'row_index': row_idx,
                    'source': pdf_path
                }
                
                # Add column-specific metadata for better filtering
                for col_idx, cell in enumerate(row):
                    if col_idx < len(headers) and headers[col_idx]:
                        header_clean = str(headers[col_idx]).strip().lower().replace(' ', '_')
                        value_clean = str(cell or "").strip()
                        if header_clean and value_clean:
                            row_metadata[f'col_{header_clean}'] = value_clean
                
                row_doc = Document(
                    page_content=row_text,
                    metadata=row_metadata
                )
                documents.append(row_doc)

    


# ---------------------- Main Processor ---------------------------------
class FixedTableManualProcessor:
    def __init__(self, db_path: str = "./fixed_table_manual_db"):
        self.db_path = db_path
        self.collection_name = "fixed_table_manuals"
        self.supported_extensions = {'.pdf', '.csv', '.txt', '.xlsx', '.xls'}

        logger.info("Initializing processor...")
        self._initialize_components()

    def delete_document_by_name(self, document_name: str) -> bool:
        """Delete all chunks for a specific document"""
        try:
            collection = getattr(self.vectorstore, '_collection', None)
            if collection:
                data = collection.get(where={"document_name": document_name})
                if data['ids']:
                    collection.delete(ids=data['ids'])
                    self.vectorstore.persist()
                    return True
            return False
        except Exception:
            return False    

    def _initialize_components(self):
        self.embeddings = OfflineHuggingFaceEmbeddings()
        self.pdf_extractor = EnhancedPDFExtractor()
        self.gpu_manager = GPUMemoryManager()

        os.makedirs(self.db_path, exist_ok=True)
        self.vectorstore = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )

    def process_document(self, file_path: str) -> Dict:
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        doc_name = file_path.name

        if file_extension not in self.supported_extensions:
            return {'status': 'error', 'document': doc_name, 'error': f'Unsupported: {file_extension}'}

        try:
            self.gpu_manager.cleanup()

            # Check if already processed (using metadata lookup)
            doc_bytes = file_path.read_bytes()
            doc_hash = hashlib.md5(doc_bytes).hexdigest()
            if self._is_document_processed(doc_hash):
                return {'status': 'already_exists', 'document': doc_name, 'doc_hash': doc_hash}

            # Extract
            documents: List[Document] = []
            if file_extension == '.pdf':
                documents = self.pdf_extractor.extract_from_pdf(str(file_path))
            elif file_extension in ['.xlsx', '.xls']:
                documents = self._extract_excel(str(file_path))
            elif file_extension == '.csv':
                documents = self._extract_csv(str(file_path))
            elif file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    raw = f.read()
                    cleaned = clean_text_content(raw)
                    documents = [Document(page_content=cleaned, metadata={'source': str(file_path)})]

            if not documents:
                return {'status': 'error', 'document': doc_name, 'error': 'No content extracted'}

            # Add global metadata fields and sanitize
            for doc in documents:
                md = doc.metadata or {}
                md.update({
                    'document_name': doc_name,
                    'file_extension': file_extension,
                    'doc_hash': doc_hash,
                    'timestamp': datetime.now().isoformat()
                })
                # sanitize metadata values
                sanitized = {}
                for k, v in md.items():
                    if isinstance(v, (str, int, float, bool, list, dict)):
                        sanitized[k] = v
                    else:
                        sanitized[k] = str(v)
                doc.metadata = sanitized

            # Add to vectorstore
            self._add_to_vectorstore_batched(documents)

            self.gpu_manager.cleanup()

            return {
                'status': 'success',
                'document': doc_name,
                'doc_hash': doc_hash,
                'chunks': len(documents),
                'stats': {
                    'total_chunks': len(documents),
                    'table_chunks': sum(1 for d in documents if d.metadata.get('is_table')),
                    'text_chunks': sum(1 for d in documents if not d.metadata.get('is_table'))
                }
            }
        except Exception:
            logger.exception("Error processing %s", doc_name)
            return {'status': 'error', 'document': doc_name, 'error': 'processing_failed'}

    def _extract_excel(self, file_path: str) -> List[Document]:
        documents: List[Document] = []
        xl_file = pd.ExcelFile(file_path)
        for sheet in xl_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet)
            text = f"Excel Sheet: {sheet}\n\n"
            for _, row in df.iterrows():
                text += " | ".join(f"{c}: {v}" for c, v in row.items() if pd.notna(v)) + "\n"
            cleaned = clean_text_content(text)    
            documents.append(Document(page_content=cleaned, metadata={'source': file_path, 'is_table': True}))
        return documents

    def _extract_csv(self, file_path: str) -> List[Document]:
        df = pd.read_csv(file_path)
        text = f"CSV File: {os.path.basename(file_path)}\n\n"
        for _, row in df.iterrows():
            text += " | ".join(f"{c}: {v}" for c, v in row.items() if pd.notna(v)) + "\n"
        cleaned = clean_text_content(text)
        return [Document(page_content=cleaned, metadata={'source': file_path, 'is_table': True})]

    def _add_to_vectorstore_batched(self, documents: List[Document]):
        # Build arrays
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [f"{doc.metadata.get('doc_hash','unknown')}_{i}" for i, doc in enumerate(documents)]

        batch_size = 64
        for i in range(0, len(texts), batch_size):
            end = min(i + batch_size, len(texts))
            self.vectorstore.add_texts(
                texts=texts[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end]
            )
        self.vectorstore.persist()

    def _is_document_processed(self, doc_hash: str) -> bool:
        try:
            # Prefer reading collection metadata if available
            collection = getattr(self.vectorstore, '_collection', None)
            if collection:
                data = collection.get()
                for md in data.get('metadatas', []):
                    if md and md.get('doc_hash') == doc_hash:
                        return True
            # fallback: try similarity_search with metadata filter
            try:
                results = self.vectorstore.similarity_search("_exists_check", k=1, filter={"doc_hash": doc_hash})
                return len(results) > 0
            except Exception:
                return False
        except Exception:
            logger.exception("_is_document_processed failed")
            return False

    def query_manuals(self, question: str, n_results: int = 20, save_context: bool = True) -> Dict:
        try:
            self.gpu_manager.cleanup()

            # Primary search
            results = self.vectorstore.similarity_search_with_score(question, k=n_results)
            # Primary search - hybrid (semantic + keyword)
            # try: ################################
            #     results = self._hybrid_search(question, k=n_results)
            # except Exception as e:
            #     logger.warning(f"Hybrid search failed, falling back: {e}")
            #     results = self.vectorstore.similarity_search_with_score(question, k=n_results)
            
            # print(f"DEBUG: Question = '{question}'")
            # # print(f"DEBUG: Tech terms found = {tech_terms}")
            # print(f"DEBUG: Total results = {len(results)}")
            # for i, (doc, score) in enumerate(results[:3]):
            #     print(f"Result {i}: Score={score:.4f}")
            #     print(f"Content preview: {doc.page_content[:150]}...")
            #     print("---")    

            # Optional table-specific search
            if any(kw in question.lower() for kw in ['table', 'load', 'watts', 'specification']):
                try:
                    table_results = self.vectorstore.similarity_search_with_score(question, k=5, filter={"is_table": True})
                    results.extend(table_results)
                except Exception:
                    logger.exception("Table-filtered search failed")
                # Also search table rows for granular matches
                try:
                    row_results = self.vectorstore.similarity_search_with_score(
                        question, k=5, filter={"is_table_row": True}
                    )
                    results.extend(row_results)
                except Exception:
                    logger.exception("Table row search failed")    

            # Normalize scores into a 0..1 similarity (heuristic)
            def _to_similarity(raw: float) -> float:
                """Convert Chroma distance to similarity (0..1)."""
                if raw is None:
                    return 0.0
                try:
                    r = float(raw)
                    if r >= 0.0:
                        return 1.0 / (1.0 + r)  # distance → similarity
                    return max(0.0, min(1.0, r))  # already similarity
                except Exception:
                    return 0.0

            normalized: List[Tuple[Document, float, float]] = []
            for doc, raw in results:
                sim = _to_similarity(raw)
                normalized.append((doc, float(raw), sim))

             # Optional keyword bias for domain-specific matches
            q = question.lower()
            q_terms = set(re.findall(r"[a-z0-9\-]+", q))
            def _keyword_bonus(text: str) -> float:
                t = text.lower()
                bonus = 0.0
                # tech_terms = re.findall(r'[A-Z][a-z]*\d+|[A-Z]{2,}|[A-Z][a-z]+\s+[A-Z][a-z]+', question)        ###### added : matching
                # for term in tech_terms:
                #     if term in text:
                #         bonus += 0.15
                for key in ("low-duty", "low duty", "compressor"):
                    if key in t and key.replace("-", " ") in q:
                        bonus += 0.08
                    elif key in t:
                        bonus += 0.05
                hits = sum(1 for term in q_terms if term in t)
                if hits:
                    bonus += min(0.05, 0.01 * hits)
                return bonus

            normalized = [
                (doc, raw, min(1.0, sim + _keyword_bonus(doc.page_content)))
                for (doc, raw, sim) in normalized
            ]

            # Deduplicate by doc_hash+snippet
            seen = set()
            unique = []
            for doc, raw, sim in normalized:
                key = f"{doc.metadata.get('doc_hash','')}_{doc.page_content[:120]}"
                if key in seen:
                    continue
                seen.add(key)
                unique.append((doc, raw, sim))

            # Sort by sim descending
            unique.sort(key=lambda x: x[2], reverse=True)
            n_results = min(n_results, 9)
            final = unique[:n_results]

            # Build context and metadata list
            context_parts = []
            metadata_out = []
            for doc, raw, sim in final:
                doc_name = doc.metadata.get('document_name', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                if doc.metadata.get('is_table'):
                    context_parts.append(f"[STRUCTURED TABLE from {doc_name}, Page {page}]:\n{doc.page_content}")
                elif doc.metadata.get('is_table_row'):
                    context_parts.append(f"[TABLE ROW from {doc_name}, Page {page}]:\n{doc.page_content}")
                else:
                    context_parts.append(f"[TEXT from {doc_name}, Page {page}]:\n{doc.page_content}")

                clean_doc_name = doc_name
                if doc_name.startswith('manual_') and '_' in doc_name[7:]:
                    parts = doc_name.split('_', 2)  # Split into 3 parts max
                    if len(parts) >= 3:
                        clean_doc_name = parts[2]

                metadata_out.append({
                    'document': clean_doc_name,                                      ############ ADDED clean_doc_name
                    'page': page,
                    'raw_score': raw,
                    'similarity_score': round(sim, 4),
                    'is_table': doc.metadata.get('is_table', False),
                    'content_type': doc.metadata.get('content_type', 'unknown')
                })

            context_text = "\n\n".join(context_parts)

            # Save context for debugging with raw scores included
            context_file = None
            if save_context and context_text:
                context_file = self._save_context(question, context_text, metadata_out)
            from collections import defaultdict
            doc_pages = defaultdict(set)
            for meta in metadata_out:
                doc = meta['document']
                page = meta['page']
                if page != 'N/A':
                    doc_pages[doc].add(int(page))
            
            grouped_metadata = [
                {"doc": doc, "pages": sorted(list(pages))}
                for doc, pages in doc_pages.items()
            ]    

            self.gpu_manager.cleanup()
            return {
                'question': question,
                'context': context_text,
                'metadata': grouped_metadata,
                'metadata_detailed':metadata_out,
                'num_results': len(final),
                'context_file': context_file
            }
        except Exception:
            logger.exception("Query error")
            return {'question': question, 'error': 'query_failed', 'num_results': 0}

    def _save_context(self, question: str, context: str, metadata: List[Dict]) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"query_context_{timestamp}.txt"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write("="*80 + "\n")
                f.write(f"QUESTION: {question}\n")
                f.write("="*80 + "\n")
                f.write("METADATA:\n")
                for idx, meta in enumerate(metadata):
                    f.write(f"\nSource {idx+1}:\n")
                    for key, value in meta.items():
                        f.write(f"  {key}: {value}\n")
                f.write("="*80 + "\n")
                f.write("CONTEXT PASSED TO LLM:\n")
                f.write(context)
                f.write("\n" + "="*80 + "\n")
            logger.info("Context saved to %s", filename)
            return filename
        except Exception:
            logger.exception("Error saving context")
            return None
    


    def get_stats(self) -> Dict:
        try:
            collection = getattr(self.vectorstore, '_collection', None)
            if collection:
                data = collection.get()
                total = len(data.get('ids', []))
                tables = sum(1 for m in data.get('metadatas', []) if m.get('is_table'))
                return {
                    'total_chunks': total,
                    'table_chunks': tables,
                    'text_chunks': total - tables,
                    'status': 'ready'
                }
            return {'total_chunks': 0, 'status': 'ready'}
        except Exception:
            logger.exception("get_stats failed")
            return {'total_chunks': 0, 'status': 'error'}


# Keep your existing function names

def initialize_fixed_processor(db_path: str = "./fixed_table_manual_db") -> FixedTableManualProcessor:
    return FixedTableManualProcessor(db_path=db_path)


def process_fixed_manual_query(processor: FixedTableManualProcessor, question: str, llm_messages: List[Dict], generate_llm_response_func) -> Dict:
    processor.gpu_manager.cleanup()

    query_result = processor.query_manuals(question, n_results=10, save_context=True)

    if query_result.get('error'):
        return query_result

    if not query_result['context']:
        response = generate_llm_response_func(llm_messages, "no context")
        return {'question': question, 'answer': response, 'source': 'llm_knowledge', 'metadata': []}

    # best_score = max([m['similarity_score'] for m in query_result['metadata']], default=0.0)
    best_score = max([m['similarity_score'] for m in query_result.get('metadata_detailed', [])], default=0.0)

    # Threshold is based on normalized sim (0..1). Tune this after inspecting saved raw scores.
    if best_score < 0.3:
        response = generate_llm_response_func(llm_messages, "no relevant context")
        return {
            'question': question,
            'answer': response,
            'source': 'llm_knowledge',
            'metadata': [],
            'context_file': query_result.get('context_file')
        }

#     context_prompt = f"""
# You are a marine engineering assistant.

# TASK:
# Answer the QUESTION strictly using the CONTEXT.

# CONTEXT:
# {query_result['context']}

# QUESTION:
# {question}

# RULES:
# 1. Use only the information provided in CONTEXT.
# 2. Preserve spacing between words exactly as in CONTEXT. Do not add, remove, or alter spaces.
# 3. Copy numbers, units, and symbols exactly as written in CONTEXT. Do not change formatting.
# 4. Do not merge words, split words, or correct spellings from CONTEXT.
# 5. If the answer is not present in CONTEXT, state: "Not found in context."
# """
    context_prompt = f"""
You are a marine engineering assistant with expertise in technical documentation analysis.

CONTEXT (Multiple sources provided):
{query_result['context']}

QUESTION:
{question}

CRITICAL TASK:
1. READ ALL CONTEXT SECTIONS CAREFULLY - there are multiple sources
2. IDENTIFY which section directly answers the question (ignore tangentially related content)
3. EXTRACT the most relevant and complete answer
4. If multiple sources mention the topic, use the CLEAREST and MOST COMPLETE explanation

EXAMPLE:
Question: "What is X?"
Bad: Using a section that only mentions X in passing
Good: Using the section that defines and explains X in detail

NOW ANSWER THE QUESTION:
- Focus on the DEFINITION and EXPLANATION
- Ignore sections that only mention the term without explaining it
- Keep answer under 150 words
- Use technical terms exactly as written in context

Your answer:"""

    messages = [
        {'role': 'system', 'content': 'You are a helpful marine engineering assistant.'},
        {'role': 'user', 'content': context_prompt}
    ]


    response = generate_llm_response_func(messages, "manual query")
    processor.gpu_manager.cleanup()

    return {
        'question': question,
        'answer': response,
        'source': 'manual_context',
        'metadata': query_result['metadata'],
        'context_file': query_result.get('context_file')
    }
