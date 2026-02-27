"""
Marine Document Extractor v2 - With Heading Preservation
=========================================================
Key fix: Section headings are ALWAYS prepended to chunk content,
so the LLM always knows what section a chunk belongs to.

Changes from v1:
1. Headings are prepended to chunk text, not just stored in metadata
2. Unstructured 'Title' elements are never skipped
3. Hierarchical heading tracking (keeps parent headings)
4. Better category handling - only skips true page artifacts
"""

import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
import re
import pandas as pd
import logging
import hashlib
from typing import List, Dict, Tuple, Any, Optional, Set
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import gc
import pdfplumber
from langchain.schema import Document
import torch

from unstructured.partition.pdf import partition_pdf
from langchain_community.vectorstores import Chroma
from embedding_service import embedding_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# EMBEDDING WRAPPER (unchanged)
# =============================================================================

class OfflineHuggingFaceEmbeddings:
    def __init__(self):
        self.client = embedding_service.get_model()

    def _ensure_list(self, arr: Any) -> Any:
        try:
            import numpy as np
            if isinstance(arr, np.ndarray):
                return arr.tolist()
        except Exception:
            pass
        return arr

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out = self.client.encode(texts)
        out = self._ensure_list(out)
        return list(out)

    def embed_query(self, text: str) -> List[float]:
        out = self.client.encode([text])
        out = self._ensure_list(out)
        return list(out[0]) if isinstance(out, (list, tuple)) and len(out) > 0 else list(out)


# =============================================================================
# EXTRACTION: v2 with Heading Preservation
# =============================================================================

class MarineDocumentExtractor:
    """
    Zero-loss extraction with heading preservation.
    
    KEY CHANGE: Every chunk starts with its section heading(s) so the LLM
    always knows what the content is about.
    
    Example output:
        "7 Wiring the power supply
         7.1 Wire the power supply
         
         1. Remove the transmitter housing cover.
         2. Open the warning flap.
         3. Connect the power supply wires to terminals 9 and 10."
    """
    
    def __init__(
        self,
        target_chunk_size: int = 1600,
        max_chunk_size: int = 2200,
        min_chunk_size: int = 100,
        overlap_size: int = 150
    ):
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        
        # Heading patterns - ordered by specificity
        self.heading_patterns = [
            re.compile(r'^(\d+\.)+\d*\s+\S'),           # 3.1.2 Fuel System
            re.compile(r'^CHAPTER\s+\d+', re.I),         # CHAPTER 5
            re.compile(r'^SECTION\s+\d+', re.I),         # SECTION 3
            re.compile(r'^\d+\.\d+\s+\S'),               # 7.1 Wire the power
            re.compile(r'^\d+\s+[A-Z]\w'),               # 7 Wiring the power
            re.compile(r'^(APPENDIX|ANNEX)\s+[A-Z0-9]', re.I),
            re.compile(r'^[A-Z][A-Z\s\-]{3,40}$'),       # MAINTENANCE PROCEDURES
        ]
        
        # Heading hierarchy levels
        self.heading_level_patterns = [
            (re.compile(r'^CHAPTER\s+\d+', re.I), 1),
            (re.compile(r'^SECTION\s+\d+', re.I), 1),
            (re.compile(r'^\d+\s+[A-Z]'), 2),                    # 7 Wiring
            (re.compile(r'^\d+\.\d+\s+'), 3),                    # 7.1 Wire
            (re.compile(r'^\d+\.\d+\.\d+\s+'), 4),               # 7.1.1 Step
            (re.compile(r'^[A-Z][A-Z\s\-]{3,40}$'), 2),          # ALL CAPS HEADING
        ]
        
        self.preserve_start_patterns = [
            re.compile(r'^(CAUTION|WARNING|DANGER|NOTE|IMPORTANT)\s*:', re.I),
            re.compile(r'^(Step|STEP)\s+\d+', re.I),
            re.compile(r'^\d+\)\s+'),
            re.compile(r'^[a-z]\)\s+'),
        ]
        
        # Categories to SKIP - be very conservative, only skip true artifacts
        # self.skip_categories = {
        #     'pagebreak', 'pagenumber'
        # }
        self.skip_categories = {
        'pagebreak', 'pagenumber', 'table'
        }
        
        # Categories that might be headings
        self.heading_categories = {
            'title', 'header'
        }
    
    def extract_from_pdf(self, pdf_path: str) -> List[Document]:
        pdf_path = str(pdf_path)
        documents = []
        
        stats = {
            'elements_extracted': 0,
            'elements_chunked': 0,
            'tables_extracted': 0,
            'total_chars_extracted': 0,
            'total_chars_chunked': 0,
            'headings_found': 0
        }
        
        # Step 1: Text extraction
        text_documents, stats = self._extract_text_unstructured(pdf_path, stats)
        documents.extend(text_documents)
        
        # Step 2: Tables
        table_documents = self._extract_tables_pdfplumber(pdf_path)
        documents.extend(table_documents)
        stats['tables_extracted'] = len([d for d in table_documents if d.metadata.get('is_table')])
        
        logger.info(
            f"Extraction complete for {Path(pdf_path).name}: "
            f"{stats['elements_extracted']} elements -> {len(text_documents)} text chunks, "
            f"{stats['tables_extracted']} tables, "
            f"{stats['headings_found']} headings preserved, "
            f"chars: {stats['total_chars_extracted']} -> {stats['total_chars_chunked']}"
        )
        
        if stats['total_chars_extracted'] > 0:
            ratio = stats['total_chars_chunked'] / stats['total_chars_extracted']
            if ratio < 0.95:
                logger.warning(f"Potential content loss: {ratio:.1%} retention")
        
        return documents
    
    def _extract_text_unstructured(self, pdf_path: str, stats: Dict) -> Tuple[List[Document], Dict]:
        documents = []
        elements = []
        
        for strategy in ["fast", "hi_res", "auto"]:
            try:
                logger.info(f"Trying Unstructured strategy: {strategy}")
                elements = partition_pdf(
                    filename=pdf_path,
                    strategy=strategy,
                    infer_table_structure=False,
                    include_metadata=True,
                )
                if elements:
                    logger.info(f"Success with strategy: {strategy}, got {len(elements)} elements")
                    break
            except Exception as e:
                logger.warning(f"Strategy '{strategy}' failed: {e}")
                continue
        
        if not elements:
            logger.error(f"All extraction strategies failed for {pdf_path}")
            return documents, stats
        
        processed_elements = []
        
        for el in elements:
            text = self._get_element_text(el)
            if not text or not text.strip():
                continue
            
            page = self._get_element_page(el)
            category = self._get_element_category(el).lower()
            
            # Only skip true page artifacts - NEVER skip titles
            if category in self.skip_categories:
                continue
            
            # Determine if this is a heading
            is_heading = False
            if category in self.heading_categories:
                is_heading = True
            elif self._is_heading(text):
                is_heading = True
            
            cleaned = self._clean_text(text)
            if cleaned:
                processed_elements.append({
                    'text': cleaned,
                    'page': page,
                    'category': category,
                    'is_heading': is_heading,
                    'original_length': len(text)
                })
                stats['total_chars_extracted'] += len(cleaned)
                if is_heading:
                    stats['headings_found'] += 1
        
        stats['elements_extracted'] = len(processed_elements)
        
        if not processed_elements:
            return documents, stats
        
        # Chunk with heading preservation
        chunks = self._chunk_elements_with_headings(processed_elements, pdf_path)
        chunks = self._add_overlap(chunks)
        
        for chunk in chunks:
            doc = Document(
                page_content=chunk['text'],
                metadata={
                    'page': chunk['page'],
                    'content_type': 'text',
                    'is_table': False,
                    'section_heading': chunk.get('heading', ''),
                    'section_context': chunk.get('section', ''),
                    'source': pdf_path,
                    'has_overlap': chunk.get('has_overlap', False)
                }
            )
            documents.append(doc)
            stats['total_chars_chunked'] += len(chunk['text'])
        
        stats['elements_chunked'] = len(documents)
        return documents, stats
    
    def _chunk_elements_with_headings(self, elements: List[Dict], source: str) -> List[Dict]:
        """
        Chunk elements with HEADING PRESERVATION.
        
        Key difference from v1: 
        - Track active headings at each level
        - When starting a new chunk, prepend the current heading hierarchy
        - This ensures every chunk knows what section it belongs to
        """
        chunks = []
        
        # Track heading hierarchy: {level: heading_text}
        active_headings = {}
        current_chunk = {
            'text': '',
            'page': elements[0]['page'] if elements else 1,
            'section': '',
            'heading': ''
        }
        
        for el in elements:
            text = el['text']
            page = el['page']
            is_heading = el.get('is_heading', False)
            
            if is_heading:
                level = self._get_heading_level(text)
                
                # Update heading hierarchy - clear lower levels
                active_headings[level] = text.strip()
                keys_to_remove = [k for k in active_headings if k > level]
                for k in keys_to_remove:
                    del active_headings[k]
                
                # If current chunk has enough content, save it
                if current_chunk['text'].strip() and len(current_chunk['text']) >= self.min_chunk_size:
                    chunks.append(current_chunk.copy())
                    # Start new chunk WITH heading
                    heading_prefix = self._build_heading_prefix(active_headings)
                    current_chunk = {
                        'text': heading_prefix,
                        'page': page,
                        'section': text.strip()[:100],
                        'heading': heading_prefix.strip()
                    }
                else:
                    # Chunk too small - add heading to existing chunk
                    separator = '\n\n' if current_chunk['text'] else ''
                    current_chunk['text'] += separator + text
                    current_chunk['section'] = text.strip()[:100]
                    current_chunk['heading'] = self._build_heading_prefix(active_headings).strip()
                continue
            
            # Regular content - check if we should preserve with next
            should_preserve = self._should_preserve_with_next(text)
            
            separator = '\n\n' if current_chunk['text'] else ''
            potential_text = current_chunk['text'] + separator + text
            potential_length = len(potential_text)
            
            # Case 1: Fits comfortably
            if potential_length <= self.target_chunk_size:
                current_chunk['text'] = potential_text
                current_chunk['page'] = current_chunk.get('page') or page
                continue
            
            # Case 2: Exceeds target but under max, and should preserve
            if potential_length <= self.max_chunk_size and should_preserve:
                current_chunk['text'] = potential_text
                continue
            
            # Case 3: Need to start new chunk
            if current_chunk['text'].strip():
                chunks.append(current_chunk.copy())
            
            # NEW CHUNK: Always prepend current heading hierarchy
            heading_prefix = self._build_heading_prefix(active_headings)
            
            if len(text) <= self.max_chunk_size:
                current_chunk = {
                    'text': heading_prefix + text if heading_prefix else text,
                    'page': page,
                    'section': current_chunk.get('section', ''),
                    'heading': heading_prefix.strip()
                }
            else:
                # Element too large - split it
                split_chunks = self._split_large_text(text, page, 
                    current_chunk.get('section', ''), heading_prefix)
                chunks.extend(split_chunks[:-1])
                if split_chunks:
                    current_chunk = split_chunks[-1]
                else:
                    current_chunk = {
                        'text': heading_prefix,
                        'page': page,
                        'section': current_chunk.get('section', ''),
                        'heading': heading_prefix.strip()
                    }
        
        # Don't forget last chunk
        if current_chunk['text'].strip():
            chunks.append(current_chunk)
        
        return chunks
    
    def _build_heading_prefix(self, active_headings: Dict[int, str]) -> str:
        """
        Build a heading prefix from the active heading hierarchy.
        
        Example output:
            "7 Wiring the power supply\n7.1 Wire the power supply\n\n"
        """
        if not active_headings:
            return ''
        
        sorted_levels = sorted(active_headings.keys())
        heading_lines = [active_headings[level] for level in sorted_levels]
        
        return '\n'.join(heading_lines) + '\n\n'
    
    def _get_heading_level(self, text: str) -> int:
        """Determine the hierarchical level of a heading."""
        text = text.strip()
        first_line = text.split('\n')[0].strip()
        
        for pattern, level in self.heading_level_patterns:
            if pattern.match(first_line):
                return level
        
        return 2  # Default level
    
    def _split_large_text(self, text: str, page: int, section: str, 
                          heading_prefix: str = '') -> List[Dict]:
        """Split text that exceeds max chunk size, prepending heading to first chunk."""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_text = heading_prefix if heading_prefix else ''
        is_first = True
        
        for sentence in sentences:
            potential = current_text + (' ' if current_text and not current_text.endswith('\n') else '') + sentence
            
            if len(potential) <= self.target_chunk_size:
                current_text = potential
            else:
                if current_text.strip():
                    chunks.append({
                        'text': current_text.strip(),
                        'page': page,
                        'section': section,
                        'heading': heading_prefix.strip() if is_first else ''
                    })
                    is_first = False
                
                if len(sentence) > self.max_chunk_size:
                    words = sentence.split()
                    current_text = heading_prefix if not chunks else ''
                    for word in words:
                        potential = current_text + (' ' if current_text else '') + word
                        if len(potential) <= self.target_chunk_size:
                            current_text = potential
                        else:
                            if current_text.strip():
                                chunks.append({
                                    'text': current_text.strip(),
                                    'page': page,
                                    'section': section,
                                    'heading': ''
                                })
                            current_text = word
                else:
                    current_text = sentence
        
        if current_text.strip():
            chunks.append({
                'text': current_text.strip(),
                'page': page,
                'section': section,
                'heading': heading_prefix.strip() if is_first else ''
            })
        
        return chunks
    
    def _add_overlap(self, chunks: List[Dict]) -> List[Dict]:
        if not chunks or self.overlap_size <= 0:
            return chunks
        
        enhanced = []
        for i, chunk in enumerate(chunks):
            new_chunk = chunk.copy()
            
            if i > 0:
                prev_text = chunks[i - 1]['text']
                prev_page = chunks[i - 1].get('page', 0)
                curr_page = chunk.get('page', 0)
                
                if abs(curr_page - prev_page) <= 1:
                    if len(prev_text) > self.overlap_size:
                        overlap = prev_text[-self.overlap_size:]
                        period_pos = overlap.find('. ')
                        if period_pos > 0 and period_pos < len(overlap) // 2:
                            overlap = overlap[period_pos + 2:]
                        else:
                            space_pos = overlap.find(' ')
                            if space_pos > 0:
                                overlap = overlap[space_pos + 1:]
                    else:
                        overlap = prev_text
                    
                    if overlap.strip():
                        new_chunk['text'] = f"[...] {overlap.strip()}\n\n{chunk['text']}"
                        new_chunk['has_overlap'] = True
            
            enhanced.append(new_chunk)
        
        return enhanced
    
    def _is_heading(self, text: str) -> bool:
        text = text.strip()
        first_line = text.split('\n')[0].strip()
        
        if len(first_line) > 80:
            return False
        
        for pattern in self.heading_patterns:
            if pattern.match(first_line):
                return True
        
        if first_line.isupper() and 3 < len(first_line) < 50 and ' ' in first_line:
            return True
        
        return False
    
    def _should_preserve_with_next(self, text: str) -> bool:
        text = text.strip()
        for pattern in self.preserve_start_patterns:
            if pattern.match(text):
                return True
        if text.rstrip().endswith(':'):
            return True
        return False
    
    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        text = re.sub(r'^\s*Page\s+\d+\s*(of\s+\d+)?\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*[•●○▪]\s*', '\n• ', text)
        return text.strip()
    
    def _get_element_text(self, el) -> str:
        try:
            if hasattr(el, 'text'):
                return el.text or ''
            if hasattr(el, 'get_text'):
                return el.get_text() or ''
            return str(el) or ''
        except Exception:
            return ''
    
    def _get_element_page(self, el) -> int:
        try:
            meta = getattr(el, 'metadata', None)
            if meta:
                if hasattr(meta, 'page_number'):
                    return int(meta.page_number or 1)
                if isinstance(meta, dict):
                    return int(meta.get('page_number') or meta.get('page') or 1)
        except Exception:
            pass
        return 1
    
    def _get_element_category(self, el) -> str:
        try:
            if hasattr(el, 'category'):
                return str(el.category or '')
            meta = getattr(el, 'metadata', None)
            if meta and hasattr(meta, 'category'):
                return str(meta.category or '')
        except Exception:
            pass
        return ''
    
    # =========================================================================
    # Table Extraction (unchanged - pdfplumber works well)
    # =========================================================================
    
    def _extract_tables_pdfplumber(self, pdf_path: str) -> List[Document]:
        documents = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        tables = page.extract_tables() or []
                        for table_idx, table in enumerate(tables):
                            if not table or len(table) == 0:
                                continue
                            
                            table_text = self._format_table(table, page_num, table_idx)
                            if table_text:
                                documents.append(Document(
                                    page_content=table_text,
                                    metadata={
                                        'page': page_num,
                                        'content_type': 'table',
                                        'is_table': True,
                                        'table_index': table_idx,
                                        'source': pdf_path
                                    }
                                ))
                            
                            row_docs = self._create_row_documents(table, page_num, table_idx, pdf_path)
                            documents.extend(row_docs)
                    except Exception as e:
                        logger.warning(f"Error extracting tables from page {page_num}: {e}")
                        continue
        except Exception as e:
            logger.error(f"pdfplumber failed for {pdf_path}: {e}")
        return documents
    
    def _format_table(self, table: List[List], page_num: int, table_idx: int) -> str:
        if not table:
            return ""
        lines = [
            f"TABLE {table_idx + 1} (Page {page_num}):",
            "=" * 50
        ]
        headers = table[0] if table else []
        for row_idx, row in enumerate(table):
            if row_idx == 0:
                header_text = " | ".join(str(cell or "").strip() for cell in row)
                lines.append(f"HEADERS: {header_text}")
                lines.append("-" * 50)
            else:
                row_data = []
                for col_idx, cell in enumerate(row):
                    if col_idx < len(headers) and headers[col_idx]:
                        header = str(headers[col_idx]).strip()
                        value = str(cell or "").strip()
                        if header and value:
                            row_data.append(f"{header}: {value}")
                if row_data:
                    lines.append(" | ".join(row_data))
        return "\n".join(lines)
    
    def _create_row_documents(self, table: List[List], page_num: int, 
                               table_idx: int, source: str) -> List[Document]:
        documents = []
        if not table or len(table) < 2:
            return documents
        headers = table[0]
        for row_idx, row in enumerate(table[1:], 1):
            row_pairs = []
            metadata = {
                'page': page_num,
                'content_type': 'table_row',
                'is_table_row': True,
                'is_table': False,
                'table_index': table_idx,
                'row_index': row_idx,
                'source': source
            }
            for col_idx, cell in enumerate(row):
                if col_idx < len(headers) and headers[col_idx]:
                    header = str(headers[col_idx]).strip()
                    value = str(cell or "").strip()
                    if header and value:
                        row_pairs.append(f"{header}: {value}")
                        header_key = re.sub(r'[^a-z0-9]', '_', header.lower())
                        metadata[f'col_{header_key}'] = value
            if row_pairs:
                row_text = f"Table Row (Page {page_num}): " + " | ".join(row_pairs)
                documents.append(Document(
                    page_content=row_text,
                    metadata=metadata
                ))
        return documents


# =============================================================================
# KEYWORD MATCHER (unchanged)
# =============================================================================

class KeywordMatcher:
    def __init__(self):
        self.stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'why',
            'how', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
            'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from', 'as',
            'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further',
            'once', 'here', 'there', 'all', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'also', 'now', 'about', 'any',
            'system', 'check', 'verify', 'ensure', 'procedure', 'manual',
            'maintenance', 'operation', 'equipment', 'vessel', 'ship'
        }
    
    def calculate_keyword_boost(self, query: str, chunk_text: str, max_boost: float = 0.4) -> float:
        query_lower = query.lower()
        chunk_lower = chunk_text.lower()
        query_terms = self._extract_terms(query)
        
        if not query_terms:
            return 0.0
        
        total_boost = 0.0
        matched_terms = 0
        
        for term, specificity in query_terms:
            if term in chunk_lower:
                matched_terms += 1
                total_boost += specificity * 0.15
        
        if matched_terms > 1:
            total_boost += 0.05 * (matched_terms - 1)
        
        if len(query_lower) > 3 and query_lower in chunk_lower:
            total_boost += 0.15
        
        identifiers = self._extract_identifiers(query)
        for identifier in identifiers:
            if identifier.lower() in chunk_lower:
                total_boost += 0.2
        
        return min(total_boost, max_boost)
    
    def _extract_terms(self, text: str) -> List[Tuple[str, float]]:
        raw_terms = re.findall(r'[a-zA-Z0-9_\-]+', text.lower())
        terms_with_scores = []
        for term in raw_terms:
            if term in self.stopwords or len(term) < 2:
                continue
            specificity = self._calculate_specificity(term)
            terms_with_scores.append((term, specificity))
        return terms_with_scores
    
    def _calculate_specificity(self, term: str) -> float:
        score = 0.5
        if re.search(r'\d', term): score += 0.3
        if term.isupper() and len(term) >= 2: score += 0.2
        if '_' in term: score += 0.3
        if len(term) > 8: score += 0.1
        if re.search(r'[a-z][A-Z]|[A-Z][a-z]', term): score += 0.1
        return min(score, 1.0)
    
    def _extract_identifiers(self, text: str) -> List[str]:
        identifiers = []
        identifiers.extend(re.findall(r'IMO\s*\d+', text, re.IGNORECASE))
        identifiers.extend(re.findall(r'(?:P/?N|Part\s*No\.?)\s*[A-Z0-9\-]+', text, re.IGNORECASE))
        identifiers.extend(re.findall(r'[A-Z0-9]+[_\-][A-Z0-9_\-]+', text))
        identifiers.extend(re.findall(r'\b[A-Z]{2,}[0-9]{2,}[A-Z0-9]*\b', text, re.IGNORECASE))
        return list(set(identifiers))


# =============================================================================
# QUERY FUNCTION (unchanged)
# =============================================================================

def query_manuals_hybrid(vectorstore, question: str, n_results: int = 10,
                         keyword_matcher: KeywordMatcher = None,
                         save_context: bool = True, gpu_manager=None) -> Dict:
    if keyword_matcher is None:
        keyword_matcher = KeywordMatcher()
    
    try:
        if gpu_manager:
            gpu_manager.cleanup()
        
        search_k = min(n_results * 3, 30)
        results = vectorstore.similarity_search_with_score(question, k=search_k)
        
        table_keywords = ['table', 'specification', 'rating', 'capacity', 'dimension', 'limit']
        if any(kw in question.lower() for kw in table_keywords):
            try:
                table_results = vectorstore.similarity_search_with_score(
                    question, k=5, filter={"is_table": True})
                results.extend(table_results)
            except Exception:
                pass
            try:
                row_results = vectorstore.similarity_search_with_score(
                    question, k=5, filter={"is_table_row": True})
                results.extend(row_results)
            except Exception:
                pass
        
        if not results:
            return {
                'question': question, 'context': '', 'metadata': [],
                'metadata_detailed': [], 'num_results': 0, 'error': None
            }
        
        scored_results = []
        for doc, raw_score in results:
            if raw_score >= 0:
                semantic_sim = 1.0 / (1.0 + raw_score)
            else:
                semantic_sim = max(0.0, min(1.0, -raw_score))
            
            keyword_boost = keyword_matcher.calculate_keyword_boost(question, doc.page_content)
            final_score = min(1.0, semantic_sim + keyword_boost)
            
            scored_results.append({
                'doc': doc, 'raw_score': raw_score, 'semantic_sim': semantic_sim,
                'keyword_boost': keyword_boost, 'final_score': final_score
            })
        
        seen = set()
        unique_results = []
        for item in scored_results:
            content_key = item['doc'].page_content[:150]
            doc_hash = item['doc'].metadata.get('doc_hash', '')
            dedup_key = f"{doc_hash}_{content_key}"
            if dedup_key not in seen:
                seen.add(dedup_key)
                unique_results.append(item)
        
        unique_results.sort(key=lambda x: x['final_score'], reverse=True)
        final_results = unique_results[:n_results]
        
        context_parts = []
        metadata_out = []
        
        for item in final_results:
            doc = item['doc']
            doc_name = doc.metadata.get('document_name', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            
            if doc.metadata.get('is_table'):
                context_parts.append(f"[TABLE from {doc_name}, Page {page}]:\n{doc.page_content}")
            elif doc.metadata.get('is_table_row'):
                context_parts.append(f"[TABLE ROW from {doc_name}, Page {page}]:\n{doc.page_content}")
            else:
                context_parts.append(f"[TEXT from {doc_name}, Page {page}]:\n{doc.page_content}")
            
            clean_doc_name = doc_name
            if doc_name.startswith('manual_') and '_' in doc_name[7:]:
                parts = doc_name.split('_', 2)
                if len(parts) >= 3:
                    clean_doc_name = parts[2]
            
            metadata_out.append({
                'document': clean_doc_name,
                'page': page,
                'raw_score': round(item['raw_score'], 4),
                'semantic_score': round(item['semantic_sim'], 4),
                'keyword_boost': round(item['keyword_boost'], 4),
                'final_score': round(item['final_score'], 4),
                'is_table': doc.metadata.get('is_table', False),
                'content_type': doc.metadata.get('content_type', 'unknown')
            })
        
        context_text = "\n\n".join(context_parts)
        
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
        
        context_file = None
        if save_context and context_text:
            context_file = _save_query_context(question, context_text, metadata_out)
        
        if gpu_manager:
            gpu_manager.cleanup()
        
        return {
            'question': question, 'context': context_text,
            'metadata': grouped_metadata, 'metadata_detailed': metadata_out,
            'num_results': len(final_results), 'context_file': context_file
        }
    except Exception as e:
        logger.exception(f"Query error: {e}")
        return {
            'question': question, 'context': '', 'metadata': [],
            'metadata_detailed': [], 'num_results': 0, 'error': str(e)
        }


def _save_query_context(question: str, context: str, metadata: List[Dict]) -> Optional[str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"query_context_{timestamp}.txt"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n")
            f.write(f"QUESTION: {question}\n")
            f.write("=" * 80 + "\n\n")
            f.write("SCORING DETAILS:\n")
            for idx, meta in enumerate(metadata):
                f.write(f"\nResult {idx + 1}:\n")
                f.write(f"  Document: {meta.get('document')}\n")
                f.write(f"  Page: {meta.get('page')}\n")
                f.write(f"  Semantic Score: {meta.get('semantic_score')}\n")
                f.write(f"  Keyword Boost: {meta.get('keyword_boost')}\n")
                f.write(f"  Final Score: {meta.get('final_score')}\n")
            f.write("\n" + "=" * 80 + "\n")
            f.write("CONTEXT PASSED TO LLM:\n\n")
            f.write(context)
            f.write("\n" + "=" * 80 + "\n")
        return filename
    except Exception:
        logger.exception("Error saving context")
        return None


class GPUMemoryManager:
    @staticmethod
    def cleanup():
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            gc.collect()


# =============================================================================
# DROP-IN REPLACEMENT
# =============================================================================

class EnhancedPDFExtractor:
    def __init__(self):
        self.extractor = MarineDocumentExtractor(
            target_chunk_size=1600,
            max_chunk_size=2200,
            min_chunk_size=100,
            overlap_size=200
        )
    
    def extract_from_pdf(self, pdf_path: str) -> List[Document]:
        return self.extractor.extract_from_pdf(pdf_path)
    
# =============================================================================
# VERIFICATION: Test zero-loss extraction
# =============================================================================

def verify_zero_loss(pdf_path: str, extractor: MarineDocumentExtractor = None) -> Dict:
    """
    Verify that extraction doesn't lose content.
    
    Run this on your test PDFs to confirm the extractor works.
    """
    if extractor is None:
        extractor = MarineDocumentExtractor()
    
    # Get raw text using simple fitz extraction (baseline)
    import fitz
    doc = fitz.open(pdf_path)
    raw_text = ""
    for page in doc:
        raw_text += page.get_text()
    doc.close()
    
    raw_chars = len(raw_text)
    raw_words = len(raw_text.split())
    
    # Get extracted documents
    documents = extractor.extract_from_pdf(pdf_path)
    text_docs = [d for d in documents if not d.metadata.get('is_table') and not d.metadata.get('is_table_row')]
    
    extracted_text = " ".join(d.page_content for d in text_docs)
    extracted_chars = len(extracted_text)
    extracted_words = len(extracted_text.split())
    
    # Account for overlap (adds ~10%)
    # Account for cleaning (removes ~5%)
    # Net should be close to 100%
    
    char_ratio = extracted_chars / raw_chars if raw_chars > 0 else 0
    word_ratio = extracted_words / raw_words if raw_words > 0 else 0
    
    result = {
        'pdf_path': pdf_path,
        'raw_chars': raw_chars,
        'raw_words': raw_words,
        'extracted_chars': extracted_chars,
        'extracted_words': extracted_words,
        'char_ratio': round(char_ratio, 3),
        'word_ratio': round(word_ratio, 3),
        'text_chunks': len(text_docs),
        'table_chunks': len([d for d in documents if d.metadata.get('is_table')]),
        'row_chunks': len([d for d in documents if d.metadata.get('is_table_row')]),
        'status': 'OK' if char_ratio >= 0.90 else 'WARNING'
    }
    
    print(f"\n{'=' * 60}")
    print(f"VERIFICATION: {Path(pdf_path).name}")
    print(f"{'=' * 60}")
    print(f"Raw:       {raw_chars:,} chars, {raw_words:,} words")
    print(f"Extracted: {extracted_chars:,} chars, {extracted_words:,} words")
    print(f"Ratio:     {char_ratio:.1%} chars, {word_ratio:.1%} words")
    print(f"Chunks:    {result['text_chunks']} text, {result['table_chunks']} tables, {result['row_chunks']} rows")
    print(f"Status:    {result['status']}")
    print(f"{'=' * 60}\n")
    
    return result    


# =============================================================================
# PROCESSOR (unchanged interface)
# =============================================================================

class FixedTableManualProcessor:
    def __init__(self, db_path: str = "./fixed_table_manual_db"):
        self.db_path = db_path
        self.collection_name = "fixed_table_manuals"
        self.supported_extensions = {'.pdf', '.csv', '.txt', '.xlsx', '.xls'}
        logger.info("Initializing processor...")
        self._initialize_components()

    def _initialize_components(self):
        self.embeddings = OfflineHuggingFaceEmbeddings()
        self.pdf_extractor = EnhancedPDFExtractor()
        self.gpu_manager = GPUMemoryManager()
        self.keyword_matcher = KeywordMatcher()
        os.makedirs(self.db_path, exist_ok=True)
        self.vectorstore = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )

    def delete_document_by_name(self, document_name: str) -> bool:
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

    def process_document(self, file_path: str) -> Dict:
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        doc_name = file_path.name

        if file_extension not in self.supported_extensions:
            return {'status': 'error', 'document': doc_name, 'error': f'Unsupported: {file_extension}'}

        try:
            self.gpu_manager.cleanup()
            doc_bytes = file_path.read_bytes()
            doc_hash = hashlib.md5(doc_bytes).hexdigest()
            if self._is_document_processed(doc_hash):
                return {'status': 'already_exists', 'document': doc_name, 'doc_hash': doc_hash}

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
                    documents = [Document(page_content=raw.strip(), metadata={'source': str(file_path)})]

            if not documents:
                return {'status': 'error', 'document': doc_name, 'error': 'No content extracted'}

            for doc in documents:
                md = doc.metadata or {}
                md.update({
                    'document_name': doc_name,
                    'file_extension': file_extension,
                    'doc_hash': doc_hash,
                    'timestamp': datetime.now().isoformat()
                })
                sanitized = {}
                for k, v in md.items():
                    if isinstance(v, (str, int, float, bool, list, dict)):
                        sanitized[k] = v
                    else:
                        sanitized[k] = str(v)
                doc.metadata = sanitized

            self._add_to_vectorstore_batched(documents)
            self.gpu_manager.cleanup()

            return {
                'status': 'success', 'document': doc_name, 'doc_hash': doc_hash,
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
        documents = []
        xl_file = pd.ExcelFile(file_path)
        for sheet in xl_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet)
            text = f"Excel Sheet: {sheet}\n\n"
            for _, row in df.iterrows():
                text += " | ".join(f"{c}: {v}" for c, v in row.items() if pd.notna(v)) + "\n"
            documents.append(Document(page_content=text.strip(), metadata={'source': file_path, 'is_table': True}))
        return documents

    def _extract_csv(self, file_path: str) -> List[Document]:
        df = pd.read_csv(file_path)
        text = f"CSV File: {os.path.basename(file_path)}\n\n"
        for _, row in df.iterrows():
            text += " | ".join(f"{c}: {v}" for c, v in row.items() if pd.notna(v)) + "\n"
        return [Document(page_content=text.strip(), metadata={'source': file_path, 'is_table': True})]

    def _add_to_vectorstore_batched(self, documents: List[Document]):
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [f"{doc.metadata.get('doc_hash','unknown')}_{i}" for i, doc in enumerate(documents)]
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            end = min(i + batch_size, len(texts))
            self.vectorstore.add_texts(
                texts=texts[i:end], metadatas=metadatas[i:end], ids=ids[i:end]
            )
        self.vectorstore.persist()

    def _is_document_processed(self, doc_hash: str) -> bool:
        try:
            collection = getattr(self.vectorstore, '_collection', None)
            if collection:
                data = collection.get()
                for md in data.get('metadatas', []):
                    if md and md.get('doc_hash') == doc_hash:
                        return True
            return False
        except Exception:
            return False

    def query_manuals(self, question: str, n_results: int = 10, save_context: bool = True) -> Dict:
        return query_manuals_hybrid(
            vectorstore=self.vectorstore,
            question=question, n_results=n_results,
            keyword_matcher=self.keyword_matcher,
            save_context=save_context, gpu_manager=self.gpu_manager
        )

    def get_stats(self) -> Dict:
        try:
            collection = getattr(self.vectorstore, '_collection', None)
            if collection:
                data = collection.get()
                total = len(data.get('ids', []))
                tables = sum(1 for m in data.get('metadatas', []) if m.get('is_table'))
                return {'total_chunks': total, 'table_chunks': tables,
                        'text_chunks': total - tables, 'status': 'ready'}
            return {'total_chunks': 0, 'status': 'ready'}
        except Exception:
            return {'total_chunks': 0, 'status': 'error'}


def initialize_fixed_processor(db_path: str = "./fixed_table_manual_db") -> FixedTableManualProcessor:
    return FixedTableManualProcessor(db_path=db_path)


# def process_fixed_manual_query(processor: FixedTableManualProcessor, question: str,
#                                 llm_messages: List[Dict], generate_llm_response_func) -> Dict:
#     processor.gpu_manager.cleanup()
#     query_result = processor.query_manuals(question, n_results=10, save_context=True)

#     if query_result.get('error'):
#         return query_result

#     if not query_result['context']:
#         response = generate_llm_response_func(llm_messages, "no context")
#         return {'question': question, 'answer': response, 'source': 'llm_knowledge', 'metadata': []}

#     best_score = max([m['final_score'] for m in query_result.get('metadata_detailed', [])], default=0.0)

#     if best_score < 0.3:
#         response = generate_llm_response_func(llm_messages, "no relevant context")
#         return {
#             'question': question, 'answer': response, 'source': 'llm_knowledge',
#             'metadata': [], 'context_file': query_result.get('context_file')
#         }

#     context_prompt = f"""You are a marine engineering assistant with expertise in technical documentation analysis.

# CONTEXT (Multiple sources provided):
# {query_result['context']}

# QUESTION:
# {question}

# INSTRUCTIONS:
# 1. Read all context sections carefully
# 2. Find the section that directly answers the question
# 3. Extract the most relevant and complete answer
# 4. If multiple sources mention the topic, use the clearest explanation
# 5. Use technical terms exactly as written in context
# 6. Keep answer under 200 words

# Your answer:"""

#     messages = [
#         {'role': 'system', 'content': 'You are a helpful marine engineering assistant.'},
#         {'role': 'user', 'content': context_prompt}
#     ]

#     response = generate_llm_response_func(messages, "manual query")
#     processor.gpu_manager.cleanup()

#     return {
#         'question': question, 'answer': response, 'source': 'manual_context',
#         'metadata': query_result['metadata'], 'context_file': query_result.get('context_file')
#     }

def _merge_overlapping_chunks(context_text: str, metadata_detailed: list) -> str:
    """
    Merge overlapping chunks from the same document/nearby pages
    into continuous blocks before sending to LLM.
    """
    import re as _re
    
    # Split into individual chunks using the markers
    chunk_pattern = _re.compile(r'\[(TEXT|TABLE ROW|TABLE|STRUCTURED TABLE) from .+?, Page .+?\]:\n')
    matches = list(chunk_pattern.finditer(context_text))
    
    if not matches:
        return context_text
    
    # Extract each chunk with its metadata
    chunks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(context_text)
        header = match.group(0)
        body = context_text[start + len(header):end].strip()
        body = _re.sub(r'^\[\.\.\.\]\s*', '', body)
        
        # Extract doc name and page from header
        info = _re.search(r'from (.+?), Page (\d+)', header)
        doc_name = info.group(1) if info else ''
        page = int(info.group(2)) if info else 0
        chunk_type = match.group(1)  # TEXT, TABLE, TABLE ROW
        
        chunks.append({
            'header': header.strip(),
            'body': body,
            'doc': doc_name,
            'page': page,
            'type': chunk_type,
            'merged': False
        })
    
    # Sort by document then page for merging
    chunks.sort(key=lambda c: (c['doc'], c['page']))
    
    # Merge overlapping text chunks
    merged = []
    i = 0
    while i < len(chunks):
        current = chunks[i]
        
        # Don't merge table chunks — they're already well-structured
        if current['type'] in ('TABLE', 'TABLE ROW', 'STRUCTURED TABLE'):
            merged.append(current)
            i += 1
            continue
        
        # Try to merge with subsequent chunks
        combined_body = current['body']
        combined_pages = {current['page']}
        j = i + 1
        
        while j < len(chunks):
            nxt = chunks[j]
            
            # Only merge if: same doc, nearby page, both are TEXT
            if (nxt['type'] not in ('TABLE', 'TABLE ROW', 'STRUCTURED TABLE')
                and nxt['doc'] == current['doc']
                and abs(nxt['page'] - max(combined_pages)) <= 1):
                
                # Check for actual text overlap
                overlap_found = False
                # Try different overlap window sizes
                for window in [150, 100, 50]:
                    if len(combined_body) >= window and len(nxt['body']) >= window:
                        tail = combined_body[-window:]
                        # Find overlap in next chunk's start
                        for k in range(len(tail), 10, -1):
                            snippet = tail[-k:]
                            pos = nxt['body'].find(snippet)
                            if pos != -1 and pos < window:
                                # Found overlap — merge by appending non-overlapping part
                                new_part = nxt['body'][pos + len(snippet):]
                                if new_part.strip():
                                    combined_body = combined_body + '\n' + new_part.strip()
                                combined_pages.add(nxt['page'])
                                overlap_found = True
                                break
                        if overlap_found:
                            break
                
                if not overlap_found:
                    # No overlap but same doc + nearby page — still merge with separator
                    # Check if next chunk adds new content (not duplicate)
                    nxt_preview = nxt['body'][:100]
                    if nxt_preview not in combined_body:
                        combined_body = combined_body + '\n\n' + nxt['body']
                        combined_pages.add(nxt['page'])
                    # else: skip duplicate
                
                j += 1
            else:
                break
        
        pages_str = ', '.join(str(p) for p in sorted(combined_pages))
        merged.append({
            'header': f"[TEXT from {current['doc']}, Page {pages_str}]:",
            'body': combined_body,
            'doc': current['doc'],
            'page': min(combined_pages),
            'type': 'TEXT',
            'merged': len(combined_pages) > 1
        })
        i = j
    
    # Rebuild context string
    parts = []
    for chunk in merged:
        # Strip overlap markers [...] from start of body
        body = _re.sub(r'^\[\.\.\.\]\s*', '', chunk['body'])
        parts.append(f"{chunk['header']}\n{body}")
    
    return '\n\n'.join(parts)

def process_fixed_manual_query(processor, question, llm_messages, generate_llm_response_func):
    print("REACHED TEST3 PROCESS_FIXED_MANUAL_QUERY")
    processor.gpu_manager.cleanup()
    query_result = processor.query_manuals(question, n_results=10, save_context=True)
    
    if query_result.get('error'):
        return query_result
    
    if not query_result['context']:
        response = generate_llm_response_func(llm_messages, "manual query without context")
        processor.gpu_manager.cleanup()
        return {
            'question': question,
            'answer': response,
            'source': 'llm_knowledge',
            'metadata': []
        }
    
    detailed_metadata = query_result.get('metadata_detailed', [])
    
    # Split context by actual chunk markers
    import re as _re
    chunk_pattern = _re.compile(r'\[(TEXT|TABLE ROW|TABLE) from .+?, Page .+?\]:\n')
    raw_context = query_result['context']
    matches = list(chunk_pattern.finditer(raw_context))
    
    chunks_with_text = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_context)
        chunks_with_text.append(raw_context[start:end].strip())

    # # Merge overlapping chunks before sending to LLM
    # merged_context = _merge_overlapping_chunks(
    #     query_result['context'],
    #     query_result.get('metadata_detailed', [])
    # )
    # with open('merged_debug.txt', 'w') as f:
    #     f.write(merged_context)
    try:
        merged_context = _merge_overlapping_chunks(
            query_result['context'],
            query_result.get('metadata_detailed', [])
        )
        print("hi")
        with open('merged_debug.txt', 'w') as f:
            f.write(merged_context)
    except Exception as e:
        logger.error(f"MERGE FAILED: {e}")
        merged_context = query_result['context']  # fallback to original        
    
    context_prompt = f"""You are answering based on manual excerpts.

RULES YOU MUST FOLLOW:
1. If the context contains a NUMBERED PROCEDURE (Step 1, Step 2... or 1., 2., 3...), you MUST list EVERY step. Do NOT summarize steps. Do NOT skip any step.
2. Multiple context sections may contain OVERLAPPING parts of the same procedure. Combine them into ONE complete list with ALL unique steps.
3. Copy ALL numbers, codes, values EXACTLY as written. Never approximate.
4. If information is not in the context, say so.

CONTEXT:
{merged_context}

QUESTION: {question}

YOUR ANSWER (list every step if it's a procedure):"""

    messages_with_context = [
        {
            'role': 'system',
            'content': 'You are a marine engineering assistant. When the context contains numbered steps, you MUST output every single step. Never summarize procedures.'
        },
        {'role': 'user', 'content': context_prompt}
    ]
    
    response = generate_llm_response_func(messages_with_context, "fixed manual query")
    processor.gpu_manager.cleanup()
    answer = response.strip()
    
    # --- PROGRAMMATIC source detection (don't rely on LLM) ---
    # Check which chunks contributed to the answer
    answer_lower = answer.lower()
    used_chunk_indices = []
    
    for i, chunk in enumerate(chunks_with_text):
        # Extract key phrases from each chunk (3+ word sequences)
        chunk_clean = chunk.split(']:\n', 1)[-1] if ']:\n' in chunk else chunk
        
        # Check for distinctive content overlap
        # Use sentences/phrases from the chunk
        sentences = [s.strip() for s in _re.split(r'[.\n]', chunk_clean) if len(s.strip()) > 20]
        
        match_count = 0
        for sentence in sentences:
            # Extract key numbers and terms from sentence
            numbers = _re.findall(r'\d+\.?\d*', sentence)
            key_words = [w for w in sentence.lower().split() if len(w) > 4]
            
            # If answer contains the same numbers AND key words, this chunk was used
            number_hits = sum(1 for n in numbers if n in answer)
            word_hits = sum(1 for w in key_words if w in answer_lower)
            
            if number_hits >= 1 and word_hits >= 2:
                match_count += 1
            elif word_hits >= 4:
                match_count += 1
        
        if match_count >= 1:
            used_chunk_indices.append(i)
    
    # Map used chunks to pages
    if used_chunk_indices and detailed_metadata:
        from collections import defaultdict
        doc_pages = defaultdict(set)
        for idx in used_chunk_indices:
            if 0 <= idx < len(detailed_metadata):
                meta = detailed_metadata[idx]
                doc = meta.get('document', 'Unknown')
                page = meta.get('page', 'N/A')
                if page != 'N/A':
                    doc_pages[doc].add(int(page))
        
        if doc_pages:
            final_metadata = [
                {"doc": doc, "pages": sorted(list(pages))}
                for doc, pages in doc_pages.items()
            ]
        else:
            final_metadata = query_result['metadata']
    else:
        final_metadata = query_result['metadata']
    
    return {
        'question': question,
        'answer': answer,
        'source': 'manual_context',
        'metadata': final_metadata,
        'context_file': query_result.get('context_file')
    }