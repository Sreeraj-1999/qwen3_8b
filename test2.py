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
# Unstructured - keep for layout-aware extraction
from unstructured.partition.pdf import partition_pdf

# LangChain / Chroma imports
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Import your shared embedding service (user provided)
from embedding_service import embedding_service

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# =============================================================================
# EXTRACTION: Zero-Loss Document Extractor
# =============================================================================

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

class MarineDocumentExtractor:
    """
    Zero-loss extraction for marine technical manuals.
    
    Design principles:
    1. Use Unstructured for layout-aware extraction (handles multi-column)
    2. Custom chunking that NEVER drops content
    3. pdfplumber for tables (proven to work well)
    4. Overlap for context continuity
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
        
        # Heading patterns for structure detection
        self.heading_patterns = [
            re.compile(r'^(\d+\.)+\d*\s+[A-Z]'),        # 3.1.2 Fuel System
            re.compile(r'^CHAPTER\s+\d+', re.I),        # CHAPTER 5
            re.compile(r'^SECTION\s+\d+', re.I),        # SECTION 3
            re.compile(r'^\d+\.\s+[A-Z]'),              # 1. Introduction
            re.compile(r'^[A-Z][A-Z\s\-]{3,40}$'),      # MAINTENANCE PROCEDURES
            re.compile(r'^(APPENDIX|ANNEX)\s+[A-Z0-9]', re.I),
        ]
        
        # Content that should stay together if possible
        self.preserve_start_patterns = [
            re.compile(r'^(CAUTION|WARNING|DANGER|NOTE|IMPORTANT)\s*:', re.I),
            re.compile(r'^(Step|STEP)\s+\d+', re.I),
            re.compile(r'^\d+\)\s+'),                   # 1) First step
            re.compile(r'^[a-z]\)\s+'),                 # a) Sub-step
        ]
    
    def extract_from_pdf(self, pdf_path: str) -> List[Document]:
        """
        Main extraction method. Returns all text and table chunks.
        """
        pdf_path = str(pdf_path)
        documents = []
        
        # Track extraction stats for verification
        stats = {
            'elements_extracted': 0,
            'elements_chunked': 0,
            'tables_extracted': 0,
            'total_chars_extracted': 0,
            'total_chars_chunked': 0
        }
        
        # Step 1: Extract text elements using Unstructured (handles multi-column)
        text_documents, stats = self._extract_text_unstructured(pdf_path, stats)
        documents.extend(text_documents)
        
        # Step 2: Extract tables using pdfplumber
        table_documents = self._extract_tables_pdfplumber(pdf_path)
        documents.extend(table_documents)
        stats['tables_extracted'] = len([d for d in table_documents if d.metadata.get('is_table')])
        
        # Log extraction stats
        logger.info(
            f"Extraction complete for {Path(pdf_path).name}: "
            f"{stats['elements_extracted']} elements -> {len(text_documents)} text chunks, "
            f"{stats['tables_extracted']} tables, "
            f"chars: {stats['total_chars_extracted']} extracted -> {stats['total_chars_chunked']} chunked"
        )
        
        # Verify no significant loss (allow for whitespace differences)
        if stats['total_chars_extracted'] > 0:
            ratio = stats['total_chars_chunked'] / stats['total_chars_extracted']
            if ratio < 0.95:
                logger.warning(
                    f"Potential content loss detected: {ratio:.1%} retention. "
                    f"Expected >= 95%"
                )
        
        return documents
    
    def _extract_text_unstructured(
        self, 
        pdf_path: str, 
        stats: Dict
    ) -> Tuple[List[Document], Dict]:
        """
        Extract text using Unstructured, then apply custom chunking.
        """
        documents = []
        
        # Try extraction strategies in order
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
        
        # Process elements - extract text and metadata
        processed_elements = []
        
        for el in elements:
            text = self._get_element_text(el)
            if not text or not text.strip():
                continue
            
            page = self._get_element_page(el)
            category = self._get_element_category(el)
            
            # Skip elements that are clearly not content
            if category.lower() in ('pagebreak', 'header', 'footer', 'pagenumber'):
                # But log that we're skipping
                logger.debug(f"Skipping {category}: {text[:50]}...")
                continue
            
            # Clean the text
            cleaned = self._clean_text(text)
            if cleaned:
                processed_elements.append({
                    'text': cleaned,
                    'page': page,
                    'category': category,
                    'original_length': len(text)
                })
                stats['total_chars_extracted'] += len(cleaned)
        
        stats['elements_extracted'] = len(processed_elements)
        
        if not processed_elements:
            return documents, stats
        
        # Now chunk the processed elements
        chunks = self._chunk_elements(processed_elements, pdf_path)
        
        # Add overlap
        chunks = self._add_overlap(chunks)
        
        # Convert to Documents
        for chunk in chunks:
            doc = Document(
                page_content=chunk['text'],
                metadata={
                    'page': chunk['page'],
                    'content_type': 'text',
                    'is_table': False,
                    'section_context': chunk.get('section', ''),
                    'source': pdf_path,
                    'has_overlap': chunk.get('has_overlap', False)
                }
            )
            documents.append(doc)
            stats['total_chars_chunked'] += len(chunk['text'])
        
        stats['elements_chunked'] = len(documents)
        
        return documents, stats
    
    def _chunk_elements(
        self, 
        elements: List[Dict], 
        source: str
    ) -> List[Dict]:
        """
        Chunk elements with structure awareness.
        
        Rules:
        1. Never drop content
        2. Try to break on headings
        3. Try to keep warnings/procedures together
        4. Respect size limits
        """
        chunks = []
        current_chunk = {
            'text': '',
            'page': elements[0]['page'] if elements else 1,
            'section': ''
        }
        current_section = ''
        
        for el in elements:
            text = el['text']
            page = el['page']
            
            # Check if this is a heading
            if self._is_heading(text):
                current_section = text.strip()[:100]
                
                # If current chunk has content, save it before starting new section
                if current_chunk['text'].strip():
                    if len(current_chunk['text']) >= self.min_chunk_size:
                        current_chunk['section'] = current_section
                        chunks.append(current_chunk.copy())
                        current_chunk = {'text': '', 'page': page, 'section': current_section}
                    # If too small, keep accumulating (will merge with heading)
            
            # Check if this should be preserved with following content
            should_preserve = self._should_preserve_with_next(text)
            
            # Calculate what happens if we add this text
            separator = '\n\n' if current_chunk['text'] else ''
            potential_text = current_chunk['text'] + separator + text
            potential_length = len(potential_text)
            
            # Case 1: Fits comfortably
            if potential_length <= self.target_chunk_size:
                current_chunk['text'] = potential_text
                current_chunk['page'] = current_chunk.get('page') or page
                continue
            
            # Case 2: Exceeds target but under max, and we should preserve
            if potential_length <= self.max_chunk_size and should_preserve:
                current_chunk['text'] = potential_text
                continue
            
            # Case 3: Would exceed limits - need to handle current chunk first
            if current_chunk['text'].strip():
                current_chunk['section'] = current_section
                chunks.append(current_chunk.copy())
            
            # Start new chunk with current element
            if len(text) <= self.max_chunk_size:
                current_chunk = {
                    'text': text,
                    'page': page,
                    'section': current_section
                }
            else:
                # Element itself is too large - must split it
                split_chunks = self._split_large_text(text, page, current_section)
                chunks.extend(split_chunks[:-1])  # Add all but last
                if split_chunks:
                    current_chunk = split_chunks[-1]  # Last one continues accumulating
                else:
                    current_chunk = {'text': '', 'page': page, 'section': current_section}
        
        # Don't forget the last chunk
        if current_chunk['text'].strip():
            current_chunk['section'] = current_section
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_large_text(
        self, 
        text: str, 
        page: int, 
        section: str
    ) -> List[Dict]:
        """
        Split text that exceeds max chunk size.
        Try to break on sentence boundaries.
        """
        chunks = []
        
        # Try to split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_text = ''
        
        for sentence in sentences:
            potential = current_text + (' ' if current_text else '') + sentence
            
            if len(potential) <= self.target_chunk_size:
                current_text = potential
            else:
                # Save current if it has content
                if current_text.strip():
                    chunks.append({
                        'text': current_text.strip(),
                        'page': page,
                        'section': section
                    })
                
                # Handle sentence that's too long by itself
                if len(sentence) > self.max_chunk_size:
                    # Force split on words
                    words = sentence.split()
                    current_text = ''
                    for word in words:
                        potential = current_text + (' ' if current_text else '') + word
                        if len(potential) <= self.target_chunk_size:
                            current_text = potential
                        else:
                            if current_text.strip():
                                chunks.append({
                                    'text': current_text.strip(),
                                    'page': page,
                                    'section': section
                                })
                            current_text = word
                else:
                    current_text = sentence
        
        # Last piece
        if current_text.strip():
            chunks.append({
                'text': current_text.strip(),
                'page': page,
                'section': section
            })
        
        return chunks
    
    def _add_overlap(self, chunks: List[Dict]) -> List[Dict]:
        """
        Add overlap from previous chunk to current chunk.
        """
        if not chunks or self.overlap_size <= 0:
            return chunks
        
        enhanced = []
        
        for i, chunk in enumerate(chunks):
            new_chunk = chunk.copy()
            
            if i > 0:
                prev_text = chunks[i - 1]['text']
                prev_page = chunks[i - 1].get('page', 0)
                curr_page = chunk.get('page', 0)
                
                # Only add overlap if pages are close
                if abs(curr_page - prev_page) <= 1:
                    # Get overlap text
                    if len(prev_text) > self.overlap_size:
                        overlap = prev_text[-self.overlap_size:]
                        # Try to start at a sentence or word boundary
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
        """Check if text appears to be a section heading."""
        text = text.strip()
        first_line = text.split('\n')[0].strip()
        
        # Too long for a heading
        if len(first_line) > 80:
            return False
        
        # Check patterns
        for pattern in self.heading_patterns:
            if pattern.match(first_line):
                return True
        
        # Short all-caps line
        if first_line.isupper() and 3 < len(first_line) < 50 and ' ' in first_line:
            return True
        
        return False
    
    def _should_preserve_with_next(self, text: str) -> bool:
        """Check if this text should stay with following content."""
        text = text.strip()
        
        for pattern in self.preserve_start_patterns:
            if pattern.match(text):
                return True
        
        # Ends with colon often means list follows
        if text.rstrip().endswith(':'):
            return True
        
        return False
    
    def _clean_text(self, text: str) -> str:
        """Clean text while preserving meaningful structure."""
        if not text:
            return ""
        
        # Remove excessive whitespace but keep paragraph structure
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        
        # Remove common page artifacts
        text = re.sub(r'^\s*Page\s+\d+\s*(of\s+\d+)?\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # Standalone numbers
        
        # Normalize bullet points
        text = re.sub(r'\s*[•●○▪]\s*', '\n• ', text)
        
        return text.strip()
    
    def _get_element_text(self, el) -> str:
        """Extract text from Unstructured element."""
        try:
            if hasattr(el, 'text'):
                return el.text or ''
            if hasattr(el, 'get_text'):
                return el.get_text() or ''
            return str(el) or ''
        except Exception:
            return ''
    
    def _get_element_page(self, el) -> int:
        """Extract page number from Unstructured element."""
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
        """Extract category from Unstructured element."""
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
    # Table Extraction (pdfplumber - keeping your working code)
    # =========================================================================
    
    def _extract_tables_pdfplumber(self, pdf_path: str) -> List[Document]:
        """Extract tables using pdfplumber."""
        documents = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        tables = page.extract_tables() or []
                        
                        for table_idx, table in enumerate(tables):
                            if not table or len(table) == 0:
                                continue
                            
                            # Create formatted table document
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
                            
                            # Create row documents for granular search
                            row_docs = self._create_row_documents(
                                table, page_num, table_idx, pdf_path
                            )
                            documents.extend(row_docs)
                            
                    except Exception as e:
                        logger.warning(f"Error extracting tables from page {page_num}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"pdfplumber failed for {pdf_path}: {e}")
        
        return documents
    
    def _format_table(self, table: List[List], page_num: int, table_idx: int) -> str:
        """Format table for readability."""
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
    
    def _create_row_documents(
        self, 
        table: List[List], 
        page_num: int, 
        table_idx: int, 
        source: str
    ) -> List[Document]:
        """Create individual documents for each table row."""
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
                        # Add to metadata for filtering
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
# KEYWORD MATCHING: Hybrid Search Enhancement
# =============================================================================

class KeywordMatcher:
    """
    Keyword matching for hybrid search.
    Boosts chunks that contain exact query terms, especially identifiers.
    """
    
    def __init__(self):
        # Common words to ignore (low specificity)
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
            # Marine common terms that are too generic
            'system', 'check', 'verify', 'ensure', 'procedure', 'manual',
            'maintenance', 'operation', 'equipment', 'vessel', 'ship'
        }
    
    def calculate_keyword_boost(
        self, 
        query: str, 
        chunk_text: str,
        max_boost: float = 0.4
    ) -> float:
        """
        Calculate keyword boost for a chunk based on query term matches.
        
        Returns a boost value between 0 and max_boost.
        """
        query_lower = query.lower()
        chunk_lower = chunk_text.lower()
        
        # Extract query terms
        query_terms = self._extract_terms(query)
        
        if not query_terms:
            return 0.0
        
        total_boost = 0.0
        matched_terms = 0
        
        for term, specificity in query_terms:
            if term in chunk_lower:
                matched_terms += 1
                # Boost proportional to term specificity
                total_boost += specificity * 0.15
        
        # Bonus for matching multiple terms
        if matched_terms > 1:
            total_boost += 0.05 * (matched_terms - 1)
        
        # Bonus for exact phrase match
        if len(query_lower) > 3 and query_lower in chunk_lower:
            total_boost += 0.15
        
        # Check for identifier-like patterns in query that match exactly
        identifiers = self._extract_identifiers(query)
        for identifier in identifiers:
            if identifier.lower() in chunk_lower:
                total_boost += 0.2  # Strong boost for identifier match
        
        return min(total_boost, max_boost)
    
    def _extract_terms(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract terms with their specificity scores.
        Returns list of (term, specificity) tuples.
        """
        # Split on whitespace and punctuation
        raw_terms = re.findall(r'[a-zA-Z0-9_\-]+', text.lower())
        
        terms_with_scores = []
        
        for term in raw_terms:
            if term in self.stopwords:
                continue
            
            if len(term) < 2:
                continue
            
            specificity = self._calculate_specificity(term)
            terms_with_scores.append((term, specificity))
        
        return terms_with_scores
    
    def _calculate_specificity(self, term: str) -> float:
        """
        Calculate how specific/unique a term is.
        Higher score = more specific = bigger boost when matched.
        """
        score = 0.5  # Base score
        
        # Contains numbers - likely an identifier
        if re.search(r'\d', term):
            score += 0.3
        
        # All uppercase - likely an acronym or code
        if term.isupper() and len(term) >= 2:
            score += 0.2
        
        # Contains underscore - likely a tag or code
        if '_' in term:
            score += 0.3
        
        # Longer terms are usually more specific
        if len(term) > 8:
            score += 0.1
        
        # Mixed case like "IMO" or camelCase
        if re.search(r'[a-z][A-Z]|[A-Z][a-z]', term):
            score += 0.1
        
        return min(score, 1.0)
    
    def _extract_identifiers(self, text: str) -> List[str]:
        """
        Extract identifier-like patterns from text.
        These get extra boost when matched.
        """
        identifiers = []
        
        # IMO numbers: IMO followed by digits
        imo_matches = re.findall(r'IMO\s*\d+', text, re.IGNORECASE)
        identifiers.extend(imo_matches)
        
        # Part numbers: P/N, PN, Part No followed by alphanumeric
        pn_matches = re.findall(r'(?:P/?N|Part\s*No\.?)\s*[A-Z0-9\-]+', text, re.IGNORECASE)
        identifiers.extend(pn_matches)
        
        # Tag-like patterns: WORD_WORD_WORD or WORD-WORD-WORD with numbers
        tag_matches = re.findall(r'[A-Z0-9]+[_\-][A-Z0-9_\-]+', text)
        identifiers.extend(tag_matches)
        
        # Alphanumeric codes: 2+ letters followed by 2+ numbers or vice versa
        code_matches = re.findall(r'\b[A-Z]{2,}[0-9]{2,}[A-Z0-9]*\b', text, re.IGNORECASE)
        identifiers.extend(code_matches)
        
        return list(set(identifiers))


# =============================================================================
# UPDATED QUERY FUNCTION
# =============================================================================

def query_manuals_hybrid(
    vectorstore,
    question: str,
    n_results: int = 10,
    keyword_matcher: KeywordMatcher = None,
    save_context: bool = True,
    gpu_manager = None
) -> Dict:
    """
    Hybrid search: semantic similarity + keyword matching.
    
    Drop-in replacement for your query_manuals method.
    """
    if keyword_matcher is None:
        keyword_matcher = KeywordMatcher()
    
    try:
        if gpu_manager:
            gpu_manager.cleanup()
        
        # Get more candidates than needed (we'll re-rank)
        search_k = min(n_results * 3, 30)
        
        # Primary semantic search
        results = vectorstore.similarity_search_with_score(question, k=search_k)
        
        # Optional: table-specific search for relevant queries
        table_keywords = ['table', 'specification', 'rating', 'capacity', 'dimension', 'limit']
        if any(kw in question.lower() for kw in table_keywords):
            try:
                table_results = vectorstore.similarity_search_with_score(
                    question, k=5, filter={"is_table": True}
                )
                results.extend(table_results)
            except Exception:
                pass
            
            try:
                row_results = vectorstore.similarity_search_with_score(
                    question, k=5, filter={"is_table_row": True}
                )
                results.extend(row_results)
            except Exception:
                pass
        
        if not results:
            return {
                'question': question,
                'context': '',
                'metadata': [],
                'metadata_detailed': [],
                'num_results': 0,
                'error': None
            }
        
        # Convert raw scores to similarity and apply keyword boost
        scored_results = []
        
        for doc, raw_score in results:
            # Convert distance to similarity (0 to 1)
            if raw_score >= 0:
                semantic_sim = 1.0 / (1.0 + raw_score)
            else:
                semantic_sim = max(0.0, min(1.0, -raw_score))
            
            # Calculate keyword boost
            keyword_boost = keyword_matcher.calculate_keyword_boost(
                question, doc.page_content
            )
            
            # Combined score
            final_score = min(1.0, semantic_sim + keyword_boost)
            
            scored_results.append({
                'doc': doc,
                'raw_score': raw_score,
                'semantic_sim': semantic_sim,
                'keyword_boost': keyword_boost,
                'final_score': final_score
            })
        
        # Deduplicate by content hash
        seen = set()
        unique_results = []
        
        for item in scored_results:
            content_key = item['doc'].page_content[:150]
            doc_hash = item['doc'].metadata.get('doc_hash', '')
            dedup_key = f"{doc_hash}_{content_key}"
            
            if dedup_key not in seen:
                seen.add(dedup_key)
                unique_results.append(item)
        
        # Sort by final score (descending)
        unique_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Take top n
        final_results = unique_results[:n_results]
        
        # Build context and metadata
        context_parts = []
        metadata_out = []
        
        for item in final_results:
            doc = item['doc']
            doc_name = doc.metadata.get('document_name', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            
            # Format context entry
            if doc.metadata.get('is_table'):
                context_parts.append(f"[TABLE from {doc_name}, Page {page}]:\n{doc.page_content}")
            elif doc.metadata.get('is_table_row'):
                context_parts.append(f"[TABLE ROW from {doc_name}, Page {page}]:\n{doc.page_content}")
            else:
                context_parts.append(f"[TEXT from {doc_name}, Page {page}]:\n{doc.page_content}")
            
            # Clean document name for output
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
        
        # Group metadata by document
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
        
        # Save context for debugging
        context_file = None
        if save_context and context_text:
            context_file = _save_query_context(question, context_text, metadata_out)
        
        if gpu_manager:
            gpu_manager.cleanup()
        
        return {
            'question': question,
            'context': context_text,
            'metadata': grouped_metadata,
            'metadata_detailed': metadata_out,
            'num_results': len(final_results),
            'context_file': context_file
        }
        
    except Exception as e:
        logger.exception(f"Query error: {e}")
        return {
            'question': question,
            'context': '',
            'metadata': [],
            'metadata_detailed': [],
            'num_results': 0,
            'error': str(e)
        }


def _save_query_context(question: str, context: str, metadata: List[Dict]) -> Optional[str]:
    """Save query context for debugging."""
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
# INTEGRATION: Drop-in replacement for your existing code
# =============================================================================

class EnhancedPDFExtractor:
    """
    Drop-in replacement for your existing EnhancedPDFExtractor.
    Same interface, new implementation.
    """
    
    def __init__(self):
        self.extractor = MarineDocumentExtractor(
            target_chunk_size=1600,
            max_chunk_size=2200,
            min_chunk_size=100,
            overlap_size=150
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
# FIXED TABLE MANUAL PROCESSOR (Your existing class, updated)
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
                texts=texts[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end]
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
            question=question,
            n_results=n_results,
            keyword_matcher=self.keyword_matcher,
            save_context=save_context,
            gpu_manager=self.gpu_manager
        )

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
            return {'total_chunks': 0, 'status': 'error'}


# =============================================================================
# KEEP THESE FUNCTIONS (called by your other scripts)
# =============================================================================

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

    best_score = max([m['final_score'] for m in query_result.get('metadata_detailed', [])], default=0.0)

    if best_score < 0.3:
        response = generate_llm_response_func(llm_messages, "no relevant context")
        return {
            'question': question,
            'answer': response,
            'source': 'llm_knowledge',
            'metadata': [],
            'context_file': query_result.get('context_file')
        }

    context_prompt = f"""You are a marine engineering assistant with expertise in technical documentation analysis.

CONTEXT (Multiple sources provided):
{query_result['context']}

QUESTION:
{question}

INSTRUCTIONS:
1. Read all context sections carefully
2. Find the section that directly answers the question
3. Extract the most relevant and complete answer
4. If multiple sources mention the topic, use the clearest explanation
5. Use technical terms exactly as written in context
6. Keep answer under 200 words

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