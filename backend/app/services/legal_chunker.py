"""
Legal Document Chunking Service

This module provides intelligent chunking for legal documents that respects
document structure (clauses, sections, articles) and never splits within a
logical unit. Supports Tamil and English legal numbering patterns.

Responsibilities:
- Detect clause and section boundaries
- Extract headings and clause numbers
- Create semantically meaningful chunks
- Preserve page numbers and metadata
- Support both Tamil and English numbering
"""

from typing import Dict, List, Tuple, Optional
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """
    Represents a chunk of a legal document
    
    Attributes:
        chunk_id: Unique identifier (sequential number)
        text: The actual text content
        clause_number: Extracted clause/section number (e.g., "1.2.3", "Article 5")
        heading: Section or clause heading if detected
        page_number: Source page number(s)
        language: Language of the chunk ('en', 'ta', 'mixed')
        token_count: Approximate number of tokens
        char_count: Number of characters
    """
    chunk_id: int
    text: str
    clause_number: Optional[str] = None
    heading: Optional[str] = None
    page_number: Optional[int] = None
    language: str = "en"
    token_count: int = 0
    char_count: int = 0
    
    def to_dict(self) -> Dict:
        """Convert chunk to dictionary"""
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'clause_number': self.clause_number,
            'heading': self.heading,
            'page_number': self.page_number,
            'language': self.language,
            'token_count': self.token_count,
            'char_count': self.char_count
        }


class LegalDocumentChunker:
    """
    Intelligent chunking engine for legal documents
    
    Chunking Strategy:
    1. Identify clause/section boundaries using legal numbering patterns
    2. Extract headings from text formatting (UPPERCASE, position)
    3. Group text into chunks that respect clause boundaries
    4. Keep chunks under ~500 tokens (configurable)
    5. Never split within a clause
    6. Preserve metadata (page numbers, clause numbers, headings)
    
    Supported Numbering Patterns:
    - Arabic numerals: 1., 2., 3. or (1), (2), (3)
    - Roman numerals: I., II., III. or (I), (II), (III)
    - Letters: a., b., c. or (a), (b), (c)
    - Hierarchical: 1.1, 1.1.1, 1.1.1.1
    - Legal terms: Article 1, Section 2, Clause 3
    - Tamil numerals: ௧., ௨., ௩. or (௧), (௨), (௩)
    
    Example:
        >>> chunker = LegalDocumentChunker(max_tokens=500)
        >>> chunks = chunker.chunk_document(pages_data, language="en")
        >>> for chunk in chunks:
        ...     print(f"Clause {chunk.clause_number}: {chunk.heading}")
    """
    
    def __init__(self, max_tokens: int = 500, min_tokens: int = 50):
        """
        Initialize legal document chunker
        
        Args:
            max_tokens: Maximum tokens per chunk (soft limit)
            min_tokens: Minimum tokens per chunk (avoid tiny chunks)
        """
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        
        # Compile regex patterns for clause detection
        self._compile_patterns()
        
        logger.info(f"LegalDocumentChunker initialized (max: {max_tokens}, min: {min_tokens} tokens)")
    
    def _compile_patterns(self):
        """
        Compile regex patterns for detecting legal structure
        
        Patterns are ordered by specificity - more specific patterns first.
        This ensures "Article 1.2" matches before generic "1.2"
        """
        
        # Pattern 1: Legal terms with numbers
        # Matches: "Article 5", "Section 2.3", "Clause 10", "Schedule A"
        self.legal_term_pattern = re.compile(
            r'\b(Article|Section|Clause|Schedule|Part|Chapter|Division|Subdivision|Appendix)\s+([A-Z0-9][\w.]*)',
            re.IGNORECASE
        )
        
        # Pattern 2: Hierarchical numbering (most common in legal docs)
        # Matches: "1.2.3", "2.1", "3.4.5.6"
        # Heuristic: At least one dot, surrounded by whitespace or start of line
        self.hierarchical_pattern = re.compile(
            r'(?:^|\n)\s*(\d+(?:\.\d+)+)\.?\s+',
            re.MULTILINE
        )
        
        # Pattern 3: Simple numbered clauses
        # Matches: "1.", "2.", "15." at start of line
        self.simple_number_pattern = re.compile(
            r'(?:^|\n)\s*(\d+)\.[\s\n]+',
            re.MULTILINE
        )
        
        # Pattern 4: Parenthetical numbering
        # Matches: "(1)", "(a)", "(i)", "(A)"
        self.parenthetical_pattern = re.compile(
            r'(?:^|\n)\s*\(([0-9]+|[a-z]+|[A-Z]+|[ivxlcdm]+)\)\s+',
            re.MULTILINE
        )
        
        # Pattern 5: Roman numerals
        # Matches: "I.", "II.", "III.", "IV.", "V."
        self.roman_pattern = re.compile(
            r'(?:^|\n)\s*([IVXLCDM]+)\.?\s+',
            re.MULTILINE
        )
        
        # Pattern 6: Tamil numerals
        # Matches: "௧.", "௨.", "௧.௧", etc.
        # Tamil digits: ௦ ௧ ௨ ௩ ௪ ௫ ௬ ௭ ௮ ௯
        self.tamil_number_pattern = re.compile(
            r'(?:^|\n)\s*([௦-௯]+(?:\.[௦-௯]+)*)\.?\s+',
            re.MULTILINE
        )
        
        # Pattern 7: Heading detection
        # Heuristic: Line with 3+ UPPERCASE words, typically short (< 80 chars)
        # Example: "TERMS AND CONDITIONS", "PAYMENT CLAUSE"
        self.heading_pattern = re.compile(
            r'(?:^|\n)\s*([A-Z][A-Z\s]{2,79})\s*(?:\n|$)',
            re.MULTILINE
        )
    
    def chunk_document(
        self, 
        pages_data: List[Dict], 
        language: str = "en"
    ) -> List[DocumentChunk]:
        """
        Chunk a legal document into semantically meaningful pieces
        
        Process:
        1. Combine all pages into single text with page markers
        2. Detect all clause boundaries
        3. Split text at boundaries
        4. Group small chunks together (if under min_tokens)
        5. Split large chunks (if over max_tokens) at paragraph boundaries
        6. Extract metadata for each chunk
        
        Args:
            pages_data: List of page dictionaries from PDFExtractor
                       Each dict should have: page_number, text, extraction_method
            language: Primary language of document ('en', 'ta', 'mixed')
        
        Returns:
            List of DocumentChunk objects with metadata
        
        Example:
            >>> pages = [
            ...     {'page_number': 1, 'text': '1. First clause...', 'extraction_method': 'direct'},
            ...     {'page_number': 2, 'text': '2. Second clause...', 'extraction_method': 'direct'}
            ... ]
            >>> chunks = chunker.chunk_document(pages, language="en")
        """
        if not pages_data:
            logger.warning("No pages provided for chunking")
            return []
        
        logger.info(f"Starting document chunking: {len(pages_data)} pages, language: {language}")
        
        # Step 1: Combine pages with markers
        full_text, page_markers = self._combine_pages(pages_data)
        
        # Step 2: Detect clause boundaries
        boundaries = self._detect_clause_boundaries(full_text, language)
        logger.debug(f"Detected {len(boundaries)} clause boundaries")
        
        # Step 3: Split text at boundaries
        raw_chunks = self._split_at_boundaries(full_text, boundaries)
        logger.debug(f"Created {len(raw_chunks)} raw chunks")
        
        # Step 4: Optimize chunk sizes
        optimized_chunks = self._optimize_chunk_sizes(raw_chunks)
        logger.debug(f"Optimized to {len(optimized_chunks)} chunks")
        
        # Step 5: Create DocumentChunk objects with metadata
        final_chunks = []
        for i, (text, clause_num, heading) in enumerate(optimized_chunks, start=1):
            page_num = self._find_page_number(text, page_markers)
            token_count = self._estimate_tokens(text)
            
            chunk = DocumentChunk(
                chunk_id=i,
                text=text.strip(),
                clause_number=clause_num,
                heading=heading,
                page_number=page_num,
                language=language,
                token_count=token_count,
                char_count=len(text)
            )
            final_chunks.append(chunk)
        
        logger.info(f"Chunking complete: {len(final_chunks)} chunks created")
        return final_chunks
    
    def _combine_pages(self, pages_data: List[Dict]) -> Tuple[str, Dict[int, int]]:
        """
        Combine all pages into single text with page markers
        
        Inserts special markers to track page boundaries.
        Returns both the combined text and a mapping of character positions to pages.
        
        Args:
            pages_data: List of page dictionaries
        
        Returns:
            Tuple of (combined_text, page_markers)
            page_markers: Dict mapping character position to page number
        """
        combined = []
        page_markers = {}  # char_position -> page_number
        char_position = 0
        
        for page in pages_data:
            page_num = page.get('page_number', 0)
            text = page.get('text', '')
            
            # Record start position of this page
            page_markers[char_position] = page_num
            
            # Add page marker comment (helps with debugging)
            marker = f"\n--- PAGE {page_num} ---\n"
            combined.append(marker)
            char_position += len(marker)
            
            # Add page text
            combined.append(text)
            char_position += len(text)
            
            # Add separator
            combined.append("\n\n")
            char_position += 2
        
        return ''.join(combined), page_markers
    
    def _detect_clause_boundaries(
        self, 
        text: str, 
        language: str
    ) -> List[Tuple[int, str, Optional[str]]]:
        """
        Detect all clause/section boundaries in text
        
        Returns list of (position, clause_number, heading) tuples.
        Position is character index where clause starts.
        
        Heuristics:
        1. Try legal term patterns first (most specific)
        2. Then hierarchical numbering (1.2.3)
        3. Then simple numbering (1., 2.)
        4. Then parenthetical ((1), (a))
        5. For Tamil: Check Tamil numeral patterns
        6. Headings are detected separately and associated with nearest clause
        
        Args:
            text: Full document text
            language: Language code to prioritize patterns
        
        Returns:
            List of (char_position, clause_number, heading) sorted by position
        """
        boundaries = []
        
        # Detect all pattern matches
        # Each pattern adds (position, clause_number, None) to list
        
        # Legal terms (highest priority)
        for match in self.legal_term_pattern.finditer(text):
            pos = match.start()
            term = match.group(1)  # "Article", "Section", etc.
            number = match.group(2)  # "5", "2.3", etc.
            clause_num = f"{term} {number}"
            boundaries.append((pos, clause_num, None))
        
        # Hierarchical numbering
        for match in self.hierarchical_pattern.finditer(text):
            pos = match.start()
            number = match.group(1)
            boundaries.append((pos, number, None))
        
        # Simple numbering
        for match in self.simple_number_pattern.finditer(text):
            pos = match.start()
            number = match.group(1)
            # Only add if not already covered by hierarchical
            if not any(b[0] == pos for b in boundaries):
                boundaries.append((pos, number, None))
        
        # Parenthetical numbering
        for match in self.parenthetical_pattern.finditer(text):
            pos = match.start()
            number = match.group(1)
            if not any(b[0] == pos for b in boundaries):
                boundaries.append((pos, f"({number})", None))
        
        # Roman numerals
        for match in self.roman_pattern.finditer(text):
            pos = match.start()
            number = match.group(1)
            if not any(b[0] == pos for b in boundaries):
                boundaries.append((pos, number, None))
        
        # Tamil numerals (if Tamil or mixed)
        if language in ['ta', 'mixed']:
            for match in self.tamil_number_pattern.finditer(text):
                pos = match.start()
                number = match.group(1)
                if not any(b[0] == pos for b in boundaries):
                    boundaries.append((pos, number, None))
        
        # Detect headings and associate with nearest clause
        headings = self._detect_headings(text)
        boundaries = self._associate_headings(boundaries, headings)
        
        # Sort by position
        boundaries.sort(key=lambda x: x[0])
        
        return boundaries
    
    def _detect_headings(self, text: str) -> List[Tuple[int, str]]:
        """
        Detect section headings in text
        
        Heuristic: Lines with multiple UPPERCASE words, short length.
        These are typically section titles like "TERMS AND CONDITIONS"
        
        Args:
            text: Document text
        
        Returns:
            List of (position, heading_text) tuples
        """
        headings = []
        
        for match in self.heading_pattern.finditer(text):
            pos = match.start()
            heading_text = match.group(1).strip()
            
            # Additional validation: heading should be short and mostly uppercase
            if len(heading_text) < 80 and sum(c.isupper() for c in heading_text) / len(heading_text) > 0.7:
                headings.append((pos, heading_text))
        
        return headings
    
    def _associate_headings(
        self, 
        boundaries: List[Tuple[int, str, Optional[str]]],
        headings: List[Tuple[int, str]]
    ) -> List[Tuple[int, str, Optional[str]]]:
        """
        Associate detected headings with clause boundaries
        
        Strategy: Assign heading to the nearest following clause boundary
        within 200 characters (typical distance between heading and clause number)
        
        Args:
            boundaries: List of clause boundaries
            headings: List of detected headings
        
        Returns:
            Updated boundaries with headings attached
        """
        if not headings:
            return boundaries
        
        # Convert to mutable list
        result = list(boundaries)
        
        for heading_pos, heading_text in headings:
            # Find nearest clause after this heading (within 200 chars)
            for i, (clause_pos, clause_num, _) in enumerate(result):
                if clause_pos > heading_pos and clause_pos - heading_pos < 200:
                    # Update this boundary with the heading
                    result[i] = (clause_pos, clause_num, heading_text)
                    break
        
        return result
    
    def _split_at_boundaries(
        self, 
        text: str, 
        boundaries: List[Tuple[int, str, Optional[str]]]
    ) -> List[Tuple[str, Optional[str], Optional[str]]]:
        """
        Split text at detected clause boundaries
        
        Each chunk includes the clause marker and text until next boundary.
        
        Args:
            text: Full document text
            boundaries: List of (position, clause_number, heading)
        
        Returns:
            List of (text, clause_number, heading) tuples
        """
        if not boundaries:
            # No boundaries detected, return entire text as one chunk
            return [(text, None, None)]
        
        chunks = []
        
        # Add chunks between boundaries
        for i in range(len(boundaries)):
            start_pos = boundaries[i][0]
            clause_num = boundaries[i][1]
            heading = boundaries[i][2]
            
            # Find end position (start of next boundary, or end of text)
            if i + 1 < len(boundaries):
                end_pos = boundaries[i + 1][0]
            else:
                end_pos = len(text)
            
            chunk_text = text[start_pos:end_pos]
            chunks.append((chunk_text, clause_num, heading))
        
        # Add any text before first boundary as preamble
        if boundaries[0][0] > 0:
            preamble = text[:boundaries[0][0]]
            if len(preamble.strip()) > 50:  # Only if substantial
                chunks.insert(0, (preamble, "Preamble", None))
        
        return chunks
    
    def _optimize_chunk_sizes(
        self, 
        raw_chunks: List[Tuple[str, Optional[str], Optional[str]]]
    ) -> List[Tuple[str, Optional[str], Optional[str]]]:
        """
        Optimize chunk sizes to fit within token limits
        
        Strategy:
        1. Merge small consecutive chunks (under min_tokens)
        2. Split large chunks (over max_tokens) at paragraph boundaries
        3. Never split within a clause - only at natural breaks
        
        Args:
            raw_chunks: List of (text, clause_number, heading)
        
        Returns:
            Optimized list of chunks
        """
        optimized = []
        buffer_text = ""
        buffer_clause = None
        buffer_heading = None
        
        for chunk_text, clause_num, heading in raw_chunks:
            token_count = self._estimate_tokens(chunk_text)
            
            # Case 1: Chunk is too large - needs splitting
            if token_count > self.max_tokens:
                # First, flush buffer if exists
                if buffer_text:
                    optimized.append((buffer_text, buffer_clause, buffer_heading))
                    buffer_text = ""
                    buffer_clause = None
                    buffer_heading = None
                
                # Split large chunk at paragraph boundaries
                sub_chunks = self._split_large_chunk(chunk_text, clause_num, heading)
                optimized.extend(sub_chunks)
            
            # Case 2: Chunk is small - add to buffer
            elif token_count < self.min_tokens:
                if not buffer_text:
                    # Start new buffer
                    buffer_text = chunk_text
                    buffer_clause = clause_num
                    buffer_heading = heading
                else:
                    # Append to existing buffer
                    buffer_text += "\n\n" + chunk_text
                    # Keep first clause number, last heading
                    if heading:
                        buffer_heading = heading
            
            # Case 3: Chunk is good size
            else:
                # Flush buffer first
                if buffer_text:
                    optimized.append((buffer_text, buffer_clause, buffer_heading))
                    buffer_text = ""
                    buffer_clause = None
                    buffer_heading = None
                
                # Add this chunk
                optimized.append((chunk_text, clause_num, heading))
        
        # Flush remaining buffer
        if buffer_text:
            optimized.append((buffer_text, buffer_clause, buffer_heading))
        
        return optimized
    
    def _split_large_chunk(
        self, 
        text: str, 
        clause_num: Optional[str],
        heading: Optional[str]
    ) -> List[Tuple[str, Optional[str], Optional[str]]]:
        """
        Split a large chunk at paragraph boundaries
        
        When a clause is too large for one chunk, we split at paragraph
        boundaries (double newlines) to maintain readability.
        
        Args:
            text: Large chunk text
            clause_num: Clause number to preserve
            heading: Heading to preserve (only on first sub-chunk)
        
        Returns:
            List of sub-chunks
        """
        # Split by paragraphs (double newline)
        paragraphs = text.split('\n\n')
        
        sub_chunks = []
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)
            
            # If adding this paragraph exceeds limit, save current chunk
            if current_tokens + para_tokens > self.max_tokens and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                # Only first sub-chunk gets heading
                h = heading if len(sub_chunks) == 0 else None
                sub_chunks.append((chunk_text, clause_num, h))
                
                # Start new chunk
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                # Add to current chunk
                current_chunk.append(para)
                current_tokens += para_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            h = heading if len(sub_chunks) == 0 else None
            sub_chunks.append((chunk_text, clause_num, h))
        
        return sub_chunks
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text
        
        Heuristic: ~4 characters per token for English, ~2 for Tamil
        This is a rough estimate - actual tokenization may vary.
        
        Args:
            text: Text to estimate
        
        Returns:
            Estimated token count
        """
        # Count Tamil characters
        tamil_chars = len(re.findall(r'[\u0B80-\u0BFF]', text))
        english_chars = len(text) - tamil_chars
        
        # Tamil has more complex characters, so fewer chars per token
        estimated_tokens = (english_chars / 4) + (tamil_chars / 2)
        
        return int(estimated_tokens)
    
    def _find_page_number(
        self, 
        text: str, 
        page_markers: Dict[int, int]
    ) -> Optional[int]:
        """
        Find page number for a chunk of text
        
        Looks for page markers in the text to determine source page.
        
        Args:
            text: Chunk text
            page_markers: Dict mapping positions to page numbers
        
        Returns:
            Page number or None
        """
        # Look for page marker in text
        match = re.search(r'--- PAGE (\d+) ---', text)
        if match:
            return int(match.group(1))
        
        # Fallback: return first page if multiple markers
        return None
    
    def get_chunking_stats(self, chunks: List[DocumentChunk]) -> Dict:
        """
        Get statistics about the chunking results
        
        Args:
            chunks: List of DocumentChunk objects
        
        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {}
        
        token_counts = [c.token_count for c in chunks]
        char_counts = [c.char_count for c in chunks]
        
        return {
            'total_chunks': len(chunks),
            'chunks_with_clause_numbers': sum(1 for c in chunks if c.clause_number),
            'chunks_with_headings': sum(1 for c in chunks if c.heading),
            'avg_tokens_per_chunk': sum(token_counts) / len(chunks),
            'max_tokens': max(token_counts),
            'min_tokens': min(token_counts),
            'avg_chars_per_chunk': sum(char_counts) / len(chunks),
            'total_tokens': sum(token_counts),
            'total_chars': sum(char_counts)
        }
