"""
Context Builder for GAP Analysis

This module builds structured context for LLM-based GAP analysis by:
1. Retrieving relevant document chunks using vector search
2. Assembling chunks into formatted clauses with metadata
3. Managing context length to stay within model token limits
4. Preserving language-specific formatting (Tamil/English)

The context builder does NOT call the LLM - it only prepares the input context.

Example Usage:
    >>> builder = GAPContextBuilder(
    ...     semantic_search=semantic_search_service,
    ...     metadata_store=metadata_store
    ... )
    >>> context = builder.build_context(
    ...     document_id="doc-123",
    ...     gap_prompt="Analyze grantor information",
    ...     top_k=10
    ... )
    >>> print(context.get_formatted_clauses())
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ClauseContext:
    """
    Single clause with metadata for GAP analysis.
    
    Attributes:
        clause_number: Sequential or hierarchical clause number
        heading: Clause heading or title (if available)
        text: Full text content of the clause
        page_number: Page number where clause appears
        language: Language of the clause (tamil or english)
        chunk_id: Original chunk identifier
        similarity_score: Relevance score from vector search
    """
    clause_number: str
    heading: Optional[str]
    text: str
    page_number: Optional[int]
    language: str
    chunk_id: int
    similarity_score: float
    
    def format(self) -> str:
        """
        Format clause for LLM consumption.
        
        Returns:
            str: Formatted clause string
        
        Example Output:
            Clause 1 (Page 2):
            [Heading: Property Description]
            The property is located at...
        """
        parts = [f"Clause {self.clause_number}"]
        
        if self.page_number is not None:
            parts[0] += f" (Page {self.page_number})"
        
        parts[0] += ":"
        
        if self.heading:
            parts.append(f"[Heading: {self.heading}]")
        
        parts.append(self.text)
        
        return "\n".join(parts)
    
    def estimate_tokens(self) -> int:
        """
        Estimate token count for this clause.
        
        Rough estimation: 1 token ≈ 4 characters for English, 2 for Tamil
        
        Returns:
            int: Estimated token count
        """
        total_chars = len(self.text)
        if self.heading:
            total_chars += len(self.heading)
        
        # Tamil characters use more tokens
        if self.language == "tamil":
            return total_chars // 2
        else:
            return total_chars // 4


@dataclass
class GAPContext:
    """
    Complete context assembled for GAP analysis.
    
    Attributes:
        document_id: Document identifier
        clauses: List of clause contexts in relevance order
        total_chunks: Total chunks retrieved
        language: Primary language of document
        total_tokens: Estimated total tokens
        truncated: Whether context was truncated to fit limits
    """
    document_id: str
    clauses: List[ClauseContext]
    total_chunks: int
    language: str
    total_tokens: int
    truncated: bool = False
    
    def get_formatted_clauses(self) -> str:
        """
        Get all clauses formatted for LLM prompt.
        
        Returns:
            str: All clauses formatted and joined
        
        Example:
            Clause 1 (Page 1):
            [Heading: Grantor Details]
            The seller Mr. John Doe...
            
            Clause 2 (Page 2):
            [Heading: Property Description]
            The property is located...
        """
        return "\n\n".join(clause.format() for clause in self.clauses)
    
    def get_clause_references(self) -> List[str]:
        """Get list of clause numbers for reference."""
        return [clause.clause_number for clause in self.clauses]
    
    def get_page_numbers(self) -> List[int]:
        """Get unique page numbers covered."""
        pages = [c.page_number for c in self.clauses if c.page_number is not None]
        return sorted(set(pages))
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of context.
        
        Returns:
            dict: Summary statistics
        """
        return {
            "document_id": self.document_id,
            "total_clauses": len(self.clauses),
            "total_chunks": self.total_chunks,
            "language": self.language,
            "estimated_tokens": self.total_tokens,
            "truncated": self.truncated,
            "pages_covered": len(self.get_page_numbers()),
            "clause_references": self.get_clause_references()
        }


class ContextLengthError(Exception):
    """Raised when context cannot fit within token limits."""
    pass


class GAPContextBuilder:
    """
    Builds structured context for GAP analysis using vector search.
    
    This builder:
    1. Uses semantic search to find relevant chunks based on GAP prompt
    2. Retrieves chunk metadata (clause info, page numbers, language)
    3. Formats chunks into structured clauses
    4. Manages token limits (default: 16k tokens, configurable)
    5. Preserves language-specific formatting
    
    Example:
        >>> builder = GAPContextBuilder(
        ...     semantic_search=semantic_search,
        ...     metadata_store=metadata_store,
        ...     max_tokens=12000
        ... )
        >>> context = builder.build_context(
        ...     document_id="doc-abc123",
        ...     gap_prompt="Find grantor identification details",
        ...     top_k=10
        ... )
        >>> formatted = context.get_formatted_clauses()
    """
    
    def __init__(
        self,
        semantic_search,  # SemanticSearchService
        metadata_store,   # VectorMetadataStore
        max_tokens: int = 16000,
        reserve_tokens: int = 4000  # Reserve for prompt + output
    ):
        """
        Initialize context builder.
        
        Args:
            semantic_search: Semantic search service instance
            metadata_store: Vector metadata store instance
            max_tokens: Maximum tokens for context (default: 16k)
            reserve_tokens: Tokens to reserve for prompt/output (default: 4k)
        """
        self.semantic_search = semantic_search
        self.metadata_store = metadata_store
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.available_tokens = max_tokens - reserve_tokens
        
        logger.info(
            f"Initialized GAPContextBuilder with max_tokens={max_tokens}, "
            f"available_for_context={self.available_tokens}"
        )
    
    def build_context(
        self,
        document_id: str,
        gap_prompt: str,
        top_k: int = 10,
        min_similarity: float = 0.3
    ) -> GAPContext:
        """
        Build context for GAP analysis.
        
        Args:
            document_id: Document UUID to analyze
            gap_prompt: User's GAP definition or analysis prompt
            top_k: Number of top relevant chunks to retrieve (default: 10)
            min_similarity: Minimum similarity score to include (default: 0.3)
        
        Returns:
            GAPContext: Assembled context with formatted clauses
        
        Raises:
            ValueError: If document_id or gap_prompt is empty
            ContextLengthError: If even 1 chunk exceeds available tokens
        
        Example:
            >>> context = builder.build_context(
            ...     document_id="doc-123",
            ...     gap_prompt="Analyze property boundaries and measurements",
            ...     top_k=15
            ... )
            >>> print(f"Found {len(context.clauses)} relevant clauses")
        """
        if not document_id or not document_id.strip():
            raise ValueError("document_id cannot be empty")
        
        if not gap_prompt or not gap_prompt.strip():
            raise ValueError("gap_prompt cannot be empty")
        
        logger.info(
            f"Building context for document '{document_id}' with prompt: "
            f"'{gap_prompt[:50]}...' (top_k={top_k})"
        )
        
        # Step 1: Retrieve relevant chunks using semantic search
        search_results = self._retrieve_chunks(
            document_id=document_id,
            query=gap_prompt,
            top_k=top_k,
            min_similarity=min_similarity
        )
        
        if not search_results:
            logger.warning(
                f"No relevant chunks found for document '{document_id}' "
                f"with similarity >= {min_similarity}"
            )
            # Return empty context
            return GAPContext(
                document_id=document_id,
                clauses=[],
                total_chunks=0,
                language="english",  # Default
                total_tokens=0,
                truncated=False
            )
        
        logger.info(f"Retrieved {len(search_results)} relevant chunks")
        
        # Step 2: Convert search results to clause contexts
        clauses = self._build_clauses(search_results)
        
        # Step 3: Detect primary language
        primary_language = self._detect_primary_language(clauses)
        
        # Step 4: Truncate to fit token limits
        clauses, truncated = self._truncate_to_token_limit(clauses)
        
        # Step 5: Calculate total tokens
        total_tokens = sum(clause.estimate_tokens() for clause in clauses)
        
        context = GAPContext(
            document_id=document_id,
            clauses=clauses,
            total_chunks=len(search_results),
            language=primary_language,
            total_tokens=total_tokens,
            truncated=truncated
        )
        
        logger.info(
            f"Built context: {len(clauses)} clauses, "
            f"{total_tokens} estimated tokens, "
            f"truncated={truncated}"
        )
        
        return context
    
    def _retrieve_chunks(
        self,
        document_id: str,
        query: str,
        top_k: int,
        min_similarity: float
    ) -> List[Any]:
        """
        Retrieve relevant chunks using semantic search.
        
        Args:
            document_id: Document to search within
            query: Search query (GAP prompt)
            top_k: Number of results to retrieve
            min_similarity: Minimum similarity threshold
        
        Returns:
            List[SearchResult]: Search results from semantic search
        """
        try:
            # Use semantic search service to find relevant chunks
            search_response = self.semantic_search.search_by_document(
                query=query,
                document_id=document_id,
                top_k=top_k
            )
            
            # Filter by minimum similarity
            filtered_results = [
                result for result in search_response.results
                if result.similarity_score >= min_similarity
            ]
            
            logger.debug(
                f"Retrieved {len(search_response.results)} chunks, "
                f"{len(filtered_results)} above similarity threshold {min_similarity}"
            )
            
            return filtered_results
        
        except Exception as e:
            logger.error(f"Failed to retrieve chunks: {str(e)}")
            raise
    
    def _build_clauses(self, search_results: List[Any]) -> List[ClauseContext]:
        """
        Convert search results to clause contexts with metadata.
        
        Args:
            search_results: List of SearchResult objects
        
        Returns:
            List[ClauseContext]: Formatted clauses with metadata
        """
        clauses = []
        
        for i, result in enumerate(search_results):
            # Extract metadata from search result
            # SearchResult has: vector_id, document_id, chunk_id, text_preview,
            # clause_reference, page_number, language, similarity_score
            
            # Use clause_reference if available, otherwise use chunk position
            clause_number = result.clause_reference or f"{i + 1}"
            
            # Extract heading from text if present (common pattern: "HEADING:\n")
            heading, text = self._extract_heading(result.text_preview)
            
            clause = ClauseContext(
                clause_number=clause_number,
                heading=heading,
                text=text,
                page_number=result.page_number,
                language=result.language or "english",
                chunk_id=result.chunk_id,
                similarity_score=result.similarity_score
            )
            
            clauses.append(clause)
        
        return clauses
    
    def _extract_heading(self, text: str) -> tuple[Optional[str], str]:
        """
        Extract heading from text if present.
        
        Common patterns:
        - "HEADING:\nBody text..."
        - "## Heading\nBody text..."
        - First line ending with colon
        
        Args:
            text: Full text content
        
        Returns:
            tuple: (heading, remaining_text)
        """
        if not text:
            return None, text
        
        lines = text.split('\n', 1)
        
        # Check if first line is a heading
        first_line = lines[0].strip()
        
        # Pattern 1: All caps (likely heading)
        if first_line.isupper() and len(first_line) > 3:
            remaining = lines[1] if len(lines) > 1 else ""
            return first_line, remaining.strip()
        
        # Pattern 2: Markdown heading
        if first_line.startswith('#'):
            heading = first_line.lstrip('#').strip()
            remaining = lines[1] if len(lines) > 1 else ""
            return heading, remaining.strip()
        
        # Pattern 3: Ends with colon (likely heading)
        if first_line.endswith(':') and len(first_line) < 100:
            heading = first_line.rstrip(':').strip()
            remaining = lines[1] if len(lines) > 1 else ""
            return heading, remaining.strip()
        
        # No heading found
        return None, text
    
    def _detect_primary_language(self, clauses: List[ClauseContext]) -> str:
        """
        Detect primary language of document from clauses.
        
        Args:
            clauses: List of clause contexts
        
        Returns:
            str: Primary language ('tamil' or 'english')
        """
        if not clauses:
            return "english"
        
        # Count languages
        language_counts = {}
        for clause in clauses:
            lang = clause.language.lower()
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        # Return most common language
        primary = max(language_counts.items(), key=lambda x: x[1])[0]
        
        logger.debug(f"Detected primary language: {primary} (counts: {language_counts})")
        
        return primary
    
    def _truncate_to_token_limit(
        self,
        clauses: List[ClauseContext]
    ) -> tuple[List[ClauseContext], bool]:
        """
        Truncate clauses to fit within available token limit.
        
        Keeps clauses in order of relevance (highest similarity first).
        
        Args:
            clauses: List of clause contexts
        
        Returns:
            tuple: (truncated_clauses, was_truncated)
        
        Raises:
            ContextLengthError: If even 1 clause exceeds limit
        """
        if not clauses:
            return clauses, False
        
        # Check if first clause alone exceeds limit
        first_tokens = clauses[0].estimate_tokens()
        if first_tokens > self.available_tokens:
            raise ContextLengthError(
                f"Single clause ({first_tokens} tokens) exceeds available "
                f"token limit ({self.available_tokens} tokens). "
                f"Consider increasing max_tokens or reducing chunk size."
            )
        
        # Add clauses until we hit the limit
        selected_clauses = []
        total_tokens = 0
        
        for clause in clauses:
            clause_tokens = clause.estimate_tokens()
            
            if total_tokens + clause_tokens <= self.available_tokens:
                selected_clauses.append(clause)
                total_tokens += clause_tokens
            else:
                # Hit the limit, stop adding
                logger.info(
                    f"Truncating context: {len(clauses) - len(selected_clauses)} "
                    f"clauses dropped to fit token limit"
                )
                return selected_clauses, True
        
        # All clauses fit
        return selected_clauses, False
    
    def build_context_for_entire_document(
        self,
        document_id: str
    ) -> GAPContext:
        """
        Build context using ALL chunks from document (no semantic search).
        
        Use this when you want full document context, not query-based retrieval.
        WARNING: May exceed token limits for large documents.
        
        Args:
            document_id: Document UUID
        
        Returns:
            GAPContext: Context with all document chunks
        """
        logger.info(
            f"Building full document context for '{document_id}' "
            "(no semantic filtering)"
        )
        
        # Get all chunks for document from metadata store
        all_metadata = self.metadata_store.get_by_document(document_id)
        
        if not all_metadata:
            logger.warning(f"No chunks found for document '{document_id}'")
            return GAPContext(
                document_id=document_id,
                clauses=[],
                total_chunks=0,
                language="english",
                total_tokens=0,
                truncated=False
            )
        
        # Convert to clause contexts
        clauses = []
        for i, metadata in enumerate(all_metadata):
            heading, text = self._extract_heading(metadata.text_preview)
            
            clause = ClauseContext(
                clause_number=metadata.clause_reference or f"{i + 1}",
                heading=heading,
                text=text,
                page_number=metadata.page_number,
                language=metadata.language or "english",
                chunk_id=metadata.chunk_id,
                similarity_score=1.0  # No similarity score for full document
            )
            clauses.append(clause)
        
        # Sort by chunk_id to maintain document order
        clauses.sort(key=lambda c: c.chunk_id)
        
        # Detect language and truncate
        primary_language = self._detect_primary_language(clauses)
        clauses, truncated = self._truncate_to_token_limit(clauses)
        total_tokens = sum(clause.estimate_tokens() for clause in clauses)
        
        context = GAPContext(
            document_id=document_id,
            clauses=clauses,
            total_chunks=len(all_metadata),
            language=primary_language,
            total_tokens=total_tokens,
            truncated=truncated
        )
        
        logger.info(
            f"Built full document context: {len(clauses)} clauses, "
            f"{total_tokens} tokens, truncated={truncated}"
        )
        
        return context


# Example usage
"""
Example 1: Build context for GAP analysis

from app.services.semantic_search import SemanticSearchService
from app.services.metadata_store import VectorMetadataStore

# Initialize services
semantic_search = SemanticSearchService(...)
metadata_store = VectorMetadataStore(...)

# Create builder
builder = GAPContextBuilder(
    semantic_search=semantic_search,
    metadata_store=metadata_store,
    max_tokens=16000
)

# Build context for specific GAP analysis
context = builder.build_context(
    document_id="doc-abc123",
    gap_prompt="Analyze grantor identification, including name, parentage, and authority to sell",
    top_k=10
)

# Get formatted clauses for LLM prompt
formatted_clauses = context.get_formatted_clauses()
print(formatted_clauses)

# Output:
# Clause 1 (Page 1):
# [Heading: GRANTOR DETAILS]
# The seller Mr. John Doe, son of Richard Doe, aged 45 years...
#
# Clause 3 (Page 2):
# [Heading: AUTHORITY TO SELL]
# The grantor possesses clear title to the property...

# Get summary
print(context.get_summary())
# {
#   "document_id": "doc-abc123",
#   "total_clauses": 10,
#   "total_chunks": 10,
#   "language": "english",
#   "estimated_tokens": 8500,
#   "truncated": False,
#   "pages_covered": 3,
#   "clause_references": ["1", "3", "5", ...]
# }


Example 2: Tamil document context

context_tamil = builder.build_context(
    document_id="doc-tamil-456",
    gap_prompt="சொத்து விளக்கம் மற்றும் எல்லைகள்",
    top_k=8
)

formatted = context_tamil.get_formatted_clauses()
# Clause 1 (Page 1):
# [Heading: சொத்து விவரம்]
# இந்த சொத்து சர்வே எண் 123/4 இல் அமைந்துள்ளது...


Example 3: Handle token limits

# Large document - will be truncated
try:
    context = builder.build_context(
        document_id="large-doc-789",
        gap_prompt="Full property analysis",
        top_k=50  # Request many chunks
    )
    
    if context.truncated:
        print(f"⚠️ Context truncated: {context.total_chunks} chunks "
              f"reduced to {len(context.clauses)} clauses")
        print(f"Covered pages: {context.get_page_numbers()}")
        
except ContextLengthError as e:
    print(f"Context too large: {e}")


Example 4: Full document context (no search)

# Get entire document without semantic filtering
context_full = builder.build_context_for_entire_document(
    document_id="doc-small-123"
)

print(f"Full document: {len(context_full.clauses)} clauses")
print(f"Language: {context_full.language}")


Example 5: Using context in prompt

from app.services.prompt_factory import PromptFactory

# Build context
context = builder.build_context(
    document_id="doc-123",
    gap_prompt="Find attestation and witness information",
    top_k=10
)

# Create prompt with context
factory = PromptFactory()
gap_definition = GAPDefinition()

# Get formatted clauses as list of strings
clauses_for_prompt = [clause.format() for clause in context.clauses]

# Create GAP prompt
prompt = factory.create_gap_prompt(
    language=Language.ENGLISH,
    gap_definition=gap_definition,
    clauses=clauses_for_prompt,
    document_context=f"Document covers pages {context.get_page_numbers()}"
)

# Send to LLM
response = ollama_client.generate(prompt)
"""
