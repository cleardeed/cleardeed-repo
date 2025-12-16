"""
Semantic Search Service

Provides similarity-based search over legal document chunks using FAISS vector store.
Queries are converted to embeddings and matched against indexed document chunks.

Features:
- Text-to-vector query conversion
- Cosine similarity search via FAISS
- Language-aware retrieval (Tamil queries prefer Tamil chunks)
- Document-level filtering
- Metadata enrichment of results

No LLM inference - pure embedding-based semantic search.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.vector_store import FAISSIndexManager, create_faiss_manager
from app.services.metadata_store import VectorMetadataStore, VectorMetadata
from app.services.language_detector import LanguageDetector
from app.config.embedding_config import DEFAULT_CONFIG


logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """
    Single search result with metadata and similarity score.
    
    Attributes:
        vector_id: FAISS vector index
        document_id: Source document UUID
        chunk_id: Chunk identifier
        text_preview: First 100 chars of chunk text
        clause_reference: Legal clause number (e.g., "1.2.3")
        page_number: Source page number
        language: Chunk language ("en", "ta", "mixed")
        similarity_score: Cosine similarity (0-1, higher is better)
        rank: Position in search results (1-indexed)
    """
    vector_id: int
    document_id: str
    chunk_id: int
    text_preview: str
    clause_reference: Optional[str]
    page_number: Optional[int]
    language: str
    similarity_score: float
    rank: int


@dataclass
class SearchResponse:
    """
    Complete search response with results and metadata.
    
    Attributes:
        query: Original query text
        query_language: Detected language of query
        results: List of SearchResult objects
        total_results: Number of results returned
        search_time_ms: Search execution time in milliseconds
    """
    query: str
    query_language: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: float


class SemanticSearchService:
    """
    Semantic search service for legal documents using FAISS vector store.
    
    This service performs similarity-based retrieval of document chunks:
    1. Converts query text to embedding vector
    2. Searches FAISS index for similar vectors
    3. Retrieves metadata for matched vectors
    4. Applies filters and language preferences
    5. Returns ranked results with scores
    
    Language-Aware Retrieval:
        - Detects query language (English, Tamil, Mixed)
        - Tamil queries boost Tamil chunk scores
        - English queries boost English chunk scores
        - Mixed queries treat all languages equally
    
    Example:
        >>> service = SemanticSearchService()
        >>> response = service.search(
        ...     query="property deed ownership",
        ...     top_k=5
        ... )
        >>> for result in response.results:
        ...     print(f"{result.rank}. {result.text_preview} ({result.similarity_score:.3f})")
    """
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[FAISSIndexManager] = None,
        metadata_store: Optional[VectorMetadataStore] = None,
        language_detector: Optional[LanguageDetector] = None
    ):
        """
        Initialize semantic search service.
        
        Args:
            embedding_service: Service for generating embeddings
            vector_store: FAISS index manager
            metadata_store: Vector metadata store
            language_detector: Language detection service
        """
        self.embedding_service = embedding_service or get_embedding_service()
        
        self.vector_store = vector_store or create_faiss_manager(
            dimension=DEFAULT_CONFIG.embedding_dimension
        )
        
        self.metadata_store = metadata_store or VectorMetadataStore()
        
        self.language_detector = language_detector or LanguageDetector()
        
        logger.info("Initialized SemanticSearchService")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None,
        language_boost: bool = True,
        min_similarity: float = 0.0
    ) -> SearchResponse:
        """
        Perform semantic search over document chunks.
        
        Args:
            query: Search query text (English or Tamil)
            top_k: Number of results to return
            document_id: Filter results to specific document (None = all docs)
            language_boost: Apply language-aware scoring
            min_similarity: Minimum similarity threshold (0-1)
        
        Returns:
            SearchResponse: Search results with metadata
        
        Example:
            >>> service = SemanticSearchService()
            >>> response = service.search("payment terms", top_k=5)
            >>> print(f"Found {response.total_results} results")
            >>> for result in response.results:
            ...     print(f"{result.clause_reference}: {result.similarity_score:.3f}")
        """
        import time
        start_time = time.time()
        
        logger.info(
            f"Search query: '{query[:50]}...' (top_k={top_k}, "
            f"document_id={document_id}, language_boost={language_boost})"
        )
        
        # Validate input
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return SearchResponse(
                query=query,
                query_language="unknown",
                results=[],
                total_results=0,
                search_time_ms=0.0
            )
        
        # Check if index is empty
        if self.vector_store.get_total_vectors() == 0:
            logger.warning("Search attempted on empty index")
            return SearchResponse(
                query=query,
                query_language="unknown",
                results=[],
                total_results=0,
                search_time_ms=0.0
            )
        
        try:
            # Step 1: Detect query language
            query_language, confidence = self.language_detector.detect_language(query)
            logger.debug(f"Query language: {query_language} (confidence: {confidence:.2f})")
            
            # Step 2: Generate query embedding
            query_embedding = self._generate_query_embedding(query)
            
            # Step 3: Search FAISS index (retrieve more than needed for filtering)
            search_k = top_k * 3 if document_id or language_boost else top_k
            raw_results = self.vector_store.search(
                query_embedding=query_embedding,
                k=search_k,
                return_scores=True
            )
            
            # Step 4: Retrieve metadata and build results
            results = self._build_results(raw_results)
            
            # Step 5: Apply filters
            if document_id:
                results = self._filter_by_document(results, document_id)
            
            # Step 6: Apply language-aware boosting
            if language_boost and query_language in ['en', 'ta']:
                results = self._apply_language_boost(results, query_language)
            
            # Step 7: Apply minimum similarity threshold
            if min_similarity > 0:
                results = [r for r in results if r.similarity_score >= min_similarity]
            
            # Step 8: Limit to top_k and re-rank
            results = results[:top_k]
            for i, result in enumerate(results, 1):
                result.rank = i
            
            search_time_ms = (time.time() - start_time) * 1000
            
            logger.info(
                f"Search completed: {len(results)} results in {search_time_ms:.2f}ms"
            )
            
            return SearchResponse(
                query=query,
                query_language=query_language,
                results=results,
                total_results=len(results),
                search_time_ms=search_time_ms
            )
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Search failed: {str(e)}") from e
    
    def _generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for query text.
        
        Args:
            query: Query text
        
        Returns:
            Embedding vector
        """
        logger.debug("Generating query embedding")
        
        try:
            embedding = self.embedding_service.embed_single(query)
            logger.debug(f"Generated embedding of dimension {len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Query embedding generation failed: {str(e)}")
            raise RuntimeError(f"Failed to generate query embedding: {str(e)}") from e
    
    def _build_results(
        self,
        raw_results: List[tuple[str, float]]
    ) -> List[SearchResult]:
        """
        Build SearchResult objects from FAISS results and metadata.
        
        Args:
            raw_results: List of (metadata_id, similarity_score) tuples from FAISS
        
        Returns:
            List of SearchResult objects
        """
        results = []
        
        for rank, (metadata_id, similarity_score) in enumerate(raw_results, 1):
            # Extract vector_id from metadata_id (format: "doc-123_chunk_1")
            # The vector_id is stored in reverse_metadata in vector_store
            # But we need to get it from metadata_store
            
            # Get all metadata and find matching entry
            # This is a limitation - we need a better way to map metadata_id to vector_id
            # For now, we'll search through metadata
            
            # Alternative: parse metadata_id to extract document_id and chunk_id
            # Format: "{document_id}_chunk_{chunk_id}"
            try:
                parts = metadata_id.rsplit("_chunk_", 1)
                if len(parts) == 2:
                    doc_id = parts[0]
                    chunk_id = int(parts[1])
                    
                    # Get metadata by document and chunk
                    metadata = self.metadata_store.get_by_chunk(doc_id, chunk_id)
                    
                    if metadata:
                        result = SearchResult(
                            vector_id=metadata.vector_id,
                            document_id=metadata.document_id,
                            chunk_id=metadata.chunk_id,
                            text_preview=metadata.text_preview or "",
                            clause_reference=metadata.clause_reference,
                            page_number=metadata.page_number,
                            language=metadata.language,
                            similarity_score=similarity_score,
                            rank=rank
                        )
                        results.append(result)
                    else:
                        logger.warning(f"No metadata found for {metadata_id}")
                else:
                    logger.warning(f"Invalid metadata_id format: {metadata_id}")
                    
            except Exception as e:
                logger.warning(f"Failed to parse metadata_id {metadata_id}: {str(e)}")
                continue
        
        return results
    
    def _filter_by_document(
        self,
        results: List[SearchResult],
        document_id: str
    ) -> List[SearchResult]:
        """
        Filter results to specific document.
        
        Args:
            results: List of search results
            document_id: Document UUID to filter by
        
        Returns:
            Filtered results
        """
        filtered = [r for r in results if r.document_id == document_id]
        
        logger.debug(
            f"Document filter: {len(filtered)} results (from {len(results)})"
        )
        
        return filtered
    
    def _apply_language_boost(
        self,
        results: List[SearchResult],
        query_language: str
    ) -> List[SearchResult]:
        """
        Apply language-aware scoring boost.
        
        Tamil queries boost Tamil chunks, English queries boost English chunks.
        
        Args:
            results: List of search results
            query_language: Detected query language ("en" or "ta")
        
        Returns:
            Re-sorted results with boosted scores
        """
        logger.debug(f"Applying language boost for query_language={query_language}")
        
        boost_factor = 1.1  # 10% boost for matching language
        
        for result in results:
            if result.language == query_language:
                result.similarity_score *= boost_factor
                logger.debug(
                    f"Boosted chunk {result.chunk_id} "
                    f"(lang={result.language}) to {result.similarity_score:.4f}"
                )
        
        # Re-sort by boosted scores
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        
        return results
    
    def search_by_document(
        self,
        query: str,
        document_id: str,
        top_k: int = 5
    ) -> SearchResponse:
        """
        Search within a specific document only.
        
        Convenience method for document-scoped search.
        
        Args:
            query: Search query
            document_id: Document UUID to search within
            top_k: Number of results
        
        Returns:
            SearchResponse with results from specified document only
        """
        return self.search(
            query=query,
            top_k=top_k,
            document_id=document_id,
            language_boost=True
        )
    
    def get_similar_chunks(
        self,
        document_id: str,
        chunk_id: int,
        top_k: int = 5
    ) -> SearchResponse:
        """
        Find chunks similar to a specific chunk.
        
        Args:
            document_id: Source document UUID
            chunk_id: Source chunk ID
            top_k: Number of similar chunks to return
        
        Returns:
            SearchResponse with similar chunks
        """
        # Get the source chunk metadata
        metadata = self.metadata_store.get_by_chunk(document_id, chunk_id)
        
        if not metadata:
            logger.warning(f"Chunk not found: {document_id}_chunk_{chunk_id}")
            return SearchResponse(
                query=f"[Similar to chunk {chunk_id}]",
                query_language="unknown",
                results=[],
                total_results=0,
                search_time_ms=0.0
            )
        
        # Use the text preview as query
        query = metadata.text_preview or ""
        
        logger.info(
            f"Finding chunks similar to {document_id}_chunk_{chunk_id}"
        )
        
        return self.search(
            query=query,
            top_k=top_k + 1,  # +1 because source chunk will be included
            document_id=None,  # Search across all documents
            language_boost=False
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the search system.
        
        Returns:
            dict: Statistics including index size, languages, etc.
        """
        return {
            'total_vectors': self.vector_store.get_total_vectors(),
            'metadata_entries': self.metadata_store.count(),
            'metadata_stats': self.metadata_store.get_stats(),
            'embedding_dimension': self.embedding_service.get_embedding_dimension(),
        }


if __name__ == "__main__":
    """
    Test and example usage of SemanticSearchService.
    """
    import sys
    from pathlib import Path
    
    # Add backend to path
    backend_dir = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(backend_dir))
    
    from app.models.schemas import TextChunk
    from app.services.vector_ingestion import VectorIngestionPipeline
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== Semantic Search Service Test ===\n")
    
    # Step 1: Create and ingest test data
    print("Step 1: Ingesting test documents...\n")
    
    test_chunks = [
        TextChunk(
            chunk_id=1,
            text="This property deed establishes ownership rights for the real estate.",
            clause_number="1.1",
            heading="DEFINITIONS",
            page_number=1,
            page_range=[1],
            language="en",
            token_count=10
        ),
        TextChunk(
            chunk_id=2,
            text="The buyer shall pay the agreed purchase amount within thirty days.",
            clause_number="2.1",
            heading="PAYMENT TERMS",
            page_number=2,
            page_range=[2],
            language="en",
            token_count=11
        ),
        TextChunk(
            chunk_id=3,
            text="இந்த சொத்து ஆவணம் ரியல் எஸ்டேட்டுக்கான உரிமையை நிறுவுகிறது.",
            clause_number="௧.௧",
            heading="வரையறைகள்",
            page_number=1,
            page_range=[1],
            language="ta",
            token_count=8
        ),
        TextChunk(
            chunk_id=4,
            text="வாங்குபவர் ஒப்புக்கொண்ட தொகையை முப்பது நாட்களுக்குள் செலுத்த வேண்டும்.",
            clause_number="௨.௧",
            heading="கட்டண விதிமுறைகள்",
            page_number=2,
            page_range=[2],
            language="ta",
            token_count=10
        ),
    ]
    
    # Ingest documents
    pipeline = VectorIngestionPipeline()
    result = pipeline.ingest_document("test-doc-001", test_chunks)
    
    print(f"Ingested {result.vectors_indexed} chunks\n")
    
    # Step 2: Initialize search service
    print("Step 2: Initializing search service...\n")
    search_service = SemanticSearchService(
        embedding_service=pipeline.embedding_service,
        vector_store=pipeline.vector_store,
        metadata_store=pipeline.metadata_store
    )
    
    # Step 3: Perform searches
    print("Step 3: Testing searches...\n")
    
    # English query
    print("Search 1: English query - 'property ownership'")
    response = search_service.search("property ownership", top_k=3)
    print(f"  Query language: {response.query_language}")
    print(f"  Results: {response.total_results}")
    print(f"  Search time: {response.search_time_ms:.2f}ms")
    for result in response.results:
        print(f"    {result.rank}. [{result.language}] {result.text_preview[:50]}...")
        print(f"       Score: {result.similarity_score:.4f}, Clause: {result.clause_reference}")
    print()
    
    # Tamil query
    print("Search 2: Tamil query - 'சொத்து உரிமை'")
    response = search_service.search("சொத்து உரிமை", top_k=3)
    print(f"  Query language: {response.query_language}")
    print(f"  Results: {response.total_results}")
    for result in response.results:
        print(f"    {result.rank}. [{result.language}] {result.text_preview[:50]}...")
        print(f"       Score: {result.similarity_score:.4f}")
    print()
    
    # Payment terms query
    print("Search 3: 'payment terms within days'")
    response = search_service.search("payment terms within days", top_k=2)
    print(f"  Results: {response.total_results}")
    for result in response.results:
        print(f"    {result.rank}. {result.text_preview[:50]}...")
        print(f"       Score: {result.similarity_score:.4f}")
    print()
    
    # Stats
    print("Search System Statistics:")
    stats = search_service.get_stats()
    print(f"  Total vectors: {stats['total_vectors']}")
    print(f"  Metadata entries: {stats['metadata_entries']}")
    print(f"  Languages: {stats['metadata_stats']['languages']}")
