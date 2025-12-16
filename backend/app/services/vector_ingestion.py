"""
Vector Ingestion Pipeline

Orchestrates the process of converting legal document chunks into searchable
vector embeddings. This pipeline coordinates embedding generation, FAISS indexing,
and metadata storage.

Pipeline Flow:
1. Accept processed document chunks (TextChunk objects)
2. Extract text from each chunk
3. Generate embeddings using local embedding model
4. Insert embeddings into FAISS vector store
5. Store metadata (document_id, chunk_id, clause, page, etc.)
6. Return indexing statistics

Design Principles:
- Document-level isolation: Each document is processed independently
- Atomic operations: All chunks for a document succeed or fail together
- Idempotent: Re-running with same document_id replaces old vectors
- Error resilience: Detailed error logging with partial success handling
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.models.schemas import TextChunk
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.vector_store import FAISSIndexManager, create_faiss_manager
from app.services.metadata_store import VectorMetadataStore, VectorMetadata
from app.config.embedding_config import DEFAULT_CONFIG


logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """
    Result of vector ingestion pipeline.
    
    Attributes:
        document_id: Document UUID
        vectors_indexed: Number of vectors successfully indexed
        chunks_processed: Number of chunks processed
        chunks_failed: Number of chunks that failed
        embedding_dimension: Dimension of generated embeddings
        success: Whether ingestion completed successfully
        error: Error message if failed
    """
    document_id: str
    vectors_indexed: int
    chunks_processed: int
    chunks_failed: int
    embedding_dimension: int
    success: bool
    error: Optional[str] = None


class VectorIngestionPipeline:
    """
    Pipeline for ingesting legal document chunks into vector search system.
    
    This class coordinates three services:
    1. EmbeddingService: Generates vector embeddings from text
    2. FAISSIndexManager: Stores embeddings in searchable index
    3. VectorMetadataStore: Maintains chunk metadata
    
    Example:
        >>> pipeline = VectorIngestionPipeline()
        >>> chunks = [...]  # List of TextChunk objects
        >>> result = pipeline.ingest_document("doc-123", chunks)
        >>> print(f"Indexed {result.vectors_indexed} vectors")
    """
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[FAISSIndexManager] = None,
        metadata_store: Optional[VectorMetadataStore] = None
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            embedding_service: Service for generating embeddings (creates if None)
            vector_store: FAISS index manager (creates if None)
            metadata_store: Metadata store (creates if None)
        """
        # Initialize services
        self.embedding_service = embedding_service or get_embedding_service()
        
        self.vector_store = vector_store or create_faiss_manager(
            dimension=DEFAULT_CONFIG.embedding_dimension
        )
        
        self.metadata_store = metadata_store or VectorMetadataStore()
        
        logger.info(
            f"Initialized VectorIngestionPipeline with "
            f"embedding_dim={self.embedding_service.get_embedding_dimension()}"
        )
    
    def ingest_document(
        self,
        document_id: str,
        chunks: List[TextChunk],
        replace_existing: bool = True
    ) -> IngestionResult:
        """
        Ingest all chunks for a document into the vector search system.
        
        This method processes all chunks atomically at the document level:
        - If replace_existing=True: Removes old vectors before adding new ones
        - Generates embeddings for all chunks
        - Adds embeddings to FAISS index
        - Stores metadata for each chunk
        
        Args:
            document_id: Document UUID
            chunks: List of TextChunk objects from document processing
            replace_existing: Whether to replace existing vectors for this document
        
        Returns:
            IngestionResult: Statistics about the ingestion process
        
        Example:
            >>> pipeline = VectorIngestionPipeline()
            >>> chunks = [
            ...     TextChunk(text="Property deed...", chunk_id=1, ...),
            ...     TextChunk(text="Payment terms...", chunk_id=2, ...)
            ... ]
            >>> result = pipeline.ingest_document("doc-123", chunks)
            >>> if result.success:
            ...     print(f"Indexed {result.vectors_indexed} chunks")
        """
        logger.info(
            f"[{document_id}] Starting ingestion for {len(chunks)} chunks "
            f"(replace_existing={replace_existing})"
        )
        
        if not chunks:
            logger.warning(f"[{document_id}] No chunks to ingest")
            return IngestionResult(
                document_id=document_id,
                vectors_indexed=0,
                chunks_processed=0,
                chunks_failed=0,
                embedding_dimension=self.embedding_service.get_embedding_dimension(),
                success=True
            )
        
        try:
            # Step 1: Remove existing vectors if requested
            if replace_existing:
                self._remove_existing_vectors(document_id)
            
            # Step 2: Extract text from chunks
            texts, chunk_metadata = self._prepare_chunks(document_id, chunks)
            
            # Step 3: Generate embeddings
            embeddings = self._generate_embeddings(document_id, texts)
            
            # Step 4: Add to FAISS index
            vector_ids = self._add_to_index(document_id, embeddings, chunk_metadata)
            
            # Step 5: Store metadata (including embeddings for rebuild)
            self._store_metadata(document_id, vector_ids, chunk_metadata, embeddings)
            
            # Success
            result = IngestionResult(
                document_id=document_id,
                vectors_indexed=len(vector_ids),
                chunks_processed=len(chunks),
                chunks_failed=0,
                embedding_dimension=len(embeddings[0]) if embeddings else 0,
                success=True
            )
            
            logger.info(
                f"[{document_id}] Ingestion completed: "
                f"{result.vectors_indexed} vectors indexed"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"[{document_id}] Ingestion failed: {str(e)}", exc_info=True)
            
            return IngestionResult(
                document_id=document_id,
                vectors_indexed=0,
                chunks_processed=len(chunks),
                chunks_failed=len(chunks),
                embedding_dimension=self.embedding_service.get_embedding_dimension(),
                success=False,
                error=str(e)
            )
    
    def _remove_existing_vectors(self, document_id: str):
        """
        Remove all existing vectors for a document.
        
        This ensures document-level isolation and idempotency.
        """
        logger.debug(f"[{document_id}] Removing existing vectors")
        
        # Remove from metadata store (which tracks vector IDs)
        removed_count = self.metadata_store.remove_by_document(document_id)
        
        # Note: FAISS IndexFlat doesn't support efficient removal
        # Vectors remain in FAISS but won't be returned since metadata is gone
        
        if removed_count > 0:
            logger.info(f"[{document_id}] Removed {removed_count} existing vectors")
    
    def _prepare_chunks(
        self,
        document_id: str,
        chunks: List[TextChunk]
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """
        Extract text and metadata from chunks.
        
        Returns:
            tuple: (list of texts, list of metadata dicts)
        """
        logger.debug(f"[{document_id}] Preparing {len(chunks)} chunks")
        
        texts = []
        metadata_list = []
        
        for chunk in chunks:
            # Extract text
            texts.append(chunk.text)
            
            # Prepare metadata
            metadata = {
                'chunk_id': chunk.chunk_id,
                'clause_reference': chunk.clause_number,
                'page_number': chunk.page_number,
                'language': chunk.language,
                'text_preview': chunk.text[:100] if chunk.text else None,
                'chunk_data': chunk  # Keep full chunk for later reference
            }
            metadata_list.append(metadata)
        
        logger.debug(f"[{document_id}] Prepared {len(texts)} texts for embedding")
        
        return texts, metadata_list
    
    def _generate_embeddings(
        self,
        document_id: str,
        texts: List[str]
    ) -> List[List[float]]:
        """
        Generate embeddings for all texts.
        
        Args:
            document_id: Document UUID (for logging)
            texts: List of text strings
        
        Returns:
            List of embedding vectors
        
        Raises:
            RuntimeError: If embedding generation fails
        """
        logger.info(f"[{document_id}] Generating embeddings for {len(texts)} texts")
        
        try:
            embeddings = self.embedding_service.embed_texts(
                texts=texts,
                show_progress_bar=False
            )
            
            logger.info(
                f"[{document_id}] Generated {len(embeddings)} embeddings "
                f"(dimension: {len(embeddings[0])})"
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"[{document_id}] Embedding generation failed: {str(e)}")
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}") from e
    
    def _add_to_index(
        self,
        document_id: str,
        embeddings: List[List[float]],
        chunk_metadata: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Add embeddings to FAISS index.
        
        Args:
            document_id: Document UUID
            embeddings: List of embedding vectors
            chunk_metadata: List of metadata dicts
        
        Returns:
            List of FAISS vector IDs
        
        Raises:
            RuntimeError: If indexing fails
        """
        logger.debug(f"[{document_id}] Adding {len(embeddings)} vectors to FAISS")
        
        try:
            # Create metadata IDs for FAISS (chunk_id as string)
            metadata_ids = [
                f"{document_id}_chunk_{meta['chunk_id']}"
                for meta in chunk_metadata
            ]
            
            # Add to FAISS index
            vector_ids = self.vector_store.add_embeddings(
                embeddings=embeddings,
                metadata_ids=metadata_ids
            )
            
            logger.info(
                f"[{document_id}] Added {len(vector_ids)} vectors to FAISS index "
                f"(total: {self.vector_store.get_total_vectors()})"
            )
            
            return vector_ids
            
        except Exception as e:
            logger.error(f"[{document_id}] FAISS indexing failed: {str(e)}")
            raise RuntimeError(f"Failed to add to FAISS index: {str(e)}") from e
    
    def _store_metadata(
        self,
        document_id: str,
        vector_ids: List[int],
        chunk_metadata: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ):
        """
        Store metadata for all vectors, including embeddings for index rebuild.
        
        Args:
            document_id: Document UUID
            vector_ids: List of FAISS vector IDs
            chunk_metadata: List of metadata dicts
            embeddings: List of embedding vectors (for rebuild support)
        
        Raises:
            RuntimeError: If metadata storage fails
        """
        logger.debug(f"[{document_id}] Storing metadata for {len(vector_ids)} vectors")
        
        try:
            # Create VectorMetadata objects
            metadata_objects = []
            
            for vector_id, chunk_meta, embedding in zip(vector_ids, chunk_metadata, embeddings):
                metadata = VectorMetadata(
                    vector_id=vector_id,
                    document_id=document_id,
                    chunk_id=chunk_meta['chunk_id'],
                    clause_reference=chunk_meta.get('clause_reference'),
                    page_number=chunk_meta.get('page_number'),
                    language=chunk_meta.get('language', 'en'),
                    text_preview=chunk_meta.get('text_preview'),
                    embedding=embedding  # Store embedding for rebuild
                )
                metadata_objects.append(metadata)
            
            # Bulk insert
            new_count = self.metadata_store.add_bulk(metadata_objects)
            
            logger.info(
                f"[{document_id}] Stored metadata for {len(metadata_objects)} vectors "
                f"({new_count} new)"
            )
            
        except Exception as e:
            logger.error(f"[{document_id}] Metadata storage failed: {str(e)}")
            raise RuntimeError(f"Failed to store metadata: {str(e)}") from e
    
    def get_document_vectors(self, document_id: str) -> List[VectorMetadata]:
        """
        Retrieve all vector metadata for a document.
        
        Args:
            document_id: Document UUID
        
        Returns:
            List of VectorMetadata objects
        """
        return self.metadata_store.get_by_document(document_id)
    
    def remove_document(self, document_id: str) -> int:
        """
        Remove all vectors for a document and rebuild FAISS index.
        
        This is the clean approach: removes the document's metadata and
        rebuilds the FAISS index without the removed vectors.
        
        Steps:
        1. Remove document metadata
        2. Get all remaining metadata (with embeddings)
        3. Rebuild FAISS index from remaining vectors
        
        Args:
            document_id: Document UUID
        
        Returns:
            int: Number of vectors removed
        """
        logger.info(f"[{document_id}] Removing document and rebuilding index")
        
        # Step 1: Remove metadata for this document
        removed_count = self.metadata_store.remove_by_document(document_id)
        logger.info(f"[{document_id}] Removed {removed_count} metadata entries")
        
        if removed_count == 0:
            logger.warning(f"[{document_id}] No vectors found to remove")
            return 0
        
        # Step 2: Get all remaining metadata with embeddings
        logger.info("Collecting remaining vectors for index rebuild...")
        all_metadata = self.metadata_store.get_all()
        
        if not all_metadata:
            logger.info("No remaining vectors, clearing index")
            self.vector_store.clear()
            return removed_count
        
        # Step 3: Prepare embeddings and metadata IDs for rebuild
        embeddings_with_metadata = []
        for meta in all_metadata:
            if meta.embedding is not None:
                metadata_id = f"{meta.document_id}_chunk_{meta.chunk_id}"
                embeddings_with_metadata.append((meta.embedding, metadata_id))
            else:
                logger.warning(
                    f"Metadata {meta.vector_id} has no embedding, skipping in rebuild"
                )
        
        # Step 4: Rebuild FAISS index
        logger.info(
            f"Rebuilding FAISS index with {len(embeddings_with_metadata)} remaining vectors"
        )
        self.vector_store.rebuild_index(embeddings_with_metadata)
        
        # Step 5: Update metadata with new vector IDs
        logger.info("Updating metadata with new vector IDs...")
        self._update_metadata_vector_ids()
        
        logger.info(
            f"[{document_id}] ✓ Removed {removed_count} vectors and rebuilt index "
            f"({self.vector_store.get_total_vectors()} vectors remaining)"
        )
        
        return removed_count
    
    def _update_metadata_vector_ids(self):
        """
        Update metadata store with new vector IDs after rebuild.
        
        After rebuilding FAISS index, vector IDs change (sequential from 0).
        This method updates the metadata store to reflect new IDs.
        """
        # Get all metadata sorted by document and chunk
        all_metadata = self.metadata_store.get_all()
        
        # FAISS assigns IDs sequentially in order of insertion
        # So we just need to reassign IDs sequentially
        for new_id, meta in enumerate(all_metadata):
            old_id = meta.vector_id
            meta.vector_id = new_id
            
            # Update in metadata store
            self.metadata_store.add(meta)  # Overwrites existing
            
            logger.debug(f"Updated vector_id {old_id} -> {new_id}")
        
        logger.info(f"Updated {len(all_metadata)} metadata entries with new vector IDs")
    
    def save_to_disk(self, index_name: str = "index", metadata_name: str = "vector_metadata.json"):
        """
        Persist both FAISS index and metadata to disk.
        
        Args:
            index_name: Base name for FAISS index files
            metadata_name: Name for metadata JSON file
        """
        logger.info(f"Saving index and metadata to disk")
        
        # Save FAISS index
        index_path = self.vector_store.save_index(index_name)
        logger.info(f"Saved FAISS index to {index_path}")
        
        # Save metadata
        metadata_path = self.metadata_store.save(metadata_name)
        logger.info(f"Saved metadata to {metadata_path}")
    
    def load_from_disk(self, index_name: str = "index", metadata_name: str = "vector_metadata.json") -> bool:
        """
        Load FAISS index and metadata from disk.
        
        Args:
            index_name: Base name for FAISS index files
            metadata_name: Name for metadata JSON file
        
        Returns:
            bool: True if loaded successfully
        """
        logger.info(f"Loading index and metadata from disk")
        
        # Load FAISS index
        index_success = self.vector_store.load_index(index_name)
        if not index_success:
            logger.error("Failed to load FAISS index")
            return False
        
        # Load metadata
        metadata_success = self.metadata_store.load(metadata_name)
        if not metadata_success:
            logger.error("Failed to load metadata")
            return False
        
        logger.info(
            f"Loaded {self.vector_store.get_total_vectors()} vectors and "
            f"{self.metadata_store.count()} metadata entries"
        )
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the ingestion system.
        
        Returns:
            dict: Statistics from all components
        """
        return {
            'embedding': self.embedding_service.get_model_info(),
            'vector_store': self.vector_store.get_stats(),
            'metadata_store': self.metadata_store.get_stats(),
        }


if __name__ == "__main__":
    """
    Test and example usage of VectorIngestionPipeline.
    """
    import sys
    from pathlib import Path
    
    # Add backend to path
    backend_dir = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(backend_dir))
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== Vector Ingestion Pipeline Test ===\n")
    
    # Create test chunks
    test_chunks = [
        TextChunk(
            chunk_id=1,
            text="This property deed establishes ownership rights.",
            clause_number="1.1",
            heading="DEFINITIONS",
            page_number=1,
            page_range=[1],
            language="en",
            token_count=8
        ),
        TextChunk(
            chunk_id=2,
            text="The buyer shall pay the agreed amount within 30 days.",
            clause_number="2.1",
            heading="PAYMENT TERMS",
            page_number=2,
            page_range=[2],
            language="en",
            token_count=10
        ),
        TextChunk(
            chunk_id=3,
            text="இந்த சொத்து ஆவணம் உரிமையை நிறுவுகிறது.",
            clause_number="௧.௧",
            heading="வரையறைகள்",
            page_number=1,
            page_range=[1],
            language="ta",
            token_count=7
        ),
    ]
    
    print(f"Created {len(test_chunks)} test chunks\n")
    
    # Initialize pipeline
    print("Initializing pipeline (this will load the embedding model)...")
    pipeline = VectorIngestionPipeline()
    print("Pipeline initialized\n")
    
    # Ingest document
    document_id = "test-doc-123"
    print(f"Ingesting document: {document_id}")
    result = pipeline.ingest_document(document_id, test_chunks)
    
    print("\nIngestion Result:")
    print(f"  Success: {result.success}")
    print(f"  Vectors Indexed: {result.vectors_indexed}")
    print(f"  Chunks Processed: {result.chunks_processed}")
    print(f"  Chunks Failed: {result.chunks_failed}")
    print(f"  Embedding Dimension: {result.embedding_dimension}")
    if result.error:
        print(f"  Error: {result.error}")
    print()
    
    # Get stats
    print("Pipeline Statistics:")
    stats = pipeline.get_stats()
    print(f"  Total Vectors: {stats['vector_store']['total_vectors']}")
    print(f"  Metadata Entries: {stats['metadata_store']['total_vectors']}")
    print(f"  Languages: {stats['metadata_store']['languages']}")
    print()
    
    # Save to disk
    print("Saving to disk...")
    pipeline.save_to_disk("test_index", "test_metadata.json")
    print("Saved successfully")
