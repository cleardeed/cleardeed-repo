"""
FAISS Vector Store Manager

This module implements a vector similarity search system using FAISS (Facebook AI 
Similarity Search) for storing and querying document embeddings.

Index Type Selection:
    - IndexFlatIP: Used for cosine similarity with L2-normalized embeddings
      - IP = Inner Product (dot product)
      - For normalized vectors, dot product equals cosine similarity
      - Exact search, no approximation
      - Best for small to medium datasets (<1M vectors)
    
    - IndexFlatL2: Alternative for L2 (Euclidean) distance
      - Use when embeddings are NOT normalized
      - Measures geometric distance in embedding space

Thread Safety:
    All operations are protected with threading locks to ensure safe concurrent access.
"""

import logging
import pickle
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import faiss

from app.config.embedding_config import DEFAULT_CONFIG


logger = logging.getLogger(__name__)


class FAISSIndexManager:
    """
    Thread-safe FAISS index manager for vector similarity search.
    
    This class manages a FAISS index for storing and searching document chunk
    embeddings. It supports:
    - Adding vectors with metadata
    - Searching for similar vectors
    - Persisting index to disk
    - Loading index from disk
    
    Index Type:
        Uses IndexFlatIP (Inner Product) because our embeddings are L2-normalized.
        For normalized vectors: cosine_similarity = dot_product
        This provides exact cosine similarity search.
    
    Thread Safety:
        All operations are protected with locks for concurrent access.
    
    Example:
        >>> manager = FAISSIndexManager(dimension=1024)
        >>> manager.add_embeddings(
        ...     embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
        ...     metadata_ids=["chunk_1", "chunk_2"]
        ... )
        >>> results = manager.search(query_embedding=[0.15, 0.25, ...], k=5)
    """
    
    def __init__(
        self, 
        dimension: int,
        index_type: str = "IndexFlatIP",
        index_path: Optional[Path] = None
    ):
        """
        Initialize FAISS index manager.
        
        Args:
            dimension: Embedding vector dimension (e.g., 1024 for multilingual-e5-large)
            index_type: Type of FAISS index ("IndexFlatIP" or "IndexFlatL2")
            index_path: Path to save/load index (defaults to data/faiss/)
        
        Raises:
            ValueError: If dimension <= 0 or unsupported index_type
        """
        if dimension <= 0:
            raise ValueError(f"Dimension must be positive, got {dimension}")
        
        if index_type not in ["IndexFlatIP", "IndexFlatL2"]:
            raise ValueError(
                f"Unsupported index_type: {index_type}. "
                "Use 'IndexFlatIP' or 'IndexFlatL2'"
            )
        
        self.dimension = dimension
        self.index_type = index_type
        self.index_path = index_path or Path("data/faiss")
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize FAISS index
        self._index: Optional[faiss.Index] = None
        self._metadata: Dict[int, str] = {}  # Maps FAISS index -> metadata ID
        self._reverse_metadata: Dict[str, int] = {}  # Maps metadata ID -> FAISS index
        self._next_id = 0
        
        # Create index
        self._create_index()
        
        logger.info(
            f"Initialized FAISSIndexManager: dimension={dimension}, "
            f"type={index_type}, path={self.index_path}"
        )
    
    def _create_index(self):
        """
        Create a new FAISS index based on the specified type.
        
        IndexFlatIP:
            - Inner Product (dot product) similarity
            - Best for normalized embeddings (cosine similarity)
            - Returns: Higher scores = more similar
        
        IndexFlatL2:
            - L2 (Euclidean) distance
            - Best for non-normalized embeddings
            - Returns: Lower scores = more similar
        """
        with self._lock:
            if self.index_type == "IndexFlatIP":
                # Inner product index for cosine similarity (normalized vectors)
                self._index = faiss.IndexFlatIP(self.dimension)
                logger.debug("Created IndexFlatIP (cosine similarity for normalized vectors)")
            
            elif self.index_type == "IndexFlatL2":
                # L2 distance index for Euclidean distance
                self._index = faiss.IndexFlatL2(self.dimension)
                logger.debug("Created IndexFlatL2 (Euclidean distance)")
    
    def add_embeddings(
        self, 
        embeddings: List[List[float]], 
        metadata_ids: List[str]
    ) -> List[int]:
        """
        Add embeddings to the FAISS index with associated metadata.
        
        Args:
            embeddings: List of embedding vectors (each vector is a list of floats)
            metadata_ids: List of metadata identifiers (e.g., chunk IDs)
        
        Returns:
            List of internal FAISS indices assigned to each embedding
        
        Raises:
            ValueError: If embeddings and metadata_ids have different lengths
            ValueError: If embedding dimension doesn't match index dimension
        
        Example:
            >>> manager = FAISSIndexManager(dimension=1024)
            >>> embeddings = [[0.1, 0.2, ...], [0.3, 0.4, ...]]
            >>> ids = ["chunk_1", "chunk_2"]
            >>> faiss_ids = manager.add_embeddings(embeddings, ids)
        """
        if len(embeddings) != len(metadata_ids):
            raise ValueError(
                f"Embeddings and metadata_ids must have same length: "
                f"got {len(embeddings)} vs {len(metadata_ids)}"
            )
        
        if not embeddings:
            logger.warning("add_embeddings called with empty list")
            return []
        
        # Convert to numpy array and validate dimensions
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        if embeddings_array.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {embeddings_array.shape[1]}"
            )
        
        with self._lock:
            # Get starting index
            start_idx = self._next_id
            
            # Add to FAISS index
            self._index.add(embeddings_array)
            
            # Store metadata mappings
            faiss_indices = []
            for i, metadata_id in enumerate(metadata_ids):
                faiss_idx = start_idx + i
                self._metadata[faiss_idx] = metadata_id
                self._reverse_metadata[metadata_id] = faiss_idx
                faiss_indices.append(faiss_idx)
            
            # Update next ID
            self._next_id += len(embeddings)
            
            logger.info(
                f"Added {len(embeddings)} embeddings to index "
                f"(total: {self._index.ntotal})"
            )
            
            return faiss_indices
    
    def search(
        self, 
        query_embedding: List[float], 
        k: int = 5,
        return_scores: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Search for the k most similar vectors in the index.
        
        Args:
            query_embedding: Query vector to search for
            k: Number of nearest neighbors to return
            return_scores: Whether to include similarity scores
        
        Returns:
            List of tuples: (metadata_id, similarity_score)
            - For IndexFlatIP: Higher scores = more similar
            - For IndexFlatL2: Lower scores = more similar
        
        Example:
            >>> results = manager.search(
            ...     query_embedding=[0.15, 0.25, ...],
            ...     k=5
            ... )
            >>> for metadata_id, score in results:
            ...     print(f"{metadata_id}: {score:.4f}")
        """
        if not query_embedding:
            logger.warning("search called with empty query_embedding")
            return []
        
        # Convert to numpy array
        query_array = np.array([query_embedding], dtype=np.float32)
        
        if query_array.shape[1] != self.dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.dimension}, "
                f"got {query_array.shape[1]}"
            )
        
        with self._lock:
            # Check if index is empty
            if self._index.ntotal == 0:
                logger.warning("Search called on empty index")
                return []
            
            # Limit k to available vectors
            k = min(k, self._index.ntotal)
            
            # Perform search
            distances, indices = self._index.search(query_array, k)
            
            # Build results with metadata
            results = []
            for i in range(k):
                faiss_idx = int(indices[0][i])
                distance = float(distances[0][i])
                
                # Get metadata ID
                if faiss_idx in self._metadata:
                    metadata_id = self._metadata[faiss_idx]
                    
                    if return_scores:
                        results.append((metadata_id, distance))
                    else:
                        results.append((metadata_id, None))
                else:
                    logger.warning(f"No metadata found for FAISS index {faiss_idx}")
            
            logger.debug(f"Search returned {len(results)} results")
            
            return results
    
    def batch_search(
        self,
        query_embeddings: List[List[float]],
        k: int = 5
    ) -> List[List[Tuple[str, float]]]:
        """
        Search for multiple query vectors at once (more efficient).
        
        Args:
            query_embeddings: List of query vectors
            k: Number of nearest neighbors per query
        
        Returns:
            List of result lists, one for each query
        """
        if not query_embeddings:
            return []
        
        # Convert to numpy array
        query_array = np.array(query_embeddings, dtype=np.float32)
        
        with self._lock:
            if self._index.ntotal == 0:
                logger.warning("Batch search called on empty index")
                return [[] for _ in query_embeddings]
            
            # Limit k
            k = min(k, self._index.ntotal)
            
            # Perform batch search
            distances, indices = self._index.search(query_array, k)
            
            # Build results for each query
            all_results = []
            for query_idx in range(len(query_embeddings)):
                query_results = []
                for i in range(k):
                    faiss_idx = int(indices[query_idx][i])
                    distance = float(distances[query_idx][i])
                    
                    if faiss_idx in self._metadata:
                        metadata_id = self._metadata[faiss_idx]
                        query_results.append((metadata_id, distance))
                
                all_results.append(query_results)
            
            logger.debug(f"Batch search processed {len(query_embeddings)} queries")
            
            return all_results
    
    def remove_by_metadata_id(self, metadata_id: str) -> bool:
        """
        Remove an embedding from the index by its metadata ID.
        
        Note: This only removes metadata mapping. To actually remove vectors
        from FAISS, call rebuild_index() after removing metadata.
        
        Args:
            metadata_id: Metadata identifier to remove
        
        Returns:
            bool: True if removed, False if not found
        """
        with self._lock:
            if metadata_id in self._reverse_metadata:
                faiss_idx = self._reverse_metadata[metadata_id]
                
                # Remove from metadata mappings
                del self._reverse_metadata[metadata_id]
                del self._metadata[faiss_idx]
                
                logger.info(f"Removed metadata for ID: {metadata_id}")
                return True
            else:
                logger.warning(f"Metadata ID not found: {metadata_id}")
                return False
    
    def rebuild_index(self, embeddings_with_metadata: List[tuple[List[float], str]]) -> int:
        """
        Rebuild FAISS index from scratch with provided embeddings.
        
        This is the clean way to remove vectors: rebuild the entire index
        with only the vectors you want to keep.
        
        Args:
            embeddings_with_metadata: List of (embedding, metadata_id) tuples
        
        Returns:
            int: Number of vectors in rebuilt index
        
        Example:
            >>> # After removing document metadata
            >>> remaining = [(emb1, "doc1_chunk1"), (emb2, "doc1_chunk2")]
            >>> manager.rebuild_index(remaining)
        """
        with self._lock:
            logger.info(f"Rebuilding FAISS index with {len(embeddings_with_metadata)} vectors")
            
            # Create new empty index
            self._create_index()
            
            # Reset metadata
            self._metadata.clear()
            self._reverse_metadata.clear()
            self._next_id = 0
            
            if not embeddings_with_metadata:
                logger.info("Rebuilt empty index")
                return 0
            
            # Separate embeddings and metadata IDs
            embeddings = [emb for emb, _ in embeddings_with_metadata]
            metadata_ids = [mid for _, mid in embeddings_with_metadata]
            
            # Add all embeddings back
            self.add_embeddings(embeddings, metadata_ids)
            
            logger.info(
                f"Index rebuilt successfully: {self._index.ntotal} vectors, "
                f"{len(self._metadata)} metadata entries"
            )
            
            return self._index.ntotal
    
    def get_total_vectors(self) -> int:
        """
        Get the total number of vectors in the index.
        
        Returns:
            int: Number of vectors in FAISS index
        """
        with self._lock:
            return self._index.ntotal if self._index else 0
    
    def get_metadata_count(self) -> int:
        """
        Get the number of metadata entries.
        
        Returns:
            int: Number of metadata mappings
        """
        with self._lock:
            return len(self._metadata)
    
    def clear(self):
        """
        Clear the index completely (remove all vectors).
        """
        with self._lock:
            logger.info("Clearing FAISS index")
            self._create_index()
            self._metadata.clear()
            self._reverse_metadata.clear()
            self._next_id = 0
            logger.info("✓ Index cleared")
    
    def save_index(self, filename: Optional[str] = None) -> Path:
        """
        Save FAISS index and metadata to disk.
        
        Saves two files:
        - {filename}.index: FAISS index binary
        - {filename}.metadata: Pickled metadata dictionary
        
        Args:
            filename: Base filename (without extension). Defaults to 'index'
        
        Returns:
            Path: Path to saved index file
        
        Example:
            >>> manager.save_index("my_index")
            # Creates: data/faiss/my_index.index and my_index.metadata
        """
        if filename is None:
            filename = "index"
        
        # Ensure directory exists
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        index_file = self.index_path / f"{filename}.index"
        metadata_file = self.index_path / f"{filename}.metadata"
        
        with self._lock:
            # Save FAISS index
            faiss.write_index(self._index, str(index_file))
            logger.info(f"Saved FAISS index to {index_file}")
            
            # Save metadata
            metadata_dict = {
                'metadata': self._metadata,
                'reverse_metadata': self._reverse_metadata,
                'next_id': self._next_id,
                'dimension': self.dimension,
                'index_type': self.index_type
            }
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata_dict, f)
            
            logger.info(f"Saved metadata to {metadata_file}")
            
            return index_file
    
    def load_index(self, filename: Optional[str] = None) -> bool:
        """
        Load FAISS index and metadata from disk.
        
        Args:
            filename: Base filename (without extension). Defaults to 'index'
        
        Returns:
            bool: True if loaded successfully, False otherwise
        
        Example:
            >>> manager = FAISSIndexManager(dimension=1024)
            >>> manager.load_index("my_index")
        """
        if filename is None:
            filename = "index"
        
        index_file = self.index_path / f"{filename}.index"
        metadata_file = self.index_path / f"{filename}.metadata"
        
        # Check if files exist
        if not index_file.exists():
            logger.error(f"Index file not found: {index_file}")
            return False
        
        if not metadata_file.exists():
            logger.error(f"Metadata file not found: {metadata_file}")
            return False
        
        try:
            with self._lock:
                # Load FAISS index
                self._index = faiss.read_index(str(index_file))
                logger.info(f"Loaded FAISS index from {index_file}")
                
                # Load metadata
                with open(metadata_file, 'rb') as f:
                    metadata_dict = pickle.load(f)
                
                # Validate dimension match
                if metadata_dict['dimension'] != self.dimension:
                    logger.error(
                        f"Dimension mismatch: expected {self.dimension}, "
                        f"got {metadata_dict['dimension']}"
                    )
                    return False
                
                # Restore metadata
                self._metadata = metadata_dict['metadata']
                self._reverse_metadata = metadata_dict['reverse_metadata']
                self._next_id = metadata_dict['next_id']
                
                logger.info(
                    f"Loaded metadata with {len(self._metadata)} entries "
                    f"(total vectors: {self._index.ntotal})"
                )
                
                return True
        
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            return False
    
    def clear(self):
        """
        Clear all data from the index and metadata.
        Creates a new empty index.
        """
        with self._lock:
            self._create_index()
            self._metadata.clear()
            self._reverse_metadata.clear()
            self._next_id = 0
            logger.info("Cleared index and metadata")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            dict: Index statistics
        """
        with self._lock:
            return {
                'dimension': self.dimension,
                'index_type': self.index_type,
                'total_vectors': self._index.ntotal if self._index else 0,
                'metadata_count': len(self._metadata),
                'next_id': self._next_id,
                'index_path': str(self.index_path),
            }


def create_faiss_manager(
    dimension: Optional[int] = None,
    index_type: str = "IndexFlatIP"
) -> FAISSIndexManager:
    """
    Factory function to create a FAISS index manager with default configuration.
    
    Args:
        dimension: Embedding dimension (uses DEFAULT_CONFIG if None)
        index_type: FAISS index type
    
    Returns:
        FAISSIndexManager: Initialized index manager
    """
    if dimension is None:
        dimension = DEFAULT_CONFIG.embedding_dimension
    
    return FAISSIndexManager(dimension=dimension, index_type=index_type)


if __name__ == "__main__":
    """
    Test and example usage of FAISSIndexManager.
    """
    import random
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== FAISS Index Manager Test ===\n")
    
    # Create manager
    dimension = 128  # Small dimension for testing
    manager = FAISSIndexManager(dimension=dimension, index_type="IndexFlatIP")
    
    print(f"Created index: dimension={dimension}, type=IndexFlatIP\n")
    
    # Generate random embeddings
    num_vectors = 10
    embeddings = []
    metadata_ids = []
    
    for i in range(num_vectors):
        # Random normalized vector
        vec = [random.random() for _ in range(dimension)]
        norm = sum(x**2 for x in vec) ** 0.5
        vec = [x/norm for x in vec]  # Normalize
        
        embeddings.append(vec)
        metadata_ids.append(f"chunk_{i}")
    
    print(f"Generated {num_vectors} random embeddings")
    
    # Add to index
    faiss_ids = manager.add_embeddings(embeddings, metadata_ids)
    print(f"Added embeddings with FAISS IDs: {faiss_ids}\n")
    
    # Print stats
    stats = manager.get_stats()
    print("Index Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Search
    query = embeddings[0]  # Use first embedding as query
    print(f"Searching with query (should match chunk_0)...")
    results = manager.search(query, k=3)
    print("Top 3 results:")
    for metadata_id, score in results:
        print(f"  {metadata_id}: {score:.4f}")
    print()
    
    # Save index
    save_path = manager.save_index("test_index")
    print(f"Saved index to {save_path}\n")
    
    # Create new manager and load
    manager2 = FAISSIndexManager(dimension=dimension)
    success = manager2.load_index("test_index")
    print(f"Loaded index: {success}")
    print(f"Loaded vectors: {manager2.get_total_vectors()}")
    print(f"Loaded metadata: {manager2.get_metadata_count()}")
