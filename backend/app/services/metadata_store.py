"""
Vector Metadata Store

Lightweight metadata storage for FAISS vector indices. Maps FAISS vector IDs
to document chunk metadata without requiring a database.

Design Philosophy:
- Simple: JSON-based persistence, no database overhead
- Fast: In-memory dictionary for O(1) lookups
- Reliable: Thread-safe operations with automatic persistence
- Scalable: Handles thousands of vectors efficiently

This store complements the FAISS index by maintaining the semantic information
associated with each vector (document ID, chunk details, etc.).
"""

import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime


logger = logging.getLogger(__name__)


@dataclass
class VectorMetadata:
    """
    Metadata associated with a single FAISS vector.
    
    Attributes:
        vector_id: FAISS internal index (int)
        document_id: Source document UUID
        chunk_id: Chunk identifier within document
        clause_reference: Legal clause number (e.g., "1.2.3", "Article 5")
        page_number: Source page number
        language: Text language ("en", "ta", "mixed")
        text_preview: First 100 chars of chunk text (for debugging)
        embedding: The actual vector embedding (stored for index rebuild)
        created_at: Timestamp when vector was added
    """
    vector_id: int
    document_id: str
    chunk_id: int
    clause_reference: Optional[str] = None
    page_number: Optional[int] = None
    language: str = "en"
    text_preview: Optional[str] = None
    embedding: Optional[List[float]] = None
    created_at: Optional[str] = None
    
    def __post_init__(self):
        """Set default created_at timestamp if not provided"""
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()
        
        # Truncate text preview if needed
        if self.text_preview and len(self.text_preview) > 100:
            self.text_preview = self.text_preview[:97] + "..."


class VectorMetadataStore:
    """
    Thread-safe metadata store for FAISS vectors.
    
    This class maintains a mapping from FAISS vector IDs to document metadata,
    persisted as JSON on disk. All operations are thread-safe.
    
    Storage Structure:
        {
            "0": {
                "vector_id": 0,
                "document_id": "doc-123",
                "chunk_id": 1,
                "clause_reference": "1.1",
                "page_number": 1,
                "language": "en",
                "text_preview": "This is the first clause...",
                "created_at": "2025-12-14T10:30:00"
            },
            "1": { ... }
        }
    
    Example:
        >>> store = VectorMetadataStore()
        >>> metadata = VectorMetadata(
        ...     vector_id=0,
        ...     document_id="doc-123",
        ...     chunk_id=1,
        ...     clause_reference="1.1",
        ...     page_number=1,
        ...     language="en"
        ... )
        >>> store.add(metadata)
        >>> retrieved = store.get(0)
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize metadata store.
        
        Args:
            storage_path: Directory to store metadata JSON files
                         (defaults to data/metadata/)
        """
        self.storage_path = storage_path or Path("data/metadata")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage: vector_id -> metadata dict
        self._store: Dict[int, Dict[str, Any]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Initialized VectorMetadataStore at {self.storage_path}")
    
    def add(self, metadata: VectorMetadata) -> bool:
        """
        Add metadata for a single vector.
        
        Args:
            metadata: VectorMetadata instance
        
        Returns:
            bool: True if added, False if already exists (overwrites)
        """
        with self._lock:
            vector_id = metadata.vector_id
            metadata_dict = asdict(metadata)
            
            exists = vector_id in self._store
            self._store[vector_id] = metadata_dict
            
            if exists:
                logger.debug(f"Updated metadata for vector_id {vector_id}")
            else:
                logger.debug(f"Added metadata for vector_id {vector_id}")
            
            return not exists
    
    def add_bulk(self, metadata_list: List[VectorMetadata]) -> int:
        """
        Add metadata for multiple vectors efficiently.
        
        Args:
            metadata_list: List of VectorMetadata instances
        
        Returns:
            int: Number of new entries added (excludes overwrites)
        """
        if not metadata_list:
            logger.warning("add_bulk called with empty list")
            return 0
        
        with self._lock:
            new_count = 0
            
            for metadata in metadata_list:
                vector_id = metadata.vector_id
                metadata_dict = asdict(metadata)
                
                if vector_id not in self._store:
                    new_count += 1
                
                self._store[vector_id] = metadata_dict
            
            logger.info(
                f"Bulk added {len(metadata_list)} metadata entries "
                f"({new_count} new, {len(metadata_list) - new_count} updated)"
            )
            
            return new_count
    
    def get(self, vector_id: int) -> Optional[VectorMetadata]:
        """
        Retrieve metadata by vector ID.
        
        Args:
            vector_id: FAISS vector index
        
        Returns:
            VectorMetadata if found, None otherwise
        """
        with self._lock:
            metadata_dict = self._store.get(vector_id)
            
            if metadata_dict is None:
                return None
            
            # Convert dict back to VectorMetadata
            return VectorMetadata(**metadata_dict)
    
    def get_bulk(self, vector_ids: List[int]) -> List[Optional[VectorMetadata]]:
        """
        Retrieve metadata for multiple vectors.
        
        Args:
            vector_ids: List of FAISS vector indices
        
        Returns:
            List of VectorMetadata (None for missing IDs)
        """
        with self._lock:
            results = []
            for vector_id in vector_ids:
                metadata_dict = self._store.get(vector_id)
                if metadata_dict:
                    results.append(VectorMetadata(**metadata_dict))
                else:
                    results.append(None)
            
            return results
    
    def get_by_document(self, document_id: str) -> List[VectorMetadata]:
        """
        Get all metadata for a specific document.
        
        Args:
            document_id: Document UUID
        
        Returns:
            List of VectorMetadata for this document
        """
        with self._lock:
            results = []
            
            for metadata_dict in self._store.values():
                if metadata_dict.get('document_id') == document_id:
                    results.append(VectorMetadata(**metadata_dict))
            
            logger.debug(f"Found {len(results)} vectors for document {document_id}")
            
            return results
    
    def get_by_chunk(self, document_id: str, chunk_id: int) -> Optional[VectorMetadata]:
        """
        Get metadata for a specific chunk.
        
        Args:
            document_id: Document UUID
            chunk_id: Chunk identifier
        
        Returns:
            VectorMetadata if found, None otherwise
        """
        with self._lock:
            for metadata_dict in self._store.values():
                if (metadata_dict.get('document_id') == document_id and 
                    metadata_dict.get('chunk_id') == chunk_id):
                    return VectorMetadata(**metadata_dict)
            
            return None
    
    def remove(self, vector_id: int) -> bool:
        """
        Remove metadata for a vector.
        
        Args:
            vector_id: FAISS vector index
        
        Returns:
            bool: True if removed, False if not found
        """
        with self._lock:
            if vector_id in self._store:
                del self._store[vector_id]
                logger.debug(f"Removed metadata for vector_id {vector_id}")
                return True
            else:
                logger.warning(f"Vector_id {vector_id} not found for removal")
                return False
    
    def remove_by_document(self, document_id: str) -> int:
        """
        Remove all metadata for a document.
        
        Args:
            document_id: Document UUID
        
        Returns:
            int: Number of entries removed
        """
        with self._lock:
            to_remove = []
            
            for vector_id, metadata_dict in self._store.items():
                if metadata_dict.get('document_id') == document_id:
                    to_remove.append(vector_id)
            
            for vector_id in to_remove:
                del self._store[vector_id]
            
            logger.info(f"Removed {len(to_remove)} vectors for document {document_id}")
            
            return len(to_remove)
    
    def count(self) -> int:
        """
        Get total number of metadata entries.
        
        Returns:
            int: Number of vectors with metadata
        """
        with self._lock:
            return len(self._store)
    
    def clear(self):
        """Clear all metadata from memory."""
        with self._lock:
            self._store.clear()
            logger.info("Cleared all metadata from store")
    
    def save(self, filename: str = "vector_metadata.json") -> Path:
        """
        Persist metadata to JSON file.
        
        Args:
            filename: Name of JSON file to create
        
        Returns:
            Path: Full path to saved file
        
        Example:
            >>> store.save("my_vectors.json")
        """
        filepath = self.storage_path / filename
        
        with self._lock:
            try:
                # Convert int keys to strings for JSON
                json_data = {
                    str(vector_id): metadata_dict
                    for vector_id, metadata_dict in self._store.items()
                }
                
                # Write to file with pretty formatting
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                
                logger.info(
                    f"Saved {len(self._store)} metadata entries to {filepath}"
                )
                
                return filepath
                
            except Exception as e:
                logger.error(f"Failed to save metadata: {str(e)}")
                raise
    
    def load(self, filename: str = "vector_metadata.json") -> bool:
        """
        Load metadata from JSON file.
        
        Args:
            filename: Name of JSON file to load
        
        Returns:
            bool: True if loaded successfully, False otherwise
        
        Example:
            >>> store.load("my_vectors.json")
        """
        filepath = self.storage_path / filename
        
        if not filepath.exists():
            logger.warning(f"Metadata file not found: {filepath}")
            return False
        
        try:
            with self._lock:
                with open(filepath, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # Convert string keys back to integers
                self._store = {
                    int(vector_id): metadata_dict
                    for vector_id, metadata_dict in json_data.items()
                }
                
                logger.info(
                    f"Loaded {len(self._store)} metadata entries from {filepath}"
                )
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to load metadata: {str(e)}")
            return False
    
    def get_all(self) -> List[VectorMetadata]:
        """
        Get all metadata entries.
        
        Returns:
            List of all VectorMetadata objects
        """
        with self._lock:
            results = []
            for metadata_dict in self._store.values():
                results.append(VectorMetadata(**metadata_dict))
            return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the metadata store.
        
        Returns:
            dict: Statistics including counts by language, document, etc.
        """
        with self._lock:
            if not self._store:
                return {
                    'total_vectors': 0,
                    'languages': {},
                    'documents': {},
                }
            
            # Count by language
            languages = {}
            documents = {}
            
            for metadata_dict in self._store.values():
                lang = metadata_dict.get('language', 'unknown')
                doc_id = metadata_dict.get('document_id', 'unknown')
                
                languages[lang] = languages.get(lang, 0) + 1
                documents[doc_id] = documents.get(doc_id, 0) + 1
            
            return {
                'total_vectors': len(self._store),
                'languages': languages,
                'documents': documents,
                'storage_path': str(self.storage_path),
            }
    
    def export_for_document(
        self, 
        document_id: str, 
        filename: Optional[str] = None
    ) -> Path:
        """
        Export metadata for a single document.
        
        Args:
            document_id: Document UUID
            filename: Output filename (auto-generated if None)
        
        Returns:
            Path: Path to exported file
        """
        if filename is None:
            filename = f"metadata_{document_id}.json"
        
        filepath = self.storage_path / filename
        
        with self._lock:
            # Filter metadata for this document
            document_metadata = {
                str(vector_id): metadata_dict
                for vector_id, metadata_dict in self._store.items()
                if metadata_dict.get('document_id') == document_id
            }
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(document_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(
                f"Exported {len(document_metadata)} metadata entries "
                f"for document {document_id} to {filepath}"
            )
            
            return filepath


if __name__ == "__main__":
    """
    Test and example usage of VectorMetadataStore.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== Vector Metadata Store Test ===\n")
    
    # Create store
    store = VectorMetadataStore()
    
    # Create sample metadata
    metadata_list = [
        VectorMetadata(
            vector_id=0,
            document_id="doc-123",
            chunk_id=1,
            clause_reference="1.1",
            page_number=1,
            language="en",
            text_preview="This is the first clause regarding definitions."
        ),
        VectorMetadata(
            vector_id=1,
            document_id="doc-123",
            chunk_id=2,
            clause_reference="1.2",
            page_number=1,
            language="en",
            text_preview="The parties agree to the following terms."
        ),
        VectorMetadata(
            vector_id=2,
            document_id="doc-456",
            chunk_id=1,
            clause_reference="௧.௧",
            page_number=1,
            language="ta",
            text_preview="இது முதல் விதிமுறை வரையறைகள் பற்றியது."
        ),
    ]
    
    # Bulk add
    print(f"Adding {len(metadata_list)} metadata entries...")
    new_count = store.add_bulk(metadata_list)
    print(f"Added {new_count} new entries\n")
    
    # Get stats
    print("Store Statistics:")
    stats = store.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Retrieve by vector ID
    print("Retrieving vector_id=0:")
    metadata = store.get(0)
    if metadata:
        print(f"  Document: {metadata.document_id}")
        print(f"  Chunk: {metadata.chunk_id}")
        print(f"  Clause: {metadata.clause_reference}")
        print(f"  Language: {metadata.language}")
        print(f"  Preview: {metadata.text_preview}")
    print()
    
    # Get by document
    print("Retrieving all vectors for doc-123:")
    doc_metadata = store.get_by_document("doc-123")
    print(f"  Found {len(doc_metadata)} vectors")
    for meta in doc_metadata:
        print(f"    - Vector {meta.vector_id}: Clause {meta.clause_reference}")
    print()
    
    # Save to disk
    print("Saving to disk...")
    filepath = store.save("test_metadata.json")
    print(f"Saved to {filepath}\n")
    
    # Create new store and load
    print("Loading from disk into new store...")
    store2 = VectorMetadataStore()
    success = store2.load("test_metadata.json")
    print(f"Load successful: {success}")
    print(f"Loaded entries: {store2.count()}\n")
    
    # Export for single document
    print("Exporting metadata for doc-123...")
    export_path = store.export_for_document("doc-123")
    print(f"Exported to {export_path}")
