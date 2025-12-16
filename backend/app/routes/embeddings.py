"""
FastAPI routes for embedding generation and vector operations.

This module provides endpoints for generating embeddings from processed documents
and indexing them in the FAISS vector store.
"""

import logging
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import FAISSIndexManager
from app.services.metadata_store import VectorMetadataStore

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/embeddings",
    tags=["embeddings"],
    responses={
        404: {"description": "Document not found"},
        500: {"description": "Internal server error"}
    }
)


class EmbeddingResponse(BaseModel):
    """Response for embedding generation."""
    document_id: str
    chunks_processed: int
    vectors_indexed: int
    success: bool
    message: Optional[str] = None


@router.post(
    "/generate/{document_id}",
    response_model=EmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate embeddings for a document",
    description="""
    Generate embeddings for all chunks of a processed document and index them in FAISS.
    
    This endpoint:
    1. Loads the processed document JSON (from data/processed/)
    2. Extracts all text chunks
    3. Generates embeddings using the embedding service
    4. Indexes vectors in FAISS
    5. Stores metadata for retrieval
    
    **Requirements:**
    - Document must be uploaded and processed first
    - Processed JSON must contain chunks
    
    **Returns:**
    - Number of chunks processed
    - Number of vectors successfully indexed
    - Success status
    """
)
async def generate_embeddings(document_id: str) -> EmbeddingResponse:
    """
    Generate embeddings for a document's chunks and index them.
    
    Args:
        document_id: Unique document identifier
    
    Returns:
        EmbeddingResponse with processing statistics
    
    Raises:
        HTTPException: If document not found or processing fails
    """
    try:
        import json
        from pathlib import Path
        from app import main
        
        logger.info(f"Generating embeddings for document: {document_id}")
        
        # Get service instances
        embedding_service = main.embedding_service
        vector_store = main.vector_store
        metadata_store = main.metadata_store
        
        if not all([embedding_service, vector_store, metadata_store]):
            raise HTTPException(
                status_code=503,
                detail="Services not initialized. Application startup may have failed."
            )
        
        # Load processed document JSON
        processed_dir = Path("data/processed")
        json_path = processed_dir / f"{document_id}.json"
        
        if not json_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Processed document not found: {document_id}"
            )
        
        with open(json_path, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
        
        # Extract chunks
        chunks = doc_data.get('chunks', [])
        if not chunks:
            return EmbeddingResponse(
                document_id=document_id,
                chunks_processed=0,
                vectors_indexed=0,
                success=True,
                message="No chunks found in document"
            )
        
        # Extract text from chunks
        texts = [chunk.get('text', '') for chunk in chunks if chunk.get('text')]
        if not texts:
            return EmbeddingResponse(
                document_id=document_id,
                chunks_processed=len(chunks),
                vectors_indexed=0,
                success=True,
                message="No text content in chunks"
            )
        
        logger.info(f"Generating embeddings for {len(texts)} chunks")
        
        # Generate embeddings
        embeddings = embedding_service.embed_texts(texts)
        
        # Create metadata IDs
        metadata_ids = [f"{document_id}_chunk_{i}" for i in range(len(embeddings))]
        
        # Add to FAISS index
        faiss_ids = vector_store.add_embeddings(embeddings, metadata_ids)
        
        # Store metadata
        for i, (chunk, meta_id, faiss_id) in enumerate(zip(chunks, metadata_ids, faiss_ids)):
            from app.services.metadata_store import VectorMetadata
            
            metadata = VectorMetadata(
                vector_id=faiss_id,
                document_id=document_id,
                chunk_id=i,
                page_number=chunk.get('page_number'),
                language=doc_data.get('metadata', {}).get('language', 'en'),
                text_preview=chunk.get('text', '')[:100]
            )
            metadata_store.add(metadata)
        
        # Save index and metadata
        vector_store.save_index("index")
        metadata_store.save("vector_metadata.json")
        
        logger.info(f"Successfully indexed {len(faiss_ids)} vectors for document {document_id}")
        
        return EmbeddingResponse(
            document_id=document_id,
            chunks_processed=len(chunks),
            vectors_indexed=len(faiss_ids),
            success=True,
            message=f"Generated and indexed {len(faiss_ids)} vectors"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating embeddings for {document_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate embeddings: {str(e)}"
        )
