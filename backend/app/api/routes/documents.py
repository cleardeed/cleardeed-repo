"""
Document Processing API Routes

This module defines all HTTP endpoints related to document upload, processing,
and analysis. Handles multipart file uploads and coordinates processing tasks.

Responsibilities:
- POST /upload - Accept PDF file uploads
- GET /documents/{id} - Retrieve document metadata
- POST /analyze - Trigger document analysis
- GET /analysis/{id} - Get analysis results
- DELETE /documents/{id} - Remove document
"""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List, Optional

logger = logging.getLogger(__name__)
from app.models.schemas import (
    DocumentMetadata,
    AnalysisRequest,
    AnalysisResult,
    ErrorResponse,
    DocumentType
)
from app.services.pdf_processor import PDFProcessor
from app.services.language_detector import LanguageDetector
from app.services.llm_service import LLMService
from app.utils.file_handler import FileHandler
from app.utils.validators import validate_file

router = APIRouter(
    prefix="/documents",
    tags=["documents"]
)

# Service instances (will be initialized properly)
pdf_processor = PDFProcessor()
language_detector = LanguageDetector()
llm_service = LLMService()
file_handler = FileHandler()


@router.post("/upload", response_model=DocumentMetadata)
async def upload_document(
    file: UploadFile = File(...),
    document_type: DocumentType = DocumentType.PROPERTY_DOCUMENT,
    process_immediately: bool = True
):
    """
    Upload a PDF document and optionally process it through the pipeline
    
    Workflow:
    1. Validate and save uploaded PDF
    2. Run document processing pipeline (extract, detect, normalize, chunk)
    3. Save processed output to JSON file
    4. Return document metadata with processing status
    
    Args:
        file: PDF file to upload
        document_type: Category of document (property, encumbrance, other)
        process_immediately: Whether to run processing pipeline immediately (default: True)
    
    Returns:
        DocumentMetadata with file details, unique ID, and processing status
    
    Raises:
        HTTPException: If file validation fails or processing error occurs
    """
    try:
        logger.info(f"Receiving file upload: {file.filename}, type: {document_type}")
        
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        # Validate file
        is_valid, error_msg = validate_file(file_content, file.filename)
        if not is_valid:
            logger.warning(f"File validation failed: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Save file and get document ID
        document_id, file_path = await file_handler.save_upload_file(
            file_content=file_content,
            original_filename=file.filename,
            document_type=document_type.value
        )
        
        logger.info(f"File saved successfully: {document_id} at {file_path}")
        
        # Initialize response metadata
        from datetime import datetime
        page_count = None
        language = None
        processing_status = "pending"
        
        # Process document if requested
        if process_immediately:
            logger.info(f"Starting processing pipeline for document: {document_id}")
            
            try:
                # Run processing pipeline
                from app.services.document_processor import DocumentProcessingPipeline
                import json
                from pathlib import Path
                
                pipeline = DocumentProcessingPipeline()
                result = pipeline.process_document(file_path, document_id)
                
                # Save processed output to JSON
                processed_dir = Path("data/processed")
                processed_dir.mkdir(parents=True, exist_ok=True)
                
                output_file = processed_dir / f"{document_id}.json"
                
                # Convert Pydantic models to dicts for JSON serialization
                result_dict = {
                    'document_id': result['document_id'],
                    'status': result['status'],
                    'pages': [page.dict() for page in result['pages']],
                    'chunks': [chunk.dict() for chunk in result['chunks']],
                    'metadata': result['metadata'],
                    'error': result['error']
                }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)
                
                logger.info(f"Processed output saved to: {output_file}")
                
                # Update metadata from processing results
                if result['status'] == 'completed':
                    page_count = result['metadata'].get('total_pages')
                    language = result['metadata'].get('primary_language')
                    processing_status = "completed"
                    logger.info(
                        f"Document {document_id} processed: "
                        f"{page_count} pages, {len(result['chunks'])} chunks, "
                        f"language: {language}"
                    )
                else:
                    processing_status = "failed"
                    logger.error(f"Document {document_id} processing failed: {result['error']}")
                    
            except Exception as e:
                logger.error(f"Processing pipeline failed for {document_id}: {str(e)}", exc_info=True)
                processing_status = "failed"
                # Don't raise - still return uploaded file info
        
        # Return metadata
        return DocumentMetadata(
            document_id=document_id,
            filename=file.filename,
            file_size=file_size,
            document_type=document_type,
            upload_timestamp=datetime.utcnow(),
            file_path=file_path,
            page_count=page_count,
            language=language if language else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@router.post("/upload-multiple", response_model=List[DocumentMetadata])
async def upload_multiple_documents(
    files: List[UploadFile] = File(...),
    document_types: Optional[List[DocumentType]] = None
):
    """
    Upload multiple PDF documents in one request
    
    Args:
        files: List of PDF files
        document_types: Corresponding document types for each file
    
    Returns:
        List of DocumentMetadata for all uploaded files
    """
    # TODO: Implement batch upload
    pass


@router.get("/documents/{document_id}")
async def get_document_metadata(document_id: str):
    """
    Retrieve processed document data
    
    Returns the processed JSON output including extracted pages and chunks.
    
    Args:
        document_id: Unique identifier for the document
    
    Returns:
        Processed document data with pages and chunks
    
    Raises:
        HTTPException: If document not found
    """
    try:
        from pathlib import Path
        import json
        
        # Load processed JSON file
        processed_file = Path(f"data/processed/{document_id}.json")
        
        if not processed_file.exists():
            logger.warning(f"Processed document not found: {document_id}")
            raise HTTPException(
                status_code=404, 
                detail=f"Document {document_id} not found or not yet processed"
            )
        
        with open(processed_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        logger.info(f"Retrieved processed document: {document_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve document: {str(e)}")


@router.post("/analyze", response_model=AnalysisResult)
async def analyze_documents(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger comprehensive analysis of uploaded documents
    
    This endpoint:
    1. Validates all document IDs exist
    2. Extracts text from PDFs (if not already done)
    3. Detects language (Tamil/English/Mixed)
    4. Sends to LLM for analysis with property context
    5. Returns analysis results
    
    Args:
        request: Analysis request with document IDs and property details
        background_tasks: FastAPI background tasks for async processing
    
    Returns:
        AnalysisResult with status (will be processed in background if async enabled)
    """
    # TODO: Implement analysis orchestration
    pass


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and its associated data
    
    Removes:
    - Uploaded PDF file
    - Processed JSON output
    
    Args:
        document_id: Unique identifier for the document
    
    Returns:
        Success message
    """
    try:
        from pathlib import Path
        import shutil
        
        deleted_items = []
        
        # Delete uploaded PDF directory
        upload_dir = Path(f"data/uploads/{document_id}")
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
            deleted_items.append("uploaded_file")
            logger.info(f"Deleted upload directory: {upload_dir}")
        
        # Delete processed JSON
        processed_file = Path(f"data/processed/{document_id}.json")
        if processed_file.exists():
            processed_file.unlink()
            deleted_items.append("processed_data")
            logger.info(f"Deleted processed file: {processed_file}")
        
        if not deleted_items:
            logger.warning(f"Document {document_id} not found")
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        return {
            "success": True,
            "document_id": document_id,
            "deleted_items": deleted_items,
            "message": f"Document {document_id} deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
@router.get("/analysis/{analysis_id}", response_model=AnalysisResult)
async def get_analysis_result(analysis_id: str):
    """
    Get results of a previously triggered analysis
    
    Args:
        analysis_id: Unique identifier for the analysis job
    
    Returns:
        AnalysisResult with current status and results (if completed)
    """
    # TODO: Implement result retrieval
    pass


@router.post("/extract-text/{document_id}")
async def extract_text_from_document(document_id: str):
    """
    Manually trigger text extraction from a document
    (Usually happens automatically during analysis)
    
    Args:
        document_id: Document to extract text from
    
    Returns:
        TextExtractionResult with extracted content
    """
    # TODO: Implement text extraction endpoint
    pass
