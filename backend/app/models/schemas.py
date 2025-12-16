"""
Pydantic Models and Schemas

This module defines all data models and validation schemas used across the API.
Uses Pydantic for automatic validation and serialization.

Responsibilities:
- Request/Response models
- Data validation schemas
- Type definitions
- Document metadata structures
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    """Types of documents that can be uploaded"""
    PROPERTY_DOCUMENT = "property-document"
    ENCUMBRANCE_CERTIFICATE = "encumbrance-certificate"
    OTHER_DOCUMENTS = "other-documents"


class LanguageCode(str, Enum):
    """Supported languages"""
    ENGLISH = "en"
    TAMIL = "ta"
    MIXED = "mixed"


class ProcessingStatus(str, Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentUploadRequest(BaseModel):
    """Request model for document upload"""
    document_type: DocumentType
    filename: str
    file_size: int
    property_id: Optional[str] = None
    
    @validator('file_size')
    def validate_file_size(cls, v):
        if v > 25 * 1024 * 1024:  # 25MB
            raise ValueError('File size exceeds 25MB limit')
        return v


class DocumentMetadata(BaseModel):
    """Metadata extracted from uploaded document"""
    document_id: str
    filename: str
    file_size: int
    document_type: DocumentType
    upload_timestamp: datetime
    language: Optional[LanguageCode] = None
    page_count: Optional[int] = None
    file_path: str


class TextExtractionResult(BaseModel):
    """Result of text extraction from PDF"""
    document_id: str
    language: LanguageCode
    page_count: int
    text_content: str
    text_length: int
    extracted_at: datetime
    confidence_score: Optional[float] = None


class AnalysisRequest(BaseModel):
    """Request model for document analysis"""
    document_ids: List[str]
    property_data: Dict[str, Any]
    analysis_type: str = "comprehensive"


class AnalysisResult(BaseModel):
    """Result of document analysis"""
    analysis_id: str
    document_ids: List[str]
    status: ProcessingStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class PropertyDetails(BaseModel):
    """Property details for context in analysis"""
    zone: str
    district: str
    sro_name: str
    taluk: str
    village: str
    survey_number: str
    subdivision: Optional[str] = None
    email: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str
    services: Dict[str, str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Document Processing Pipeline Models

class UploadedDocument(BaseModel):
    """
    Represents a document uploaded to the system
    
    This model tracks the uploaded file and its metadata throughout
    the processing pipeline. It serves as the source record for all
    subsequent processing steps.
    
    Validation:
    - document_id must be valid UUID format
    - file_size must be positive
    - file_path must be non-empty string
    """
    document_id: str = Field(
        ..., 
        description="Unique identifier (UUID) for the document",
        min_length=36,
        max_length=36
    )
    
    filename: str = Field(
        ..., 
        description="Original filename with extension",
        min_length=1,
        max_length=255
    )
    
    file_path: str = Field(
        ..., 
        description="Absolute path where file is stored on disk",
        min_length=1
    )
    
    file_size: int = Field(
        ..., 
        description="Size of file in bytes",
        gt=0,
        le=26214400  # 25MB max
    )
    
    document_type: DocumentType = Field(
        ...,
        description="Category of document (property-document, encumbrance-certificate, other-documents)"
    )
    
    upload_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the document was uploaded"
    )
    
    mime_type: str = Field(
        default="application/pdf",
        description="MIME type of the uploaded file"
    )
    
    checksum: Optional[str] = Field(
        None,
        description="SHA256 checksum of file for integrity verification"
    )
    
    uploaded_by: Optional[str] = Field(
        None,
        description="User identifier who uploaded the document"
    )
    
    processing_status: ProcessingStatus = Field(
        default=ProcessingStatus.PENDING,
        description="Current processing status of the document"
    )
    
    error_message: Optional[str] = Field(
        None,
        description="Error message if processing failed"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "a7b8c9d0-1234-5678-9abc-def012345678",
                "filename": "property_deed.pdf",
                "file_path": "data/uploads/a7b8c9d0-1234-5678-9abc-def012345678/property_deed.pdf",
                "file_size": 1048576,
                "document_type": "property-document",
                "upload_timestamp": "2025-12-14T10:30:00",
                "mime_type": "application/pdf",
                "processing_status": "pending"
            }
        }


class ExtractedPage(BaseModel):
    """
    Represents a single page extracted from a PDF document
    
    This model captures the result of text extraction for one page,
    including the extraction method used (direct or OCR) and metadata
    about the extraction process.
    
    Validation:
    - page_number must be positive
    - text content can be empty for blank pages
    - word_count must be non-negative
    """
    page_number: int = Field(
        ..., 
        description="Page number within the document (1-indexed)",
        ge=1
    )
    
    text: str = Field(
        ..., 
        description="Extracted text content from the page"
    )
    
    extraction_method: str = Field(
        ..., 
        description="Method used to extract text ('direct' or 'ocr')",
        pattern="^(direct|ocr)$"
    )
    
    word_count: int = Field(
        ..., 
        description="Number of words extracted from page",
        ge=0
    )
    
    char_count: int = Field(
        default=0,
        description="Number of characters in extracted text",
        ge=0
    )
    
    language: Optional[LanguageCode] = Field(
        None,
        description="Detected language of the page content"
    )
    
    language_confidence: Optional[float] = Field(
        None,
        description="Confidence score for language detection (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    has_images: bool = Field(
        default=False,
        description="Whether the page contains embedded images"
    )
    
    has_tables: bool = Field(
        default=False,
        description="Whether the page contains table structures"
    )
    
    extraction_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the text extraction was performed"
    )
    
    ocr_confidence: Optional[float] = Field(
        None,
        description="OCR confidence score if OCR was used (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    @validator('text')
    def validate_text_length(cls, v, values):
        """Ensure text is not unreasonably long for a single page"""
        if len(v) > 50000:  # ~10k words max per page
            raise ValueError("Text content exceeds reasonable page size (50k chars)")
        return v
    
    @validator('char_count', always=True)
    def set_char_count(cls, v, values):
        """Auto-calculate char_count if not provided"""
        if 'text' in values and v == 0:
            return len(values['text'])
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "page_number": 1,
                "text": "This is the extracted text from page 1...",
                "extraction_method": "direct",
                "word_count": 245,
                "char_count": 1450,
                "language": "en",
                "language_confidence": 0.95,
                "has_images": False,
                "has_tables": True,
                "extraction_timestamp": "2025-12-14T10:31:00"
            }
        }


class TextChunk(BaseModel):
    """
    Represents a semantic chunk of document text
    
    This model represents a chunk created by the legal document chunker.
    Chunks respect document structure (clauses, sections) and never split
    within a logical unit.
    
    Validation:
    - chunk_id must be positive
    - text must not be empty
    - token_count should be within reasonable bounds
    """
    chunk_id: int = Field(
        ..., 
        description="Sequential identifier for the chunk within document",
        ge=1
    )
    
    document_id: str = Field(
        ..., 
        description="Parent document UUID this chunk belongs to",
        min_length=36,
        max_length=36
    )
    
    text: str = Field(
        ..., 
        description="The actual text content of the chunk",
        min_length=1
    )
    
    clause_number: Optional[str] = Field(
        None,
        description="Legal clause/section number if detected (e.g., '1.2.3', 'Article 5')",
        max_length=100
    )
    
    heading: Optional[str] = Field(
        None,
        description="Section heading if detected (e.g., 'TERMS AND CONDITIONS')",
        max_length=200
    )
    
    page_number: Optional[int] = Field(
        None,
        description="Source page number from original document",
        ge=1
    )
    
    page_range: Optional[List[int]] = Field(
        None,
        description="List of page numbers if chunk spans multiple pages"
    )
    
    language: LanguageCode = Field(
        ...,
        description="Primary language of the chunk"
    )
    
    token_count: int = Field(
        ..., 
        description="Estimated number of tokens in the chunk",
        ge=0,
        le=1000  # Reasonable upper bound
    )
    
    char_count: int = Field(
        ..., 
        description="Number of characters in the chunk",
        ge=1
    )
    
    chunk_type: Optional[str] = Field(
        None,
        description="Type of content (e.g., 'preamble', 'clause', 'schedule', 'signature')",
        max_length=50
    )
    
    embedding: Optional[List[float]] = Field(
        None,
        description="Vector embedding for semantic search (future use)"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the chunk"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the chunk was created"
    )
    
    @validator('text')
    def validate_text_not_empty(cls, v):
        """Ensure text is not just whitespace"""
        if not v.strip():
            raise ValueError("Chunk text cannot be empty or whitespace-only")
        return v
    
    @validator('char_count', always=True)
    def validate_char_count(cls, v, values):
        """Ensure char_count matches text length"""
        if 'text' in values:
            actual_count = len(values['text'])
            if v != actual_count and v != 0:
                # Allow some tolerance or auto-correct
                return actual_count
        return v
    
    @validator('page_range')
    def validate_page_range(cls, v, values):
        """Ensure page_range is sorted and contains valid pages"""
        if v is not None:
            if not all(p > 0 for p in v):
                raise ValueError("Page numbers must be positive")
            if len(v) != len(set(v)):
                raise ValueError("Page range contains duplicates")
            # Auto-sort for consistency
            return sorted(v)
        return v
    
    @validator('embedding')
    def validate_embedding_dimension(cls, v):
        """Ensure embedding has consistent dimensions if provided"""
        if v is not None:
            if len(v) < 128 or len(v) > 4096:
                raise ValueError("Embedding dimension should be between 128 and 4096")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": 1,
                "document_id": "a7b8c9d0-1234-5678-9abc-def012345678",
                "text": "1. DEFINITIONS\n\nIn this agreement, the following terms shall have the meanings set forth below...",
                "clause_number": "1",
                "heading": "DEFINITIONS",
                "page_number": 2,
                "language": "en",
                "token_count": 487,
                "char_count": 2340,
                "chunk_type": "clause",
                "metadata": {
                    "extraction_method": "direct",
                    "has_legal_terms": True
                },
                "created_at": "2025-12-14T10:32:00"
            }
        }
