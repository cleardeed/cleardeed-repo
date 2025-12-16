"""
Document Processing Pipeline Service

This module orchestrates the complete document processing workflow from
PDF upload to structured chunks ready for LLM analysis. It coordinates
multiple services to extract, detect, normalize, and chunk document text.

Responsibilities:
- Coordinate the complete processing pipeline
- Extract text from PDFs (direct or OCR)
- Detect language per page
- Normalize text for legal documents
- Chunk text respecting legal structure
- Return structured chunks with metadata
- Comprehensive logging at each step
"""

from typing import List, Dict, Optional
import logging
from pathlib import Path

from app.services.pdf_processor import PDFExtractor
from app.services.language_detector import LanguageDetector
from app.services.text_normalizer import TextNormalizer
from app.services.legal_chunker import LegalDocumentChunker, DocumentChunk
from app.services.vector_ingestion import VectorIngestionPipeline, IngestionResult
from app.models.schemas import (
    UploadedDocument,
    ExtractedPage,
    TextChunk,
    LanguageCode,
    ProcessingStatus
)

logger = logging.getLogger(__name__)


class DocumentProcessingPipeline:
    """
    End-to-end document processing pipeline
    
    Orchestrates the complete workflow:
    1. PDF Text Extraction (PDFExtractor)
    2. Language Detection (LanguageDetector)
    3. Text Normalization (TextNormalizer)
    4. Legal Chunking (LegalDocumentChunker)
    5. Vector Ingestion (VectorIngestionPipeline) - NEW
    6. Index Persistence (FAISS + Metadata)
    
    Each step is logged and errors are captured for debugging.
    
    Example:
        >>> pipeline = DocumentProcessingPipeline()
        >>> result = pipeline.process_document("path/to/doc.pdf", "doc-123")
        >>> print(f"Processed {len(result['chunks'])} chunks")
        >>> print(f"Indexed {result['metadata']['vectors_indexed']} vectors")
    """
    
    def __init__(
        self,
        max_chunk_tokens: int = 500,
        min_chunk_tokens: int = 50,
        enable_ocr: bool = True,
        enable_vector_ingestion: bool = True
    ):
        """
        Initialize document processing pipeline
        
        Args:
            max_chunk_tokens: Maximum tokens per chunk
            min_chunk_tokens: Minimum tokens per chunk
            enable_ocr: Whether to use OCR fallback for scanned PDFs
            enable_vector_ingestion: Whether to generate embeddings and index vectors
        """
        logger.info("Initializing Document Processing Pipeline")
        
        # Initialize all service components
        self.pdf_extractor = PDFExtractor()
        self.language_detector = LanguageDetector()
        self.text_normalizer = TextNormalizer()
        self.chunker = LegalDocumentChunker(
            max_tokens=max_chunk_tokens,
            min_tokens=min_chunk_tokens
        )
        
        self.enable_ocr = enable_ocr
        self.enable_vector_ingestion = enable_vector_ingestion
        
        # Initialize vector ingestion pipeline (lazy loaded to avoid loading embedding model if not needed)
        self._vector_pipeline: Optional[VectorIngestionPipeline] = None
        
        logger.info(
            f"Pipeline initialized: OCR={enable_ocr}, "
            f"chunk_size={min_chunk_tokens}-{max_chunk_tokens} tokens, "
            f"vector_ingestion={enable_vector_ingestion}"
        )
    
    def process_document(
        self, 
        pdf_path: str, 
        document_id: str
    ) -> Dict:
        """
        Process a document through the complete pipeline
        
        This is the main entry point. It executes all pipeline steps in order
        and returns structured chunks ready for LLM analysis.
        
        Pipeline Steps:
        1. Extract text from PDF (page by page)
        2. Detect language for each page
        3. Normalize text (language-specific)
        4. Chunk text into legal clauses
        5. Generate embeddings and index vectors (FAISS)
        6. Persist index and metadata to disk
        7. Package results with metadata
        
        Args:
            pdf_path: Path to uploaded PDF file
            document_id: Unique identifier for the document
        
        Returns:
            Dictionary containing:
            - document_id: Document identifier
            - status: Processing status
            - pages: List of ExtractedPage objects
            - chunks: List of TextChunk objects
            - metadata: Processing statistics
            - error: Error message if processing failed
        
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If any processing step fails
        
        Example:
            >>> pipeline = DocumentProcessingPipeline()
            >>> result = pipeline.process_document("contract.pdf", "doc-123")
            >>> if result['status'] == 'completed':
            ...     for chunk in result['chunks']:
            ...         print(f"Clause {chunk.clause_number}")
        """
        logger.info(f"Starting document processing: {document_id}")
        logger.info(f"PDF path: {pdf_path}")
        
        # Validate file exists
        if not Path(pdf_path).exists():
            error_msg = f"PDF file not found: {pdf_path}"
            logger.error(error_msg)
            return self._create_error_result(document_id, error_msg)
        
        try:
            # Step 1: Extract text from PDF
            logger.info(f"[{document_id}] Step 1/6: Extracting text from PDF")
            pages_data = self._extract_text(pdf_path, document_id)
            
            if not pages_data:
                error_msg = "No text extracted from PDF"
                logger.error(f"[{document_id}] {error_msg}")
                return self._create_error_result(document_id, error_msg)
            
            logger.info(
                f"[{document_id}] Extracted {len(pages_data)} pages, "
                f"total words: {sum(p['word_count'] for p in pages_data)}"
            )
            
            # Step 2: Detect language
            logger.info(f"[{document_id}] Step 2/6: Detecting language")
            language = self._detect_language(pages_data, document_id)
            logger.info(f"[{document_id}] Primary language: {language}")
            
            # Step 3: Normalize text
            logger.info(f"[{document_id}] Step 3/6: Normalizing text")
            normalized_pages = self._normalize_text(pages_data, language, document_id)
            
            # Step 4: Chunk into legal clauses
            logger.info(f"[{document_id}] Step 4/6: Chunking into legal clauses")
            chunks = self._chunk_document(normalized_pages, language, document_id)
            
            logger.info(
                f"[{document_id}] Created {len(chunks)} chunks, "
                f"avg tokens: {sum(c.token_count for c in chunks) / len(chunks):.0f}"
            )
            
            # Step 5: Convert to Pydantic models
            logger.info(f"[{document_id}] Converting to Pydantic models")
            extracted_pages = self._create_page_models(normalized_pages)
            text_chunks = self._create_chunk_models(chunks, document_id)
            
            # Step 6: Vector ingestion (embeddings + FAISS indexing)
            ingestion_result = None
            if self.enable_vector_ingestion:
                ingestion_result = self._ingest_vectors(document_id, text_chunks)
            else:
                logger.info(f"[{document_id}] Vector ingestion disabled, skipping")
            
            # Step 7: Compile results
            result = {
                'document_id': document_id,
                'status': 'completed',
                'pages': extracted_pages,
                'chunks': text_chunks,
                'metadata': self._create_metadata(
                    pages_data, 
                    chunks, 
                    language, 
                    ingestion_result
                ),
                'error': None
            }
            
            logger.info(f"[{document_id}] ✓ Processing completed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logger.error(f"[{document_id}] {error_msg}", exc_info=True)
            return self._create_error_result(document_id, error_msg)
    
    def _extract_text(self, pdf_path: str, document_id: str) -> List[Dict]:
        """
        Step 1: Extract text from PDF
        
        Uses PDFExtractor to get text from each page. Automatically falls back
        to OCR if direct extraction yields insufficient text.
        
        Args:
            pdf_path: Path to PDF file
            document_id: Document identifier for logging
        
        Returns:
            List of page dictionaries with text and metadata
        """
        try:
            pages_data = self.pdf_extractor.extract_text_from_pdf(pdf_path)
            
            # Log extraction statistics
            direct_count = sum(1 for p in pages_data if p['extraction_method'] == 'direct')
            ocr_count = sum(1 for p in pages_data if p['extraction_method'] == 'ocr')
            
            logger.debug(
                f"[{document_id}] Extraction methods: "
                f"direct={direct_count}, ocr={ocr_count}"
            )
            
            return pages_data
            
        except Exception as e:
            logger.error(f"[{document_id}] Text extraction failed: {e}")
            raise
    
    def _detect_language(self, pages_data: List[Dict], document_id: str) -> str:
        """
        Step 2: Detect document language
        
        Analyzes all pages to determine the primary language. For mixed-language
        documents, determines which language is dominant.
        
        Args:
            pages_data: List of extracted pages
            document_id: Document identifier for logging
        
        Returns:
            Language code: 'en', 'ta', or 'mixed'
        """
        try:
            # Combine all page text for language detection
            combined_text = ' '.join(p['text'] for p in pages_data if p['text'])
            
            if not combined_text.strip():
                logger.warning(f"[{document_id}] No text for language detection, defaulting to English")
                return 'en'
            
            # Detect language
            language, confidence = self.language_detector.detect_language(combined_text)
            
            logger.debug(
                f"[{document_id}] Language detected: {language} "
                f"(confidence: {confidence:.2f})"
            )
            
            # If mixed, get distribution
            if language == 'mixed':
                distribution = self.language_detector.analyze_mixed_content(combined_text)
                logger.debug(
                    f"[{document_id}] Language distribution: "
                    f"Tamil={distribution['ta']:.1%}, English={distribution['en']:.1%}"
                )
            
            return language
            
        except Exception as e:
            logger.warning(f"[{document_id}] Language detection failed: {e}, defaulting to English")
            return 'en'
    
    def _normalize_text(
        self, 
        pages_data: List[Dict], 
        language: str, 
        document_id: str
    ) -> List[Dict]:
        """
        Step 3: Normalize text for legal documents
        
        Applies language-specific normalization to clean up OCR artifacts,
        fix formatting issues, and prepare text for chunking.
        
        Args:
            pages_data: List of extracted pages
            language: Detected language
            document_id: Document identifier for logging
        
        Returns:
            Updated pages with normalized text
        """
        try:
            normalized_pages = []
            
            for page in pages_data:
                original_text = page['text']
                
                # Apply normalization
                normalized_text = self.text_normalizer.normalize(
                    original_text, 
                    language=language
                )
                
                # Create updated page dict
                normalized_page = page.copy()
                normalized_page['text'] = normalized_text
                normalized_page['original_length'] = len(original_text)
                normalized_page['normalized_length'] = len(normalized_text)
                
                # Log significant changes
                reduction = len(original_text) - len(normalized_text)
                if reduction > 100:
                    logger.debug(
                        f"[{document_id}] Page {page['page_number']}: "
                        f"removed {reduction} chars during normalization"
                    )
                
                normalized_pages.append(normalized_page)
            
            total_reduction = sum(
                p['original_length'] - p['normalized_length'] 
                for p in normalized_pages
            )
            
            logger.debug(
                f"[{document_id}] Normalization complete: "
                f"removed {total_reduction} chars total"
            )
            
            return normalized_pages
            
        except Exception as e:
            logger.warning(f"[{document_id}] Normalization failed: {e}, using original text")
            return pages_data
    
    def _chunk_document(
        self, 
        pages_data: List[Dict], 
        language: str, 
        document_id: str
    ) -> List[DocumentChunk]:
        """
        Step 4: Chunk document into legal clauses
        
        Uses LegalDocumentChunker to create semantically meaningful chunks
        that respect clause boundaries and never split within a logical unit.
        
        Args:
            pages_data: List of normalized pages
            language: Document language
            document_id: Document identifier for logging
        
        Returns:
            List of DocumentChunk objects
        """
        try:
            chunks = self.chunker.chunk_document(pages_data, language=language)
            
            # Log chunking statistics
            stats = self.chunker.get_chunking_stats(chunks)
            
            logger.debug(
                f"[{document_id}] Chunking stats: "
                f"total={stats['total_chunks']}, "
                f"with_clauses={stats['chunks_with_clause_numbers']}, "
                f"with_headings={stats['chunks_with_headings']}, "
                f"avg_tokens={stats['avg_tokens_per_chunk']:.0f}"
            )
            
            return chunks
            
        except Exception as e:
            logger.error(f"[{document_id}] Chunking failed: {e}")
            raise
    
    def _create_page_models(self, pages_data: List[Dict]) -> List[ExtractedPage]:
        """
        Convert page dictionaries to ExtractedPage Pydantic models
        
        Args:
            pages_data: List of page dictionaries
        
        Returns:
            List of ExtractedPage objects
        """
        extracted_pages = []
        
        for page in pages_data:
            extracted_page = ExtractedPage(
                page_number=page['page_number'],
                text=page['text'],
                extraction_method=page['extraction_method'],
                word_count=page['word_count'],
                char_count=len(page['text']),
                language=None,  # Will be set if per-page detection is added
                language_confidence=None
            )
            extracted_pages.append(extracted_page)
        
        return extracted_pages
    
    def _create_chunk_models(
        self, 
        chunks: List[DocumentChunk], 
        document_id: str
    ) -> List[TextChunk]:
        """
        Convert DocumentChunk objects to TextChunk Pydantic models
        
        Args:
            chunks: List of DocumentChunk objects from chunker
            document_id: Parent document identifier
        
        Returns:
            List of TextChunk Pydantic models
        """
        text_chunks = []
        
        for chunk in chunks:
            text_chunk = TextChunk(
                chunk_id=chunk.chunk_id,
                document_id=document_id,
                text=chunk.text,
                clause_number=chunk.clause_number,
                heading=chunk.heading,
                page_number=chunk.page_number,
                language=chunk.language,
                token_count=chunk.token_count,
                char_count=chunk.char_count,
                metadata={
                    'extraction_method': 'pipeline',
                    'chunker_version': '1.0'
                }
            )
            text_chunks.append(text_chunk)
        
        return text_chunks
    
    def _create_metadata(
        self, 
        pages_data: List[Dict], 
        chunks: List[DocumentChunk],
        language: str,
        ingestion_result: Optional[IngestionResult] = None
    ) -> Dict:
        """
        Create metadata summary of processing results
        
        Args:
            pages_data: Extracted pages
            chunks: Created chunks
            language: Detected language
            ingestion_result: Vector ingestion result (if enabled)
        
        Returns:
            Dictionary with processing metadata
        """
        metadata = {
            'total_pages': len(pages_data),
            'total_chunks': len(chunks),
            'primary_language': language,
            'extraction_methods': {
                'direct': sum(1 for p in pages_data if p['extraction_method'] == 'direct'),
                'ocr': sum(1 for p in pages_data if p['extraction_method'] == 'ocr')
            },
            'total_words': sum(p['word_count'] for p in pages_data),
            'total_tokens': sum(c.token_count for c in chunks),
            'avg_tokens_per_chunk': sum(c.token_count for c in chunks) / len(chunks) if chunks else 0,
            'chunks_with_clause_numbers': sum(1 for c in chunks if c.clause_number),
            'chunks_with_headings': sum(1 for c in chunks if c.heading)
        }
        
        # Add vector ingestion statistics if available
        if ingestion_result:
            metadata['vector_ingestion'] = {
                'vectors_indexed': ingestion_result.vectors_indexed,
                'chunks_processed': ingestion_result.chunks_processed,
                'chunks_failed': ingestion_result.chunks_failed,
                'embedding_dimension': ingestion_result.embedding_dimension,
                'success': ingestion_result.success,
                'error': ingestion_result.error
            }
        
        return metadata
    
    def _ingest_vectors(
        self,
        document_id: str,
        text_chunks: List[TextChunk]
    ) -> Optional[IngestionResult]:
        """
        Step 5: Generate embeddings and index vectors
        
        This step:
        1. Initializes VectorIngestionPipeline (lazy loads embedding model)
        2. Generates embeddings for all chunks
        3. Indexes vectors in FAISS
        4. Stores metadata
        5. Persists index and metadata to disk
        
        Re-ingestion Safety:
        - If document already exists, old vectors are removed first
        - This ensures clean re-processing without duplicate vectors
        
        Args:
            document_id: Document identifier
            text_chunks: List of TextChunk Pydantic models
        
        Returns:
            IngestionResult with statistics, or None if failed
        """
        logger.info(f"[{document_id}] Step 5/6: Generating embeddings and indexing vectors")
        
        try:
            # Lazy initialize vector pipeline (loads embedding model on first use)
            if self._vector_pipeline is None:
                logger.info(f"[{document_id}] Initializing vector ingestion pipeline...")
                self._vector_pipeline = VectorIngestionPipeline()
                logger.info(f"[{document_id}] Vector pipeline initialized")
            
            # Ingest document (replace_existing=True ensures safe re-ingestion)
            logger.info(f"[{document_id}] Generating embeddings for {len(text_chunks)} chunks...")
            ingestion_result = self._vector_pipeline.ingest_document(
                document_id=document_id,
                chunks=text_chunks,
                replace_existing=True  # Safe re-ingestion
            )
            
            if ingestion_result.success:
                logger.info(
                    f"[{document_id}] ✓ Indexed {ingestion_result.vectors_indexed} vectors "
                    f"(dimension: {ingestion_result.embedding_dimension})"
                )
                
                # Step 6: Persist to disk
                logger.info(f"[{document_id}] Step 6/6: Persisting index and metadata to disk")
                self._persist_index(document_id)
                
            else:
                logger.error(
                    f"[{document_id}] Vector ingestion failed: {ingestion_result.error}"
                )
            
            return ingestion_result
            
        except Exception as e:
            logger.error(
                f"[{document_id}] Vector ingestion error: {str(e)}",
                exc_info=True
            )
            # Return failed result instead of crashing
            return IngestionResult(
                document_id=document_id,
                vectors_indexed=0,
                chunks_processed=len(text_chunks),
                chunks_failed=len(text_chunks),
                embedding_dimension=0,
                success=False,
                error=str(e)
            )
    
    def _persist_index(self, document_id: str):
        """
        Persist FAISS index and metadata to disk
        
        Saves:
        - FAISS index: data/faiss/index.index
        - Metadata: data/metadata/vector_metadata.json
        
        Args:
            document_id: Document identifier (for logging)
        """
        try:
            if self._vector_pipeline:
                self._vector_pipeline.save_to_disk(
                    index_name="index",
                    metadata_name="vector_metadata.json"
                )
                logger.info(f"[{document_id}] ✓ Persisted index and metadata to disk")
        except Exception as e:
            logger.error(
                f"[{document_id}] Failed to persist index: {str(e)}",
                exc_info=True
            )
            # Don't crash on persistence failure
    
    def _create_error_result(self, document_id: str, error_message: str) -> Dict:
        """
        Create error result dictionary
        
        Args:
            document_id: Document identifier
            error_message: Error description
        
        Returns:
            Error result dictionary
        """
        return {
            'document_id': document_id,
            'status': 'failed',
            'pages': [],
            'chunks': [],
            'metadata': {},
            'error': error_message
        }
    
    def process_document_async(
        self, 
        pdf_path: str, 
        document_id: str,
        callback=None
    ):
        """
        Process document asynchronously (future implementation)
        
        This method is a placeholder for async processing support.
        Could be implemented using asyncio or background task queues.
        
        Args:
            pdf_path: Path to PDF file
            document_id: Document identifier
            callback: Optional callback function for status updates
        """
        # TODO: Implement async processing with progress callbacks
        logger.info(f"[{document_id}] Async processing not yet implemented")
        raise NotImplementedError("Async processing coming soon")
