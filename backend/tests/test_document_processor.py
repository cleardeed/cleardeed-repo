"""
Unit Tests for Document Processing Pipeline

Tests cover:
- PDF text extraction (direct and OCR)
- Language detection (English, Tamil, Mixed)
- Text normalization
- Legal document chunking
- End-to-end pipeline processing
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from app.services.document_processor import DocumentProcessingPipeline
from app.services.pdf_processor import PDFExtractor
from app.services.language_detector import LanguageDetector
from app.services.text_normalizer import TextNormalizer
from app.services.legal_chunker import LegalDocumentChunker, DocumentChunk


# Fixtures

@pytest.fixture
def sample_english_pages():
    """Sample extracted pages in English"""
    return [
        {
            'page_number': 1,
            'text': '1. DEFINITIONS\n\nIn this agreement, the following terms shall have the meanings set forth below.',
            'extraction_method': 'direct',
            'word_count': 15
        },
        {
            'page_number': 2,
            'text': '2. PAYMENT TERMS\n\nThe buyer shall pay the seller within thirty (30) days.',
            'extraction_method': 'direct',
            'word_count': 14
        }
    ]


@pytest.fixture
def sample_tamil_pages():
    """Sample extracted pages in Tamil"""
    return [
        {
            'page_number': 1,
            'text': '௧. வரையறைகள்\n\nஇந்த ஒப்பந்தத்தில், பின்வரும் சொற்கள் கீழே குறிப்பிடப்பட்ட அர்த்தங்களைக் கொண்டிருக்கும்.',
            'extraction_method': 'direct',
            'word_count': 8
        },
        {
            'page_number': 2,
            'text': '௨. கட்டண விதிமுறைகள்\n\nவாங்குபவர் முப்பது நாட்களுக்குள் விற்பவருக்கு பணம் செலுத்த வேண்டும்.',
            'extraction_method': 'direct',
            'word_count': 9
        }
    ]


@pytest.fixture
def sample_scanned_pages():
    """Sample pages from scanned PDF (OCR)"""
    return [
        {
            'page_number': 1,
            'text': 'PROPERTY DEED\n\nThis deed is made on December 14, 2025 between Party A and Party B.',
            'extraction_method': 'ocr',
            'word_count': 17
        }
    ]


@pytest.fixture
def sample_mixed_pages():
    """Sample mixed-language pages (Tamil and English)"""
    return [
        {
            'page_number': 1,
            'text': '1. DEFINITIONS / வரையறைகள்\n\nProperty means the land / சொத்து என்பது நிலம்',
            'extraction_method': 'direct',
            'word_count': 11
        }
    ]


@pytest.fixture
def mock_pdf_extractor():
    """Mock PDFExtractor"""
    with patch('app.services.document_processor.PDFExtractor') as mock:
        yield mock


@pytest.fixture
def mock_language_detector():
    """Mock LanguageDetector"""
    with patch('app.services.document_processor.LanguageDetector') as mock:
        yield mock


@pytest.fixture
def mock_text_normalizer():
    """Mock TextNormalizer"""
    with patch('app.services.document_processor.TextNormalizer') as mock:
        yield mock


@pytest.fixture
def mock_legal_chunker():
    """Mock LegalDocumentChunker"""
    with patch('app.services.document_processor.LegalDocumentChunker') as mock:
        yield mock


# Test PDFExtractor

class TestPDFExtractor:
    """Tests for PDF text extraction"""
    
    def test_extract_english_pdf(self, sample_english_pages):
        """Test extraction of English PDF with direct text"""
        extractor = PDFExtractor()
        
        # Mock the extraction
        with patch.object(extractor, 'extract_text_from_pdf', return_value=sample_english_pages):
            result = extractor.extract_text_from_pdf('test.pdf')
        
        assert len(result) == 2
        assert result[0]['page_number'] == 1
        assert result[0]['extraction_method'] == 'direct'
        assert 'DEFINITIONS' in result[0]['text']
    
    def test_extract_tamil_pdf(self, sample_tamil_pages):
        """Test extraction of Tamil PDF"""
        extractor = PDFExtractor()
        
        with patch.object(extractor, 'extract_text_from_pdf', return_value=sample_tamil_pages):
            result = extractor.extract_text_from_pdf('tamil.pdf')
        
        assert len(result) == 2
        assert 'வரையறைகள்' in result[0]['text']
        assert result[0]['extraction_method'] == 'direct'
    
    def test_extract_scanned_pdf_with_ocr(self, sample_scanned_pages):
        """Test OCR fallback for scanned PDF"""
        extractor = PDFExtractor()
        
        with patch.object(extractor, 'extract_text_from_pdf', return_value=sample_scanned_pages):
            result = extractor.extract_text_from_pdf('scanned.pdf')
        
        assert len(result) == 1
        assert result[0]['extraction_method'] == 'ocr'
        assert 'PROPERTY DEED' in result[0]['text']
    
    def test_empty_pdf(self):
        """Test handling of empty PDF"""
        extractor = PDFExtractor()
        
        empty_pages = [
            {
                'page_number': 1,
                'text': '',
                'extraction_method': 'direct',
                'word_count': 0
            }
        ]
        
        with patch.object(extractor, 'extract_text_from_pdf', return_value=empty_pages):
            result = extractor.extract_text_from_pdf('empty.pdf')
        
        assert len(result) == 1
        assert result[0]['word_count'] == 0


# Test LanguageDetector

class TestLanguageDetector:
    """Tests for language detection"""
    
    def test_detect_english(self):
        """Test detection of English text"""
        detector = LanguageDetector()
        
        with patch.object(detector, 'detect_language', return_value=('en', 0.95)):
            lang, conf = detector.detect_language("This is English text")
        
        assert lang == 'en'
        assert conf > 0.9
    
    def test_detect_tamil(self):
        """Test detection of Tamil text"""
        detector = LanguageDetector()
        
        with patch.object(detector, 'detect_language', return_value=('ta', 0.92)):
            lang, conf = detector.detect_language("இது தமிழ் உரை")
        
        assert lang == 'ta'
        assert conf > 0.9
    
    def test_detect_mixed_language(self):
        """Test detection of mixed Tamil-English text"""
        detector = LanguageDetector()
        
        with patch.object(detector, 'detect_language', return_value=('mixed', 0.85)):
            lang, conf = detector.detect_language("This is English மற்றும் தமிழ்")
        
        assert lang == 'mixed'
        assert conf > 0.8


# Test TextNormalizer

class TestTextNormalizer:
    """Tests for text normalization"""
    
    def test_normalize_english(self):
        """Test English text normalization"""
        normalizer = TextNormalizer()
        
        text = "Page 1\n\nSection  1\n\nPage 1\n\nText here"
        
        with patch.object(normalizer, 'normalize', return_value="Section 1\n\nText here"):
            result = normalizer.normalize(text, language='en')
        
        # Headers removed, whitespace normalized
        assert 'Page 1' not in result
        assert 'Section 1' in result
    
    def test_normalize_tamil(self):
        """Test Tamil text normalization"""
        normalizer = TextNormalizer()
        
        # Tamil text with broken word splits
        text = "வண க் க ம்"
        
        with patch.object(normalizer, 'normalize', return_value="வணக்கம்"):
            result = normalizer.normalize(text, language='ta')
        
        # Word splits fixed
        assert result == "வணக்கம்"


# Test LegalDocumentChunker

class TestLegalDocumentChunker:
    """Tests for legal document chunking"""
    
    def test_chunk_with_clause_numbers(self, sample_english_pages):
        """Test chunking with detected clause numbers"""
        chunker = LegalDocumentChunker()
        
        expected_chunks = [
            DocumentChunk(
                chunk_id=1,
                text='1. DEFINITIONS\n\nIn this agreement...',
                clause_number='1',
                heading='DEFINITIONS',
                page_number=1,
                language='en',
                token_count=15,
                char_count=80
            ),
            DocumentChunk(
                chunk_id=2,
                text='2. PAYMENT TERMS\n\nThe buyer shall pay...',
                clause_number='2',
                heading='PAYMENT TERMS',
                page_number=2,
                language='en',
                token_count=14,
                char_count=75
            )
        ]
        
        with patch.object(chunker, 'chunk_document', return_value=expected_chunks):
            result = chunker.chunk_document(sample_english_pages, language='en')
        
        assert len(result) == 2
        assert result[0].clause_number == '1'
        assert result[0].heading == 'DEFINITIONS'
        assert result[1].clause_number == '2'
    
    def test_chunk_tamil_document(self, sample_tamil_pages):
        """Test chunking Tamil document with Tamil numerals"""
        chunker = LegalDocumentChunker()
        
        expected_chunks = [
            DocumentChunk(
                chunk_id=1,
                text='௧. வரையறைகள்...',
                clause_number='௧',
                heading='வரையறைகள்',
                page_number=1,
                language='ta',
                token_count=10,
                char_count=60
            )
        ]
        
        with patch.object(chunker, 'chunk_document', return_value=expected_chunks):
            result = chunker.chunk_document(sample_tamil_pages, language='ta')
        
        assert len(result) == 1
        assert result[0].language == 'ta'


# Test DocumentProcessingPipeline (End-to-End)

class TestDocumentProcessingPipeline:
    """End-to-end tests for document processing pipeline"""
    
    def test_process_english_document(
        self, 
        sample_english_pages,
        mock_pdf_extractor,
        mock_language_detector,
        mock_text_normalizer,
        mock_legal_chunker
    ):
        """Test complete pipeline for English document"""
        
        # Setup mocks
        mock_extractor_instance = MagicMock()
        mock_extractor_instance.extract_text_from_pdf.return_value = sample_english_pages
        mock_pdf_extractor.return_value = mock_extractor_instance
        
        mock_detector_instance = MagicMock()
        mock_detector_instance.detect_language.return_value = ('en', 0.95)
        mock_language_detector.return_value = mock_detector_instance
        
        mock_normalizer_instance = MagicMock()
        mock_normalizer_instance.normalize.side_effect = lambda text, language: text
        mock_text_normalizer.return_value = mock_normalizer_instance
        
        mock_chunker_instance = MagicMock()
        expected_chunks = [
            DocumentChunk(
                chunk_id=1,
                text='1. DEFINITIONS...',
                clause_number='1',
                heading='DEFINITIONS',
                page_number=1,
                language='en',
                token_count=15,
                char_count=80
            )
        ]
        mock_chunker_instance.chunk_document.return_value = expected_chunks
        mock_chunker_instance.get_chunking_stats.return_value = {
            'total_chunks': 1,
            'chunks_with_clause_numbers': 1,
            'chunks_with_headings': 1,
            'avg_tokens_per_chunk': 15
        }
        mock_legal_chunker.return_value = mock_chunker_instance
        
        # Test pipeline
        pipeline = DocumentProcessingPipeline()
        
        with patch('pathlib.Path.exists', return_value=True):
            result = pipeline.process_document('test.pdf', 'doc-123')
        
        # Assertions
        assert result['status'] == 'completed'
        assert result['document_id'] == 'doc-123'
        assert len(result['pages']) == 2
        assert len(result['chunks']) == 1
        assert result['metadata']['primary_language'] == 'en'
        assert result['metadata']['total_pages'] == 2
        assert result['error'] is None
    
    def test_process_tamil_document(
        self,
        sample_tamil_pages,
        mock_pdf_extractor,
        mock_language_detector,
        mock_text_normalizer,
        mock_legal_chunker
    ):
        """Test complete pipeline for Tamil document"""
        
        # Setup mocks
        mock_extractor_instance = MagicMock()
        mock_extractor_instance.extract_text_from_pdf.return_value = sample_tamil_pages
        mock_pdf_extractor.return_value = mock_extractor_instance
        
        mock_detector_instance = MagicMock()
        mock_detector_instance.detect_language.return_value = ('ta', 0.92)
        mock_language_detector.return_value = mock_detector_instance
        
        mock_normalizer_instance = MagicMock()
        mock_normalizer_instance.normalize.side_effect = lambda text, language: text
        mock_text_normalizer.return_value = mock_normalizer_instance
        
        mock_chunker_instance = MagicMock()
        expected_chunks = [
            DocumentChunk(
                chunk_id=1,
                text='வரையறைகள்...',
                clause_number='௧',
                heading='வரையறைகள்',
                page_number=1,
                language='ta',
                token_count=10,
                char_count=50
            )
        ]
        mock_chunker_instance.chunk_document.return_value = expected_chunks
        mock_chunker_instance.get_chunking_stats.return_value = {
            'total_chunks': 1,
            'chunks_with_clause_numbers': 1,
            'chunks_with_headings': 1,
            'avg_tokens_per_chunk': 10
        }
        mock_legal_chunker.return_value = mock_chunker_instance
        
        # Test pipeline
        pipeline = DocumentProcessingPipeline()
        
        with patch('pathlib.Path.exists', return_value=True):
            result = pipeline.process_document('tamil.pdf', 'doc-456')
        
        # Assertions
        assert result['status'] == 'completed'
        assert result['metadata']['primary_language'] == 'ta'
        assert len(result['chunks']) == 1
    
    def test_process_scanned_document(
        self,
        sample_scanned_pages,
        mock_pdf_extractor,
        mock_language_detector,
        mock_text_normalizer,
        mock_legal_chunker
    ):
        """Test pipeline with scanned PDF (OCR)"""
        
        # Setup mocks
        mock_extractor_instance = MagicMock()
        mock_extractor_instance.extract_text_from_pdf.return_value = sample_scanned_pages
        mock_pdf_extractor.return_value = mock_extractor_instance
        
        mock_detector_instance = MagicMock()
        mock_detector_instance.detect_language.return_value = ('en', 0.88)
        mock_language_detector.return_value = mock_detector_instance
        
        mock_normalizer_instance = MagicMock()
        mock_normalizer_instance.normalize.side_effect = lambda text, language: text
        mock_text_normalizer.return_value = mock_normalizer_instance
        
        mock_chunker_instance = MagicMock()
        expected_chunks = [
            DocumentChunk(
                chunk_id=1,
                text='PROPERTY DEED...',
                clause_number=None,
                heading='PROPERTY DEED',
                page_number=1,
                language='en',
                token_count=17,
                char_count=85
            )
        ]
        mock_chunker_instance.chunk_document.return_value = expected_chunks
        mock_chunker_instance.get_chunking_stats.return_value = {
            'total_chunks': 1,
            'chunks_with_clause_numbers': 0,
            'chunks_with_headings': 1,
            'avg_tokens_per_chunk': 17
        }
        mock_legal_chunker.return_value = mock_chunker_instance
        
        # Test pipeline
        pipeline = DocumentProcessingPipeline()
        
        with patch('pathlib.Path.exists', return_value=True):
            result = pipeline.process_document('scanned.pdf', 'doc-789')
        
        # Assertions
        assert result['status'] == 'completed'
        assert result['metadata']['extraction_methods']['ocr'] == 1
        assert result['chunks'][0].heading == 'PROPERTY DEED'
    
    def test_process_mixed_language_document(
        self,
        sample_mixed_pages,
        mock_pdf_extractor,
        mock_language_detector,
        mock_text_normalizer,
        mock_legal_chunker
    ):
        """Test pipeline with mixed Tamil-English document"""
        
        # Setup mocks
        mock_extractor_instance = MagicMock()
        mock_extractor_instance.extract_text_from_pdf.return_value = sample_mixed_pages
        mock_pdf_extractor.return_value = mock_extractor_instance
        
        mock_detector_instance = MagicMock()
        mock_detector_instance.detect_language.return_value = ('mixed', 0.85)
        mock_detector_instance.analyze_mixed_content.return_value = {'en': 0.6, 'ta': 0.4}
        mock_language_detector.return_value = mock_detector_instance
        
        mock_normalizer_instance = MagicMock()
        mock_normalizer_instance.normalize.side_effect = lambda text, language: text
        mock_text_normalizer.return_value = mock_normalizer_instance
        
        mock_chunker_instance = MagicMock()
        expected_chunks = [
            DocumentChunk(
                chunk_id=1,
                text='DEFINITIONS / வரையறைகள்...',
                clause_number='1',
                heading='DEFINITIONS',
                page_number=1,
                language='mixed',
                token_count=11,
                char_count=60
            )
        ]
        mock_chunker_instance.chunk_document.return_value = expected_chunks
        mock_chunker_instance.get_chunking_stats.return_value = {
            'total_chunks': 1,
            'chunks_with_clause_numbers': 1,
            'chunks_with_headings': 1,
            'avg_tokens_per_chunk': 11
        }
        mock_legal_chunker.return_value = mock_chunker_instance
        
        # Test pipeline
        pipeline = DocumentProcessingPipeline()
        
        with patch('pathlib.Path.exists', return_value=True):
            result = pipeline.process_document('mixed.pdf', 'doc-999')
        
        # Assertions
        assert result['status'] == 'completed'
        assert result['metadata']['primary_language'] == 'mixed'
        assert result['chunks'][0].language == 'mixed'
    
    def test_process_nonexistent_file(self):
        """Test error handling for non-existent file"""
        pipeline = DocumentProcessingPipeline()
        
        with patch('pathlib.Path.exists', return_value=False):
            result = pipeline.process_document('missing.pdf', 'doc-404')
        
        # Should return error result
        assert result['status'] == 'failed'
        assert 'not found' in result['error'].lower()
        assert len(result['pages']) == 0
        assert len(result['chunks']) == 0
    
    def test_process_with_extraction_error(
        self,
        mock_pdf_extractor,
        mock_language_detector,
        mock_text_normalizer,
        mock_legal_chunker
    ):
        """Test error handling when extraction fails"""
        
        # Setup mock to raise exception
        mock_extractor_instance = MagicMock()
        mock_extractor_instance.extract_text_from_pdf.side_effect = Exception("Extraction failed")
        mock_pdf_extractor.return_value = mock_extractor_instance
        
        pipeline = DocumentProcessingPipeline()
        
        with patch('pathlib.Path.exists', return_value=True):
            result = pipeline.process_document('corrupt.pdf', 'doc-error')
        
        # Should return error result
        assert result['status'] == 'failed'
        assert 'failed' in result['error'].lower()


# Test Fixtures and Utilities

def test_sample_english_pages_fixture(sample_english_pages):
    """Validate English pages fixture"""
    assert len(sample_english_pages) == 2
    assert all('page_number' in page for page in sample_english_pages)
    assert all('text' in page for page in sample_english_pages)
    assert all('extraction_method' in page for page in sample_english_pages)


def test_sample_tamil_pages_fixture(sample_tamil_pages):
    """Validate Tamil pages fixture"""
    assert len(sample_tamil_pages) == 2
    assert 'வரையறைகள்' in sample_tamil_pages[0]['text']


def test_sample_scanned_pages_fixture(sample_scanned_pages):
    """Validate scanned pages fixture"""
    assert len(sample_scanned_pages) == 1
    assert sample_scanned_pages[0]['extraction_method'] == 'ocr'


def test_sample_mixed_pages_fixture(sample_mixed_pages):
    """Validate mixed-language pages fixture"""
    assert len(sample_mixed_pages) == 1
    text = sample_mixed_pages[0]['text']
    # Should contain both English and Tamil
    assert 'DEFINITIONS' in text
    assert 'வரையறைகள்' in text
