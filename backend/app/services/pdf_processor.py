"""
PDF Processing Service

This module handles all PDF-related operations including text extraction,
page analysis, and metadata extraction. Works with both Tamil and English PDFs.

Responsibilities:
- Extract text from PDF files (using pdfplumber)
- Extract metadata (page count, title, author, dates)
- Handle OCR for scanned PDFs
- Process Tamil unicode text correctly
- Extract images/tables if needed
- Page-by-page processing for large documents
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import re
import pdfplumber
from PIL import Image
import io

# Optional Tesseract OCR support
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None

logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Extracts text from PDF documents with automatic fallback to OCR
    
    Features:
    - Digital PDF text extraction using pdfplumber
    - Automatic OCR fallback for scanned PDFs
    - Tamil and English language support
    - Page-by-page extraction with page numbers
    - OCR noise cleaning
    - Configurable text detection threshold
    """
    
    def __init__(self, min_text_length: int = 50):
        """
        Initialize PDF extractor
        
        Args:
            min_text_length: Minimum text length to consider a page has text.
                           If extracted text is shorter, OCR is triggered.
        """
        self.min_text_length = min_text_length
        # Configure Tesseract for Tamil + English
        self.tesseract_config = r'--oem 3 --psm 6 -l tam+eng'
        logger.info("PDFExtractor initialized")
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Extract text from PDF with automatic OCR fallback
        
        This is the main entry point. It attempts direct text extraction first,
        and falls back to OCR if insufficient text is found.
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            List of dictionaries, each containing:
            - page_number (int): Page number (1-indexed)
            - text (str): Extracted text content
            - extraction_method (str): 'direct' or 'ocr'
            - word_count (int): Number of words extracted
        
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If PDF cannot be processed
        
        Example:
            >>> extractor = PDFExtractor()
            >>> pages = extractor.extract_text_from_pdf("document.pdf")
            >>> for page in pages:
            ...     print(f"Page {page['page_number']}: {page['text'][:100]}")
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Starting text extraction from: {pdf_path}")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"PDF has {total_pages} pages")
                
                pages_data = []
                
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_data = self._extract_page_text(page, page_num, pdf_path)
                    pages_data.append(page_data)
                    
                    logger.debug(
                        f"Page {page_num}/{total_pages}: "
                        f"{page_data['word_count']} words, "
                        f"method: {page_data['extraction_method']}"
                    )
                
                logger.info(f"Extraction complete: {total_pages} pages processed")
                return pages_data
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def _extract_page_text(
        self, 
        page, 
        page_num: int, 
        pdf_path: Path
    ) -> Dict[str, any]:
        """
        Extract text from a single page with OCR fallback
        
        Args:
            page: pdfplumber page object
            page_num: Page number (1-indexed)
            pdf_path: Path to original PDF (for OCR)
        
        Returns:
            Dictionary with page data
        """
        # Try direct text extraction first
        text = page.extract_text()
        
        # Check if we got meaningful text
        if text and len(text.strip()) >= self.min_text_length:
            # Direct extraction successful
            cleaned_text = self._clean_text(text)
            return {
                'page_number': page_num,
                'text': cleaned_text,
                'extraction_method': 'direct',
                'word_count': len(cleaned_text.split())
            }
        else:
            # Fallback to OCR
            logger.info(f"Page {page_num}: Insufficient text found, using OCR")
            ocr_text = self._ocr_page(page)
            cleaned_text = self._clean_ocr_text(ocr_text)
            
            return {
                'page_number': page_num,
                'text': cleaned_text,
                'extraction_method': 'ocr',
                'word_count': len(cleaned_text.split())
            }
    
    def _ocr_page(self, page) -> str:
        """
        Perform OCR on a PDF page
        
        Args:
            page: pdfplumber page object
        
        Returns:
            OCR extracted text
        """
        try:
            # Convert PDF page to image
            image = page.to_image(resolution=300)
            pil_image = image.original
            
            # Use the helper function for OCR
            text = self.perform_ocr_on_image(pil_image)
            
            return text
            
        except Exception as e:
            logger.error(f"OCR failed: {str(e)}")
            return ""
    
    def perform_ocr_on_image(
        self, 
        image: Image.Image, 
        auto_detect_tamil: bool = True
    ) -> str:
        """
        Perform OCR on an image with automatic Tamil detection
        
        This helper function performs OCR with intelligent language selection.
        It can automatically detect if Tamil characters are present and adjust
        the OCR language accordingly.
        
        Args:
            image: PIL Image object to perform OCR on
            auto_detect_tamil: If True, performs initial English OCR to detect
                             Tamil presence, then re-runs with Tamil if needed
        
        Returns:
            Extracted text from the image
        
        Raises:
            None - Returns empty string on failure
        
        Example:
            >>> extractor = PDFExtractor()
            >>> from PIL import Image
            >>> img = Image.open("document_page.png")
            >>> text = extractor.perform_ocr_on_image(img)
        """
        try:
            if auto_detect_tamil:
                # Two-pass approach: Try English first to detect script
                logger.debug("Performing initial OCR for language detection")
                
                # First pass: Quick English OCR
                eng_text = self._run_tesseract(image, language='eng', psm=6)
                
                # Check if result suggests Tamil content
                # Tamil content in English OCR often produces gibberish or very few words
                if self._should_use_tamil(eng_text, image):
                    logger.info("Tamil characters detected, using Tamil+English OCR")
                    # Second pass: Full Tamil + English OCR
                    return self._run_tesseract(image, language='tam+eng', psm=6)
                else:
                    logger.debug("Using English OCR")
                    return eng_text
            else:
                # Direct Tamil + English OCR
                return self._run_tesseract(image, language='tam+eng', psm=6)
                
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}", exc_info=True)
            return ""
    
    def _run_tesseract(
        self, 
        image: Image.Image, 
        language: str = 'eng',
        psm: int = 6,
        oem: int = 3
    ) -> str:
        """
        Run Tesseract OCR with specified configuration
        
        Args:
            image: PIL Image object
            language: Tesseract language code(s). Examples:
                     'eng' - English only
                     'tam' - Tamil only
                     'tam+eng' - Tamil and English
            psm: Page segmentation mode (default 6 = uniform text block)
                 1 = Automatic page segmentation with OSD
                 3 = Fully automatic page segmentation (default)
                 6 = Assume uniform block of text
                 11 = Sparse text
            oem: OCR Engine mode (default 3 = Default, based on what is available)
                 0 = Legacy engine only
                 1 = Neural nets LSTM engine only
                 2 = Legacy + LSTM engines
                 3 = Default, based on what is available
        
        Returns:
            Extracted text
        """
        if not TESSERACT_AVAILABLE or pytesseract is None:
            logger.warning(
                "Tesseract not available. OCR will be skipped for image-based PDFs. "
                "To enable OCR, install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki"
            )
            return ""
            
        try:
            config = f'--oem {oem} --psm {psm} -l {language}'
            text = pytesseract.image_to_string(image, config=config)
            return text.strip()
            
        except Exception as e:
            if "TesseractNotFoundError" in str(type(e).__name__):
                logger.warning(
                    "Tesseract not found. OCR skipped. Install Tesseract OCR:\n"
                    "Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
                    "Linux: sudo apt-get install tesseract-ocr tesseract-ocr-tam"
                )
            else:
                logger.error(f"Tesseract execution failed: {str(e)}")
            return ""
    
    def _should_use_tamil(self, english_ocr_text: str, image: Image.Image) -> bool:
        """
        Determine if Tamil OCR should be used based on initial English OCR results
        
        Strategy:
        1. If English OCR produces very little text, likely non-English script
        2. If text has low word-to-character ratio, likely Tamil
        3. Sample a small region with Tamil-specific OCR to confirm
        
        Args:
            english_ocr_text: Result from English-only OCR
            image: Original image
        
        Returns:
            True if Tamil OCR should be used
        """
        # Check 1: Very little text extracted (< 20 chars)
        if len(english_ocr_text.strip()) < 20:
            logger.debug("Very little text extracted, trying Tamil")
            return True
        
        # Check 2: Low word count despite characters (Tamil has more chars per word)
        words = english_ocr_text.split()
        chars = len(english_ocr_text.replace(' ', ''))
        
        if chars > 0:
            char_to_word_ratio = chars / max(len(words), 1)
            # Tamil typically has higher ratio due to complex characters
            if char_to_word_ratio > 15:
                logger.debug(f"High char/word ratio ({char_to_word_ratio:.1f}), likely Tamil")
                return True
        
        # Check 3: Look for Tamil unicode characters in a test OCR
        # This is a fallback - do a small test with Tamil OCR on image crop
        try:
            # Test on a small region (top 1/4 of image)
            width, height = image.size
            test_region = image.crop((0, 0, width, height // 4))
            test_text = self._run_tesseract(test_region, language='tam', psm=6)
            
            # If Tamil OCR extracts significantly more text, use Tamil
            if len(test_text.strip()) > len(english_ocr_text.strip()) * 1.5:
                logger.debug("Tamil OCR extracts more text, using Tamil")
                return True
                
        except Exception as e:
            logger.debug(f"Tamil detection test failed: {e}")
        
        return False
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text (basic cleaning)
        
        Args:
            text: Raw extracted text
        
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _clean_ocr_text(self, text: str) -> str:
        """
        Clean OCR text by removing common noise patterns
        
        OCR often introduces artifacts like:
        - Random single characters
        - Weird spacing
        - Misrecognized symbols
        
        Args:
            text: Raw OCR text
        
        Returns:
            Cleaned text with noise removed
        """
        if not text:
            return ""
        
        # Basic cleaning
        text = self._clean_text(text)
        
        # Remove standalone single characters (except common ones)
        # Keep: I, a, A, numbers, Tamil characters
        text = re.sub(r'\b[b-hj-zB-HJ-Z]\b', '', text)
        
        # Remove lines with only special characters
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Keep line if it has at least some alphanumeric or Tamil characters
            if re.search(r'[\w\u0B80-\u0BFF]', line):
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Remove excessive punctuation
        text = re.sub(r'([!?.,])\1+', r'\1', text)
        
        # Final whitespace cleanup
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def get_page_text(self, pages_data: List[Dict], page_number: int) -> str:
        """
        Get text from a specific page
        
        Args:
            pages_data: List of page dictionaries from extract_text_from_pdf
            page_number: Page number to retrieve (1-indexed)
        
        Returns:
            Text content of the specified page, or empty string if not found
        """
        for page in pages_data:
            if page['page_number'] == page_number:
                return page['text']
        return ""
    
    def get_full_text(self, pages_data: List[Dict]) -> str:
        """
        Combine all pages into a single text string
        
        Args:
            pages_data: List of page dictionaries from extract_text_from_pdf
        
        Returns:
            All pages concatenated with page breaks
        """
        full_text = []
        for page in pages_data:
            full_text.append(f"--- Page {page['page_number']} ---")
            full_text.append(page['text'])
            full_text.append("")  # Empty line between pages
        
        return '\n'.join(full_text)
    
    def get_extraction_stats(self, pages_data: List[Dict]) -> Dict:
        """
        Get statistics about the extraction
        
        Args:
            pages_data: List of page dictionaries from extract_text_from_pdf
        
        Returns:
            Dictionary with extraction statistics
        """
        total_pages = len(pages_data)
        direct_pages = sum(1 for p in pages_data if p['extraction_method'] == 'direct')
        ocr_pages = sum(1 for p in pages_data if p['extraction_method'] == 'ocr')
        total_words = sum(p['word_count'] for p in pages_data)
        
        return {
            'total_pages': total_pages,
            'direct_extraction_pages': direct_pages,
            'ocr_pages': ocr_pages,
            'total_words': total_words,
            'avg_words_per_page': total_words / total_pages if total_pages > 0 else 0
        }


# Keep the old PDFProcessor class for backward compatibility
class PDFProcessor(PDFExtractor):
    """
    Alias for PDFExtractor to maintain backward compatibility
    
    Use PDFExtractor for new code.
    """
    pass
