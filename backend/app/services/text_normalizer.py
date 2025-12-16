"""
Text Normalization Service for Legal Documents

This module provides text normalization and cleaning functions specifically
designed for legal documents in Tamil and English. It handles common issues
from OCR extraction and prepares text for LLM analysis.

Responsibilities:
- Normalize whitespace and line breaks
- Remove repeated headers/footers
- Fix Tamil Unicode normalization issues
- Repair broken OCR word splits
- Clean formatting artifacts
"""

from typing import Dict, List, Tuple, Optional
import logging
import re
from collections import Counter

# Try to import Indic NLP library for Tamil processing
try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    from indicnlp.tokenize import indic_tokenize
    INDIC_NLP_AVAILABLE = True
except ImportError:
    INDIC_NLP_AVAILABLE = False
    logging.warning("Indic NLP library not available. Tamil normalization will be limited.")

logger = logging.getLogger(__name__)


class TextNormalizer:
    """
    Normalizes and cleans extracted text from legal documents
    
    Handles language-specific normalization:
    - English: Whitespace, headers/footers, common OCR errors
    - Tamil: Unicode normalization, word splits, character fixes
    
    Example:
        >>> normalizer = TextNormalizer()
        >>> cleaned = normalizer.normalize("raw text...", language="en")
    """
    
    def __init__(self):
        """Initialize text normalizer with language-specific tools"""
        self.tamil_normalizer = None
        
        # Initialize Tamil normalizer if Indic NLP is available
        if INDIC_NLP_AVAILABLE:
            try:
                factory = IndicNormalizerFactory()
                self.tamil_normalizer = factory.get_normalizer("ta")
                logger.info("Tamil normalizer initialized with Indic NLP")
            except Exception as e:
                logger.warning(f"Could not initialize Tamil normalizer: {e}")
        
        # Common legal document headers/footers patterns
        # Example: "Page 1 of 10", "Document ID: 12345", etc.
        self.header_footer_patterns = [
            r'Page\s+\d+\s+of\s+\d+',
            r'Page\s+\d+',
            r'Document\s+(ID|Number):\s*\S+',
            r'Confidential',
            r'©\s*\d{4}',
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # Dates at top/bottom
        ]
        
        logger.info("TextNormalizer initialized")
    
    def normalize(self, text: str, language: str = "en") -> str:
        """
        Normalize text based on detected language
        
        This is the main entry point. Routes to language-specific normalization.
        
        Args:
            text: Raw extracted text
            language: Language code ('en', 'ta', or 'mixed')
        
        Returns:
            Cleaned and normalized text
        
        Example:
            >>> normalizer = TextNormalizer()
            >>> # English text
            >>> text = "Page 1\\n\\nThis is   a test\\n\\nPage 1"
            >>> clean = normalizer.normalize(text, language="en")
            >>> print(clean)
            This is a test
            
            >>> # Tamil text
            >>> tamil = "வணக்கம்   உலகம்"
            >>> clean = normalizer.normalize(tamil, language="ta")
        """
        if not text:
            return ""
        
        logger.debug(f"Normalizing text ({len(text)} chars) for language: {language}")
        
        # Apply common normalization first
        text = self._normalize_common(text)
        
        # Language-specific normalization
        if language == "ta":
            text = self._normalize_tamil(text)
        elif language == "en":
            text = self._normalize_english(text)
        elif language == "mixed":
            # For mixed content, apply both (Tamil first, then English)
            text = self._normalize_tamil(text)
            text = self._normalize_english(text)
        
        # Final cleanup
        text = self._final_cleanup(text)
        
        logger.debug(f"Normalized text: {len(text)} chars")
        return text
    
    def _normalize_common(self, text: str) -> str:
        """
        Apply common normalization for all languages
        
        - Normalize line endings
        - Remove excessive blank lines
        - Fix spacing around punctuation
        
        Args:
            text: Input text
        
        Returns:
            Text with common normalizations applied
        """
        # Normalize line endings to \n
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive blank lines (more than 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove trailing whitespace from lines
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text
    
    def _normalize_english(self, text: str) -> str:
        """
        Normalize English text from legal documents
        
        Handles:
        - Repeated headers/footers
        - Whitespace normalization
        - Common OCR errors
        - Legal formatting
        
        Example:
            Input:  "Page 1\\n\\nSECTION  1\\n\\nPage 1\\n\\nText here"
            Output: "SECTION 1\\n\\nText here"
        
        Args:
            text: English text
        
        Returns:
            Normalized English text
        """
        # Remove common headers/footers
        text = self._remove_headers_footers(text)
        
        # Normalize whitespace (multiple spaces to single)
        text = re.sub(r' {2,}', ' ', text)
        
        # Fix common OCR errors in English
        # Example: "w hich" -> "which", "t he" -> "the"
        text = self._fix_common_ocr_errors(text)
        
        # Normalize section numbering
        # Example: "Section1" -> "Section 1"
        text = re.sub(r'(Section|Article|Clause)(\d+)', r'\1 \2', text)
        
        # Fix broken hyphenation at line breaks
        # Example: "docu-\nment" -> "document"
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        return text
    
    def _normalize_tamil(self, text: str) -> str:
        """
        Normalize Tamil text from legal documents
        
        Handles:
        - Unicode normalization (various forms of same character)
        - Broken word splits from OCR
        - Character combination fixes
        
        Example:
            Input:  "வண க் க ம்"  (broken word)
            Output: "வணக்கம்"     (properly joined)
        
        Args:
            text: Tamil text
        
        Returns:
            Normalized Tamil text
        """
        # Use Indic NLP normalizer if available
        if self.tamil_normalizer:
            try:
                text = self.tamil_normalizer.normalize(text)
                logger.debug("Applied Indic NLP normalization")
            except Exception as e:
                logger.warning(f"Indic NLP normalization failed: {e}")
        
        # Fix broken Tamil words (common OCR issue)
        # Tamil words often broken by spaces: "வண க்கம்" -> "வணக்கம்"
        text = self._fix_tamil_word_splits(text)
        
        # Normalize Tamil digits to ASCII digits for consistency
        # Tamil digits: ௦ ௧ ௨ ௩ ௪ ௫ ௬ ௭ ௮ ௯
        tamil_digits = '௦௧௨௩௪௫௬௭௮௯'
        for i, tamil_digit in enumerate(tamil_digits):
            text = text.replace(tamil_digit, str(i))
        
        # Fix spacing around Tamil punctuation
        # Example: "வணக்கம் ." -> "வணக்கம்."
        text = re.sub(r'([\u0B80-\u0BFF])\s+([.,!?;:])', r'\1\2', text)
        
        # Remove extra spaces in Tamil text
        # But preserve space between words
        text = re.sub(r' {2,}', ' ', text)
        
        return text
    
    def _fix_tamil_word_splits(self, text: str) -> str:
        """
        Fix broken Tamil words caused by OCR
        
        OCR often incorrectly splits Tamil words with spaces between
        base characters and modifiers (vowel signs, etc.)
        
        Example:
            Input:  "வண க் க ம்"
            Output: "வணக்கம்"
        
        Strategy:
        - Identify Tamil base characters followed by space and modifier
        - Remove the space if it's breaking a word
        
        Args:
            text: Tamil text with potential word splits
        
        Returns:
            Text with word splits repaired
        """
        # Pattern: Tamil character + space + Tamil vowel sign/modifier
        # Tamil vowel signs: U+0BBE-U+0BCD
        # This pattern joins them back together
        
        # Join base character with following vowel signs
        text = re.sub(r'([\u0B80-\u0BFF])\s+([\u0BBE-\u0BCD])', r'\1\2', text)
        
        # Join consonant + virama + space + consonant
        # Example: "க் க" -> "க்க"
        text = re.sub(r'([\u0B95-\u0BB9]\u0BCD)\s+([\u0B95-\u0BB9])', r'\1\2', text)
        
        return text
    
    def _remove_headers_footers(self, text: str) -> str:
        """
        Remove repeated headers and footers from document
        
        Legal documents often have:
        - Page numbers: "Page 1", "Page 2 of 10"
        - Document IDs: "Document ID: 12345"
        - Confidentiality notices
        - Dates
        
        This function identifies and removes lines that repeat across pages.
        
        Example:
            Input:
                "Page 1\\nContent here\\nPage 2\\nMore content\\nPage 3"
            Output:
                "Content here\\nMore content"
        
        Args:
            text: Full document text
        
        Returns:
            Text with headers/footers removed
        """
        lines = text.split('\n')
        
        # Count frequency of each line
        line_counts = Counter(lines)
        
        # Lines that appear more than 2 times are likely headers/footers
        repeated_lines = {line for line, count in line_counts.items() 
                         if count > 2 and len(line.strip()) < 100}
        
        # Also remove lines matching known patterns
        pattern_lines = set()
        for line in lines:
            for pattern in self.header_footer_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    pattern_lines.add(line)
        
        # Combine both sets
        lines_to_remove = repeated_lines | pattern_lines
        
        # Filter out these lines
        cleaned_lines = [line for line in lines if line not in lines_to_remove]
        
        return '\n'.join(cleaned_lines)
    
    def _fix_common_ocr_errors(self, text: str) -> str:
        """
        Fix common OCR misrecognitions in English text
        
        Common OCR errors:
        - "l" (lowercase L) confused with "1" or "I"
        - "O" (letter O) confused with "0" (zero)
        - Broken words: "w hich" -> "which"
        
        Example:
            Input:  "The c ontract states that..."
            Output: "The contract states that..."
        
        Args:
            text: English text with potential OCR errors
        
        Returns:
            Text with common errors fixed
        """
        # Fix broken common words (space in the middle)
        common_words = [
            ('t he', 'the'),
            ('w hich', 'which'),
            ('s hall', 'shall'),
            ('a nd', 'and'),
            ('c ontract', 'contract'),
            ('p roperty', 'property'),
            ('agre ement', 'agreement'),
            ('part y', 'party'),
            ('part ies', 'parties'),
        ]
        
        for broken, fixed in common_words:
            # Case insensitive replacement
            text = re.sub(r'\b' + broken + r'\b', fixed, text, flags=re.IGNORECASE)
        
        # Fix common number/letter confusion in legal contexts
        # Example: "Section l" -> "Section 1"
        text = re.sub(r'\b(Section|Article|Clause)\s+l\b', r'\1 1', text)
        
        # Fix "O" vs "0" in years
        # Example: "2O23" -> "2023"
        text = re.sub(r'\b(19|20)([O0])(\d)\b', r'\1O\3', text)
        text = text.replace('O', '0')  # Then normalize all Os in that context
        
        return text
    
    def _final_cleanup(self, text: str) -> str:
        """
        Final cleanup pass on normalized text
        
        - Remove leading/trailing whitespace
        - Ensure single space between sentences
        - Remove orphaned punctuation
        
        Args:
            text: Text after language-specific normalization
        
        Returns:
            Final cleaned text
        """
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Ensure single space after punctuation
        text = re.sub(r'([.!?])\s+', r'\1 ', text)
        
        # Remove orphaned punctuation at start of lines
        text = re.sub(r'\n\s*[.,;:]\s*', '\n', text)
        
        # Collapse multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove spaces before punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        
        return text
    
    def normalize_batch(
        self, 
        texts: List[str], 
        languages: List[str]
    ) -> List[str]:
        """
        Normalize multiple texts in batch
        
        Args:
            texts: List of text strings
            languages: Corresponding language codes
        
        Returns:
            List of normalized texts
        
        Example:
            >>> normalizer = TextNormalizer()
            >>> texts = ["English text", "Tamil text"]
            >>> langs = ["en", "ta"]
            >>> cleaned = normalizer.normalize_batch(texts, langs)
        """
        if len(texts) != len(languages):
            raise ValueError("texts and languages must have same length")
        
        return [self.normalize(text, lang) for text, lang in zip(texts, languages)]
    
    def get_normalization_stats(self, original: str, normalized: str) -> Dict:
        """
        Get statistics about normalization changes
        
        Args:
            original: Original text
            normalized: Normalized text
        
        Returns:
            Dictionary with statistics
        """
        return {
            'original_length': len(original),
            'normalized_length': len(normalized),
            'chars_removed': len(original) - len(normalized),
            'reduction_percentage': (len(original) - len(normalized)) / len(original) * 100 if original else 0,
            'original_lines': original.count('\n') + 1,
            'normalized_lines': normalized.count('\n') + 1,
        }
