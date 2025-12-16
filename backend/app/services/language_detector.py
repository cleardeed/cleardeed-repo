"""
Language Detection Service

This module detects the language of text content, with special support for
Tamil and English. Can handle mixed-language documents.

Responsibilities:
- Detect if text is Tamil, English, or mixed
- Calculate confidence scores
- Identify language distribution in mixed documents
- Handle unicode Tamil characters correctly
"""

from typing import Dict, Tuple, Optional
import logging
import re

# Try to import fasttext, fall back to langdetect
try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False
    try:
        import langdetect
        LANGDETECT_AVAILABLE = True
    except ImportError:
        LANGDETECT_AVAILABLE = False

logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Detects language in text content with focus on Tamil and English
    
    Uses a hybrid approach:
    1. Unicode range detection for Tamil characters (U+0B80 to U+0BFF)
    2. fastText language identification model (primary)
    3. langdetect library (fallback)
    4. Character frequency analysis (final fallback)
    
    Language codes:
    - 'ta': Tamil
    - 'en': English
    - 'mixed': Mixed Tamil and English (both > 20%)
    """
    
    def __init__(self, fasttext_model_path: Optional[str] = None):
        """
        Initialize language detection models
        
        Args:
            fasttext_model_path: Path to fastText model file (optional).
                               If None, uses built-in detection methods.
                               Download model from:
                               https://fasttext.cc/docs/en/language-identification.html
        """
        self.fasttext_model = None
        
        # Try to load fastText model if available
        if FASTTEXT_AVAILABLE and fasttext_model_path:
            try:
                self.fasttext_model = fasttext.load_model(fasttext_model_path)
                logger.info("fastText model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load fastText model: {e}")
        
        # Tamil unicode range: U+0B80 to U+0BFF
        self.tamil_pattern = re.compile(r'[\u0B80-\u0BFF]')
        
        # English pattern (basic ASCII letters)
        self.english_pattern = re.compile(r'[a-zA-Z]')
        
        logger.info("LanguageDetector initialized")
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect primary language of text with confidence score
        
        This is the main entry point. Uses multiple detection methods:
        1. Unicode analysis for Tamil detection
        2. fastText model (if available)
        3. langdetect library (fallback)
        4. Character frequency analysis (final fallback)
        
        Args:
            text: Text content to analyze (minimum 10 characters recommended)
        
        Returns:
            Tuple of (language_code, confidence_score)
            - language_code: 'en', 'ta', or 'mixed'
            - confidence_score: 0.0 to 1.0 (higher = more confident)
        
        Example:
            >>> detector = LanguageDetector()
            >>> lang, conf = detector.detect_language("This is English text")
            >>> print(f"Language: {lang}, Confidence: {conf:.2f}")
            Language: en, Confidence: 0.95
        """
        if not text or len(text.strip()) < 3:
            logger.warning("Text too short for reliable detection")
            return 'en', 0.1  # Default to English with low confidence
        
        # Step 1: Check for mixed content first (most specific)
        tamil_ratio = self.get_tamil_percentage(text)
        english_ratio = self._get_english_percentage(text)
        
        # If both languages are significantly present, it's mixed
        if tamil_ratio > 0.2 and english_ratio > 0.2:
            confidence = 0.8  # High confidence for mixed detection
            logger.debug(f"Mixed language detected: Tamil {tamil_ratio:.2%}, English {english_ratio:.2%}")
            return 'mixed', confidence
        
        # Step 2: Use unicode detection for clear Tamil content
        if tamil_ratio > 0.5:
            # Predominantly Tamil
            confidence = min(0.95, tamil_ratio)
            logger.debug(f"Tamil detected via unicode: {tamil_ratio:.2%}")
            return 'ta', confidence
        
        # Step 3: Try fastText if available (most accurate for general text)
        if self.fasttext_model:
            fasttext_result = self._detect_with_fasttext(text)
            if fasttext_result:
                return fasttext_result
        
        # Step 4: Try langdetect library (good fallback)
        if LANGDETECT_AVAILABLE:
            langdetect_result = self._detect_with_langdetect(text)
            if langdetect_result:
                return langdetect_result
        
        # Step 5: Final fallback - character frequency analysis
        if tamil_ratio > english_ratio:
            confidence = min(0.7, tamil_ratio)
            return 'ta', confidence
        else:
            confidence = min(0.7, english_ratio)
            return 'en', confidence
    
    def _detect_with_fasttext(self, text: str) -> Optional[Tuple[str, float]]:
        """
        Detect language using fastText model
        
        Args:
            text: Text to analyze
        
        Returns:
            Tuple of (language_code, confidence) or None if detection fails
        """
        try:
            # fastText returns labels like '__label__en' and probabilities
            predictions = self.fasttext_model.predict(text.replace('\n', ' '), k=1)
            
            # Extract language code (remove __label__ prefix)
            lang_label = predictions[0][0].replace('__label__', '')
            confidence = float(predictions[1][0])
            
            # Map fastText language codes to our codes
            if lang_label == 'ta' or lang_label == 'tam':
                return 'ta', confidence
            elif lang_label == 'en' or lang_label == 'eng':
                return 'en', confidence
            else:
                # Unknown language, return None to try other methods
                logger.debug(f"fastText detected unsupported language: {lang_label}")
                return None
                
        except Exception as e:
            logger.debug(f"fastText detection failed: {e}")
            return None
    
    def _detect_with_langdetect(self, text: str) -> Optional[Tuple[str, float]]:
        """
        Detect language using langdetect library
        
        Args:
            text: Text to analyze
        
        Returns:
            Tuple of (language_code, confidence) or None if detection fails
        """
        try:
            # langdetect can be unstable, so we catch all exceptions
            detected_lang = langdetect.detect(text)
            
            # Get confidence using detect_langs (returns probabilities)
            lang_probs = langdetect.detect_langs(text)
            confidence = 0.5  # Default medium confidence
            
            for lang_prob in lang_probs:
                if lang_prob.lang == detected_lang:
                    confidence = lang_prob.prob
                    break
            
            # Map language codes
            if detected_lang in ['ta', 'tam']:
                return 'ta', confidence
            elif detected_lang in ['en', 'eng']:
                return 'en', confidence
            else:
                logger.debug(f"langdetect detected unsupported language: {detected_lang}")
                return None
                
        except Exception as e:
            logger.debug(f"langdetect detection failed: {e}")
            return None
    
    def analyze_mixed_content(self, text: str) -> Dict[str, float]:
        """
        Analyze language distribution in mixed-language documents
        
        Calculates the proportion of Tamil vs English characters in the text.
        Useful for understanding bilingual documents.
        
        Args:
            text: Text content to analyze
        
        Returns:
            Dictionary with language codes and their percentages (0.0 to 1.0)
            Example: {'en': 0.65, 'ta': 0.35}
        """
        tamil_ratio = self.get_tamil_percentage(text)
        english_ratio = self._get_english_percentage(text)
        
        # Normalize so they sum to 1.0 (if both are present)
        total = tamil_ratio + english_ratio
        
        if total > 0:
            return {
                'ta': tamil_ratio / total,
                'en': english_ratio / total
            }
        else:
            # No recognizable characters
            return {'en': 0.5, 'ta': 0.5}
    
    def is_tamil_text(self, text: str, threshold: float = 0.1) -> bool:
        """
        Check if text contains Tamil characters
        
        Args:
            text: Text to check
            threshold: Minimum percentage of Tamil characters (default 10%)
        
        Returns:
            True if Tamil characters detected above threshold
        """
        tamil_percentage = self.get_tamil_percentage(text)
        return tamil_percentage >= threshold
    
    def get_tamil_percentage(self, text: str) -> float:
        """
        Calculate percentage of Tamil characters in text
        
        Scans text for characters in Tamil unicode range (U+0B80 to U+0BFF).
        Only counts alphabetic characters (ignores numbers, punctuation, spaces).
        
        Args:
            text: Text to analyze
        
        Returns:
            Percentage of Tamil characters (0.0 to 1.0)
        """
        if not text:
            return 0.0
        
        # Count Tamil characters using unicode range
        tamil_chars = len(self.tamil_pattern.findall(text))
        
        # Count total alphabetic characters (ignore spaces, numbers, punctuation)
        # Include both English and Tamil alphabetic characters
        total_alpha_chars = len(self.english_pattern.findall(text)) + tamil_chars
        
        if total_alpha_chars == 0:
            return 0.0
        
        return tamil_chars / total_alpha_chars
    
    def _get_english_percentage(self, text: str) -> float:
        """
        Calculate percentage of English characters in text
        
        Args:
            text: Text to analyze
        
        Returns:
            Percentage of English characters (0.0 to 1.0)
        """
        if not text:
            return 0.0
        
        # Count English alphabetic characters
        english_chars = len(self.english_pattern.findall(text))
        tamil_chars = len(self.tamil_pattern.findall(text))
        
        # Total alphabetic characters
        total_alpha_chars = english_chars + tamil_chars
        
        if total_alpha_chars == 0:
            return 0.0
        
        return english_chars / total_alpha_chars
    
    def detect(self, text: str) -> Tuple[str, float]:
        """
        Alias for detect_language() for backward compatibility
        
        Args:
            text: Text content to analyze
        
        Returns:
            Tuple of (language_code, confidence_score)
        """
        return self.detect_language(text)
