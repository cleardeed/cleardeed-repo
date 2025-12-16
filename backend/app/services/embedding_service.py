"""
Local Embedding Service

Service for generating vector embeddings from text using local transformer models.
Uses sentence-transformers library with multilingual support for Tamil and English.

This service implements a singleton pattern to load the model only once and
reuse it across multiple embedding requests.
"""

import logging
import time
from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config.embedding_config import (
    EmbeddingConfig,
    DEFAULT_CONFIG,
    get_device
)


logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Singleton service for generating text embeddings using local models.
    
    This service loads a sentence-transformer model once and reuses it for all
    embedding requests. Supports both Tamil and English text with automatic
    L2 normalization for cosine similarity.
    
    Example:
        >>> service = EmbeddingService.get_instance()
        >>> texts = ["This is English", "இது தமிழ்"]
        >>> embeddings = service.embed_texts(texts)
        >>> print(len(embeddings))  # 2
        >>> print(len(embeddings[0]))  # 1024 (for multilingual-e5-large)
    """
    
    _instance: Optional['EmbeddingService'] = None
    _model: Optional[SentenceTransformer] = None
    _config: Optional[EmbeddingConfig] = None
    _is_loaded: bool = False
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the embedding service.
        
        Note: Use get_instance() instead of direct instantiation to ensure
        singleton behavior.
        
        Args:
            config: Embedding configuration (uses DEFAULT_CONFIG if None)
        """
        if EmbeddingService._instance is not None:
            logger.warning(
                "EmbeddingService instance already exists. "
                "Use get_instance() instead of direct instantiation."
            )
        
        self._config = config or DEFAULT_CONFIG
        logger.info(f"Initializing EmbeddingService with model: {self._config.model_name}")
    
    @classmethod
    def get_instance(cls, config: Optional[EmbeddingConfig] = None) -> 'EmbeddingService':
        """
        Get or create the singleton instance of EmbeddingService.
        
        Args:
            config: Configuration to use (only used on first call)
        
        Returns:
            EmbeddingService: The singleton instance
        """
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """
        Reset the singleton instance (useful for testing or model switching).
        This will unload the current model from memory.
        """
        if cls._instance is not None:
            logger.info("Resetting EmbeddingService instance")
            if cls._model is not None:
                # Clear model from memory
                del cls._model
                cls._model = None
            cls._instance = None
            cls._is_loaded = False
    
    def _load_model(self):
        """
        Load the sentence-transformer model into memory.
        
        This method is called automatically on first use. It loads the model
        onto the configured device (CPU or CUDA) and logs the load time.
        """
        if self._is_loaded and self._model is not None:
            logger.debug("Model already loaded, skipping...")
            return
        
        logger.info(f"Loading embedding model: {self._config.model_name}")
        logger.info(f"Target device: {self._config.device}")
        
        start_time = time.time()
        
        try:
            # Load model from HuggingFace
            self._model = SentenceTransformer(
                self._config.model_name,
                device=self._config.device
            )
            
            # Configure model settings
            self._model.max_seq_length = self._config.max_sequence_length
            
            # Enable FP16 if configured and on CUDA
            if self._config.use_fp16 and self._config.device == "cuda":
                logger.info("Enabling FP16 (half-precision) mode for GPU")
                self._model.half()
            
            load_time = time.time() - start_time
            
            logger.info(
                f"Model loaded successfully in {load_time:.2f} seconds "
                f"(dimension: {self._config.embedding_dimension})"
            )
            
            self._is_loaded = True
            
            # Store in class variable for singleton access
            EmbeddingService._model = self._model
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise RuntimeError(f"Could not load embedding model: {str(e)}") from e
    
    def embed_texts(
        self, 
        texts: Union[str, List[str]], 
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for one or more texts.
        
        This method handles Tamil, English, and mixed-language text. Embeddings
        are automatically normalized using L2 normalization for cosine similarity.
        
        Args:
            texts: Single text string or list of text strings
            batch_size: Override default batch size (uses config if None)
            show_progress_bar: Show progress bar for large batches
        
        Returns:
            List of embedding vectors (each vector is a list of floats)
        
        Raises:
            ValueError: If texts is empty or contains only invalid inputs
            RuntimeError: If model loading fails
        
        Example:
            >>> service = EmbeddingService.get_instance()
            >>> embeddings = service.embed_texts([
            ...     "Property deed document",
            ...     "சொத்து ஆவணம்"
            ... ])
            >>> len(embeddings)  # 2
            >>> len(embeddings[0])  # 1024
        """
        # Ensure model is loaded
        if not self._is_loaded or self._model is None:
            self._load_model()
        
        # Handle empty input
        if not texts:
            logger.warning("embed_texts called with empty input")
            return []
        
        # Convert single string to list
        if isinstance(texts, str):
            texts = [texts]
        
        # Validate and clean inputs
        cleaned_texts = self._validate_and_clean_texts(texts)
        
        if not cleaned_texts:
            logger.warning("All input texts were invalid or empty")
            return []
        
        # Get batch size
        if batch_size is None:
            batch_size = self._config.batch_size
        
        try:
            logger.debug(f"Encoding {len(cleaned_texts)} texts with batch_size={batch_size}")
            
            # Generate embeddings
            embeddings = self._model.encode(
                cleaned_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=self._config.normalize_embeddings,
                convert_to_numpy=True
            )
            
            # Convert numpy array to list of lists
            embeddings_list = embeddings.tolist()
            
            logger.debug(
                f"Generated {len(embeddings_list)} embeddings "
                f"of dimension {len(embeddings_list[0])}"
            )
            
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Error during embedding generation: {str(e)}")
            raise RuntimeError(f"Embedding generation failed: {str(e)}") from e
    
    def _validate_and_clean_texts(self, texts: List[str]) -> List[str]:
        """
        Validate and clean input texts.
        
        This method:
        - Removes None values
        - Converts non-string types to strings
        - Strips whitespace
        - Replaces empty strings with a placeholder
        
        Args:
            texts: List of input texts
        
        Returns:
            List of cleaned, valid texts
        """
        cleaned = []
        
        for i, text in enumerate(texts):
            # Handle None
            if text is None:
                logger.warning(f"Text at index {i} is None, skipping")
                continue
            
            # Convert to string if needed
            if not isinstance(text, str):
                logger.debug(f"Converting text at index {i} from {type(text)} to str")
                text = str(text)
            
            # Strip whitespace
            text = text.strip()
            
            # Handle empty strings
            if not text:
                logger.warning(f"Text at index {i} is empty after cleaning, using placeholder")
                text = "[EMPTY]"
            
            cleaned.append(text)
        
        return cleaned
    
    def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text (convenience method).
        
        Args:
            text: Input text string
        
        Returns:
            Single embedding vector as list of floats
        
        Example:
            >>> service = EmbeddingService.get_instance()
            >>> embedding = service.embed_single("Property deed")
            >>> len(embedding)  # 1024
        """
        embeddings = self.embed_texts([text])
        return embeddings[0] if embeddings else []
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this service.
        
        Returns:
            int: Embedding dimension (e.g., 1024 for multilingual-e5-large)
        """
        return self._config.embedding_dimension
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information including name, dimension, device, etc.
        """
        return {
            "model_name": self._config.model_name,
            "embedding_dimension": self._config.embedding_dimension,
            "device": self._config.device,
            "max_sequence_length": self._config.max_sequence_length,
            "normalize_embeddings": self._config.normalize_embeddings,
            "is_loaded": self._is_loaded,
            "batch_size": self._config.batch_size,
            "use_fp16": self._config.use_fp16,
        }
    
    def compute_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Since embeddings are L2-normalized, this is simply the dot product.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            float: Similarity score between -1 and 1 (higher is more similar)
        
        Example:
            >>> service = EmbeddingService.get_instance()
            >>> emb1 = service.embed_single("Property deed")
            >>> emb2 = service.embed_single("Real estate document")
            >>> similarity = service.compute_similarity(emb1, emb2)
            >>> print(f"Similarity: {similarity:.3f}")
        """
        # Convert to numpy for efficient computation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # If embeddings are normalized, dot product = cosine similarity
        if self._config.normalize_embeddings:
            similarity = np.dot(vec1, vec2)
        else:
            # Manual cosine similarity calculation
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        
        return float(similarity)


# Convenience function for quick access
def get_embedding_service(config: Optional[EmbeddingConfig] = None) -> EmbeddingService:
    """
    Get the singleton EmbeddingService instance.
    
    Args:
        config: Configuration (only used on first call)
    
    Returns:
        EmbeddingService: The singleton instance
    """
    return EmbeddingService.get_instance(config)


if __name__ == "__main__":
    """
    Test and example usage of the EmbeddingService.
    Run this file directly to test embedding generation.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== Embedding Service Test ===\n")
    
    # Get service instance
    service = EmbeddingService.get_instance()
    
    # Print model info
    print("Model Information:")
    info = service.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()
    
    # Test English text
    print("Testing English text:")
    english_text = "This is a property deed document"
    english_embedding = service.embed_single(english_text)
    print(f"  Text: '{english_text}'")
    print(f"  Embedding dimension: {len(english_embedding)}")
    print(f"  First 5 values: {english_embedding[:5]}")
    print()
    
    # Test Tamil text
    print("Testing Tamil text:")
    tamil_text = "இது ஒரு சொத்து ஆவணம்"
    tamil_embedding = service.embed_single(tamil_text)
    print(f"  Text: '{tamil_text}'")
    print(f"  Embedding dimension: {len(tamil_embedding)}")
    print(f"  First 5 values: {tamil_embedding[:5]}")
    print()
    
    # Test batch embedding
    print("Testing batch embedding:")
    batch_texts = [
        "Property deed",
        "Sale agreement",
        "சொத்து ஆவணம்",
        "விற்பனை ஒப்பந்தம்"
    ]
    batch_embeddings = service.embed_texts(batch_texts)
    print(f"  Number of texts: {len(batch_texts)}")
    print(f"  Number of embeddings: {len(batch_embeddings)}")
    print()
    
    # Test similarity
    print("Testing similarity computation:")
    sim1 = service.compute_similarity(english_embedding, tamil_embedding)
    print(f"  English vs Tamil: {sim1:.4f}")
    
    sim2 = service.compute_similarity(batch_embeddings[0], batch_embeddings[1])
    print(f"  'Property deed' vs 'Sale agreement': {sim2:.4f}")
    
    sim3 = service.compute_similarity(batch_embeddings[2], batch_embeddings[3])
    print(f"  'சொத்து ஆவணம்' vs 'விற்பனை ஒப்பந்தம்': {sim3:.4f}")
