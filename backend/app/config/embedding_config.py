"""
Embedding Model Configuration

Configuration module for local embedding models used in document similarity
and semantic search. This module defines model parameters without loading
the actual models, allowing for easy switching between different embedding
architectures.

Supported Models:
- multilingual-e5-large: Best for multilingual (Tamil + English)
- multilingual-e5-base: Lighter alternative
- paraphrase-multilingual-mpnet-base-v2: Another multilingual option
- all-MiniLM-L6-v2: Fast, English-only fallback
"""

from enum import Enum
from typing import Dict, Optional
from pydantic import BaseModel, Field
import torch


class EmbeddingModel(str, Enum):
    """
    Available embedding models with their HuggingFace identifiers.
    
    All models are from sentence-transformers library and support
    local inference without API calls.
    """
    # Best for Tamil + English legal documents (recommended)
    MULTILINGUAL_E5_LARGE = "intfloat/multilingual-e5-large"
    
    # Lighter alternative with good multilingual support
    MULTILINGUAL_E5_BASE = "intfloat/multilingual-e5-base"
    
    # Alternative multilingual model
    PARAPHRASE_MULTILINGUAL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    
    # Fast English-only model (fallback)
    MINILM_L6 = "sentence-transformers/all-MiniLM-L6-v2"


class DeviceType(str, Enum):
    """
    Computation device options for model inference.
    
    - CPU: Works everywhere, slower
    - CUDA: GPU acceleration, requires NVIDIA GPU
    - AUTO: Automatically selects CUDA if available, else CPU
    """
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class EmbeddingConfig(BaseModel):
    """
    Configuration for embedding model inference.
    
    This class defines all parameters needed for embedding generation
    without actually loading the model. Modify these settings to switch
    models or change inference parameters.
    """
    
    # Model Selection
    model_name: EmbeddingModel = Field(
        default=EmbeddingModel.MULTILINGUAL_E5_LARGE,
        description="HuggingFace model identifier for the embedding model"
    )
    
    # Embedding Dimensions (model-specific)
    embedding_dimension: int = Field(
        default=1024,
        description="Output dimension of embedding vectors (model-dependent)",
        ge=128,
        le=4096
    )
    
    # Device Configuration
    device: DeviceType = Field(
        default=DeviceType.AUTO,
        description="Computation device: 'cpu', 'cuda', or 'auto'"
    )
    
    # Batch Processing
    batch_size: int = Field(
        default=32,
        description="Number of text chunks to process in parallel",
        ge=1,
        le=128
    )
    
    # Maximum Sequence Length
    max_sequence_length: int = Field(
        default=512,
        description="Maximum token length for input text (truncates longer texts)",
        ge=128,
        le=8192
    )
    
    # Normalization
    normalize_embeddings: bool = Field(
        default=True,
        description="Whether to L2-normalize embeddings (recommended for similarity)"
    )
    
    # Pooling Strategy
    pooling_mode: str = Field(
        default="mean",
        description="How to pool token embeddings: 'mean', 'cls', or 'max'"
    )
    
    # Precision (for GPU inference)
    use_fp16: bool = Field(
        default=False,
        description="Use half-precision (FP16) for faster GPU inference"
    )
    
    class Config:
        """Pydantic configuration"""
        use_enum_values = True


# Model-Specific Configurations
# These dictionaries map each model to its actual embedding dimensions
# and recommended settings. Use this when switching models.

MODEL_DIMENSIONS: Dict[EmbeddingModel, int] = {
    EmbeddingModel.MULTILINGUAL_E5_LARGE: 1024,    # Large model, best quality
    EmbeddingModel.MULTILINGUAL_E5_BASE: 768,      # Base model, good balance
    EmbeddingModel.PARAPHRASE_MULTILINGUAL: 768,   # Alternative option
    EmbeddingModel.MINILM_L6: 384,                 # Fast, English-only
}


MODEL_MAX_SEQ_LENGTH: Dict[EmbeddingModel, int] = {
    EmbeddingModel.MULTILINGUAL_E5_LARGE: 512,
    EmbeddingModel.MULTILINGUAL_E5_BASE: 512,
    EmbeddingModel.PARAPHRASE_MULTILINGUAL: 512,
    EmbeddingModel.MINILM_L6: 384,
}


MODEL_DESCRIPTIONS: Dict[EmbeddingModel, str] = {
    EmbeddingModel.MULTILINGUAL_E5_LARGE: 
        "Best quality for Tamil+English legal documents. 1024-dim embeddings. ~2GB model size.",
    
    EmbeddingModel.MULTILINGUAL_E5_BASE: 
        "Good balance of quality and speed. 768-dim embeddings. ~1GB model size.",
    
    EmbeddingModel.PARAPHRASE_MULTILINGUAL: 
        "Alternative multilingual model. 768-dim embeddings. Good for paraphrase detection.",
    
    EmbeddingModel.MINILM_L6: 
        "Fast English-only model. 384-dim embeddings. ~80MB model size. Use as fallback.",
}


def get_device() -> str:
    """
    Auto-detect the best available device for inference.
    
    Returns:
        str: 'cuda' if NVIDIA GPU available, else 'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def create_embedding_config(
    model_name: Optional[EmbeddingModel] = None,
    device: Optional[DeviceType] = None,
    **kwargs
) -> EmbeddingConfig:
    """
    Factory function to create embedding configuration with auto-adjusted settings.
    
    This function automatically sets the embedding dimension and max sequence
    length based on the selected model, preventing configuration mismatches.
    
    Args:
        model_name: Model to use (defaults to MULTILINGUAL_E5_LARGE)
        device: Device type (defaults to AUTO)
        **kwargs: Additional configuration parameters
    
    Returns:
        EmbeddingConfig: Configured embedding settings
    
    Example:
        >>> config = create_embedding_config(
        ...     model_name=EmbeddingModel.MULTILINGUAL_E5_BASE,
        ...     batch_size=16
        ... )
        >>> print(config.embedding_dimension)  # Automatically set to 768
    """
    if model_name is None:
        model_name = EmbeddingModel.MULTILINGUAL_E5_LARGE
    
    # Auto-set dimension based on model
    if 'embedding_dimension' not in kwargs:
        kwargs['embedding_dimension'] = MODEL_DIMENSIONS[model_name]
    
    # Auto-set max sequence length based on model
    if 'max_sequence_length' not in kwargs:
        kwargs['max_sequence_length'] = MODEL_MAX_SEQ_LENGTH[model_name]
    
    # Auto-detect device if set to AUTO
    if device == DeviceType.AUTO or device is None:
        detected_device = get_device()
        device = DeviceType.CUDA if detected_device == "cuda" else DeviceType.CPU
    
    return EmbeddingConfig(
        model_name=model_name,
        device=device,
        **kwargs
    )


def get_model_info(model: EmbeddingModel) -> Dict[str, any]:
    """
    Get comprehensive information about a specific model.
    
    Args:
        model: The embedding model enum
    
    Returns:
        dict: Model information including dimensions, description, etc.
    """
    return {
        "name": model.value,
        "dimension": MODEL_DIMENSIONS[model],
        "max_seq_length": MODEL_MAX_SEQ_LENGTH[model],
        "description": MODEL_DESCRIPTIONS[model],
    }


# Default Configuration Instance
# Use this for consistent configuration across the application
DEFAULT_CONFIG = create_embedding_config()


# Example Configurations for Different Use Cases

# High Quality Configuration (recommended for production)
HIGH_QUALITY_CONFIG = create_embedding_config(
    model_name=EmbeddingModel.MULTILINGUAL_E5_LARGE,
    batch_size=16,
    use_fp16=True,  # Enable FP16 if using GPU
)

# Fast Configuration (for development/testing)
FAST_CONFIG = create_embedding_config(
    model_name=EmbeddingModel.MULTILINGUAL_E5_BASE,
    batch_size=64,
    max_sequence_length=256,
)

# CPU-Only Configuration (for servers without GPU)
CPU_CONFIG = create_embedding_config(
    model_name=EmbeddingModel.MULTILINGUAL_E5_BASE,
    device=DeviceType.CPU,
    batch_size=8,
    use_fp16=False,
)


if __name__ == "__main__":
    """
    Example usage and testing of configuration module.
    Run this file directly to see configuration examples.
    """
    print("=== Embedding Configuration Examples ===\n")
    
    # Default configuration
    print("1. DEFAULT CONFIGURATION:")
    print(f"   Model: {DEFAULT_CONFIG.model_name}")
    print(f"   Dimension: {DEFAULT_CONFIG.embedding_dimension}")
    print(f"   Device: {DEFAULT_CONFIG.device}")
    print(f"   Batch Size: {DEFAULT_CONFIG.batch_size}")
    print(f"   Available Device: {get_device()}")
    print()
    
    # High quality configuration
    print("2. HIGH QUALITY CONFIGURATION:")
    print(f"   Model: {HIGH_QUALITY_CONFIG.model_name}")
    print(f"   Dimension: {HIGH_QUALITY_CONFIG.embedding_dimension}")
    print(f"   FP16: {HIGH_QUALITY_CONFIG.use_fp16}")
    print()
    
    # Fast configuration
    print("3. FAST CONFIGURATION:")
    print(f"   Model: {FAST_CONFIG.model_name}")
    print(f"   Dimension: {FAST_CONFIG.embedding_dimension}")
    print(f"   Max Length: {FAST_CONFIG.max_sequence_length}")
    print()
    
    # Model information
    print("4. AVAILABLE MODELS:")
    for model in EmbeddingModel:
        info = get_model_info(model)
        print(f"   {model.name}:")
        print(f"     - Dimension: {info['dimension']}")
        print(f"     - {info['description']}")
    print()
    
    # Custom configuration
    print("5. CUSTOM CONFIGURATION:")
    custom = create_embedding_config(
        model_name=EmbeddingModel.PARAPHRASE_MULTILINGUAL,
        batch_size=32,
    )
    print(f"   Model: {custom.model_name}")
    print(f"   Dimension: {custom.embedding_dimension} (auto-set)")
    print(f"   Max Length: {custom.max_sequence_length} (auto-set)")
