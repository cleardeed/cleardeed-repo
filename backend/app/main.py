"""
FastAPI Application Entry Point

This module initializes and configures the FastAPI application for the legal document
analysis system. It sets up middleware, CORS, and registers all API routes.

Responsibilities:
- Initialize FastAPI app with metadata
- Configure CORS for frontend communication
- Register API routers
- Set up middleware (logging, error handling)
- Health check endpoints
- Startup/shutdown event handlers
- Load vector search components (FAISS, embeddings, metadata) at startup

Lifecycle Management:
    STARTUP:
    1. Initialize embedding service (loads model into memory once)
    2. Load FAISS index from disk (data/faiss/)
    3. Load vector metadata from disk (data/metadata/)
    4. Fail fast if any component fails to load
    5. All services are singletons - initialized once, reused globally
    
    RUNTIME:
    - Embedding service: Reused for all embedding requests
    - FAISS index: Shared across all search requests
    - Metadata store: Shared for all metadata lookups
    
    SHUTDOWN:
    - No special cleanup needed (services are stateless)
    - Index already persisted to disk
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import documents
from app.routes import gap_analysis, embeddings
from app.config.settings import settings
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import FAISSIndexManager
from app.services.metadata_store import VectorMetadataStore
from app.services.semantic_search import SemanticSearchService
from app.services.gap_detection_engine import GAPDetectionEngine
from app.services.ollama_client import OllamaClient, OllamaConfig
from app.config.embedding_config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# Global service instances (initialized at startup)
# These are singletons shared across all requests
embedding_service: EmbeddingService = None
vector_store: FAISSIndexManager = None
metadata_store: VectorMetadataStore = None
gap_detection_engine: GAPDetectionEngine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown logic.
    
    Modern FastAPI approach (replaces @app.on_event decorators).
    Ensures proper initialization order and fail-fast behavior.
    """
    # STARTUP
    logger.info("=" * 60)
    logger.info("ClearDeed Backend - Starting up...")
    logger.info("=" * 60)
    
    try:
        # Step 1: Initialize Embedding Service (loads model into memory)
        # This is the most expensive operation - happens once at startup
        logger.info("1/4 Initializing embedding service (loading model)...")
        global embedding_service
        embedding_service = EmbeddingService(DEFAULT_CONFIG)
        logger.info(f"  ✓ Embedding service ready: {embedding_service.get_model_info()['model_name']}")
        
        # Step 2: Initialize FAISS Vector Store
        logger.info("2/4 Initializing FAISS vector store...")
        global vector_store
        from pathlib import Path
        vector_store = FAISSIndexManager(
            dimension=embedding_service.get_embedding_dimension(),
            index_type="IndexFlatIP",
            index_path=Path("data/faiss")
        )
        
        # Try to load existing index from disk (graceful failure if not found)
        index_loaded = vector_store.load_index("index")
        if index_loaded:
            stats = vector_store.get_stats()
            logger.info(
                f"  ✓ Loaded existing FAISS index: {stats['total_vectors']} vectors"
            )
        else:
            logger.info(
                f"  ✓ FAISS index initialized (no existing index found - will create on first upload)"
            )
        
        # Step 3: Initialize Metadata Store
        logger.info("3/4 Initializing vector metadata store...")
        global metadata_store
        metadata_store = VectorMetadataStore()
        
        # Try to load existing metadata from disk (graceful failure if not found)
        metadata_loaded = metadata_store.load("vector_metadata.json")
        if metadata_loaded:
            count = metadata_store.count()
            logger.info(
                f"  ✓ Loaded existing metadata: {count} entries"
            )
        else:
            logger.info(
                f"  ✓ Metadata store initialized (no existing metadata found)"
            )
        
        # Step 4: Initialize GAP Detection Engine with Ollama
        logger.info("4/4 Initializing GAP detection engine with Ollama...")
        global gap_detection_engine
        try:
            from app.services.gap_context_builder import GAPContextBuilder
            from app.services.prompt_factory import PromptFactory
            
            # Initialize Ollama client
            ollama_config = OllamaConfig(
                model_name=settings.OLLAMA_MODEL,  # gap-english
                base_url=settings.OLLAMA_BASE_URL,
                use_custom_model=True  # Using custom gap-english model
            )
            ollama_client = OllamaClient(ollama_config)
            
            # Initialize semantic search service
            semantic_search = SemanticSearchService(
                embedding_service=embedding_service,
                vector_store=vector_store,
                metadata_store=metadata_store
            )
            
            # Initialize context builder for clause retrieval
            context_builder = GAPContextBuilder(
                semantic_search=semantic_search,
                metadata_store=metadata_store
            )
            
            # Initialize prompt factory for generating LLM prompts
            prompt_factory = PromptFactory()
            
            # Initialize GAP detection engine
            gap_detection_engine = GAPDetectionEngine(
                context_builder=context_builder,
                prompt_factory=prompt_factory,
                ollama_client=ollama_client,
                metadata_store=metadata_store
            )
            
            # Register engine with gap_analysis router
            from app.routes import gap_analysis
            gap_analysis.set_gap_detection_engine(gap_detection_engine)
            
            logger.info(f"  ✓ GAP detection engine ready with model: {settings.OLLAMA_MODEL}")
        except Exception as e:
            logger.warning(f"  ⚠ GAP detection engine initialization failed: {e}")
            logger.warning("  GAP analysis will not be available")
            gap_detection_engine = None
        
        logger.info("=" * 60)
        logger.info("✓ ClearDeed Backend - Startup complete!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"✗ STARTUP FAILED: {str(e)}")
        logger.error("=" * 60)
        raise  # Fail fast - don't start if initialization fails
    
    # Application runs here (yield control to FastAPI)
    yield
    
    # SHUTDOWN
    logger.info("ClearDeed Backend - Shutting down...")
    logger.info("✓ Shutdown complete")


# Initialize FastAPI app with lifespan handler
app = FastAPI(
    title="ClearDeed Legal Document Analyzer",
    description="Backend API for processing and analyzing legal documents",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "status": "healthy",
        "service": "ClearDeed Document Analyzer",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """
    Detailed health check endpoint.
    
    Returns service status and statistics about loaded components.
    """
    # Check if services are initialized
    services_ready = all([
        embedding_service is not None,
        vector_store is not None,
        metadata_store is not None
    ])
    
    health_data = {
        "status": "healthy" if services_ready else "degraded",
        "services": {
            "api": "up",
            "embedding_service": "up" if embedding_service else "down",
            "vector_store": "up" if vector_store else "down",
            "metadata_store": "up" if metadata_store else "down",
        }
    }
    
    # Add statistics if services are ready
    if services_ready:
        try:
            health_data["statistics"] = {
                "embedding_dimension": embedding_service.get_embedding_dimension(),
                "embedding_model": embedding_service.get_model_info()["model_name"],
                "total_vectors": vector_store.get_total_vectors(),
                "metadata_entries": metadata_store.count(),
                "metadata_stats": metadata_store.get_stats()
            }
        except Exception as e:
            logger.error(f"Error collecting health stats: {str(e)}")
            health_data["status"] = "degraded"
            health_data["error"] = str(e)
    
    return health_data


def get_embedding_service() -> EmbeddingService:
    """
    Get the global embedding service instance.
    
    Dependency injection helper for FastAPI routes.
    
    Raises:
        HTTPException: If service is not initialized
    """
    if embedding_service is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding service not initialized. Application startup may have failed."
        )
    return embedding_service


def get_vector_store() -> FAISSIndexManager:
    """
    Get the global FAISS vector store instance.
    
    Dependency injection helper for FastAPI routes.
    
    Raises:
        HTTPException: If service is not initialized
    """
    if vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store not initialized. Application startup may have failed."
        )
    return vector_store


def get_metadata_store() -> VectorMetadataStore:
    """
    Get the global metadata store instance.
    
    Dependency injection helper for FastAPI routes.
    
    Raises:
        HTTPException: If service is not initialized
    """
    if metadata_store is None:
        raise HTTPException(
            status_code=503,
            detail="Metadata store not initialized. Application startup may have failed."
        )
    return metadata_store


# Register API routes (using /api prefix to match frontend expectations)
app.include_router(documents.router, prefix="/api", tags=["documents"])
app.include_router(gap_analysis.router, prefix="/api", tags=["gap-analysis"])
app.include_router(embeddings.router, prefix="/api", tags=["embeddings"])
