"""
FastAPI routes for GAP (Grantor, Attestation, Property) analysis.

This module provides endpoints for analyzing legal documents using LLM-based
GAP detection to identify missing or incomplete information.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Path, Body, status
from pydantic import BaseModel, Field

from app.services.gap_detection_engine import (
    GAPDetectionEngine,
    GAPDetectionResult,
    GAPDetectionError
)
from app.services.gap_context_builder import GAPContextBuilder, ContextLengthError
from app.services.prompt_factory import PromptFactory, Language
from app.services.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaModelError,
    OllamaTimeoutError
)
from app.models.gap_schema import RiskLevel
from app.services.language_detector import LanguageDetector

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["gap-analysis"],
    responses={
        404: {"description": "Document not found"},
        500: {"description": "Internal server error"},
        503: {"description": "LLM service unavailable"}
    }
)


# ============================================================================
# Request/Response Models
# ============================================================================

class GAPAnalysisRequest(BaseModel):
    """
    Request body for GAP analysis.
    
    Attributes:
        gap_prompt: User's GAP definition or analysis instructions
        top_k: Number of relevant chunks to retrieve (default: 10)
    """
    gap_prompt: str = Field(
        ...,
        description="GAP analysis prompt or definition. Describe what aspects to analyze.",
        min_length=10,
        max_length=2000,
        examples=[
            "Analyze grantor identification and authority to sell property",
            "Find missing attestation and witness signatures",
            "Check property boundaries and measurements"
        ]
    )
    
    top_k: Optional[int] = Field(
        default=10,
        description="Number of top relevant document chunks to retrieve",
        ge=1,
        le=50,
        examples=[10, 15, 20]
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "gap_prompt": "Analyze grantor identification, including name, parentage, and legal authority",
                "top_k": 10
            }
        }


class GAPItemResponse(BaseModel):
    """Single GAP item in the response."""
    gap_id: str = Field(..., description="Unique GAP identifier (e.g., GAP-001)")
    title: str = Field(..., description="Short descriptive title")
    clause_reference: str = Field(..., description="Reference to document clause(s)")
    description: str = Field(..., description="Detailed description of the issue")
    risk_level: RiskLevel = Field(..., description="Risk severity level")
    recommendation: str = Field(..., description="Suggested action or remedy")
    
    class Config:
        json_schema_extra = {
            "example": {
                "gap_id": "GAP-001",
                "title": "Missing grantor identification",
                "clause_reference": "Clause 1",
                "description": "The document does not specify the grantor's father's name",
                "risk_level": "High",
                "recommendation": "Verify grantor identity through government ID"
            }
        }


class GAPAnalysisSummary(BaseModel):
    """Summary statistics of GAP analysis."""
    total_gaps: int = Field(..., description="Total number of gaps found")
    high_risk: int = Field(..., description="Number of high-risk gaps")
    medium_risk: int = Field(..., description="Number of medium-risk gaps")
    low_risk: int = Field(..., description="Number of low-risk gaps")
    clauses_analyzed: int = Field(..., description="Number of clauses analyzed")
    analysis_time_ms: float = Field(..., description="Analysis time in milliseconds")
    language_detected: str = Field(..., description="Detected document language")


class GAPAnalysisResponse(BaseModel):
    """
    Response containing GAP analysis results.
    
    Attributes:
        document_id: Document UUID
        gaps: List of identified issues
        summary: Analysis statistics
        success: Whether analysis succeeded
        error: Error message if failed
    """
    document_id: str = Field(..., description="Document UUID")
    gaps: List[GAPItemResponse] = Field(..., description="List of identified gaps")
    summary: GAPAnalysisSummary = Field(..., description="Analysis summary")
    success: bool = Field(..., description="Whether analysis succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc-abc123",
                "gaps": [
                    {
                        "gap_id": "GAP-001",
                        "title": "Missing witness signatures",
                        "clause_reference": "Attestation section",
                        "description": "Document lacks required witness signatures",
                        "risk_level": "High",
                        "recommendation": "Obtain signatures from two witnesses"
                    }
                ],
                "summary": {
                    "total_gaps": 1,
                    "high_risk": 1,
                    "medium_risk": 0,
                    "low_risk": 0,
                    "clauses_analyzed": 10,
                    "analysis_time_ms": 5420.5,
                    "language_detected": "english"
                },
                "success": True,
                "error": None
            }
        }


# ============================================================================
# Dependency Injection
# ============================================================================

# Global service instances (initialized in main.py)
_gap_detection_engine: Optional[GAPDetectionEngine] = None


def set_gap_detection_engine(engine: GAPDetectionEngine):
    """Set the global GAP detection engine instance."""
    global _gap_detection_engine
    _gap_detection_engine = engine
    logger.info("GAP detection engine registered with router")


def get_gap_detection_engine() -> GAPDetectionEngine:
    """
    Dependency to get GAP detection engine.
    
    Returns:
        GAPDetectionEngine: Initialized engine
    
    Raises:
        HTTPException: If engine not initialized
    """
    if _gap_detection_engine is None:
        logger.error("GAP detection engine not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GAP analysis service is not available. Please try again later."
        )
    return _gap_detection_engine


# ============================================================================
# Routes
# ============================================================================

@router.post(
    "/{document_id}/gap-analysis",
    response_model=GAPAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze document for GAPs",
    description="""
    Perform GAP (Grantor, Attestation, Property) analysis on a legal document.
    
    This endpoint:
    1. Retrieves relevant document clauses using semantic search
    2. Analyzes clauses using local LLM (Ollama)
    3. Identifies missing or incomplete information
    4. Returns validated GAP list with risk levels
    
    **Language Detection:**
    - Automatically detects document language (English or Tamil)
    - Uses language-specific prompts
    
    **GAP Analysis:**
    - **G (Grantor)**: Seller/transferor identification and authority
    - **A (Attestation)**: Witness signatures, notarization, registration
    - **P (Property)**: Property description, boundaries, measurements
    
    **Risk Levels:**
    - **High**: Critical information missing, significant legal risk
    - **Medium**: Information incomplete, needs clarification
    - **Low**: Minor issues, low impact on document validity
    
    **Example Prompts:**
    - "Analyze grantor identification and legal authority"
    - "Check for missing witness signatures and attestation"
    - "Verify property boundaries and measurements"
    - "Find gaps in property transfer documentation"
    
    **Requirements:**
    - Document must be uploaded and processed first
    - Ollama must be running locally (default: http://localhost:11434)
    - At least one relevant clause must be found
    """,
    responses={
        200: {
            "description": "GAP analysis completed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "document_id": "doc-abc123",
                        "gaps": [
                            {
                                "gap_id": "GAP-001",
                                "title": "Missing grantor father's name",
                                "clause_reference": "Clause 1",
                                "description": "Grantor's father's name not specified",
                                "risk_level": "High",
                                "recommendation": "Verify identity through ID documents"
                            }
                        ],
                        "summary": {
                            "total_gaps": 1,
                            "high_risk": 1,
                            "medium_risk": 0,
                            "low_risk": 0,
                            "clauses_analyzed": 10,
                            "analysis_time_ms": 5420.5,
                            "language_detected": "english"
                        },
                        "success": True,
                        "error": None
                    }
                }
            }
        },
        404: {
            "description": "Document not found",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Document with ID 'doc-xyz' not found"
                    }
                }
            }
        },
        503: {
            "description": "LLM service unavailable",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Cannot connect to Ollama. Is it running at http://localhost:11434?"
                    }
                }
            }
        }
    }
)
async def analyze_document_gaps(
    document_id: str = Path(
        ...,
        description="Document UUID to analyze",
        examples=["doc-abc123", "550e8400-e29b-41d4-a716-446655440000"]
    ),
    request: GAPAnalysisRequest = Body(...),
    engine: GAPDetectionEngine = Depends(get_gap_detection_engine)
) -> GAPAnalysisResponse:
    """
    Analyze document for GAPs using LLM.
    
    This endpoint performs GAP (Grantor, Attestation, Property) analysis
    to identify missing or incomplete information in legal documents.
    
    Args:
        document_id: Document UUID
        request: Analysis request with gap_prompt and top_k
        engine: GAP detection engine (injected)
    
    Returns:
        GAPAnalysisResponse: Analysis results with gaps and summary
    
    Raises:
        HTTPException: On errors (404, 500, 503)
    """
    logger.info(
        f"[API] GAP analysis request for document '{document_id}' "
        f"with prompt: '{request.gap_prompt[:100]}...'"
    )
    
    try:
        # Step 1: Detect language from GAP prompt
        # This determines which prompt template to use (English or Tamil)
        language = _detect_language(request.gap_prompt)
        
        logger.info(
            f"[API] Detected language: {language.value} for document '{document_id}'"
        )
        
        # Step 2: Call GAP detection engine
        # ⚠️ CRITICAL: NO LLM logic in route layer
        # All LLM calls, prompt building, and response parsing happen in service layer
        # This route ONLY handles HTTP concerns (validation, errors, response formatting)
        result: GAPDetectionResult = engine.detect_gaps(
            document_id=document_id,
            gap_prompt=request.gap_prompt,
            language=language,
            top_k=request.top_k or 10
        )
        
        # ✓ Service layer guarantees:
        # 1. Schema validation (Pydantic models)
        # 2. JSON retry on malformed output
        # 3. Clause reference verification
        # 4. Language consistency enforcement
        
        # Step 3: Build response
        if not result.validation_passed:
            logger.error(
                f"[API] GAP analysis failed for document '{document_id}': "
                f"{result.error}"
            )
            
            # Return error response with empty gaps
            return GAPAnalysisResponse(
                document_id=document_id,
                gaps=[],
                summary=GAPAnalysisSummary(
                    total_gaps=0,
                    high_risk=0,
                    medium_risk=0,
                    low_risk=0,
                    clauses_analyzed=result.clauses_analyzed,
                    analysis_time_ms=result.analysis_time_ms,
                    language_detected=result.language
                ),
                success=False,
                error=result.error
            )
        
        # Success - convert to response format
        gap_responses = [
            GAPItemResponse(
                gap_id=gap.gap_id,
                title=gap.title,
                clause_reference=gap.clause_reference,
                description=gap.description,
                risk_level=gap.risk_level,
                recommendation=gap.recommendation
            )
            for gap in result.gaps
        ]
        
        summary = GAPAnalysisSummary(
            total_gaps=len(result.gaps),
            high_risk=len([g for g in result.gaps if g.risk_level == RiskLevel.HIGH]),
            medium_risk=len([g for g in result.gaps if g.risk_level == RiskLevel.MEDIUM]),
            low_risk=len([g for g in result.gaps if g.risk_level == RiskLevel.LOW]),
            clauses_analyzed=result.clauses_analyzed,
            analysis_time_ms=result.analysis_time_ms,
            language_detected=result.language
        )
        
        logger.info(
            f"[API] ✓ GAP analysis completed for document '{document_id}': "
            f"{len(result.gaps)} gaps found in {result.analysis_time_ms:.0f}ms"
        )
        
        return GAPAnalysisResponse(
            document_id=document_id,
            gaps=gap_responses,
            summary=summary,
            success=True,
            error=None
        )
    
    except ContextLengthError as e:
        logger.error(f"[API] Context too large for document '{document_id}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Document context exceeds token limits: {str(e)}"
        )
    
    except OllamaConnectionError as e:
        logger.error(f"[API] Cannot connect to Ollama: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cannot connect to Ollama. Is it running at http://localhost:11434? Error: {str(e)}"
        )
    
    except OllamaModelError as e:
        logger.error(f"[API] Ollama model error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM model error: {str(e)}"
        )
    
    except OllamaTimeoutError as e:
        logger.error(f"[API] Ollama request timeout: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"LLM request timed out: {str(e)}"
        )
    
    except GAPDetectionError as e:
        logger.error(f"[API] GAP detection error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"GAP detection failed: {str(e)}"
        )
    
    except ValueError as e:
        logger.error(f"[API] Invalid input: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}"
        )
    
    except Exception as e:
        logger.exception(f"[API] Unexpected error during GAP analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during GAP analysis: {str(e)}"
        )


# ============================================================================
# Helper Functions
# ============================================================================

def _detect_language(text: str) -> Language:
    """
    Detect language from text.
    
    Uses LanguageDetector to identify Tamil vs English.
    Defaults to English if detection fails.
    
    Args:
        text: Text to analyze
    
    Returns:
        Language: Detected language enum
    """
    try:
        detector = LanguageDetector()
        detected_lang = detector.detect(text)
        
        if detected_lang == "tamil":
            return Language.TAMIL
        else:
            return Language.ENGLISH
    
    except Exception as e:
        logger.warning(f"Language detection failed, defaulting to English: {str(e)}")
        return Language.ENGLISH


# ============================================================================
# Health Check
# ============================================================================

@router.get(
    "/gap-analysis/health",
    summary="Check GAP analysis service health",
    description="Check if GAP analysis service and Ollama LLM are available",
    tags=["gap-analysis", "health"]
)
async def gap_analysis_health(
    engine: GAPDetectionEngine = Depends(get_gap_detection_engine)
) -> dict:
    """
    Check GAP analysis service health.
    
    Returns:
        dict: Service health status
    """
    try:
        # Check Ollama connectivity
        health = engine.ollama_client.health_check()
        
        return {
            "service": "gap-analysis",
            "status": "healthy" if health["available"] else "degraded",
            "ollama": health,
            "components": {
                "context_builder": "ok",
                "prompt_factory": "ok",
                "gap_detection_engine": "ok"
            }
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "service": "gap-analysis",
            "status": "unhealthy",
            "error": str(e)
        }
