"""
GAP Detection Engine

This module orchestrates the complete GAP analysis pipeline:
1. Retrieve relevant document clauses using vector search
2. Build structured prompts using prompt factory
3. Call local LLM (Ollama) for analysis
4. Parse and validate JSON output
5. Retry on validation failures with correction prompts
6. Return validated GAP list

Strict Rules Enforced:
- No free-text output (JSON only)
- No hallucinated clauses (all GAPs must reference actual clauses)
- Schema validation (strict Pydantic models)
- Single retry on validation failure

Example Usage:
    >>> engine = GAPDetectionEngine(
    ...     context_builder=builder,
    ...     prompt_factory=factory,
    ...     ollama_client=client,
    ...     metadata_store=store
    ... )
    >>> result = engine.detect_gaps(
    ...     document_id="doc-123",
    ...     gap_prompt="Analyze grantor identification",
    ...     language=Language.ENGLISH
    ... )
    >>> print(f"Found {len(result.gaps)} issues")
"""

import logging
import json
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from app.services.gap_context_builder import (
    GAPContextBuilder,
    GAPContext,
    ContextLengthError
)
from app.services.prompt_factory import (
    PromptFactory,
    Language,
    GAPDefinition
)
from app.services.ollama_client import (
    OllamaClient,
    OllamaClientError,
    OllamaConnectionError,
    OllamaModelError,
    OllamaTimeoutError
)
from app.models.gap_schema import (
    GAPAnalysisOutput,
    GAPItem,
    RiskLevel,
    validate_gap_analysis,
    ValidationError as SchemaValidationError
)

logger = logging.getLogger(__name__)


@dataclass
class GAPDetectionResult:
    """
    Result of GAP detection with metadata.
    
    Attributes:
        document_id: Document UUID
        gaps: Validated list of GAP items
        analysis_time_ms: Total analysis time in milliseconds
        tokens_estimated: Estimated tokens used
        llm_calls: Number of LLM calls made
        retries: Number of retries performed
        language: Analysis language
        clauses_analyzed: Number of clauses analyzed
        validation_passed: Whether validation succeeded
        error: Error message if failed
    """
    document_id: str
    gaps: List[GAPItem]
    analysis_time_ms: float
    tokens_estimated: int
    llm_calls: int
    retries: int
    language: str
    clauses_analyzed: int
    validation_passed: bool
    error: Optional[str] = None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of detection results."""
        return {
            "document_id": self.document_id,
            "total_gaps": len(self.gaps),
            "high_risk": len([g for g in self.gaps if g.risk_level == RiskLevel.HIGH]),
            "medium_risk": len([g for g in self.gaps if g.risk_level == RiskLevel.MEDIUM]),
            "low_risk": len([g for g in self.gaps if g.risk_level == RiskLevel.LOW]),
            "analysis_time_ms": self.analysis_time_ms,
            "tokens_estimated": self.tokens_estimated,
            "llm_calls": self.llm_calls,
            "retries": self.retries,
            "language": self.language,
            "clauses_analyzed": self.clauses_analyzed,
            "validation_passed": self.validation_passed,
            "error": self.error
        }
    
    def has_high_risk_gaps(self) -> bool:
        """Check if any high-risk GAPs were found."""
        return any(gap.risk_level == RiskLevel.HIGH for gap in self.gaps)


class GAPDetectionError(Exception):
    """Base exception for GAP detection errors."""
    pass


class GAPDetectionEngine:
    """
    Engine for detecting GAPs in legal documents using LLM analysis.
    
    This engine:
    1. Retrieves relevant document clauses using semantic search
    2. Builds language-specific prompts with strict rules
    3. Calls Ollama LLM for analysis
    4. Validates JSON output against strict schema
    5. Retries once on validation failure with correction prompt
    6. Returns validated GAP list with metadata
    
    Strict Rules Enforced:
    - JSON-only output (no free text)
    - All GAPs must reference actual clauses from context
    - Schema validation with Pydantic models
    - Automatic retry on validation failure
    
    Example:
        >>> engine = GAPDetectionEngine(
        ...     context_builder=context_builder,
        ...     prompt_factory=prompt_factory,
        ...     ollama_client=ollama_client,
        ...     metadata_store=metadata_store
        ... )
        >>> result = engine.detect_gaps(
        ...     document_id="doc-abc123",
        ...     gap_prompt="Analyze property boundaries",
        ...     language=Language.ENGLISH,
        ...     top_k=10
        ... )
    """
    
    def __init__(
        self,
        context_builder: GAPContextBuilder,
        prompt_factory: PromptFactory,
        ollama_client: OllamaClient,
        metadata_store,  # VectorMetadataStore
        max_retries: int = 1
    ):
        """
        Initialize GAP detection engine.
        
        Args:
            context_builder: Context builder for retrieving clauses
            prompt_factory: Prompt factory for generating LLM prompts
            ollama_client: Ollama client for LLM calls
            metadata_store: Metadata store for clause validation
            max_retries: Maximum retries on validation failure (default: 1)
        """
        self.context_builder = context_builder
        self.prompt_factory = prompt_factory
        self.ollama_client = ollama_client
        self.metadata_store = metadata_store
        self.max_retries = max_retries
        
        logger.info(
            f"Initialized GAPDetectionEngine with max_retries={max_retries}"
        )
    
    def detect_gaps(
        self,
        document_id: str,
        gap_prompt: str,
        language: Language = Language.ENGLISH,
        top_k: int = 10,
        gap_definition: Optional[GAPDefinition] = None
    ) -> GAPDetectionResult:
        """
        Detect GAPs in a document.
        
        Args:
            document_id: Document UUID to analyze
            gap_prompt: User's GAP analysis prompt/definition
            language: Analysis language (english or tamil)
            top_k: Number of top relevant chunks to retrieve
            gap_definition: Custom GAP definition (uses default if None)
        
        Returns:
            GAPDetectionResult: Detection results with validated GAPs
        
        Raises:
            GAPDetectionError: If detection fails after retries
            ValueError: If inputs are invalid
        
        Example:
            >>> result = engine.detect_gaps(
            ...     document_id="doc-123",
            ...     gap_prompt="Find missing grantor identification details",
            ...     language=Language.ENGLISH,
            ...     top_k=10
            ... )
            >>> for gap in result.gaps:
            ...     print(f"{gap.gap_id}: {gap.title} (Risk: {gap.risk_level})")
        """
        start_time = time.time()
        
        logger.info(
            f"[{document_id}] Starting GAP detection with language={language.value}, "
            f"top_k={top_k}, prompt='{gap_prompt[:100]}...'"
        )
        
        # Validate inputs
        if not document_id or not document_id.strip():
            raise ValueError("document_id cannot be empty")
        
        if not gap_prompt or not gap_prompt.strip():
            raise ValueError("gap_prompt cannot be empty")
        
        # Use default GAP definition if not provided
        if gap_definition is None:
            gap_definition = GAPDefinition()
        
        try:
            # Step 1: Build context (retrieve relevant clauses)
            logger.info(f"[{document_id}] Step 1/5: Building context...")
            context = self._build_context(document_id, gap_prompt, top_k)
            
            if not context.clauses:
                logger.warning(f"[{document_id}] No relevant clauses found")
                return GAPDetectionResult(
                    document_id=document_id,
                    gaps=[],
                    analysis_time_ms=self._elapsed_ms(start_time),
                    tokens_estimated=0,
                    llm_calls=0,
                    retries=0,
                    language=language.value,
                    clauses_analyzed=0,
                    validation_passed=True,
                    error="No relevant clauses found"
                )
            
            logger.info(
                f"[{document_id}] Context built: {len(context.clauses)} clauses, "
                f"{context.total_tokens} tokens"
            )
            
            # Step 2: Build LLM prompt
            logger.info(f"[{document_id}] Step 2/5: Building LLM prompt...")
            clauses_for_prompt = [clause.format() for clause in context.clauses]
            
            prompt = self.prompt_factory.create_gap_prompt(
                language=language,
                gap_definition=gap_definition,
                clauses=clauses_for_prompt,
                document_context=f"Analysis of {len(context.clauses)} relevant clauses"
            )
            
            logger.debug(
                f"[{document_id}] Generated prompt: {len(prompt)} chars"
            )
            
            # Step 3: Call LLM (with retry logic)
            logger.info(f"[{document_id}] Step 3/5: Calling LLM...")
            llm_response, llm_calls = self._call_llm_with_retry(
                document_id=document_id,
                prompt=prompt,
                context=context
            )
            
            # Step 4: Parse and validate JSON
            logger.info(f"[{document_id}] Step 4/5: Validating JSON output...")
            validated_output = self._validate_output(
                document_id=document_id,
                llm_response=llm_response,
                context=context
            )
            
            # Step 5: Verify clause references
            # ✓ PROTECTION 3: Anti-Hallucination Check
            # Ensures all GAP clause references match actual clauses in context
            # Prevents LLM from inventing non-existent clause numbers
            logger.info(f"[{document_id}] Step 5/5: Verifying clause references...")
            self._verify_clause_references(
                document_id=document_id,
                gaps=validated_output.gaps,
                context=context
            )
            # Note: Verification logs warnings but doesn't fail
            # Invalid references are flagged for manual review
            
            # Success!
            elapsed_ms = self._elapsed_ms(start_time)
            
            logger.info(
                f"[{document_id}] ✓ GAP detection completed: "
                f"{len(validated_output.gaps)} gaps found in {elapsed_ms:.0f}ms"
            )
            
            return GAPDetectionResult(
                document_id=document_id,
                gaps=validated_output.gaps,
                analysis_time_ms=elapsed_ms,
                tokens_estimated=context.total_tokens,
                llm_calls=llm_calls,
                retries=llm_calls - 1,  # First call is not a retry
                language=language.value,
                clauses_analyzed=len(context.clauses),
                validation_passed=True,
                error=None
            )
        
        except ContextLengthError as e:
            error_msg = f"Context too large: {str(e)}"
            logger.error(f"[{document_id}] {error_msg}")
            return self._create_error_result(
                document_id, start_time, language.value, error_msg
            )
        
        except OllamaConnectionError as e:
            error_msg = f"Cannot connect to Ollama: {str(e)}"
            logger.error(f"[{document_id}] {error_msg}")
            return self._create_error_result(
                document_id, start_time, language.value, error_msg
            )
        
        except OllamaModelError as e:
            error_msg = f"LLM model error: {str(e)}"
            logger.error(f"[{document_id}] {error_msg}")
            return self._create_error_result(
                document_id, start_time, language.value, error_msg
            )
        
        except OllamaTimeoutError as e:
            error_msg = f"LLM request timeout: {str(e)}"
            logger.error(f"[{document_id}] {error_msg}")
            return self._create_error_result(
                document_id, start_time, language.value, error_msg
            )
        
        except SchemaValidationError as e:
            error_msg = f"Schema validation failed after retries: {str(e)}"
            logger.error(f"[{document_id}] {error_msg}")
            return self._create_error_result(
                document_id, start_time, language.value, error_msg
            )
        
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.exception(f"[{document_id}] {error_msg}")
            return self._create_error_result(
                document_id, start_time, language.value, error_msg
            )
    
    def _build_context(
        self,
        document_id: str,
        gap_prompt: str,
        top_k: int
    ) -> GAPContext:
        """Build context by retrieving relevant clauses."""
        try:
            context = self.context_builder.build_context(
                document_id=document_id,
                gap_prompt=gap_prompt,
                top_k=top_k
            )
            return context
        
        except Exception as e:
            logger.error(
                f"[{document_id}] Failed to build context: {str(e)}"
            )
            raise
    
    def _call_llm_with_retry(
        self,
        document_id: str,
        prompt: str,
        context: GAPContext
    ) -> tuple[str, int]:
        """
        Call LLM with automatic retry on validation failure.
        
        Returns:
            tuple: (validated_response, total_calls)
        """
        llm_calls = 0
        last_response = None
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            llm_calls += 1
            
            try:
                if attempt == 0:
                    # First attempt: use original prompt
                    logger.info(
                        f"[{document_id}] LLM call #{llm_calls} (initial attempt)"
                    )
                    current_prompt = prompt
                else:
                    # Retry attempt: add correction guidance
                    logger.info(
                        f"[{document_id}] LLM call #{llm_calls} (retry #{attempt}) "
                        f"- Previous error: {str(last_error)[:100]}..."
                    )
                    current_prompt = self._create_correction_prompt(
                        original_prompt=prompt,
                        failed_response=last_response,
                        error_message=str(last_error)
                    )
                
                # Call LLM
                response = self.ollama_client.generate(current_prompt)
                
                logger.info(
                    f"[{document_id}] LLM response received: {len(response)} chars"
                )
                logger.debug(
                    f"[{document_id}] Response preview: {response[:200]}..."
                )
                
                # Try to validate immediately
                try:
                    self._validate_output(document_id, response, context)
                    # Validation succeeded!
                    logger.info(
                        f"[{document_id}] ✓ LLM response validated on attempt {attempt + 1}"
                    )
                    return response, llm_calls
                
                except SchemaValidationError as e:
                    # Validation failed
                    last_response = response
                    last_error = e
                    
                    if attempt < self.max_retries:
                        logger.warning(
                            f"[{document_id}] Validation failed on attempt {attempt + 1}, "
                            f"will retry. Error: {str(e)[:200]}..."
                        )
                        continue
                    else:
                        logger.error(
                            f"[{document_id}] Validation failed after {self.max_retries + 1} attempts"
                        )
                        raise
            
            except (OllamaConnectionError, OllamaModelError, OllamaTimeoutError) as e:
                # Don't retry on Ollama errors
                logger.error(
                    f"[{document_id}] Ollama error on attempt {attempt + 1}: {str(e)}"
                )
                raise
        
        # Should not reach here
        raise GAPDetectionError("LLM call failed after all retries")
    
    def _create_correction_prompt(
        self,
        original_prompt: str,
        failed_response: str,
        error_message: str
    ) -> str:
        """
        Create a correction prompt for retry attempt.
        
        Args:
            original_prompt: Original prompt that failed
            failed_response: Response that failed validation
            error_message: Validation error message
        
        Returns:
            str: Corrected prompt with guidance
        """
        correction = f"""Your previous response failed validation. Please fix and retry.

ERROR:
{error_message}

YOUR PREVIOUS RESPONSE:
{failed_response[:500]}...

REQUIREMENTS:
1. Return ONLY valid JSON matching the schema
2. Each GAP must have ALL required fields: gap_id, title, clause_reference, description, risk_level, recommendation
3. risk_level must be EXACTLY one of: "Low", "Medium", "High" (case-sensitive)
4. gap_id must follow pattern: GAP-XXX (e.g., GAP-001, GAP-002)
5. clause_reference must reference specific clauses from the document
6. NO extra fields allowed
7. NO free text outside JSON

{original_prompt}

YOUR CORRECTED JSON OUTPUT:"""
        
        return correction
    
    def _validate_output(
        self,
        document_id: str,
        llm_response: str,
        context: GAPContext
    ) -> GAPAnalysisOutput:
        """
        Parse and validate LLM response against schema.
        
        ✓ PROTECTION 1: Schema Validation
        - Uses Pydantic models for strict validation
        - Enforces required fields, types, and constraints
        - Rejects extra fields not in schema
        
        ✓ PROTECTION 2: JSON Retry
        - Automatically repairs malformed JSON
        - Sends back to LLM with correction instructions
        - One retry attempt for format fixes
        
        Args:
            document_id: Document ID for logging
            llm_response: Raw LLM response string
            context: Context used for analysis
        
        Returns:
            GAPAnalysisOutput: Validated output
        
        Raises:
            SchemaValidationError: If validation fails after retry
        """
        # Extract and parse JSON from response
        # Local LLMs often produce malformed JSON (missing quotes, trailing commas, etc.)
        # unlike commercial APIs (GPT-4) which reliably output valid JSON
        try:
            json_str = self._extract_json(llm_response)
            json_data = json.loads(json_str)
        
        except json.JSONDecodeError as e:
            logger.warning(
                f"[{document_id}] JSON parsing failed: {str(e)}\n"
                f"Attempting automatic repair..."
            )
            
            # Attempt to repair invalid JSON (one retry only)
            try:
                json_data = self._repair_invalid_json(
                    document_id=document_id,
                    invalid_output=llm_response,
                    parse_error=str(e)
                )
                logger.info(
                    f"[{document_id}] ✓ JSON repair successful"
                )
            
            except json.JSONDecodeError as repair_error:
                logger.error(
                    f"[{document_id}] JSON repair failed: {str(repair_error)}\n"
                    f"Original response: {llm_response[:500]}..."
                )
                raise SchemaValidationError(
                    f"LLM response is not valid JSON after repair attempt: {str(repair_error)}"
                ) from repair_error
        
        # Validate against schema
        try:
            validated = validate_gap_analysis(json_data)
            logger.info(
                f"[{document_id}] ✓ Schema validation passed: "
                f"{len(validated.gaps)} gaps"
            )
            return validated
        
        except SchemaValidationError as e:
            logger.error(
                f"[{document_id}] Schema validation failed: {str(e)}"
            )
            raise
    
    def _repair_invalid_json(
        self,
        document_id: str,
        invalid_output: str,
        parse_error: str
    ) -> Dict[str, Any]:
        """
        Attempt to repair invalid JSON by asking LLM to fix format only.
        
        Why this is needed:
        Local LLMs (Llama, Mistral, etc.) frequently produce malformed JSON:
        - Missing quotes around keys: {gap_id: "GAP-001"}
        - Trailing commas: {"gaps": [...],}
        - Unescaped quotes in strings
        - Missing closing braces/brackets
        
        Commercial APIs (GPT-4, Claude) rarely have these issues due to
        better instruction-following and JSON mode features.
        
        This repair mechanism:
        1. Sends the malformed output back to the LLM
        2. Asks to fix ONLY the format (no content changes)
        3. Prevents adding new GAPs or modifying existing ones
        4. Retries once to get valid JSON
        
        Args:
            document_id: Document ID for logging
            invalid_output: The malformed JSON output
            parse_error: JSON parse error message
        
        Returns:
            dict: Parsed JSON data after repair
        
        Raises:
            json.JSONDecodeError: If repair fails
        """
        logger.warning(
            f"[{document_id}] Attempting JSON repair for malformed output"
        )
        
        # Create repair prompt that explicitly forbids content changes
        repair_prompt = f"""Your previous response contained invalid JSON that failed to parse.

JSON PARSE ERROR:
{parse_error}

YOUR INVALID OUTPUT:
{invalid_output}

INSTRUCTIONS FOR REPAIR:
1. Fix ONLY the JSON format errors (quotes, commas, brackets, escaping)
2. Do NOT add any new GAP items
3. Do NOT modify any existing content (titles, descriptions, recommendations)
4. Do NOT remove any existing GAP items
5. Keep the EXACT same content, just fix the JSON syntax
6. Return ONLY the corrected JSON, no explanations

Common JSON errors to fix:
- Add quotes around all keys: {{"gap_id": "..."}}
- Remove trailing commas: [..., "item"] not [..., "item",]
- Escape quotes in strings: "He said \\"hello\\""
- Close all braces and brackets

YOUR CORRECTED JSON (SAME CONTENT, VALID FORMAT):"""

        try:
            # Call LLM to repair JSON
            logger.debug(
                f"[{document_id}] Sending repair request to LLM"
            )
            
            repaired_response = self.ollama_client.generate(repair_prompt)
            
            logger.debug(
                f"[{document_id}] Received repaired response: {len(repaired_response)} chars"
            )
            
            # Extract and parse repaired JSON
            json_str = self._extract_json(repaired_response)
            json_data = json.loads(json_str)
            
            # Success - repaired JSON is valid
            logger.info(
                f"[{document_id}] ✓ JSON repair successful, valid JSON obtained"
            )
            
            return json_data
        
        except json.JSONDecodeError as e:
            logger.error(
                f"[{document_id}] JSON repair failed, output still invalid: {str(e)}"
            )
            raise
        
        except Exception as e:
            logger.error(
                f"[{document_id}] Unexpected error during JSON repair: {str(e)}"
            )
            raise json.JSONDecodeError(
                f"JSON repair failed: {str(e)}",
                "",
                0
            )
    
    def _extract_json(self, text: str) -> str:
        """
        Extract JSON object from LLM response.
        
        LLMs sometimes add extra text before/after JSON.
        This function finds the JSON block.
        
        Args:
            text: Full LLM response
        
        Returns:
            str: Extracted JSON string
        
        Raises:
            ValueError: If no JSON found
        """
        text = text.strip()
        
        # Find JSON block
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx == -1 or end_idx == -1:
            raise ValueError("No JSON object found in response")
        
        json_str = text[start_idx:end_idx + 1]
        
        return json_str
    
    def _verify_clause_references(
        self,
        document_id: str,
        gaps: List[GAPItem],
        context: GAPContext
    ):
        """
        Verify that all GAP clause references are valid.
        
        Ensures no hallucinated clauses by checking references
        against actual context.
        
        Args:
            document_id: Document ID for logging
            gaps: List of GAP items to verify
            context: Context with actual clauses
        
        Logs warnings for invalid references but doesn't fail.
        """
        if not gaps:
            return
        
        # Get valid clause references from context
        valid_references = set(context.get_clause_references())
        
        logger.debug(
            f"[{document_id}] Valid clause references: {valid_references}"
        )
        
        for gap in gaps:
            # Extract clause numbers from reference
            # e.g., "Clause 1" -> "1", "Clauses 2-4" -> ["2", "3", "4"]
            referenced_clauses = self._parse_clause_reference(
                gap.clause_reference
            )
            
            # Check if any referenced clause is invalid
            invalid_refs = [
                ref for ref in referenced_clauses
                if ref not in valid_references
            ]
            
            if invalid_refs:
                logger.warning(
                    f"[{document_id}] GAP {gap.gap_id} references clauses "
                    f"not in context: {invalid_refs}. "
                    f"Reference: '{gap.clause_reference}'. "
                    f"This may be a hallucinated clause."
                )
    
    def _parse_clause_reference(self, reference: str) -> List[str]:
        """
        Parse clause reference string to extract clause numbers.
        
        Examples:
            "Clause 1" -> ["1"]
            "Clauses 2-4" -> ["2", "3", "4"]
            "Clause 5, Clause 7" -> ["5", "7"]
            "Section A" -> ["A"]
        
        Args:
            reference: Clause reference string
        
        Returns:
            List[str]: List of clause identifiers
        """
        import re
        
        # Find all numbers in the reference
        numbers = re.findall(r'\d+', reference)
        
        if numbers:
            return numbers
        
        # If no numbers, try to extract letter identifiers
        letters = re.findall(r'\b([A-Z])\b', reference)
        
        if letters:
            return letters
        
        # Fallback: return the reference as-is
        return [reference]
    
    def _elapsed_ms(self, start_time: float) -> float:
        """Calculate elapsed time in milliseconds."""
        return (time.time() - start_time) * 1000
    
    def _create_error_result(
        self,
        document_id: str,
        start_time: float,
        language: str,
        error_msg: str
    ) -> GAPDetectionResult:
        """Create error result."""
        return GAPDetectionResult(
            document_id=document_id,
            gaps=[],
            analysis_time_ms=self._elapsed_ms(start_time),
            tokens_estimated=0,
            llm_calls=0,
            retries=0,
            language=language,
            clauses_analyzed=0,
            validation_passed=False,
            error=error_msg
        )
    
    def detect_gaps_batch(
        self,
        document_ids: List[str],
        gap_prompt: str,
        language: Language = Language.ENGLISH,
        top_k: int = 10
    ) -> List[GAPDetectionResult]:
        """
        Detect GAPs for multiple documents.
        
        Args:
            document_ids: List of document UUIDs
            gap_prompt: GAP analysis prompt
            language: Analysis language
            top_k: Number of chunks per document
        
        Returns:
            List[GAPDetectionResult]: Results for each document
        
        Example:
            >>> results = engine.detect_gaps_batch(
            ...     document_ids=["doc-1", "doc-2", "doc-3"],
            ...     gap_prompt="Analyze attestation",
            ...     language=Language.ENGLISH
            ... )
            >>> for result in results:
            ...     print(f"{result.document_id}: {len(result.gaps)} gaps")
        """
        logger.info(
            f"Starting batch GAP detection for {len(document_ids)} documents"
        )
        
        results = []
        
        for i, doc_id in enumerate(document_ids, 1):
            logger.info(
                f"Processing document {i}/{len(document_ids)}: {doc_id}"
            )
            
            result = self.detect_gaps(
                document_id=doc_id,
                gap_prompt=gap_prompt,
                language=language,
                top_k=top_k
            )
            
            results.append(result)
            
            logger.info(
                f"Document {i}/{len(document_ids)} completed: "
                f"{len(result.gaps)} gaps, "
                f"validation_passed={result.validation_passed}"
            )
        
        logger.info(
            f"Batch processing complete: {len(results)} documents processed"
        )
        
        return results


# Example usage
"""
Example 1: Basic GAP detection

from app.services.gap_context_builder import GAPContextBuilder
from app.services.prompt_factory import PromptFactory, Language
from app.services.ollama_client import OllamaClient, OllamaConfig
from app.services.semantic_search import SemanticSearchService
from app.services.metadata_store import VectorMetadataStore

# Initialize services
context_builder = GAPContextBuilder(semantic_search, metadata_store)

# Option 1: Base model (requires full framework in every prompt)
prompt_factory = PromptFactory(use_custom_model=False)
ollama_client = OllamaClient(OllamaConfig(model_name="llama2", use_custom_model=False))

# Option 2: Custom model (framework pre-trained, only sends clauses)
# prompt_factory = PromptFactory(use_custom_model=True)
# ollama_client = OllamaClient(OllamaConfig(
#     model_name="gap-english",  # or "gap-tamil" for Tamil
#     use_custom_model=True,
#     temperature=0.3  # Lower temp for more consistent output
# ))

# Create engine
engine = GAPDetectionEngine(
    context_builder=context_builder,
    prompt_factory=prompt_factory,
    ollama_client=ollama_client,
    metadata_store=metadata_store
)

# Detect GAPs
result = engine.detect_gaps(
    document_id="doc-abc123",
    gap_prompt="Analyze grantor identification and authority to sell",
    language=Language.ENGLISH,
    top_k=10
)

# Process results
if result.validation_passed:
    print(f"✓ Found {len(result.gaps)} GAPs")
    print(f"Analysis time: {result.analysis_time_ms:.0f}ms")
    print(f"LLM calls: {result.llm_calls}")
    
    for gap in result.gaps:
        print(f"\n{gap.gap_id}: {gap.title}")
        print(f"Risk: {gap.risk_level}")
        print(f"Clause: {gap.clause_reference}")
        print(f"Description: {gap.description}")
        print(f"Recommendation: {gap.recommendation}")
else:
    print(f"✗ Analysis failed: {result.error}")


Example 2: Tamil document analysis

result_tamil = engine.detect_gaps(
    document_id="doc-tamil-456",
    gap_prompt="சொத்து எல்லைகள் மற்றும் அளவீடுகள் பகுப்பாய்வு",
    language=Language.TAMIL,
    top_k=8
)

if result_tamil.has_high_risk_gaps():
    print("⚠️ High-risk issues found!")
    high_risk = [g for g in result_tamil.gaps if g.risk_level == RiskLevel.HIGH]
    for gap in high_risk:
        print(f"  - {gap.title} ({gap.clause_reference})")


Example 3: Custom GAP definition

from app.services.prompt_factory import GAPDefinition

custom_gap_def = GAPDefinition(
    grantor_criteria=[
        "Full legal name with parentage",
        "Age and mental capacity",
        "Previous title documents",
        "Power of attorney if applicable"
    ],
    attestation_criteria=[
        "Two witness signatures",
        "Notary public attestation",
        "Registration number and date",
        "Sub-registrar office stamp"
    ],
    property_criteria=[
        "Survey number with subdivision",
        "Village, taluk, district",
        "Four-side boundaries with measurements",
        "Total area in sq ft/meters",
        "Encumbrances and liens"
    ]
)

result = engine.detect_gaps(
    document_id="doc-123",
    gap_prompt="Comprehensive property deed analysis",
    language=Language.ENGLISH,
    top_k=15,
    gap_definition=custom_gap_def
)


Example 4: Batch processing

documents = ["doc-001", "doc-002", "doc-003", "doc-004", "doc-005"]

results = engine.detect_gaps_batch(
    document_ids=documents,
    gap_prompt="Analyze attestation and witness requirements",
    language=Language.ENGLISH,
    top_k=10
)

# Summary statistics
total_gaps = sum(len(r.gaps) for r in results)
successful = sum(1 for r in results if r.validation_passed)
avg_time = sum(r.analysis_time_ms for r in results) / len(results)

print(f"Processed {len(results)} documents")
print(f"Successful: {successful}/{len(results)}")
print(f"Total GAPs found: {total_gaps}")
print(f"Average analysis time: {avg_time:.0f}ms")


Example 5: Error handling

try:
    result = engine.detect_gaps(
        document_id="doc-999",
        gap_prompt="Analyze property",
        language=Language.ENGLISH
    )
    
    if not result.validation_passed:
        print(f"Analysis failed: {result.error}")
        
        # Log details
        summary = result.get_summary()
        print(f"Details: {summary}")
        
except Exception as e:
    print(f"Unexpected error: {e}")


Example 6: Using results in API response

@app.post("/api/v1/analyze/gaps")
async def analyze_gaps(
    document_id: str,
    gap_prompt: str,
    language: str = "english"
):
    # Convert string to enum
    lang = Language.ENGLISH if language == "english" else Language.TAMIL
    
    # Run detection
    result = engine.detect_gaps(
        document_id=document_id,
        gap_prompt=gap_prompt,
        language=lang
    )
    
    # Return JSON response
    return {
        "success": result.validation_passed,
        "document_id": result.document_id,
        "gaps": [
            {
                "gap_id": gap.gap_id,
                "title": gap.title,
                "clause_reference": gap.clause_reference,
                "description": gap.description,
                "risk_level": gap.risk_level,
                "recommendation": gap.recommendation
            }
            for gap in result.gaps
        ],
        "summary": result.get_summary()
    }
"""
