"""
Strict JSON Schema for GAP Analysis Output

This module defines Pydantic models for validating GAP analysis responses from LLMs.
The schema enforces strict structure and rejects any extra fields not defined.

GAP Analysis Structure:
- List of GAP items (identified issues or concerns)
- Each GAP has: ID, title, clause reference, description, risk level, recommendation
- Risk levels: Low, Medium, High only
- No extra fields allowed (strict validation)

Example Valid JSON:
{
  "gaps": [
    {
      "gap_id": "GAP-001",
      "title": "Missing grantor identification",
      "clause_reference": "Clause 1",
      "description": "The document does not specify the grantor's father's name",
      "risk_level": "High",
      "recommendation": "Verify grantor identity through additional documents"
    }
  ]
}

Example Invalid JSON (will be rejected):
{
  "gaps": [
    {
      "gap_id": "GAP-001",
      "title": "Missing info",
      "risk_level": "Critical",  # ❌ Invalid: must be Low/Medium/High
      "extra_field": "value"     # ❌ Invalid: extra field not allowed
    }
  ]
}
"""

import logging
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict
)

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """
    Allowed risk levels for GAP analysis.
    
    - LOW: Minor issue, low impact on document validity
    - MEDIUM: Moderate concern requiring attention
    - HIGH: Critical issue with significant legal implications
    """
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class GAPItem(BaseModel):
    """
    Single GAP (issue/concern) identified in legal document analysis.
    
    This model enforces strict field validation:
    - All fields are required (no optional fields)
    - Extra fields are forbidden (will raise validation error)
    - gap_id must follow pattern: GAP-XXX (e.g., GAP-001, GAP-042)
    - risk_level must be exactly: Low, Medium, or High
    - All text fields must be non-empty
    
    Attributes:
        gap_id: Unique identifier for this GAP (format: GAP-XXX)
        title: Short descriptive title (max 200 chars)
        clause_reference: Reference to document clause(s) where issue found
        description: Detailed description of the issue (max 1000 chars)
        risk_level: Risk severity (Low, Medium, or High)
        recommendation: Suggested action or remedy (max 1000 chars)
    
    Example:
        >>> gap = GAPItem(
        ...     gap_id="GAP-001",
        ...     title="Missing witness signatures",
        ...     clause_reference="Attestation section",
        ...     description="Document lacks required witness signatures",
        ...     risk_level=RiskLevel.HIGH,
        ...     recommendation="Obtain signatures from two witnesses"
        ... )
    """
    
    # Strict configuration: no extra fields allowed
    model_config = ConfigDict(
        extra='forbid',  # Reject any extra fields not defined
        str_strip_whitespace=True,  # Auto-strip whitespace
        validate_assignment=True,  # Validate on assignment
        use_enum_values=True  # Use enum values in output
    )
    
    gap_id: str = Field(
        ...,
        description="Unique GAP identifier (format: GAP-XXX)",
        min_length=7,
        max_length=10,
        pattern=r'^GAP-\d{3,6}$',
        examples=["GAP-001", "GAP-042", "GAP-123"]
    )
    
    title: str = Field(
        ...,
        description="Short descriptive title of the issue",
        min_length=5,
        max_length=200,
        examples=["Missing grantor identification", "Incomplete property boundaries"]
    )
    
    clause_reference: str = Field(
        ...,
        description="Reference to clause(s) where issue was found",
        min_length=1,
        max_length=100,
        examples=["Clause 1", "Clauses 2-4", "Section A", "Page 3, Line 12"]
    )
    
    description: str = Field(
        ...,
        description="Detailed description of the identified issue",
        min_length=10,
        max_length=1000,
        examples=[
            "The document does not specify the grantor's complete legal name and parentage",
            "Property boundaries are described vaguely without survey number reference"
        ]
    )
    
    risk_level: RiskLevel = Field(
        ...,
        description="Risk severity level (Low, Medium, or High only)"
    )
    
    recommendation: str = Field(
        ...,
        description="Suggested action or remedy for addressing this issue",
        min_length=10,
        max_length=1000,
        examples=[
            "Verify grantor identity through government-issued ID",
            "Obtain official survey report for accurate boundary description"
        ]
    )
    
    @field_validator('gap_id')
    @classmethod
    def validate_gap_id_format(cls, v: str) -> str:
        """
        Validate GAP ID follows correct format.
        
        Format: GAP-XXX where XXX is 3-6 digits
        Examples: GAP-001, GAP-042, GAP-123456
        """
        if not v.startswith('GAP-'):
            raise ValueError(
                f"gap_id must start with 'GAP-' (got: {v}). "
                "Example: GAP-001"
            )
        
        # Extract numeric part
        numeric_part = v[4:]
        if not numeric_part.isdigit():
            raise ValueError(
                f"gap_id must be GAP-XXX where XXX is numeric (got: {v}). "
                "Example: GAP-001"
            )
        
        if len(numeric_part) < 3:
            raise ValueError(
                f"gap_id numeric part must be at least 3 digits (got: {v}). "
                "Example: GAP-001"
            )
        
        return v
    
    @field_validator('title', 'description', 'recommendation')
    @classmethod
    def validate_non_empty_text(cls, v: str, info) -> str:
        """Ensure text fields are not empty or just whitespace."""
        if not v or not v.strip():
            raise ValueError(
                f"{info.field_name} cannot be empty or whitespace only"
            )
        return v.strip()
    
    @field_validator('clause_reference')
    @classmethod
    def validate_clause_reference(cls, v: str) -> str:
        """Validate clause reference is meaningful."""
        v = v.strip()
        if not v:
            raise ValueError("clause_reference cannot be empty")
        
        # Check for common placeholder values that should be rejected
        invalid_placeholders = ['n/a', 'na', 'none', 'null', 'unknown', '-']
        if v.lower() in invalid_placeholders:
            raise ValueError(
                f"clause_reference must be a specific reference, not '{v}'. "
                "Example: 'Clause 1' or 'Section A'"
            )
        
        return v


class GAPAnalysisOutput(BaseModel):
    """
    Complete GAP analysis output containing list of identified issues.
    
    This is the root model for LLM output validation. It enforces:
    - Must contain a 'gaps' field with list of GAPItem objects
    - No extra fields allowed at root level
    - gaps list can be empty (no issues found) but must exist
    - Each GAP item is strictly validated
    
    Attributes:
        gaps: List of identified GAP items (can be empty list)
    
    Example:
        >>> output = GAPAnalysisOutput(gaps=[
        ...     GAPItem(
        ...         gap_id="GAP-001",
        ...         title="Missing attestation",
        ...         clause_reference="Clause 5",
        ...         description="No witness signatures found",
        ...         risk_level=RiskLevel.HIGH,
        ...         recommendation="Obtain witness attestations"
        ...     )
        ... ])
        >>> print(f"Found {len(output.gaps)} issues")
    """
    
    # Strict configuration: no extra fields allowed
    model_config = ConfigDict(
        extra='forbid',  # Reject any extra fields
        validate_assignment=True
    )
    
    gaps: List[GAPItem] = Field(
        ...,
        description="List of identified GAP items (can be empty if no issues found)",
        examples=[
            [],  # No issues found
            [{"gap_id": "GAP-001", "title": "...", "clause_reference": "...", 
              "description": "...", "risk_level": "High", "recommendation": "..."}]
        ]
    )
    
    @field_validator('gaps')
    @classmethod
    def validate_gaps_list(cls, v: List[GAPItem]) -> List[GAPItem]:
        """Validate gaps list and ensure unique gap_ids."""
        if v is None:
            raise ValueError("gaps field is required (use empty list if no issues)")
        
        # Check for duplicate gap_ids
        gap_ids = [gap.gap_id for gap in v]
        duplicates = [gid for gid in gap_ids if gap_ids.count(gid) > 1]
        
        if duplicates:
            raise ValueError(
                f"Duplicate gap_ids found: {set(duplicates)}. "
                "Each gap_id must be unique."
            )
        
        return v
    
    @model_validator(mode='after')
    def validate_risk_distribution(self) -> 'GAPAnalysisOutput':
        """
        Optional: Log warning if all gaps have same risk level.
        
        This is not a validation error, just a quality check.
        """
        if len(self.gaps) > 3:
            risk_levels = [gap.risk_level for gap in self.gaps]
            unique_risks = set(risk_levels)
            
            if len(unique_risks) == 1:
                logger.warning(
                    f"All {len(self.gaps)} gaps have same risk level: "
                    f"{risk_levels[0]}. Consider reviewing risk assessment."
                )
        
        return self
    
    def get_high_risk_gaps(self) -> List[GAPItem]:
        """Return only high-risk GAP items."""
        return [gap for gap in self.gaps if gap.risk_level == RiskLevel.HIGH]
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of GAP analysis.
        
        Returns:
            dict: Summary with total gaps and count by risk level
        """
        return {
            "total_gaps": len(self.gaps),
            "high_risk": len([g for g in self.gaps if g.risk_level == RiskLevel.HIGH]),
            "medium_risk": len([g for g in self.gaps if g.risk_level == RiskLevel.MEDIUM]),
            "low_risk": len([g for g in self.gaps if g.risk_level == RiskLevel.LOW]),
            "gap_ids": [g.gap_id for g in self.gaps]
        }


class ValidationError(Exception):
    """Custom exception for GAP schema validation errors."""
    pass


def validate_gap_analysis(json_data: Dict[str, Any]) -> GAPAnalysisOutput:
    """
    Validate GAP analysis JSON against strict schema.
    
    This function:
    1. Parses JSON data into Pydantic model
    2. Validates all fields and constraints
    3. Rejects extra fields
    4. Returns validated model or raises detailed error
    
    Args:
        json_data: Dictionary representing GAP analysis output
    
    Returns:
        GAPAnalysisOutput: Validated GAP analysis object
    
    Raises:
        ValidationError: If JSON doesn't match schema with detailed message
    
    Example:
        >>> json_data = {
        ...     "gaps": [
        ...         {
        ...             "gap_id": "GAP-001",
        ...             "title": "Missing grantor info",
        ...             "clause_reference": "Clause 1",
        ...             "description": "Grantor's father name not mentioned",
        ...             "risk_level": "High",
        ...             "recommendation": "Verify through ID documents"
        ...         }
        ...     ]
        ... }
        >>> validated = validate_gap_analysis(json_data)
        >>> print(f"Validated {len(validated.gaps)} gaps")
    
    Example Error Cases:
        >>> # Missing required field
        >>> bad_data = {"gaps": [{"gap_id": "GAP-001"}]}
        >>> validate_gap_analysis(bad_data)  # Raises ValidationError
        
        >>> # Invalid risk level
        >>> bad_data = {"gaps": [{"risk_level": "Critical", ...}]}
        >>> validate_gap_analysis(bad_data)  # Raises ValidationError
        
        >>> # Extra field not allowed
        >>> bad_data = {"gaps": [...], "extra_field": "value"}
        >>> validate_gap_analysis(bad_data)  # Raises ValidationError
    """
    try:
        # Parse and validate using Pydantic
        validated = GAPAnalysisOutput.model_validate(json_data)
        
        logger.info(
            f"✓ Successfully validated GAP analysis: "
            f"{len(validated.gaps)} gaps found"
        )
        
        return validated
    
    except Exception as e:
        # Transform Pydantic error into user-friendly message
        error_msg = _format_validation_error(e, json_data)
        logger.error(f"GAP schema validation failed: {error_msg}")
        raise ValidationError(error_msg) from e


def validate_gap_analysis_from_string(json_string: str) -> GAPAnalysisOutput:
    """
    Validate GAP analysis from JSON string.
    
    Args:
        json_string: JSON string representing GAP analysis
    
    Returns:
        GAPAnalysisOutput: Validated GAP analysis object
    
    Raises:
        ValidationError: If JSON is invalid or doesn't match schema
    
    Example:
        >>> json_str = '{"gaps": [{"gap_id": "GAP-001", ...}]}'
        >>> validated = validate_gap_analysis_from_string(json_str)
    """
    import json
    
    try:
        json_data = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValidationError(
            f"Invalid JSON string: {str(e)}\n"
            f"Expected valid JSON with 'gaps' field."
        ) from e
    
    return validate_gap_analysis(json_data)


def _format_validation_error(error: Exception, json_data: Dict[str, Any]) -> str:
    """
    Format Pydantic validation error into user-friendly message.
    
    Args:
        error: Pydantic validation exception
        json_data: Original JSON data that failed validation
    
    Returns:
        str: Formatted error message with suggestions
    """
    from pydantic import ValidationError as PydanticValidationError
    
    if not isinstance(error, PydanticValidationError):
        return f"Validation failed: {str(error)}"
    
    error_messages = []
    
    for err in error.errors():
        loc = " -> ".join(str(l) for l in err['loc'])
        msg = err['msg']
        error_type = err['type']
        
        # Build helpful message based on error type
        if error_type == 'extra_forbidden':
            error_messages.append(
                f"❌ Extra field not allowed: '{loc}'\n"
                f"   Only these fields are allowed: gap_id, title, clause_reference, "
                f"description, risk_level, recommendation"
            )
        
        elif error_type == 'missing':
            error_messages.append(
                f"❌ Missing required field: '{loc}'\n"
                f"   This field must be provided."
            )
        
        elif 'risk_level' in loc:
            error_messages.append(
                f"❌ Invalid risk_level: {msg}\n"
                f"   Must be exactly one of: 'Low', 'Medium', 'High'\n"
                f"   (case-sensitive)"
            )
        
        elif 'gap_id' in loc:
            error_messages.append(
                f"❌ Invalid gap_id format: {msg}\n"
                f"   Must follow pattern: GAP-XXX (e.g., GAP-001, GAP-042)"
            )
        
        else:
            error_messages.append(f"❌ {loc}: {msg}")
    
    # Add context about what was received
    error_summary = "\n".join(error_messages)
    
    return (
        f"GAP Analysis Schema Validation Failed:\n\n"
        f"{error_summary}\n\n"
        f"Received data structure: {list(json_data.keys())}\n"
        f"Expected structure: {{'gaps': [list of GAP items]}}\n\n"
        f"Each GAP item must have:\n"
        f"  - gap_id: 'GAP-XXX' format\n"
        f"  - title: descriptive title (5-200 chars)\n"
        f"  - clause_reference: specific clause reference\n"
        f"  - description: detailed description (10-1000 chars)\n"
        f"  - risk_level: 'Low', 'Medium', or 'High' (case-sensitive)\n"
        f"  - recommendation: suggested action (10-1000 chars)\n"
        f"\n"
        f"No extra fields are allowed."
    )


# Example usage and test cases
"""
Example 1: Valid GAP Analysis

valid_json = {
    "gaps": [
        {
            "gap_id": "GAP-001",
            "title": "Missing grantor identification details",
            "clause_reference": "Clause 1, Line 3",
            "description": "The document does not specify the grantor's father's name, which is required for legal identification in property transfers.",
            "risk_level": "High",
            "recommendation": "Verify grantor identity through Aadhaar card or other government-issued identification documents and amend the deed."
        },
        {
            "gap_id": "GAP-002",
            "title": "Incomplete property boundary description",
            "clause_reference": "Clause 5",
            "description": "Property boundaries are described only for North and South directions. East and West boundaries are missing.",
            "risk_level": "Medium",
            "recommendation": "Obtain survey report with complete boundary measurements and update the property description clause."
        }
    ]
}

try:
    validated = validate_gap_analysis(valid_json)
    print(f"✓ Validation successful: {len(validated.gaps)} gaps")
    print(f"Summary: {validated.get_summary()}")
    
    # Access individual gaps
    for gap in validated.gaps:
        print(f"{gap.gap_id}: {gap.title} (Risk: {gap.risk_level})")
        
except ValidationError as e:
    print(f"Validation failed: {e}")


Example 2: Invalid - Extra Field

invalid_json = {
    "gaps": [
        {
            "gap_id": "GAP-001",
            "title": "Missing info",
            "clause_reference": "Clause 1",
            "description": "Some description here",
            "risk_level": "High",
            "recommendation": "Do something",
            "extra_field": "This will be rejected"  # ❌ Not allowed
        }
    ]
}

# This will raise ValidationError
validate_gap_analysis(invalid_json)


Example 3: Invalid - Wrong Risk Level

invalid_json = {
    "gaps": [
        {
            "gap_id": "GAP-001",
            "title": "Missing info",
            "clause_reference": "Clause 1",
            "description": "Some description here",
            "risk_level": "Critical",  # ❌ Must be Low, Medium, or High
            "recommendation": "Do something"
        }
    ]
}

# This will raise ValidationError
validate_gap_analysis(invalid_json)


Example 4: Invalid - Missing Required Field

invalid_json = {
    "gaps": [
        {
            "gap_id": "GAP-001",
            "title": "Missing info",
            # ❌ Missing: clause_reference, description, risk_level, recommendation
        }
    ]
}

# This will raise ValidationError
validate_gap_analysis(invalid_json)


Example 5: Valid - No Issues Found

no_issues_json = {
    "gaps": []  # ✓ Empty list is valid (no issues found)
}

validated = validate_gap_analysis(no_issues_json)
print(f"No gaps found: {len(validated.gaps)} issues")


Example 6: Using in LLM Pipeline

from app.services.ollama_client import OllamaClient
from app.services.prompt_factory import PromptFactory
import json

# Generate prompt
factory = PromptFactory()
prompt = factory.create_gap_prompt(...)

# Get LLM response
client = OllamaClient()
response = client.generate(prompt)

# Parse JSON from response
try:
    json_data = json.loads(response)
    validated = validate_gap_analysis(json_data)
    
    # Process validated gaps
    high_risk_gaps = validated.get_high_risk_gaps()
    if high_risk_gaps:
        print(f"⚠️ Found {len(high_risk_gaps)} high-risk issues")
        
except json.JSONDecodeError:
    print("LLM did not return valid JSON")
except ValidationError as e:
    print(f"LLM response doesn't match schema: {e}")
"""
