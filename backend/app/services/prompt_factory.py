"""
Prompt Factory for Legal GAP Analysis

This module generates structured prompts for LLM-based analysis of legal documents.
GAP stands for:
- G: Grantor (seller/transferor)
- A: Attestation (witnesses, signatures, legal validation)
- P: Property (description, boundaries, rights)

The prompts enforce strict rules:
1. No assumptions - analyze only what's explicitly stated
2. No legal advice - descriptive analysis only
3. Clause-based analysis - reference specific clauses
4. JSON-only output - structured data for parsing

Example Usage:
    >>> factory = PromptFactory()
    >>> prompt = factory.create_gap_prompt(
    ...     language="english",
    ...     gap_definition=GAPDefinition(...),
    ...     clauses=["Clause 1: The grantor hereby...", ...]
    ... )
    >>> # Send prompt to LLM
    >>> response = llm.generate(prompt)
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Language(str, Enum):
    """Supported languages for prompt generation."""
    ENGLISH = "english"
    TAMIL = "tamil"


class RiskLevel(str, Enum):
    """Risk levels for GAP analysis."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class GAPDefinition:
    """
    Definition of GAP analysis components.
    
    Attributes:
        grantor_criteria: What to look for in grantor analysis
        attestation_criteria: What to look for in attestation analysis
        property_criteria: What to look for in property analysis
    """
    grantor_criteria: List[str] = field(default_factory=lambda: [
        "Seller/transferor identity",
        "Authority to sell",
        "Title ownership proof",
        "Legal capacity to transfer"
    ])
    
    attestation_criteria: List[str] = field(default_factory=lambda: [
        "Witness signatures",
        "Notary attestation",
        "Registration details",
        "Legal validation marks"
    ])
    
    property_criteria: List[str] = field(default_factory=lambda: [
        "Property description",
        "Boundaries and measurements",
        "Rights and encumbrances",
        "Location and address"
    ])


@dataclass
class OutputSchema:
    """
    Expected JSON output schema for GAP analysis.
    
    Example output:
    {
        "grantor": {
            "found": true,
            "clauses": ["1", "2"],
            "summary": "...",
            "risk_level": "low"
        },
        "attestation": {...},
        "property": {...},
        "overall_risk": "low",
        "missing_elements": []
    }
    """
    
    @staticmethod
    def get_schema() -> Dict[str, Any]:
        """Return the JSON schema for GAP analysis output."""
        return {
            "grantor": {
                "found": "boolean - whether grantor information is present",
                "clauses": "array of clause numbers/references where found",
                "summary": "string - brief summary of what was found",
                "risk_level": "string - one of: high, medium, low, none",
                "issues": "array of strings - specific concerns or missing items"
            },
            "attestation": {
                "found": "boolean",
                "clauses": "array of clause numbers/references",
                "summary": "string",
                "risk_level": "string",
                "issues": "array of strings"
            },
            "property": {
                "found": "boolean",
                "clauses": "array of clause numbers/references",
                "summary": "string",
                "risk_level": "string",
                "issues": "array of strings"
            },
            "overall_risk": "string - highest risk level from all three categories",
            "missing_elements": "array of strings - critical missing elements",
            "confidence": "string - confidence in analysis (high, medium, low)"
        }


class PromptFactory:
    """
    Factory for generating structured prompts for legal GAP analysis.
    
    This factory creates language-specific prompts that:
    - Enforce strict analysis rules (no assumptions, no advice)
    - Define clear output schemas (JSON only)
    - Inject domain knowledge (GAP definitions, risk levels)
    - Support multilingual analysis (English, Tamil)
    
    Example:
        >>> factory = PromptFactory()
        >>> gap_def = GAPDefinition()
        >>> clauses = [
        ...     "Clause 1: The seller Mr. John Doe hereby transfers...",
        ...     "Clause 2: Property located at Survey No. 123..."
        ... ]
        >>> prompt = factory.create_gap_prompt(
        ...     language=Language.ENGLISH,
        ...     gap_definition=gap_def,
        ...     clauses=clauses
        ... )
    """
    
    def __init__(self, use_custom_model: bool = False):
        """Initialize the prompt factory.
        
        Args:
            use_custom_model: If True, use simplified prompts for custom GAP models
                            that have the framework pre-trained in system prompt
        """
        self.output_schema = OutputSchema()
        self.use_custom_model = use_custom_model
        logger.info(f"Initialized PromptFactory (custom_model={use_custom_model})")
    
    def create_gap_prompt(
        self,
        language: Language,
        gap_definition: GAPDefinition,
        clauses: List[str],
        document_context: Optional[str] = None
    ) -> str:
        """
        Create a GAP analysis prompt for the LLM.
        
        If use_custom_model=True, creates a simplified prompt (just clauses)
        assuming the GAP framework is already in the model's system prompt.
        
        Args:
            language: Language for the prompt (english or tamil)
            gap_definition: GAP analysis criteria (ignored if use_custom_model=True)
            clauses: List of document clauses to analyze
            document_context: Optional context about the document
        
        Returns:
            str: Complete prompt ready to send to LLM
        
        Example:
            >>> prompt = factory.create_gap_prompt(
            ...     language=Language.ENGLISH,
            ...     gap_definition=GAPDefinition(),
            ...     clauses=["Clause 1: The grantor Mr. Smith..."],
            ...     document_context="Property deed dated 2023-01-15"
            ... )
        """
        # Use simplified prompts for custom models with pre-trained framework
        if self.use_custom_model:
            return self._create_simplified_prompt(clauses, document_context, language)
        
        # Use full prompts for base models
        if language == Language.ENGLISH:
            return self._create_english_prompt(
                gap_definition, clauses, document_context
            )
        elif language == Language.TAMIL:
            return self._create_tamil_prompt(
                gap_definition, clauses, document_context
            )
        else:
            raise ValueError(f"Unsupported language: {language}")
    
    def _create_english_prompt(
        self,
        gap_definition: GAPDefinition,
        clauses: List[str],
        document_context: Optional[str] = None
    ) -> str:
        """Generate English GAP analysis prompt."""
        
        # Format clauses with numbers
        formatted_clauses = self._format_clauses(clauses)
        
        # Build prompt
        prompt = f"""You are a legal document analyzer specializing in GAP analysis.

# STRICT RULES (MUST FOLLOW):
1. NO ASSUMPTIONS - Analyze ONLY what is explicitly stated in the provided clauses
2. NO LEGAL ADVICE - Provide descriptive analysis only, not recommendations
3. CLAUSE-BASED ANALYSIS - Always cite specific clause numbers in your findings
4. JSON OUTPUT ONLY - Return only valid JSON, no additional text or explanations

# GAP ANALYSIS FRAMEWORK:

## G - GRANTOR (Seller/Transferor)
Analyze for:
{self._format_criteria(gap_definition.grantor_criteria)}

## A - ATTESTATION (Legal Validation)
Analyze for:
{self._format_criteria(gap_definition.attestation_criteria)}

## P - PROPERTY (Description & Rights)
Analyze for:
{self._format_criteria(gap_definition.property_criteria)}

# RISK LEVELS:
- HIGH: Critical element missing or unclear, significant gaps
- MEDIUM: Element present but incomplete or vague
- LOW: Element present with minor concerns
- NONE: Element complete and clear

# OUTPUT SCHEMA:
Return a JSON object with this exact structure:
{self._format_json_schema()}

# DOCUMENT CLAUSES TO ANALYZE:
{formatted_clauses}

{f"# DOCUMENT CONTEXT:\\n{document_context}\\n" if document_context else ""}
# YOUR ANALYSIS (JSON ONLY):"""
        
        return prompt
    
    def _create_tamil_prompt(
        self,
        gap_definition: GAPDefinition,
        clauses: List[str],
        document_context: Optional[str] = None
    ) -> str:
        """
        Generate Tamil GAP analysis prompt with formal language.
        
        ✓ PROTECTION 4: Language Consistency
        - Uses formal Tamil (செந்தமிழ்) for instructions
        - Enforces Tamil content in title/description/recommendation
        - Requires English field names (JSON keys)
        - Prevents mixing languages in output
        
        Uses formal Tamil (செந்தமிழ்) appropriate for legal document analysis.
        Enforces strict JSON-only output with no explanations.
        """
        
        # Format clauses with numbers
        formatted_clauses = self._format_clauses(clauses)
        
        # Build prompt with formal Tamil instructions
        prompt = f"""நீங்கள் GAP பகுப்பாய்வு முறையில் நிபுணத்துவம் பெற்ற சட்ட ஆவண பகுப்பாய்வாளர் ஆவீர்கள்.

# GAP பகுப்பாய்வு முறை என்றால் என்ன?

GAP என்பது சொத்து பத்திரங்களில் உள்ள மூன்று முக்கிய கூறுகளின் பகுப்பாய்வு முறை:

**G - வழங்குபவர் (GRANTOR)**
சொத்தை விற்பவர் அல்லது மாற்றுபவரின் விவரங்கள்:
  • முழு பெயர் மற்றும் தந்தை/தாய் பெயர்
  • சட்டபூர்வமாக விற்கும் அதிகாரம்
  • உரிமை ஆவணங்களின் சான்றுகள்
  • மனநிலை மற்றும் வயது தகுதி

**A - சான்றளிப்பு (ATTESTATION)**
சட்டப்படி செல்லுபடியாகும் சான்றுகள்:
  • சாட்சிகளின் கையெழுத்துகள்
  • நோட்டரி பொது அதிகாரியின் உறுதிப்படுத்தல்
  • பதிவு எண் மற்றும் தேதி
  • துணை பதிவாளர் அலுவலக முத்திரை

**P - சொத்து (PROPERTY)**
சொத்தின் விளக்கம் மற்றும் உரிமைகள்:
  • சர்வே எண் மற்றும் உட்பிரிவு
  • கிராமம், தாலுக்கா, மாவட்டம்
  • நான்கு பக்க எல்லைகள் மற்றும் அளவீடுகள்
  • மொத்த பரப்பளவு
  • அடமானங்கள் மற்றும் கடன்கள்

# கண்டிப்பான விதிமுறைகள் (இவற்றை கட்டாயம் பின்பற்ற வேண்டும்):

1. **அனுமானங்கள் தடை** 
   வழங்கப்பட்ட விதிகளில் வெளிப்படையாகக் குறிப்பிடப்பட்டவற்றை மட்டுமே பகுப்பாய்வு செய்யவும்.
   உங்கள் சொந்த கருத்துக்களை சேர்க்க வேண்டாம்.

2. **சட்ட ஆலோசனை தடை**
   விளக்கமான பகுப்பாய்வு மட்டும் வழங்கவும்.
   சட்ட பரிந்துரைகள் அல்லது தீர்வுகள் வழங்க வேண்டாம்.

3. **விதி குறிப்பு கட்டாயம்**
   ஒவ்வொரு GAP உருப்படியும் குறிப்பிட்ட விதி எண்களை மேற்கோள் காட்ட வேண்டும்.
   "பொதுவாக" அல்லது "ஆவணத்தில்" என்று குறிப்பிட வேண்டாம்.

4. **JSON வெளியீடு மட்டுமே (மிக முக்கியம்)**
   செல்லுபடியாகும் JSON மட்டுமே வழங்கவும்.
   கூடுதல் உரை, விளக்கங்கள், அல்லது கருத்துரைகள் கூடாது.
   JSON-க்கு வெளியே எந்த தமிழ் வார்த்தைகளும் எழுத வேண்டாம்.

# பகுப்பாய்வு செய்ய வேண்டிய கூறுகள்:

## G - வழங்குபவர் விவரங்கள்
{self._format_criteria_tamil(gap_definition.grantor_criteria)}

## A - சான்றளிப்பு விவரங்கள்
{self._format_criteria_tamil(gap_definition.attestation_criteria)}

## P - சொத்து விவரங்கள்
{self._format_criteria_tamil(gap_definition.property_criteria)}

# இடர் நிலை மதிப்பீடு:

- **High** (அதிக இடர்): முக்கியமான தகவல் காணவில்லை, சட்டப்படி செல்லுபடியாகாத அபாயம்
- **Medium** (நடுத்தர இடர்): தகவல் முழுமையற்றது, தெளிவுபடுத்தல் தேவை
- **Low** (குறைந்த இடர்): தகவல் உள்ளது ஆனால் சிறிய மேம்பாடு தேவை

# JSON வெளியீடு வடிவம்:

கீழே உள்ள சரியான கட்டமைப்பில் JSON மட்டுமே வழங்கவும்:

{self._format_json_schema()}

**முக்கிய குறிப்பு:** 
- "gaps" என்ற பட்டியலில் கண்டறியப்பட்ட பிரச்சினைகள் மட்டும் இருக்க வேண்டும்
- ஒவ்வொரு GAP உருப்படியும் gap_id, title, clause_reference, description, risk_level, recommendation ஆகிய அனைத்து புலங்களையும் கொண்டிருக்க வேண்டும்
- risk_level "Low", "Medium", அல்லது "High" மட்டுமே (ஆங்கிலத்தில்)
- கூடுதல் புலங்கள் அனுமதிக்கப்படாது

# பகுப்பாய்வு செய்ய வேண்டிய ஆவண விதிகள்:

{formatted_clauses}

{f"# ஆவண சூழல்:\\n{document_context}\\n" if document_context else ""}

# உங்கள் பகுப்பாய்வு (JSON வடிவத்தில் மட்டும், வேறு எதுவும் எழுத வேண்டாம்):"""
        
        return prompt
    
    def _create_simplified_prompt(
        self,
        clauses: List[str],
        document_context: Optional[str] = None,
        language: Language = Language.ENGLISH
    ) -> str:
        """
        Create a simplified prompt for custom GAP models.
        
        Custom models have the GAP framework embedded in their system prompt,
        so we only need to send the clauses to analyze. This reduces prompt
        size by ~90% and speeds up inference.
        
        Args:
            clauses: List of document clauses to analyze
            document_context: Optional context about the document
            language: Language hint for the model
        
        Returns:
            str: Minimal prompt with just clauses
        """
        formatted_clauses = self._format_clauses(clauses)
        
        if language == Language.TAMIL:
            prompt = f"""# பகுப்பாய்வு செய்ய வேண்டிய ஆவண விதிகள்:

{formatted_clauses}"""
        else:
            prompt = f"""# Document Clauses to Analyze:

{formatted_clauses}"""
        
        if document_context:
            if language == Language.TAMIL:
                prompt += f"\n\n# ஆவண சூழல்:\n{document_context}"
            else:
                prompt += f"\n\n# Document Context:\n{document_context}"
        
        if language == Language.TAMIL:
            prompt += "\n\n# உங்கள் GAP பகுப்பாய்வு (JSON வடிவத்தில் மட்டும்):"
        else:
            prompt += "\n\n# Your GAP Analysis (JSON only):"
        
        return prompt
    
    def _format_clauses(self, clauses: List[str]) -> str:
        """Format clauses with numbering."""
        if not clauses:
            return "No clauses provided."
        
        formatted = []
        for i, clause in enumerate(clauses, start=1):
            # Add clause number if not already present
            if clause.strip().startswith(("Clause", "CLAUSE", "Section", "விதி")):
                formatted.append(f"{clause}")
            else:
                formatted.append(f"Clause {i}: {clause}")
        
        return "\n\n".join(formatted)
    
    def _format_criteria(self, criteria: List[str]) -> str:
        """Format criteria as bullet points."""
        return "\n".join(f"  - {item}" for item in criteria)
    
    def _format_criteria_tamil(self, criteria: List[str]) -> str:
        """Format criteria as bullet points (Tamil version uses same format)."""
        return "\n".join(f"  - {item}" for item in criteria)
    
    def _format_json_schema(self) -> str:
        """Format the JSON schema as a readable string."""
        schema = self.output_schema.get_schema()
        
        # Create a formatted example
        return """{
  "grantor": {
    "found": true/false,
    "clauses": ["1", "2", ...],
    "summary": "Brief description of grantor information",
    "risk_level": "high/medium/low/none",
    "issues": ["Specific concern 1", "Specific concern 2"]
  },
  "attestation": {
    "found": true/false,
    "clauses": ["3", "4", ...],
    "summary": "Brief description of attestation",
    "risk_level": "high/medium/low/none",
    "issues": []
  },
  "property": {
    "found": true/false,
    "clauses": ["5", "6", ...],
    "summary": "Brief description of property",
    "risk_level": "high/medium/low/none",
    "issues": []
  },
  "overall_risk": "high/medium/low/none",
  "missing_elements": ["Element 1", "Element 2"],
  "confidence": "high/medium/low"
}"""
    
    def create_clarification_prompt(
        self,
        language: Language,
        original_analysis: Dict[str, Any],
        clarification_needed: str
    ) -> str:
        """
        Create a follow-up prompt for clarification.
        
        Use this when the initial analysis needs more detail.
        
        Args:
            language: Language for the prompt
            original_analysis: The original GAP analysis result
            clarification_needed: What needs clarification
        
        Returns:
            str: Clarification prompt
        
        Example:
            >>> clarification = factory.create_clarification_prompt(
            ...     language=Language.ENGLISH,
            ...     original_analysis={"grantor": {"found": true, ...}},
            ...     clarification_needed="Provide more detail about grantor's authority"
            ... )
        """
        if language == Language.ENGLISH:
            return f"""Based on your previous analysis:
{original_analysis}

Please provide clarification on:
{clarification_needed}

Rules:
- Reference specific clauses
- Explain your reasoning
- Maintain JSON format for any new analysis

Your clarification:"""
        
        else:  # Tamil
            return f"""உங்கள் முந்தைய பகுப்பாய்வின் அடிப்படையில்:
{original_analysis}

தயவுசெய்து விளக்கம் வழங்கவும்:
{clarification_needed}

விதிகள்:
- குறிப்பிட்ட விதிகளைக் குறிப்பிடவும்
- உங்கள் காரணத்தை விளக்கவும்
- எந்த புதிய பகுப்பாய்விற்கும் JSON வடிவத்தை பராமரிக்கவும்

உங்கள் விளக்கம்:"""
    
    def validate_response(self, response: str) -> Dict[str, Any]:
        """
        Validate and parse LLM response.
        
        Args:
            response: Raw response from LLM
        
        Returns:
            dict: Parsed JSON response
        
        Raises:
            ValueError: If response is not valid JSON or missing required fields
        
        Example:
            >>> response = llm.generate(prompt)
            >>> analysis = factory.validate_response(response)
        """
        import json
        
        # Try to extract JSON from response (in case LLM adds extra text)
        response = response.strip()
        
        # Find JSON block
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx == -1 or end_idx == -1:
            raise ValueError("No JSON object found in response")
        
        json_str = response[start_idx:end_idx + 1]
        
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {str(e)}")
        
        # Validate required fields
        required_fields = ["grantor", "attestation", "property", "overall_risk"]
        missing_fields = [f for f in required_fields if f not in parsed]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Validate each GAP component
        for component in ["grantor", "attestation", "property"]:
            if component in parsed:
                self._validate_component(parsed[component], component)
        
        return parsed
    
    def _validate_component(self, component: Dict[str, Any], name: str):
        """Validate a GAP component structure."""
        required_fields = ["found", "clauses", "summary", "risk_level"]
        
        for field in required_fields:
            if field not in component:
                raise ValueError(
                    f"Missing field '{field}' in '{name}' component"
                )
        
        # Validate risk level
        if component["risk_level"] not in ["high", "medium", "low", "none"]:
            raise ValueError(
                f"Invalid risk_level in '{name}': {component['risk_level']}"
            )


# Tamil GAP Output Examples (for reference)
"""
Example Tamil GAP Output:

{
  "gaps": [
    {
      "gap_id": "GAP-001",
      "title": "வழங்குபவரின் தந்தை பெயர் குறிப்பிடப்படவில்லை",
      "clause_reference": "விதி 1",
      "description": "ஆவணத்தில் விற்பவரின் தந்தை பெயர் தெளிவாக குறிப்பிடப்படவில்லை. சட்டப்படி அடையாள சரிபார்ப்புக்கு இது அவசியம்.",
      "risk_level": "High",
      "recommendation": "விற்பவரின் முழுமையான அடையாள விவரங்களை ஆதார் அட்டை அல்லது பிற அரசு ஆவணங்கள் மூலம் சரிபார்க்கவும்."
    },
    {
      "gap_id": "GAP-002",
      "title": "சொத்து எல்லைகள் முழுமையாக குறிப்பிடப்படவில்லை",
      "clause_reference": "விதி 5",
      "description": "வடக்கு மற்றும் தெற்கு எல்லைகள் மட்டும் குறிப்பிடப்பட்டுள்ளன. கிழக்கு மற்றும் மேற்கு எல்லைகள் காணவில்லை.",
      "risk_level": "Medium",
      "recommendation": "முழுமையான சர்வே அறிக்கை மூலம் நான்கு பக்க எல்லைகளையும் பெற்று ஆவணத்தை புதுப்பிக்கவும்."
    },
    {
      "gap_id": "GAP-003",
      "title": "இரண்டாவது சாட்சி கையெழுத்து தெளிவற்றது",
      "clause_reference": "விதி 8",
      "description": "சாட்சி திருமதி. ராதாவின் கையெழுத்து தெளிவாக படிக்க முடியவில்லை மற்றும் முழு பெயர் குறிப்பிடப்படவில்லை.",
      "risk_level": "Low",
      "recommendation": "சாட்சியின் முழுமையான விவரங்களை சரிபார்த்து தெளிவான கையெழுத்துடன் உறுதிப்படுத்தல் பெறவும்."
    }
  ]
}

Note: All field names (gap_id, title, etc.) must be in English.
Only the content values (title, description, recommendation) are in Tamil.
risk_level must be exactly "Low", "Medium", or "High" in English.
"""


# Example prompts for reference
"""
Example 1: English Property Deed Analysis

factory = PromptFactory()
gap_def = GAPDefinition()
clauses = [
    "Clause 1: The seller Mr. John Doe, son of Richard Doe, aged 45 years...",
    "Clause 2: Witnessed by Mr. Smith and Mrs. Jones on 2023-01-15...",
    "Clause 3: Property located at Survey No. 123/4, Village: Mylapore..."
]

prompt = factory.create_gap_prompt(
    language=Language.ENGLISH,
    gap_definition=gap_def,
    clauses=clauses,
    document_context="Sale deed dated 2023-01-15, registered at Sub-Registrar Office"
)

# Send to LLM
response = ollama_client.generate(prompt)

# Validate and parse
analysis = factory.validate_response(response)
print(f"Overall risk: {analysis['overall_risk']}")


Example 2: Tamil Document Analysis

clauses_tamil = [
    "விதி 1: விற்பனையாளர் திரு. ராமன், திரு. கிருஷ்ணன் மகன்...",
    "விதி 2: சாட்சிகள்: திரு. முருகன் மற்றும் திருமதி. லட்சுமி...",
    "விதி 3: சொத்து அமைவிடம்: சர்வே எண். 456/7, கிராமம்: மயிலாப்பூர்..."
]

prompt_tamil = factory.create_gap_prompt(
    language=Language.TAMIL,
    gap_definition=gap_def,
    clauses=clauses_tamil,
    document_context="விற்பனை பத்திரம் தேதி 2023-01-15"
)


Example 3: Custom GAP Definition

custom_gap = GAPDefinition(
    grantor_criteria=[
        "Full name and parentage",
        "Age and legal capacity",
        "Title documents reference",
        "Power of attorney (if applicable)"
    ],
    attestation_criteria=[
        "Minimum 2 witnesses",
        "Notary seal and signature",
        "Registration number and date",
        "Sub-registrar office details"
    ],
    property_criteria=[
        "Survey number and subdivision",
        "Village and district",
        "Boundaries (North, South, East, West)",
        "Area in square feet/meters",
        "Encumbrances if any"
    ]
)

prompt_custom = factory.create_gap_prompt(
    language=Language.ENGLISH,
    gap_definition=custom_gap,
    clauses=clauses
)


Example 4: Response Validation

try:
    analysis = factory.validate_response(llm_response)
    
    # Check grantor risk
    if analysis['grantor']['risk_level'] == 'high':
        print(f"⚠️ Grantor issues: {analysis['grantor']['issues']}")
    
    # Check overall risk
    if analysis['overall_risk'] in ['high', 'medium']:
        print(f"⚠️ Overall risk: {analysis['overall_risk']}")
        print(f"Missing: {analysis['missing_elements']}")
        
except ValueError as e:
    print(f"Invalid LLM response: {e}")
"""
