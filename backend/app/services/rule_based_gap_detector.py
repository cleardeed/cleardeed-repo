"""
Rule-Based GAP Detection

This module implements heuristic-based detection of missing or incomplete information
in legal documents without using LLM. It complements LLM-based analysis by catching
obvious issues through pattern matching and keyword detection.

The rule-based detector:
- Identifies missing mandatory clauses (grantor, attestation, property)
- Detects empty or placeholder text
- Uses language-aware rules (Tamil and English)
- Returns gaps in the same schema as LLM output (can be merged)

Why Rule-Based Detection?
1. Fast: No LLM calls required (milliseconds vs seconds)
2. Reliable: Deterministic rules for common patterns
3. Complementary: Catches obvious issues LLM might miss
4. Offline: Works without Ollama connection

Example Usage:
    >>> detector = RuleBasedGAPDetector(metadata_store)
    >>> result = detector.detect_gaps(document_id="doc-123")
    >>> print(f"Found {len(result.gaps)} issues using rules")
"""

import logging
import re
from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass

from app.models.gap_schema import (
    GAPAnalysisOutput,
    GAPItem,
    RiskLevel
)

logger = logging.getLogger(__name__)


@dataclass
class RuleMatch:
    """Result of a rule evaluation."""
    matched: bool
    gap_title: str
    description: str
    recommendation: str
    risk_level: RiskLevel
    clause_reference: Optional[str] = None


class RuleBasedGAPDetector:
    """
    Rule-based GAP detector using heuristics and pattern matching.
    
    This detector analyzes document text for common issues:
    1. Missing mandatory sections (grantor, attestation, property)
    2. Empty or very short clauses
    3. Placeholder text (N/A, TBD, [blank], etc.)
    4. Missing critical information (names, dates, measurements)
    
    Rules are language-aware and support both English and Tamil.
    
    Example:
        >>> detector = RuleBasedGAPDetector(metadata_store)
        >>> gaps = detector.detect_gaps(document_id="doc-abc123")
        >>> # Returns GAPAnalysisOutput compatible with LLM output
        >>> for gap in gaps.gaps:
        ...     print(f"{gap.gap_id}: {gap.title}")
    """
    
    # Common placeholder patterns (case-insensitive)
    PLACEHOLDER_PATTERNS = [
        r'\b(n/?a|not applicable|nil)\b',
        r'\b(tbd|to be determined|pending)\b',
        r'\[\s*blank\s*\]',
        r'\[\s*\]',
        r'_{3,}',  # Multiple underscores
        r'\.{3,}',  # Multiple dots
        r'\(\s*\)',  # Empty parentheses
    ]
    
    # Grantor keywords (English)
    GRANTOR_KEYWORDS_EN = [
        'grantor', 'seller', 'vendor', 'transferor',
        'first party', 'party of the first part',
        'executant', 'conveyor'
    ]
    
    # Grantor keywords (Tamil)
    GRANTOR_KEYWORDS_TA = [
        'வழங்குபவர்', 'விற்பவர்', 'மாற்றுபவர்',
        'முதல் தரப்பினர்', 'விற்பனையாளர்'
    ]
    
    # Attestation keywords (English)
    ATTESTATION_KEYWORDS_EN = [
        'witness', 'attestation', 'notary', 'signature',
        'registered', 'registration', 'sub-registrar',
        'executed', 'signed', 'sworn'
    ]
    
    # Attestation keywords (Tamil)
    ATTESTATION_KEYWORDS_TA = [
        'சாட்சி', 'சான்றளிப்பு', 'நோட்டரி', 'கையெழுத்து',
        'பதிவு', 'பதிவாளர்', 'துணை பதிவாளர்',
        'உறுதிப்படுத்தல்'
    ]
    
    # Property keywords (English)
    PROPERTY_KEYWORDS_EN = [
        'property', 'land', 'plot', 'survey',
        'boundary', 'boundaries', 'measurement', 'area',
        'square feet', 'sq ft', 'acres', 'cents',
        'north', 'south', 'east', 'west'
    ]
    
    # Property keywords (Tamil)
    PROPERTY_KEYWORDS_TA = [
        'சொத்து', 'நிலம்', 'மனை', 'சர்வே',
        'எல்லை', 'எல்லைகள்', 'அளவீடு', 'பரப்பளவு',
        'சதுர அடி', 'ஏக்கர்', 'சென்ட்',
        'வடக்கு', 'தெற்கு', 'கிழக்கு', 'மேற்கு'
    ]
    
    def __init__(self, metadata_store):
        """
        Initialize rule-based detector.
        
        Args:
            metadata_store: VectorMetadataStore for retrieving document chunks
        """
        self.metadata_store = metadata_store
        
        # Compile regex patterns for performance
        self.placeholder_regex = re.compile(
            '|'.join(self.PLACEHOLDER_PATTERNS),
            re.IGNORECASE
        )
        
        logger.info("Initialized RuleBasedGAPDetector")
    
    def detect_gaps(
        self,
        document_id: str,
        min_clause_length: int = 20
    ) -> GAPAnalysisOutput:
        """
        Detect gaps using rule-based heuristics.
        
        Args:
            document_id: Document UUID to analyze
            min_clause_length: Minimum clause length to not be considered "empty"
        
        Returns:
            GAPAnalysisOutput: Detected gaps in standard schema
        
        Example:
            >>> gaps = detector.detect_gaps("doc-123")
            >>> if gaps.gaps:
            ...     print(f"Found {len(gaps.gaps)} issues")
        """
        logger.info(f"[{document_id}] Starting rule-based GAP detection")
        
        # Get all document chunks
        chunks = self.metadata_store.get_by_document(document_id)
        
        if not chunks:
            logger.warning(f"[{document_id}] No chunks found for document")
            return GAPAnalysisOutput(gaps=[])
        
        logger.info(f"[{document_id}] Analyzing {len(chunks)} chunks")
        
        # Detect language (majority wins)
        language = self._detect_language(chunks)
        logger.info(f"[{document_id}] Detected language: {language}")
        
        # Collect all gaps
        gaps: List[GAPItem] = []
        gap_counter = 1
        
        # Rule 1: Check for missing mandatory sections
        missing_sections = self._check_missing_sections(chunks, language)
        for section_gap in missing_sections:
            if section_gap.matched:
                gaps.append(self._create_gap_item(
                    gap_id=f"RULE-{gap_counter:03d}",
                    title=section_gap.gap_title,
                    clause_reference=section_gap.clause_reference or "Document-wide",
                    description=section_gap.description,
                    risk_level=section_gap.risk_level,
                    recommendation=section_gap.recommendation
                ))
                gap_counter += 1
        
        # Rule 2: Check for placeholder text
        placeholder_gaps = self._check_placeholders(chunks, language)
        for placeholder_gap in placeholder_gaps:
            gaps.append(self._create_gap_item(
                gap_id=f"RULE-{gap_counter:03d}",
                title=placeholder_gap.gap_title,
                clause_reference=placeholder_gap.clause_reference or "Unknown",
                description=placeholder_gap.description,
                risk_level=placeholder_gap.risk_level,
                recommendation=placeholder_gap.recommendation
            ))
            gap_counter += 1
        
        # Rule 3: Check for empty or very short clauses
        empty_gaps = self._check_empty_clauses(chunks, min_clause_length, language)
        for empty_gap in empty_gaps:
            gaps.append(self._create_gap_item(
                gap_id=f"RULE-{gap_counter:03d}",
                title=empty_gap.gap_title,
                clause_reference=empty_gap.clause_reference or "Unknown",
                description=empty_gap.description,
                risk_level=empty_gap.risk_level,
                recommendation=empty_gap.recommendation
            ))
            gap_counter += 1
        
        logger.info(
            f"[{document_id}] ✓ Rule-based detection complete: "
            f"{len(gaps)} gaps found"
        )
        
        return GAPAnalysisOutput(gaps=gaps)
    
    def _detect_language(self, chunks: List) -> str:
        """
        Detect primary language from chunks.
        
        Args:
            chunks: List of VectorMetadata objects
        
        Returns:
            str: "tamil" or "english"
        """
        tamil_count = sum(1 for c in chunks if c.language == "tamil")
        english_count = len(chunks) - tamil_count
        
        return "tamil" if tamil_count > english_count else "english"
    
    def _check_missing_sections(
        self,
        chunks: List,
        language: str
    ) -> List[RuleMatch]:
        """
        Check for missing mandatory GAP sections.
        
        Looks for presence of grantor, attestation, and property keywords
        in the document. If keywords are absent, section is likely missing.
        
        Args:
            chunks: List of VectorMetadata objects
            language: Document language
        
        Returns:
            List[RuleMatch]: Matches for missing sections
        """
        # Combine all text
        full_text = " ".join(c.text_preview for c in chunks)
        full_text_lower = full_text.lower()
        
        # Select keywords based on language
        if language == "tamil":
            grantor_kw = self.GRANTOR_KEYWORDS_TA
            attestation_kw = self.ATTESTATION_KEYWORDS_TA
            property_kw = self.PROPERTY_KEYWORDS_TA
        else:
            grantor_kw = self.GRANTOR_KEYWORDS_EN
            attestation_kw = self.ATTESTATION_KEYWORDS_EN
            property_kw = self.PROPERTY_KEYWORDS_EN
        
        matches = []
        
        # Check grantor
        grantor_found = any(kw.lower() in full_text_lower for kw in grantor_kw)
        if not grantor_found:
            matches.append(RuleMatch(
                matched=True,
                gap_title="Missing grantor information" if language == "english" else "வழங்குபவர் தகவல் காணவில்லை",
                description=(
                    "Document does not contain clear grantor/seller information. "
                    "No keywords found: " + ", ".join(grantor_kw[:3])
                ) if language == "english" else (
                    "ஆவணத்தில் வழங்குபவர்/விற்பவர் தகவல் தெளிவாக இல்லை. "
                    "முக்கிய சொற்கள் காணவில்லை: " + ", ".join(grantor_kw[:3])
                ),
                recommendation=(
                    "Add clear grantor identification section with name, parentage, "
                    "and legal authority to transfer"
                ) if language == "english" else (
                    "பெயர், தந்தை பெயர், மற்றும் சட்டப்படி மாற்றும் அதிகாரத்துடன் "
                    "தெளிவான வழங்குபவர் விவரங்களைச் சேர்க்கவும்"
                ),
                risk_level=RiskLevel.HIGH,
                clause_reference="Document-wide"
            ))
        
        # Check attestation
        attestation_found = any(kw.lower() in full_text_lower for kw in attestation_kw)
        if not attestation_found:
            matches.append(RuleMatch(
                matched=True,
                gap_title="Missing attestation section" if language == "english" else "சான்றளிப்பு பகுதி காணவில்லை",
                description=(
                    "Document lacks attestation/witness section. "
                    "No keywords found: " + ", ".join(attestation_kw[:3])
                ) if language == "english" else (
                    "ஆவணத்தில் சான்றளிப்பு/சாட்சி பகுதி இல்லை. "
                    "முக்கிய சொற்கள் காணவில்லை: " + ", ".join(attestation_kw[:3])
                ),
                recommendation=(
                    "Add attestation section with witness signatures, notarization, "
                    "and registration details"
                ) if language == "english" else (
                    "சாட்சி கையெழுத்துகள், நோட்டரி உறுதிப்படுத்தல், "
                    "மற்றும் பதிவு விவரங்களுடன் சான்றளிப்பு பகுதியைச் சேர்க்கவும்"
                ),
                risk_level=RiskLevel.HIGH,
                clause_reference="Document-wide"
            ))
        
        # Check property description
        property_found = any(kw.lower() in full_text_lower for kw in property_kw)
        if not property_found:
            matches.append(RuleMatch(
                matched=True,
                gap_title="Missing property description" if language == "english" else "சொத்து விளக்கம் காணவில்லை",
                description=(
                    "Document lacks clear property description. "
                    "No keywords found: " + ", ".join(property_kw[:3])
                ) if language == "english" else (
                    "ஆவணத்தில் தெளிவான சொத்து விளக்கம் இல்லை. "
                    "முக்கிய சொற்கள் காணவில்லை: " + ", ".join(property_kw[:3])
                ),
                recommendation=(
                    "Add detailed property description with survey numbers, "
                    "boundaries, measurements, and location"
                ) if language == "english" else (
                    "சர்வே எண்கள், எல்லைகள், அளவீடுகள், மற்றும் இடத்துடன் "
                    "விரிவான சொத்து விளக்கத்தைச் சேர்க்கவும்"
                ),
                risk_level=RiskLevel.HIGH,
                clause_reference="Document-wide"
            ))
        
        return matches
    
    def _check_placeholders(
        self,
        chunks: List,
        language: str
    ) -> List[RuleMatch]:
        """
        Check for placeholder text that needs to be filled in.
        
        Detects patterns like: N/A, TBD, [blank], ___, etc.
        
        Args:
            chunks: List of VectorMetadata objects
            language: Document language
        
        Returns:
            List[RuleMatch]: Matches for placeholder text
        """
        matches = []
        
        for chunk in chunks:
            text = chunk.text_preview
            
            # Find all placeholder matches
            placeholder_matches = list(self.placeholder_regex.finditer(text))
            
            if placeholder_matches:
                # Get context around first match
                first_match = placeholder_matches[0]
                context_start = max(0, first_match.start() - 30)
                context_end = min(len(text), first_match.end() + 30)
                context = text[context_start:context_end]
                
                clause_ref = chunk.clause_reference or f"Chunk {chunk.chunk_id}"
                
                matches.append(RuleMatch(
                    matched=True,
                    gap_title="Placeholder text found" if language == "english" else "இடங்காட்டி உரை கண்டறியப்பட்டது",
                    clause_reference=clause_ref,
                    description=(
                        f"Clause contains placeholder text that needs to be completed. "
                        f"Found {len(placeholder_matches)} placeholder(s). "
                        f"Context: ...{context}..."
                    ) if language == "english" else (
                        f"விதியில் நிரப்ப வேண்டிய இடங்காட்டி உரை உள்ளது. "
                        f"{len(placeholder_matches)} இடங்காட்டிகள் கண்டறியப்பட்டன. "
                        f"சூழல்: ...{context}..."
                    ),
                    recommendation=(
                        "Replace placeholder text with actual information"
                    ) if language == "english" else (
                        "இடங்காட்டி உரையை உண்மையான தகவலுடன் மாற்றவும்"
                    ),
                    risk_level=RiskLevel.MEDIUM
                ))
        
        return matches
    
    def _check_empty_clauses(
        self,
        chunks: List,
        min_length: int,
        language: str
    ) -> List[RuleMatch]:
        """
        Check for empty or very short clauses.
        
        Args:
            chunks: List of VectorMetadata objects
            min_length: Minimum acceptable clause length
            language: Document language
        
        Returns:
            List[RuleMatch]: Matches for empty clauses
        """
        matches = []
        
        for chunk in chunks:
            text = chunk.text_preview.strip()
            
            # Check if clause is too short
            if len(text) < min_length and len(text) > 0:
                clause_ref = chunk.clause_reference or f"Chunk {chunk.chunk_id}"
                
                matches.append(RuleMatch(
                    matched=True,
                    gap_title="Very short clause" if language == "english" else "மிகக் குறுகிய விதி",
                    clause_reference=clause_ref,
                    description=(
                        f"Clause is unusually short ({len(text)} characters). "
                        f"Content: '{text}'. This may indicate incomplete information."
                    ) if language == "english" else (
                        f"விதி அசாதாரணமாக குறுகியது ({len(text)} எழுத்துக்கள்). "
                        f"உள்ளடக்கம்: '{text}'. இது முழுமையற்ற தகவலைக் குறிக்கலாம்."
                    ),
                    recommendation=(
                        "Review and expand clause with complete information"
                    ) if language == "english" else (
                        "மீண்டும் பார்த்து முழுமையான தகவலுடன் விதியை விரிவாக்கவும்"
                    ),
                    risk_level=RiskLevel.LOW
                ))
        
        return matches
    
    def _create_gap_item(
        self,
        gap_id: str,
        title: str,
        clause_reference: str,
        description: str,
        risk_level: RiskLevel,
        recommendation: str
    ) -> GAPItem:
        """
        Create a GAPItem object.
        
        Ensures all gaps follow the same schema as LLM-generated gaps.
        
        Args:
            gap_id: Unique identifier (RULE-XXX format)
            title: Short title
            clause_reference: Reference to clause
            description: Detailed description
            risk_level: Risk level enum
            recommendation: Suggested action
        
        Returns:
            GAPItem: Validated gap item
        """
        return GAPItem(
            gap_id=gap_id,
            title=title,
            clause_reference=clause_reference,
            description=description,
            risk_level=risk_level,
            recommendation=recommendation
        )


def merge_gaps(
    rule_based_gaps: GAPAnalysisOutput,
    llm_gaps: GAPAnalysisOutput
) -> GAPAnalysisOutput:
    """
    Merge rule-based and LLM-based gaps into a single unified output.
    
    Merge Strategy:
    
    1. **Prioritization**: LLM gaps are preferred over rule-based gaps because:
       - LLM has semantic understanding of context
       - LLM provides more detailed descriptions
       - LLM can identify nuanced issues rules can't catch
    
    2. **Deduplication**: Prevents showing same issue twice by:
       - Comparing titles using fuzzy matching (key word overlap)
       - Comparing clause references (same location = likely duplicate)
       - Keeping LLM version when duplicate found
    
    3. **Complementary Coverage**: Rule-based gaps add value by:
       - Catching obvious issues LLM might miss (placeholders, empty text)
       - Providing instant feedback without LLM latency
       - Working offline when Ollama unavailable
    
    4. **Consistent Numbering**: Reassigns gap_id to maintain sequence:
       - GAP-001, GAP-002, GAP-003... (not mixed RULE-/GAP- prefixes)
       - Sorted by risk level: High → Medium → Low
       - Within same risk level, sorted by clause reference
    
    5. **Risk Ordering**: Ensures critical issues appear first:
       - High-risk gaps at top (immediate attention needed)
       - Medium-risk gaps in middle (should be addressed)
       - Low-risk gaps at bottom (minor improvements)
    
    Args:
        rule_based_gaps: Gaps from rule-based heuristic detector
        llm_gaps: Gaps from LLM semantic analysis
    
    Returns:
        GAPAnalysisOutput: Merged, deduplicated, and sorted gaps
    
    Example:
        >>> rule_gaps = rule_detector.detect_gaps("doc-123")
        >>> llm_gaps = llm_engine.detect_gaps("doc-123", "Analyze grantor")
        >>> merged = merge_gaps(rule_gaps, llm_gaps)
        >>> print(f"Total: {len(merged.gaps)} gaps (High: {sum(1 for g in merged.gaps if g.risk_level == 'High')})")
    """
    logger.info(
        f"Starting GAP merge: {len(rule_based_gaps.gaps)} rule-based + "
        f"{len(llm_gaps.gaps)} LLM"
    )
    
    # Step 1: Start with all LLM gaps (priority)
    # LLM gaps are more detailed and contextually aware
    merged_gaps = list(llm_gaps.gaps)
    
    logger.debug(f"Added {len(llm_gaps.gaps)} LLM gaps to merge pool")
    
    # Step 2: Add non-duplicate rule-based gaps
    # Build index of LLM gaps for efficient duplicate detection
    llm_gap_index = _build_gap_index(llm_gaps.gaps)
    
    added_count = 0
    duplicate_count = 0
    
    for rule_gap in rule_based_gaps.gaps:
        # Check if this rule gap is a duplicate of any LLM gap
        if _is_duplicate(rule_gap, llm_gap_index):
            duplicate_count += 1
            logger.debug(
                f"Skipped duplicate: {rule_gap.gap_id} - {rule_gap.title} "
                f"(already covered by LLM)"
            )
        else:
            merged_gaps.append(rule_gap)
            added_count += 1
            logger.debug(
                f"Added unique rule gap: {rule_gap.gap_id} - {rule_gap.title}"
            )
    
    logger.info(
        f"Deduplication complete: added {added_count} rule gaps, "
        f"skipped {duplicate_count} duplicates"
    )
    
    # Step 3: Sort by risk level (High → Medium → Low)
    # Within same risk level, sort by clause reference for consistency
    risk_order = {RiskLevel.HIGH: 0, RiskLevel.MEDIUM: 1, RiskLevel.LOW: 2}
    
    sorted_gaps = sorted(
        merged_gaps,
        key=lambda g: (
            risk_order.get(g.risk_level, 3),  # Primary: risk level
            g.clause_reference or "zzz"  # Secondary: clause reference
        )
    )
    
    logger.debug(
        f"Sorted {len(sorted_gaps)} gaps by risk level: "
        f"High={sum(1 for g in sorted_gaps if g.risk_level == RiskLevel.HIGH)}, "
        f"Medium={sum(1 for g in sorted_gaps if g.risk_level == RiskLevel.MEDIUM)}, "
        f"Low={sum(1 for g in sorted_gaps if g.risk_level == RiskLevel.LOW)}"
    )
    
    # Step 4: Reassign gap_id with consistent numbering
    # Convert all gap IDs to GAP-001, GAP-002, etc. format
    renumbered_gaps = []
    
    for i, gap in enumerate(sorted_gaps, start=1):
        # Create new gap with updated ID
        renumbered_gap = GAPItem(
            gap_id=f"GAP-{i:03d}",  # GAP-001, GAP-002, etc.
            title=gap.title,
            clause_reference=gap.clause_reference,
            description=gap.description,
            risk_level=gap.risk_level,
            recommendation=gap.recommendation
        )
        renumbered_gaps.append(renumbered_gap)
    
    logger.info(
        f"✓ Merge complete: {len(renumbered_gaps)} total gaps "
        f"(renumbered GAP-001 to GAP-{len(renumbered_gaps):03d})"
    )
    
    return GAPAnalysisOutput(gaps=renumbered_gaps)


def _build_gap_index(gaps: List[GAPItem]) -> Dict[str, List[GAPItem]]:
    """
    Build an index of gaps for efficient duplicate detection.
    
    Creates multiple indexes:
    - By normalized title (key words only)
    - By clause reference
    
    This allows O(1) duplicate lookups instead of O(n) comparisons.
    
    Args:
        gaps: List of gap items to index
    
    Returns:
        dict: Index with keys: 'by_title_words', 'by_clause'
    """
    index = {
        'by_title_words': {},  # {word: [gaps containing word]}
        'by_clause': {}  # {clause_ref: [gaps for that clause]}
    }
    
    for gap in gaps:
        # Index by title words
        key_words = _extract_key_words(gap.title)
        for word in key_words:
            if word not in index['by_title_words']:
                index['by_title_words'][word] = []
            index['by_title_words'][word].append(gap)
        
        # Index by clause reference
        if gap.clause_reference:
            clause_key = gap.clause_reference.lower().strip()
            if clause_key not in index['by_clause']:
                index['by_clause'][clause_key] = []
            index['by_clause'][clause_key].append(gap)
    
    return index


def _is_duplicate(gap: GAPItem, gap_index: Dict[str, List[GAPItem]]) -> bool:
    """
    Check if gap is a duplicate of any indexed gap.
    
    Duplicate Detection Strategy:
    
    1. **Title Similarity** (primary check):
       - Extract key words from title (ignore common words)
       - If 2+ key words match, consider similar
       - Example: "Missing grantor name" vs "Grantor identification missing"
    
    2. **Clause Reference Match** (secondary check):
       - If same clause reference, likely duplicate
       - Example: Both refer to "Clause 1"
    
    3. **Combined Match** (strongest signal):
       - If both title similar AND clause matches, definitely duplicate
    
    Args:
        gap: Gap item to check
        gap_index: Pre-built index of existing gaps
    
    Returns:
        bool: True if duplicate found, False if unique
    """
    # Extract key words from gap title
    gap_key_words = _extract_key_words(gap.title)
    
    if not gap_key_words:
        # No key words to compare, can't determine similarity
        return False
    
    # Check title similarity using word overlap
    matching_gaps = set()
    
    for word in gap_key_words:
        if word in gap_index['by_title_words']:
            matching_gaps.update(gap_index['by_title_words'][word])
    
    if not matching_gaps:
        # No title overlap, definitely not duplicate
        return False
    
    # Check each matching gap for sufficient similarity
    for existing_gap in matching_gaps:
        existing_words = _extract_key_words(existing_gap.title)
        
        # Count word overlap
        overlap = len(gap_key_words & existing_words)
        total_words = len(gap_key_words | existing_words)
        
        if total_words == 0:
            continue
        
        # Calculate similarity ratio
        similarity = overlap / total_words
        
        # Threshold: 40% word overlap = similar enough
        if similarity >= 0.4:
            logger.debug(
                f"Found similar title: '{gap.title}' ≈ '{existing_gap.title}' "
                f"({similarity:.0%} similarity)"
            )
            return True
        
        # Also check clause reference match
        if gap.clause_reference and existing_gap.clause_reference:
            gap_clause = gap.clause_reference.lower().strip()
            existing_clause = existing_gap.clause_reference.lower().strip()
            
            # If same clause AND some word overlap, likely duplicate
            if gap_clause == existing_clause and overlap > 0:
                logger.debug(
                    f"Found clause match: '{gap.title}' and '{existing_gap.title}' "
                    f"both reference '{gap.clause_reference}'"
                )
                return True
    
    # No duplicates found, gap is unique
    return False


def _extract_key_words(title: str) -> Set[str]:
    """
    Extract key words from title for similarity comparison.
    
    Key words are:
    - Longer than 3 characters (ignore short words like "the", "and")
    - Not common filler words (missing, incomplete, etc.)
    - Normalized to lowercase
    
    Args:
        title: Gap title to extract words from
    
    Returns:
        set: Set of key words
    """
    title_lower = title.lower()
    
    # Common filler words to ignore (English + Tamil)
    stop_words = {
        'missing', 'incomplete', 'lack', 'lacks', 'absent',
        'found', 'section', 'clause', 'information', 'details',
        'காணவில்லை', 'இல்லை', 'குறிப்பிடப்படவில்லை'
    }
    
    # Extract words
    words = re.findall(r'\w+', title_lower)
    
    # Filter key words
    key_words = {
        word for word in words
        if len(word) > 3 and word not in stop_words
    }
    
    return key_words


def _is_similar_to_any(title: str, existing_titles: Set[str]) -> bool:
    """
    Check if title is similar to any existing title.
    
    DEPRECATED: Use _is_duplicate() with gap_index for better performance.
    This function kept for backward compatibility.
    
    Uses simple substring matching for deduplication.
    
    Args:
        title: Title to check
        existing_titles: Set of existing titles (lowercase)
    
    Returns:
        bool: True if similar title exists
    """
    key_words = _extract_key_words(title)
    
    if not key_words:
        return False
    
    # Check if any key word appears in existing titles
    for existing in existing_titles:
        for word in key_words:
            if word in existing:
                return True
    
    return False


# Example usage
"""
Example 1: Rule-based detection only

from app.services.metadata_store import VectorMetadataStore

metadata_store = VectorMetadataStore()
detector = RuleBasedGAPDetector(metadata_store)

# Detect gaps using rules (no LLM)
gaps = detector.detect_gaps(document_id="doc-abc123")

print(f"Found {len(gaps.gaps)} issues using rules")
for gap in gaps.gaps:
    print(f"{gap.gap_id}: {gap.title} (Risk: {gap.risk_level})")


Example 2: Merge with LLM gaps

# Get rule-based gaps (fast, no LLM)
rule_gaps = detector.detect_gaps("doc-123")

# Get LLM gaps (slower, more accurate)
llm_gaps = llm_engine.detect_gaps("doc-123", "Analyze grantor info")

# Merge both (best of both worlds)
merged = merge_gaps(rule_gaps, llm_gaps)

print(f"Total gaps: {len(merged.gaps)}")
print(f"  - {len([g for g in merged.gaps if 'RULE' in g.gap_id])} from rules")
print(f"  - {len([g for g in merged.gaps if 'GAP' in g.gap_id])} from LLM")


Example 3: Fast pre-screening

# Use rules for quick pre-screening
rule_gaps = detector.detect_gaps("doc-new")

if len(rule_gaps.gaps) > 0:
    print("⚠️ Document has obvious issues, running full LLM analysis...")
    llm_gaps = llm_engine.detect_gaps("doc-new", "Full analysis")
    final = merge_gaps(rule_gaps, llm_gaps)
else:
    print("✓ No obvious issues found with rules")
    final = rule_gaps


Example 4: Language-specific detection

# English document
gaps_en = detector.detect_gaps("doc-english")
# Uses English keywords and messages

# Tamil document  
gaps_ta = detector.detect_gaps("doc-tamil")
# Uses Tamil keywords and messages


Example 5: Custom minimum clause length

# Stricter empty clause detection
gaps_strict = detector.detect_gaps(
    document_id="doc-123",
    min_clause_length=50  # Flag clauses < 50 chars
)
"""
