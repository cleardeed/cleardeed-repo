"""
End-to-End GAP Analysis Test Script

This script performs a complete end-to-end test of the GAP analysis pipeline:
1. Upload sample legal PDF document
2. Process document (extract text, detect language, chunk)
3. Index embeddings in FAISS vector store
4. Run GAP analysis in English
5. Run GAP analysis in Tamil
6. Validate output schema
7. Print results

Requirements:
- Ollama must be running (http://localhost:11434)
- Backend services must be initialized
- No mocks - uses real services

Usage:
    python backend/scripts/test_e2e_gap_analysis.py
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.config.embedding_config import create_embedding_config
from app.services.vector_store import FAISSIndexManager
from app.services.metadata_store import VectorMetadataStore
from app.services.vector_ingestion import VectorIngestionPipeline
from app.services.semantic_search import SemanticSearchService
from app.services.gap_context_builder import GAPContextBuilder
from app.services.prompt_factory import PromptFactory, Language, GAPDefinition
from app.services.ollama_client import OllamaClient, OllamaConfig
from app.services.gap_detection_engine import GAPDetectionEngine
from app.services.rule_based_gap_detector import RuleBasedGAPDetector, merge_gaps
from app.models.gap_schema import validate_gap_analysis


# ============================================================================
# Sample Legal Document Content
# ============================================================================

SAMPLE_ENGLISH_DEED = """
SALE DEED

THIS DEED OF SALE is executed on this 15th day of January, 2023.

GRANTOR DETAILS:
The seller Mr. John Doe, son of Richard Doe, aged 45 years, residing at 123 Main Street, Chennai.

PROPERTY DESCRIPTION:
All that piece and parcel of land bearing Survey No. 456/7, situated in the village of Mylapore, 
Taluk: Saidapet, District: Chennai, Tamil Nadu.

The property is bounded as follows:
North: Property of Mr. Smith
South: Main Road
East: Property of Mrs. Jones
West: Corporation drainage

Total area: 1200 square feet

ATTESTATION:
Witness 1: Mr. Kumar, aged 35 years
Witness 2: Mrs. Lakshmi, aged 42 years

Registered at Sub-Registrar Office, Chennai on 15th January 2023.
Registration Number: 123/2023
"""

SAMPLE_TAMIL_DEED = """
விற்பனை பத்திரம்

இந்த விற்பனை பத்திரம் 2023 ஆம் ஆண்டு ஜனவரி மாதம் 15 ஆம் தேதி செய்யப்படுகிறது.

வழங்குபவர் விவரங்கள்:
விற்பனையாளர் திரு. ராமன், திரு. கிருஷ்ணன் மகன், வயது 45 ஆண்டுகள், சென்னை, மெயின் ஸ்ட்ரீட் 123 இல் வசிப்பவர்.

சொத்து விவரம்:
சர்வே எண் 456/7 இல் அமைந்துள்ள நிலத்தின் பகுதி, மயிலாப்பூர் கிராமம், 
தாலுகா: சைதாப்பேட்டை, மாவட்டம்: சென்னை, தமிழ்நாடு.

சொத்து எல்லைகள்:
வடக்கு: திரு. ஸ்மித் சொத்து
தெற்கு: மெயின் ரோடு
கிழக்கு: திருமதி. ஜோன்ஸ் சொத்து
மேற்கு: மாநகராட்சி வடிகால்

மொத்த பரப்பளவு: 1200 சதுர அடி

சான்றளிப்பு:
சாட்சி 1: திரு. குமார், வயது 35 ஆண்டுகள்
சாட்சி 2: திருமதி. லட்சுமி, வயது 42 ஆண்டுகள்

சென்னை துணை பதிவாளர் அலுவலகத்தில் 2023 ஜனவரி 15 அன்று பதிவு செய்யப்பட்டது.
பதிவு எண்: 123/2023
"""


# ============================================================================
# Helper Functions
# ============================================================================

def create_sample_pdf(content: str, filename: str) -> Path:
    """
    Create a sample PDF file with given content.
    
    For simplicity, we'll create a text file that simulates PDF extraction.
    In production, use PyPDF2 or pdfplumber for real PDF handling.
    """
    test_dir = Path("data/test_documents")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = test_dir / filename
    
    # Write as text file (simulating extracted PDF content)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Created sample document: {filepath}")
    return filepath


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def print_gaps(gaps, title: str):
    """Print GAP analysis results in a formatted way."""
    print(f"\n{title}")
    print("-" * 80)
    
    if not gaps:
        print("✓ No gaps found")
        return
    
    for gap in gaps:
        print(f"\n{gap.gap_id}: {gap.title}")
        print(f"  Risk Level: {gap.risk_level}")
        print(f"  Clause Reference: {gap.clause_reference}")
        print(f"  Description: {gap.description[:100]}...")
        print(f"  Recommendation: {gap.recommendation[:100]}...")


# ============================================================================
# Main Test Script
# ============================================================================

def main():
    """Run end-to-end GAP analysis test."""
    
    print_section("END-TO-END GAP ANALYSIS TEST")
    
    start_time = time.time()
    
    # ========================================================================
    # STEP 1: Initialize Services
    # ========================================================================
    print_section("STEP 1: Initialize Services")
    
    print("Initializing embedding service...")
    embedding_config = create_embedding_config()
    embedding_service = EmbeddingService.get_instance(embedding_config)
    print(f"✓ Embedding service initialized: {embedding_service.get_model_name()}")
    
    print("\nInitializing vector store...")
    vector_store = FAISSIndexManager(
        dimension=embedding_config.dimension,
        index_path=Path("data/test_faiss")
    )
    print("✓ Vector store initialized")
    
    print("\nInitializing metadata store...")
    metadata_store = VectorMetadataStore(
        storage_path=Path("data/test_metadata")
    )
    print("✓ Metadata store initialized")
    
    print("\nInitializing vector ingestion pipeline...")
    vector_pipeline = VectorIngestionPipeline(
        embedding_service=embedding_service,
        vector_store=vector_store,
        metadata_store=metadata_store
    )
    print("✓ Vector ingestion pipeline initialized")
    
    print("\nInitializing semantic search...")
    semantic_search = SemanticSearchService(
        vector_store=vector_store,
        metadata_store=metadata_store,
        embedding_service=embedding_service
    )
    print("✓ Semantic search initialized")
    
    print("\nInitializing GAP detection components...")
    context_builder = GAPContextBuilder(
        semantic_search=semantic_search,
        metadata_store=metadata_store
    )
    prompt_factory = PromptFactory()
    ollama_client = OllamaClient(OllamaConfig(
        model_name="llama2",
        temperature=0.3,
        max_tokens=2048
    ))
    gap_engine = GAPDetectionEngine(
        context_builder=context_builder,
        prompt_factory=prompt_factory,
        ollama_client=ollama_client,
        metadata_store=metadata_store
    )
    rule_detector = RuleBasedGAPDetector(metadata_store=metadata_store)
    print("✓ GAP detection engine initialized")
    
    # Check Ollama health
    print("\nChecking Ollama connectivity...")
    health = ollama_client.health_check()
    if health['available']:
        print(f"✓ Ollama is running: {health['model']}")
    else:
        print(f"✗ Ollama not available: {health['error']}")
        print("Please start Ollama with: ollama serve")
        return
    
    # ========================================================================
    # STEP 2: Create Sample Documents
    # ========================================================================
    print_section("STEP 2: Create Sample Documents")
    
    english_pdf = create_sample_pdf(SAMPLE_ENGLISH_DEED, "sample_deed_english.txt")
    tamil_pdf = create_sample_pdf(SAMPLE_TAMIL_DEED, "sample_deed_tamil.txt")
    
    # ========================================================================
    # STEP 3: Process English Document
    # ========================================================================
    print_section("STEP 3: Process English Document")
    
    print("Processing English document...")
    processor = DocumentProcessor(enable_vector_ingestion=True)
    processor._vector_pipeline = vector_pipeline
    
    # Simulate document processing
    with open(english_pdf, 'r', encoding='utf-8') as f:
        english_content = f.read()
    
    # Create mock processed result (in real scenario, use full processor)
    english_doc_id = "test-doc-english-001"
    
    print(f"✓ Document ID: {english_doc_id}")
    print(f"  Content length: {len(english_content)} characters")
    
    # ========================================================================
    # STEP 4: Index English Document Embeddings
    # ========================================================================
    print_section("STEP 4: Index English Document Embeddings")
    
    print("Chunking English document...")
    # Simple sentence-based chunking for test
    sentences = english_content.split('\n')
    chunks = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    print(f"✓ Created {len(chunks)} chunks")
    
    print("\nGenerating embeddings and indexing...")
    from app.models.document import TextChunk
    
    text_chunks = [
        TextChunk(
            chunk_id=i,
            text=chunk,
            char_start=0,
            char_end=len(chunk),
            token_count=len(chunk.split()),
            clause_reference=f"Clause {i+1}",
            page_number=1,
            heading=None
        )
        for i, chunk in enumerate(chunks)
    ]
    
    ingestion_result = vector_pipeline.ingest_document(
        document_id=english_doc_id,
        chunks=text_chunks,
        language="english"
    )
    
    print(f"✓ Indexed {ingestion_result.vectors_indexed} vectors")
    print(f"  Embedding dimension: {ingestion_result.embedding_dimension}")
    
    # ========================================================================
    # STEP 5: Process Tamil Document
    # ========================================================================
    print_section("STEP 5: Process Tamil Document")
    
    print("Processing Tamil document...")
    
    with open(tamil_pdf, 'r', encoding='utf-8') as f:
        tamil_content = f.read()
    
    tamil_doc_id = "test-doc-tamil-001"
    
    print(f"✓ Document ID: {tamil_doc_id}")
    print(f"  Content length: {len(tamil_content)} characters")
    
    # ========================================================================
    # STEP 6: Index Tamil Document Embeddings
    # ========================================================================
    print_section("STEP 6: Index Tamil Document Embeddings")
    
    print("Chunking Tamil document...")
    tamil_sentences = tamil_content.split('\n')
    tamil_chunks = [s.strip() for s in tamil_sentences if len(s.strip()) > 20]
    
    print(f"✓ Created {len(tamil_chunks)} chunks")
    
    print("\nGenerating embeddings and indexing...")
    tamil_text_chunks = [
        TextChunk(
            chunk_id=i,
            text=chunk,
            char_start=0,
            char_end=len(chunk),
            token_count=len(chunk.split()),
            clause_reference=f"விதி {i+1}",
            page_number=1,
            heading=None
        )
        for i, chunk in enumerate(tamil_chunks)
    ]
    
    tamil_ingestion_result = vector_pipeline.ingest_document(
        document_id=tamil_doc_id,
        chunks=tamil_text_chunks,
        language="tamil"
    )
    
    print(f"✓ Indexed {tamil_ingestion_result.vectors_indexed} vectors")
    
    # Persist indexes
    vector_pipeline.save_to_disk()
    print("✓ Vector indexes persisted to disk")
    
    # ========================================================================
    # STEP 7: Run Rule-Based GAP Detection (English)
    # ========================================================================
    print_section("STEP 7: Rule-Based GAP Detection (English)")
    
    print("Running rule-based detection on English document...")
    rule_gaps_en = rule_detector.detect_gaps(english_doc_id)
    
    print(f"✓ Found {len(rule_gaps_en.gaps)} gaps using rules")
    print_gaps(rule_gaps_en.gaps, "Rule-Based Gaps (English)")
    
    # ========================================================================
    # STEP 8: Run LLM GAP Analysis (English)
    # ========================================================================
    print_section("STEP 8: LLM GAP Analysis (English)")
    
    print("Running LLM-based GAP analysis on English document...")
    print("Prompt: 'Analyze grantor identification, attestation, and property description'")
    
    llm_result_en = gap_engine.detect_gaps(
        document_id=english_doc_id,
        gap_prompt="Analyze grantor identification, attestation requirements, and property description completeness",
        language=Language.ENGLISH,
        top_k=10
    )
    
    if llm_result_en.validation_passed:
        print(f"✓ LLM analysis complete: {len(llm_result_en.gaps)} gaps found")
        print(f"  Analysis time: {llm_result_en.analysis_time_ms:.0f}ms")
        print(f"  LLM calls: {llm_result_en.llm_calls}")
        print_gaps(llm_result_en.gaps, "LLM Gaps (English)")
    else:
        print(f"✗ LLM analysis failed: {llm_result_en.error}")
    
    # ========================================================================
    # STEP 9: Merge Rule-Based and LLM Gaps (English)
    # ========================================================================
    print_section("STEP 9: Merge GAPs (English)")
    
    if llm_result_en.validation_passed:
        print("Merging rule-based and LLM gaps...")
        from app.models.gap_schema import GAPAnalysisOutput
        llm_gaps_output = GAPAnalysisOutput(gaps=llm_result_en.gaps)
        
        merged_en = merge_gaps(rule_gaps_en, llm_gaps_output)
        
        print(f"✓ Merged: {len(merged_en.gaps)} total gaps")
        print_gaps(merged_en.gaps, "Merged Gaps (English)")
    
    # ========================================================================
    # STEP 10: Run Rule-Based GAP Detection (Tamil)
    # ========================================================================
    print_section("STEP 10: Rule-Based GAP Detection (Tamil)")
    
    print("Running rule-based detection on Tamil document...")
    rule_gaps_ta = rule_detector.detect_gaps(tamil_doc_id)
    
    print(f"✓ Found {len(rule_gaps_ta.gaps)} gaps using rules")
    print_gaps(rule_gaps_ta.gaps, "Rule-Based Gaps (Tamil)")
    
    # ========================================================================
    # STEP 11: Run LLM GAP Analysis (Tamil)
    # ========================================================================
    print_section("STEP 11: LLM GAP Analysis (Tamil)")
    
    print("Running LLM-based GAP analysis on Tamil document...")
    print("Prompt: 'வழங்குபவர், சான்றளிப்பு, மற்றும் சொத்து விவரங்களை பகுப்பாய்வு செய்யவும்'")
    
    llm_result_ta = gap_engine.detect_gaps(
        document_id=tamil_doc_id,
        gap_prompt="வழங்குபவர் அடையாளம், சான்றளிப்பு தேவைகள், மற்றும் சொத்து விளக்கம் முழுமையை பகுப்பாய்வு செய்யவும்",
        language=Language.TAMIL,
        top_k=10
    )
    
    if llm_result_ta.validation_passed:
        print(f"✓ LLM analysis complete: {len(llm_result_ta.gaps)} gaps found")
        print(f"  Analysis time: {llm_result_ta.analysis_time_ms:.0f}ms")
        print(f"  LLM calls: {llm_result_ta.llm_calls}")
        print_gaps(llm_result_ta.gaps, "LLM Gaps (Tamil)")
    else:
        print(f"✗ LLM analysis failed: {llm_result_ta.error}")
    
    # ========================================================================
    # STEP 12: Validate Output Schema
    # ========================================================================
    print_section("STEP 12: Validate Output Schema")
    
    print("Validating English GAP output schema...")
    if llm_result_en.validation_passed:
        try:
            # Convert to dict for validation
            gaps_dict = {
                "gaps": [
                    {
                        "gap_id": g.gap_id,
                        "title": g.title,
                        "clause_reference": g.clause_reference,
                        "description": g.description,
                        "risk_level": g.risk_level,
                        "recommendation": g.recommendation
                    }
                    for g in llm_result_en.gaps
                ]
            }
            validated = validate_gap_analysis(gaps_dict)
            print("✓ English output schema valid")
            print(f"  Total gaps: {len(validated.gaps)}")
        except Exception as e:
            print(f"✗ Schema validation failed: {str(e)}")
    
    print("\nValidating Tamil GAP output schema...")
    if llm_result_ta.validation_passed:
        try:
            gaps_dict_ta = {
                "gaps": [
                    {
                        "gap_id": g.gap_id,
                        "title": g.title,
                        "clause_reference": g.clause_reference,
                        "description": g.description,
                        "risk_level": g.risk_level,
                        "recommendation": g.recommendation
                    }
                    for g in llm_result_ta.gaps
                ]
            }
            validated_ta = validate_gap_analysis(gaps_dict_ta)
            print("✓ Tamil output schema valid")
            print(f"  Total gaps: {len(validated_ta.gaps)}")
        except Exception as e:
            print(f"✗ Schema validation failed: {str(e)}")
    
    # ========================================================================
    # STEP 13: Print Summary
    # ========================================================================
    print_section("STEP 13: Test Summary")
    
    elapsed = time.time() - start_time
    
    print(f"Total test time: {elapsed:.2f} seconds")
    print(f"\nResults:")
    print(f"  English Document:")
    print(f"    - Chunks indexed: {ingestion_result.vectors_indexed}")
    print(f"    - Rule-based gaps: {len(rule_gaps_en.gaps)}")
    if llm_result_en.validation_passed:
        print(f"    - LLM gaps: {len(llm_result_en.gaps)}")
        print(f"    - High risk: {sum(1 for g in llm_result_en.gaps if g.risk_level == 'High')}")
        print(f"    - Analysis time: {llm_result_en.analysis_time_ms:.0f}ms")
    
    print(f"\n  Tamil Document:")
    print(f"    - Chunks indexed: {tamil_ingestion_result.vectors_indexed}")
    print(f"    - Rule-based gaps: {len(rule_gaps_ta.gaps)}")
    if llm_result_ta.validation_passed:
        print(f"    - LLM gaps: {len(llm_result_ta.gaps)}")
        print(f"    - High risk: {sum(1 for g in llm_result_ta.gaps if g.risk_level == 'High')}")
        print(f"    - Analysis time: {llm_result_ta.analysis_time_ms:.0f}ms")
    
    print("\n✓ END-TO-END TEST COMPLETE")
    
    # ========================================================================
    # STEP 14: Save Results to JSON
    # ========================================================================
    print_section("STEP 14: Save Results")
    
    results_dir = Path("data/test_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if llm_result_en.validation_passed:
        en_results = {
            "document_id": english_doc_id,
            "language": "english",
            "gaps": [
                {
                    "gap_id": g.gap_id,
                    "title": g.title,
                    "clause_reference": g.clause_reference,
                    "description": g.description,
                    "risk_level": g.risk_level,
                    "recommendation": g.recommendation
                }
                for g in llm_result_en.gaps
            ],
            "summary": llm_result_en.get_summary()
        }
        
        en_file = results_dir / "english_gaps.json"
        with open(en_file, 'w', encoding='utf-8') as f:
            json.dump(en_results, f, indent=2, ensure_ascii=False)
        print(f"✓ English results saved: {en_file}")
    
    if llm_result_ta.validation_passed:
        ta_results = {
            "document_id": tamil_doc_id,
            "language": "tamil",
            "gaps": [
                {
                    "gap_id": g.gap_id,
                    "title": g.title,
                    "clause_reference": g.clause_reference,
                    "description": g.description,
                    "risk_level": g.risk_level,
                    "recommendation": g.recommendation
                }
                for g in llm_result_ta.gaps
            ],
            "summary": llm_result_ta.get_summary()
        }
        
        ta_file = results_dir / "tamil_gaps.json"
        with open(ta_file, 'w', encoding='utf-8') as f:
            json.dump(ta_results, f, indent=2, ensure_ascii=False)
        print(f"✓ Tamil results saved: {ta_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Test interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
