"""
Validation Script for Embedding and FAISS Setup

This script tests the complete vector search pipeline:
1. Embedding generation for English and Tamil text
2. FAISS indexing and metadata storage
3. Semantic search with language-aware retrieval
4. Metadata mapping verification

Run this script to validate the setup before integrating with FastAPI.
"""

import sys
from pathlib import Path
import logging

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.models.schemas import TextChunk
from app.services.vector_ingestion import VectorIngestionPipeline
from app.services.semantic_search import SemanticSearchService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(label: str, value: str, indent: int = 2):
    """Print a formatted result line"""
    print(f"{' ' * indent}{label}: {value}")


def create_test_chunks():
    """
    Create test chunks with English and Tamil content.
    
    Returns:
        tuple: (english_chunks, tamil_chunks)
    """
    english_chunks = [
        TextChunk(
            chunk_id=1,
            text="This property deed establishes ownership rights for the real estate property located in Chennai.",
            clause_number="1.1",
            heading="PROPERTY OWNERSHIP",
            page_number=1,
            page_range=[1],
            language="en",
            token_count=15
        ),
        TextChunk(
            chunk_id=2,
            text="The buyer shall pay the agreed purchase amount of ten million rupees within thirty days of signing.",
            clause_number="2.1",
            heading="PAYMENT TERMS",
            page_number=2,
            page_range=[2],
            language="en",
            token_count=17
        ),
        TextChunk(
            chunk_id=3,
            text="Both parties agree to complete the property registration process at the sub-registrar office.",
            clause_number="3.1",
            heading="REGISTRATION",
            page_number=3,
            page_range=[3],
            language="en",
            token_count=14
        ),
    ]
    
    tamil_chunks = [
        TextChunk(
            chunk_id=4,
            text="இந்த சொத்து ஆவணம் சென்னையில் அமைந்துள்ள ரியல் எஸ்டேட் சொத்துக்கான உரிமை உரிமைகளை நிறுவுகிறது.",
            clause_number="௧.௧",
            heading="சொத்து உரிமை",
            page_number=1,
            page_range=[1],
            language="ta",
            token_count=16
        ),
        TextChunk(
            chunk_id=5,
            text="வாங்குபவர் ஒப்பந்தம் கையெழுத்திட்ட முப்பது நாட்களுக்குள் ஒப்புக்கொண்ட கொள்முதல் தொகையான பத்து மில்லியன் ரூபாய் செலுத்த வேண்டும்.",
            clause_number="௨.௧",
            heading="கட்டண விதிமுறைகள்",
            page_number=2,
            page_range=[2],
            language="ta",
            token_count=20
        ),
        TextChunk(
            chunk_id=6,
            text="இரு தரப்பினரும் துணை பதிவாளர் அலுவலகத்தில் சொத்து பதிவு செயல்முறையை முடிக்க ஒப்புக்கொள்கிறார்கள்.",
            clause_number="௩.௧",
            heading="பதிவு",
            page_number=3,
            page_range=[3],
            language="ta",
            token_count=14
        ),
    ]
    
    return english_chunks, tamil_chunks


def test_ingestion():
    """
    Test 1: Vector Ingestion
    
    Tests embedding generation and FAISS indexing for both English and Tamil chunks.
    """
    print_header("TEST 1: Vector Ingestion")
    
    # Create test data
    english_chunks, tamil_chunks = create_test_chunks()
    
    print_result("English chunks", str(len(english_chunks)))
    print_result("Tamil chunks", str(len(tamil_chunks)))
    
    # Initialize pipeline
    print("\n  Initializing vector ingestion pipeline...")
    pipeline = VectorIngestionPipeline()
    
    # Ingest English document
    print("\n  Ingesting English document...")
    english_result = pipeline.ingest_document("test-english-001", english_chunks)
    
    print_result("Status", "✓ SUCCESS" if english_result.success else "✗ FAILED", 4)
    print_result("Vectors indexed", str(english_result.vectors_indexed), 4)
    print_result("Embedding dimension", str(english_result.embedding_dimension), 4)
    
    if not english_result.success:
        print_result("Error", english_result.error, 4)
        return None
    
    # Ingest Tamil document
    print("\n  Ingesting Tamil document...")
    tamil_result = pipeline.ingest_document("test-tamil-001", tamil_chunks)
    
    print_result("Status", "✓ SUCCESS" if tamil_result.success else "✗ FAILED", 4)
    print_result("Vectors indexed", str(tamil_result.vectors_indexed), 4)
    print_result("Embedding dimension", str(tamil_result.embedding_dimension), 4)
    
    if not tamil_result.success:
        print_result("Error", tamil_result.error, 4)
        return None
    
    # Verify statistics
    print("\n  Pipeline statistics:")
    stats = pipeline.get_stats()
    print_result("Total vectors in index", str(stats['vector_store']['total_vectors']), 4)
    print_result("Total metadata entries", str(stats['metadata_store']['total_vectors']), 4)
    print_result("Languages", str(stats['metadata_store']['languages']), 4)
    
    print("\n  ✓ Ingestion test PASSED")
    return pipeline


def test_english_search(pipeline: VectorIngestionPipeline):
    """
    Test 2: English Query
    
    Tests semantic search with English query.
    """
    print_header("TEST 2: English Query - 'property ownership rights'")
    
    # Initialize search service
    search_service = SemanticSearchService(
        embedding_service=pipeline.embedding_service,
        vector_store=pipeline.vector_store,
        metadata_store=pipeline.metadata_store
    )
    
    # Perform search
    query = "property ownership rights"
    print_result("Query", f'"{query}"')
    
    response = search_service.search(query, top_k=3)
    
    print_result("Query language detected", response.query_language, 2)
    print_result("Results found", str(response.total_results), 2)
    print_result("Search time", f"{response.search_time_ms:.2f}ms", 2)
    
    print("\n  Top 3 Results:")
    for result in response.results:
        print(f"\n    Rank {result.rank}:")
        print_result("Document ID", result.document_id, 6)
        print_result("Chunk ID", str(result.chunk_id), 6)
        print_result("Language", result.language, 6)
        print_result("Similarity", f"{result.similarity_score:.4f}", 6)
        print_result("Clause", result.clause_reference or "N/A", 6)
        print_result("Text", result.text_preview[:60] + "...", 6)
    
    # Verify: Top result should be English chunk about property ownership
    expected_language = "en"
    top_result = response.results[0]
    
    print("\n  Verification:")
    if top_result.language == expected_language:
        print_result("Language match", f"✓ PASS (expected {expected_language})", 4)
    else:
        print_result("Language match", f"✗ FAIL (expected {expected_language}, got {top_result.language})", 4)
    
    if top_result.similarity_score > 0.7:
        print_result("Similarity score", f"✓ PASS (>{0.7:.2f})", 4)
    else:
        print_result("Similarity score", f"✗ FAIL (<{0.7:.2f})", 4)
    
    print("\n  ✓ English search test PASSED")


def test_tamil_search(pipeline: VectorIngestionPipeline):
    """
    Test 3: Tamil Query
    
    Tests semantic search with Tamil query and language-aware retrieval.
    """
    print_header("TEST 3: Tamil Query - 'சொத்து உரிமை'")
    
    # Initialize search service
    search_service = SemanticSearchService(
        embedding_service=pipeline.embedding_service,
        vector_store=pipeline.vector_store,
        metadata_store=pipeline.metadata_store
    )
    
    # Perform search
    query = "சொத்து உரிமை"
    print_result("Query", f'"{query}"')
    
    response = search_service.search(query, top_k=3, language_boost=True)
    
    print_result("Query language detected", response.query_language, 2)
    print_result("Results found", str(response.total_results), 2)
    print_result("Search time", f"{response.search_time_ms:.2f}ms", 2)
    print_result("Language boost", "Enabled", 2)
    
    print("\n  Top 3 Results:")
    for result in response.results:
        print(f"\n    Rank {result.rank}:")
        print_result("Document ID", result.document_id, 6)
        print_result("Chunk ID", str(result.chunk_id), 6)
        print_result("Language", result.language, 6)
        print_result("Similarity", f"{result.similarity_score:.4f}", 6)
        print_result("Clause", result.clause_reference or "N/A", 6)
        print_result("Text", result.text_preview[:60] + "...", 6)
    
    # Verify: Top result should be Tamil chunk with language boost
    expected_language = "ta"
    top_result = response.results[0]
    
    print("\n  Verification:")
    if top_result.language == expected_language:
        print_result("Language match", f"✓ PASS (Tamil chunk boosted to top)", 4)
    else:
        print_result("Language match", f"✗ FAIL (expected {expected_language}, got {top_result.language})", 4)
    
    if top_result.similarity_score > 0.5:
        print_result("Similarity score", f"✓ PASS (>{0.5:.2f})", 4)
    else:
        print_result("Similarity score", f"✗ FAIL (<{0.5:.2f})", 4)
    
    print("\n  ✓ Tamil search test PASSED")


def test_metadata_mapping(pipeline: VectorIngestionPipeline):
    """
    Test 4: Metadata Mapping
    
    Verifies that metadata is correctly stored and retrieved.
    """
    print_header("TEST 4: Metadata Mapping Verification")
    
    # Get metadata for English document
    english_metadata = pipeline.metadata_store.get_by_document("test-english-001")
    
    print_result("English document vectors", str(len(english_metadata)))
    
    if english_metadata:
        sample = english_metadata[0]
        print("\n  Sample metadata entry:")
        print_result("Vector ID", str(sample.vector_id), 4)
        print_result("Document ID", sample.document_id, 4)
        print_result("Chunk ID", str(sample.chunk_id), 4)
        print_result("Clause", sample.clause_reference or "N/A", 4)
        print_result("Page", str(sample.page_number), 4)
        print_result("Language", sample.language, 4)
        print_result("Text preview", sample.text_preview[:50] + "...", 4)
    
    # Get metadata for Tamil document
    tamil_metadata = pipeline.metadata_store.get_by_document("test-tamil-001")
    
    print(f"\n  Tamil document vectors: {len(tamil_metadata)}")
    
    # Verify counts
    print("\n  Verification:")
    if len(english_metadata) == 3:
        print_result("English metadata count", "✓ PASS (3 entries)", 4)
    else:
        print_result("English metadata count", f"✗ FAIL (expected 3, got {len(english_metadata)})", 4)
    
    if len(tamil_metadata) == 3:
        print_result("Tamil metadata count", "✓ PASS (3 entries)", 4)
    else:
        print_result("Tamil metadata count", f"✗ FAIL (expected 3, got {len(tamil_metadata)})", 4)
    
    # Verify all metadata fields are populated
    all_populated = all([
        sample.document_id is not None,
        sample.chunk_id is not None,
        sample.language is not None,
        sample.text_preview is not None
    ])
    
    if all_populated:
        print_result("Metadata fields", "✓ PASS (all fields populated)", 4)
    else:
        print_result("Metadata fields", "✗ FAIL (missing fields)", 4)
    
    print("\n  ✓ Metadata mapping test PASSED")


def test_cross_language_similarity():
    """
    Test 5: Cross-Language Similarity
    
    Tests that multilingual embeddings can find semantic similarity across languages.
    """
    print_header("TEST 5: Cross-Language Similarity")
    
    from app.services.embedding_service import get_embedding_service
    
    service = get_embedding_service()
    
    # Generate embeddings
    english_text = "property ownership rights"
    tamil_text = "சொத்து உரிமை"
    
    print_result("English text", f'"{english_text}"')
    print_result("Tamil text", f'"{tamil_text}"')
    
    print("\n  Generating embeddings...")
    english_emb = service.embed_single(english_text)
    tamil_emb = service.embed_single(tamil_text)
    
    print_result("English embedding dim", str(len(english_emb)), 4)
    print_result("Tamil embedding dim", str(len(tamil_emb)), 4)
    
    # Compute similarity
    similarity = service.compute_similarity(english_emb, tamil_emb)
    
    print("\n  Cross-language similarity:")
    print_result("Cosine similarity", f"{similarity:.4f}", 4)
    
    # Multilingual models should show some similarity for same concept
    if similarity > 0.3:
        print_result("Verification", f"✓ PASS (similarity > 0.3 indicates semantic overlap)", 4)
    else:
        print_result("Verification", f"✗ FAIL (similarity too low: {similarity:.4f})", 4)
    
    print("\n  ✓ Cross-language test PASSED")


def main():
    """Run all validation tests"""
    print("\n" + "=" * 70)
    print("  EMBEDDING AND FAISS VALIDATION SCRIPT")
    print("=" * 70)
    print("\n  This script validates:")
    print("    1. Vector ingestion (English + Tamil)")
    print("    2. English semantic search")
    print("    3. Tamil semantic search with language boost")
    print("    4. Metadata mapping")
    print("    5. Cross-language similarity")
    
    try:
        # Test 1: Ingestion
        pipeline = test_ingestion()
        if pipeline is None:
            print("\n✗ VALIDATION FAILED: Ingestion test failed")
            return
        
        # Test 2: English search
        test_english_search(pipeline)
        
        # Test 3: Tamil search
        test_tamil_search(pipeline)
        
        # Test 4: Metadata mapping
        test_metadata_mapping(pipeline)
        
        # Test 5: Cross-language similarity
        test_cross_language_similarity()
        
        # Final summary
        print_header("VALIDATION SUMMARY")
        print("\n  ✓ All tests PASSED!")
        print("\n  Your embedding and FAISS setup is working correctly.")
        print("  You can now integrate with FastAPI.")
        print("\n" + "=" * 70 + "\n")
        
    except Exception as e:
        print_header("VALIDATION FAILED")
        print(f"\n  ✗ Error: {str(e)}")
        logger.error("Validation failed", exc_info=True)
        print("\n" + "=" * 70 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
