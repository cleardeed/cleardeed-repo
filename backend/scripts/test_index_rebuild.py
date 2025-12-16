"""
Test script to validate FAISS index rebuild when documents are removed.

This script tests:
1. Ingest two documents (3 chunks each)
2. Verify 6 vectors in FAISS
3. Remove first document
4. Verify only 3 vectors remain (no orphaned vectors)
5. Search should only return results from remaining document
"""

import sys
import os
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import FAISSIndexManager
from app.services.metadata_store import VectorMetadataStore
from app.services.vector_ingestion import VectorIngestionPipeline
from app.models.document_models import TextChunk, Language


def print_section(title: str):
    """Print section header"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result"""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n{status}: {test_name}")
    if details:
        print(f"  {details}")


def main():
    """Run index rebuild tests"""
    
    print("\n" + "=" * 70)
    print("  FAISS Index Rebuild Test")
    print("=" * 70)
    
    try:
        # Initialize services
        print("\n[1/7] Initializing services...")
        embedding_service = EmbeddingService.get_instance()
        vector_store = FAISSIndexManager(embedding_dim=1024)
        metadata_store = VectorMetadataStore()
        
        # Clear any existing data
        vector_store.clear()
        metadata_store._store.clear()
        
        pipeline = VectorIngestionPipeline(
            embedding_service=embedding_service,
            vector_store=vector_store,
            metadata_store=metadata_store
        )
        print("  ✓ Services initialized")
        
        # Test 1: Ingest first document (3 chunks)
        print_section("Test 1: Ingest Document 1 (3 chunks)")
        
        doc1_chunks = [
            TextChunk(
                chunk_id=0,
                text="This agreement is made on January 1, 2024 between the parties.",
                language=Language.ENGLISH,
                page_number=1,
                clause_reference="Section 1"
            ),
            TextChunk(
                chunk_id=1,
                text="The property is located at 123 Main Street, City, State.",
                language=Language.ENGLISH,
                page_number=1,
                clause_reference="Section 2"
            ),
            TextChunk(
                chunk_id=2,
                text="Both parties agree to the terms and conditions specified herein.",
                language=Language.ENGLISH,
                page_number=2,
                clause_reference="Section 3"
            )
        ]
        
        doc1_result = pipeline.ingest_document(
            document_id="doc1",
            chunks=doc1_chunks
        )
        
        passed_1 = doc1_result.success and doc1_result.vectors_indexed == 3
        print_result(
            "Document 1 ingestion",
            passed_1,
            f"Indexed {doc1_result.vectors_indexed} vectors"
        )
        
        # Test 2: Ingest second document (3 chunks)
        print_section("Test 2: Ingest Document 2 (3 chunks)")
        
        doc2_chunks = [
            TextChunk(
                chunk_id=0,
                text="The seller hereby transfers ownership of the property.",
                language=Language.ENGLISH,
                page_number=1,
                clause_reference="Clause 1"
            ),
            TextChunk(
                chunk_id=1,
                text="Payment shall be made in full within 30 days of signing.",
                language=Language.ENGLISH,
                page_number=1,
                clause_reference="Clause 2"
            ),
            TextChunk(
                chunk_id=2,
                text="Any disputes shall be resolved through arbitration.",
                language=Language.ENGLISH,
                page_number=2,
                clause_reference="Clause 3"
            )
        ]
        
        doc2_result = pipeline.ingest_document(
            document_id="doc2",
            chunks=doc2_chunks
        )
        
        passed_2 = doc2_result.success and doc2_result.vectors_indexed == 3
        print_result(
            "Document 2 ingestion",
            passed_2,
            f"Indexed {doc2_result.vectors_indexed} vectors"
        )
        
        # Test 3: Verify 6 vectors total
        print_section("Test 3: Verify Total Vectors Before Removal")
        
        total_before = vector_store.get_total_vectors()
        metadata_count_before = len(metadata_store._store)
        
        passed_3 = total_before == 6 and metadata_count_before == 6
        print_result(
            "Total vectors before removal",
            passed_3,
            f"FAISS: {total_before} vectors, Metadata: {metadata_count_before} entries"
        )
        
        # Test 4: Remove first document
        print_section("Test 4: Remove Document 1 (Rebuild Index)")
        
        removed_count = pipeline.remove_document("doc1")
        
        total_after = vector_store.get_total_vectors()
        metadata_count_after = len(metadata_store._store)
        
        passed_4 = (
            removed_count == 3 and 
            total_after == 3 and 
            metadata_count_after == 3
        )
        print_result(
            "Document removal and rebuild",
            passed_4,
            f"Removed: {removed_count}, Remaining FAISS: {total_after}, "
            f"Remaining metadata: {metadata_count_after}"
        )
        
        # Test 5: Verify no orphaned vectors
        print_section("Test 5: Verify No Orphaned Vectors")
        
        # FAISS count should match metadata count
        no_orphans = total_after == metadata_count_after
        print_result(
            "No orphaned vectors",
            no_orphans,
            f"FAISS vectors ({total_after}) == Metadata entries ({metadata_count_after})"
        )
        
        # Test 6: Verify search only returns doc2 results
        print_section("Test 6: Verify Search Results")
        
        # Generate query embedding
        query_embedding = embedding_service.embed_single("property ownership transfer")
        
        # Search
        results, scores = vector_store.search(query_embedding, k=5)
        
        # Get metadata for results
        all_doc2 = True
        for vector_id in results:
            metadata_id = vector_store._metadata.get(vector_id)
            if metadata_id:
                # Metadata ID format: "{document_id}_chunk_{chunk_id}"
                doc_id = metadata_id.split("_chunk_")[0]
                if doc_id != "doc2":
                    all_doc2 = False
                    print(f"  Found vector from {doc_id}, expected doc2 only")
        
        passed_6 = all_doc2 and len(results) <= 3
        print_result(
            "Search returns only doc2 results",
            passed_6,
            f"Found {len(results)} results, all from doc2"
        )
        
        # Test 7: Verify metadata completeness
        print_section("Test 7: Verify Metadata Completeness")
        
        doc2_metadata = metadata_store.get_by_document("doc2")
        
        all_have_embeddings = all(meta.embedding is not None for meta in doc2_metadata)
        all_have_texts = all(meta.text_preview for meta in doc2_metadata)
        
        passed_7 = (
            len(doc2_metadata) == 3 and
            all_have_embeddings and
            all_have_texts
        )
        print_result(
            "Metadata completeness",
            passed_7,
            f"Doc2 has {len(doc2_metadata)} metadata entries, "
            f"all with embeddings and text"
        )
        
        # Final summary
        print_section("Test Summary")
        
        all_tests = [passed_1, passed_2, passed_3, passed_4, no_orphans, passed_6, passed_7]
        passed_count = sum(all_tests)
        total_tests = len(all_tests)
        
        print(f"\n  Tests Passed: {passed_count}/{total_tests}")
        
        if passed_count == total_tests:
            print("\n  ✓ ALL TESTS PASSED - Index rebuild working correctly!")
            print("  No orphaned vectors, clean document isolation maintained.")
        else:
            print("\n  ✗ SOME TESTS FAILED - Review output above")
        
        print("\n" + "=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
