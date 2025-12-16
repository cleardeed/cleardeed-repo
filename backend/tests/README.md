# Running Tests

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Run All Tests
```bash
pytest
```

## Run Specific Test File
```bash
pytest tests/test_document_processor.py
```

## Run Specific Test Class
```bash
pytest tests/test_document_processor.py::TestDocumentProcessingPipeline
```

## Run Specific Test Function
```bash
pytest tests/test_document_processor.py::test_process_english_document
```

## Run with Coverage Report
```bash
pytest --cov=app --cov-report=html
```

## Run in Verbose Mode
```bash
pytest -v
```

## Run Tests Matching Pattern
```bash
pytest -k "english"  # Runs all tests with "english" in name
```

## Run Only Unit Tests
```bash
pytest -m unit
```

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures
├── test_document_processor.py     # Pipeline tests
└── pytest.ini                     # Pytest configuration
```

## Test Coverage

The tests cover:
- ✅ English PDF processing
- ✅ Tamil PDF processing
- ✅ Scanned PDF with OCR
- ✅ Mixed-language documents
- ✅ Error handling
- ✅ Empty documents
- ✅ Non-existent files
- ✅ Extraction failures

## Mocking Strategy

- **PDFExtractor**: Mocked to return predefined page data
- **LanguageDetector**: Mocked to return specific language codes
- **TextNormalizer**: Mocked to pass-through or apply simple transforms
- **LegalDocumentChunker**: Mocked to return predefined chunks
- **File I/O**: Mocked using `patch('pathlib.Path.exists')`

## Example Output

```
tests/test_document_processor.py::TestPDFExtractor::test_extract_english_pdf PASSED
tests/test_document_processor.py::TestPDFExtractor::test_extract_tamil_pdf PASSED
tests/test_document_processor.py::TestPDFExtractor::test_extract_scanned_pdf_with_ocr PASSED
tests/test_document_processor.py::TestLanguageDetector::test_detect_english PASSED
tests/test_document_processor.py::TestLanguageDetector::test_detect_tamil PASSED
tests/test_document_processor.py::TestLanguageDetector::test_detect_mixed_language PASSED
tests/test_document_processor.py::TestDocumentProcessingPipeline::test_process_english_document PASSED
tests/test_document_processor.py::TestDocumentProcessingPipeline::test_process_tamil_document PASSED
tests/test_document_processor.py::TestDocumentProcessingPipeline::test_process_scanned_document PASSED
tests/test_document_processor.py::TestDocumentProcessingPipeline::test_process_mixed_language_document PASSED
tests/test_document_processor.py::TestDocumentProcessingPipeline::test_process_nonexistent_file PASSED
tests/test_document_processor.py::TestDocumentProcessingPipeline::test_process_with_extraction_error PASSED

================== 20 passed in 2.45s ==================
```
