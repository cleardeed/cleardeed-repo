# ClearDeed Backend API

Production-ready FastAPI backend for legal document analysis with Tamil and English support.

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app entry point
│   ├── api/
│   │   └── routes/
│   │       └── documents.py    # Document upload/analysis endpoints
│   ├── models/
│   │   └── schemas.py          # Pydantic models
│   ├── services/
│   │   ├── pdf_processor.py    # PDF text extraction
│   │   ├── language_detector.py # Tamil/English detection
│   │   ├── text_extractor.py   # Extraction orchestration
│   │   └── llm_service.py      # Ollama/LM Studio integration
│   ├── utils/
│   │   ├── file_handler.py     # File operations
│   │   └── validators.py       # Input validation
│   └── config/
│       └── settings.py         # Configuration management
├── uploads/                     # Document upload directory (created at runtime)
├── logs/                        # Application logs (created at runtime)
├── requirements.txt
├── .env.example
└── README.md
```

## Features

- **PDF Processing**: Text extraction from Tamil and English PDFs
- **Language Detection**: Automatic detection of Tamil/English/Mixed content
- **Local LLM Integration**: Ready for Ollama or LM Studio
- **Async Processing**: Background job processing for large documents
- **File Upload**: Secure multipart file upload with validation
- **API Documentation**: Auto-generated Swagger/OpenAPI docs

## Setup Instructions

### 1. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Install Tesseract OCR (for scanned PDFs)

**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Install and add to PATH
- Download Tamil language data: https://github.com/tesseract-ocr/tessdata

**Linux:**
```bash
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-tam  # Tamil language
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 4. Install LM Studio (or Ollama)

**LM Studio:**
- Download from: https://lmstudio.ai/
- Load a model (recommended: Llama 2 or Mistral)
- Start local server (default: http://localhost:1234)
- Update `.env`: `LM_STUDIO_ENABLED=true`

**Ollama (Alternative):**
- Install from: https://ollama.ai/
- Pull a model: `ollama pull llama2`
- Server runs at: http://localhost:11434

### 5. Run the Server

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### Base URL: `http://localhost:8000`

- **GET** `/` - Health check
- **GET** `/health` - Detailed health status
- **POST** `/api/v1/upload` - Upload single document
- **POST** `/api/v1/upload-multiple` - Upload multiple documents
- **GET** `/api/v1/documents/{id}` - Get document metadata
- **POST** `/api/v1/analyze` - Trigger document analysis
- **GET** `/api/v1/analysis/{id}` - Get analysis results
- **DELETE** `/api/v1/documents/{id}` - Delete document

### Interactive API Docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Next Steps (Implementation TODO)

1. **PDF Processing** (`pdf_processor.py`):
   - Implement text extraction with PyMuPDF
   - Add OCR support for scanned PDFs
   - Handle Tamil unicode properly

2. **Language Detection** (`language_detector.py`):
   - Implement Tamil unicode range detection (U+0B80 - U+0BFF)
   - Add confidence scoring

3. **LLM Integration** (`llm_service.py`):
   - Connect to LM Studio/Ollama API
   - Implement prompt engineering for legal analysis
   - Add streaming support

4. **File Upload** (`documents.py`):
   - Complete upload endpoint
   - Add file validation
   - Implement document storage

5. **Analysis Pipeline** (`documents.py`):
   - Orchestrate PDF → Text → Language → LLM flow
   - Add background task processing
   - Return structured results

## Development Notes

- All service classes have clear docstrings explaining their responsibilities
- No business logic implemented yet - only structure
- Ready for Ollama or LM Studio integration
- Designed for Tamil and English document processing
- Uses async/await for scalability

## Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests (when implemented)
pytest
```

## License

Internal project - ClearDeed Hackathon
