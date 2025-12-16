# Startup script for ClearDeed backend
Write-Host "Starting ClearDeed Backend..." -ForegroundColor Cyan

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Start FastAPI server
Write-Host "Starting FastAPI server on http://localhost:8000" -ForegroundColor Green
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
