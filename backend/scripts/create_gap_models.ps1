# Setup script for creating custom GAP analysis Ollama models
# Run this: .\scripts\create_gap_models.ps1

Write-Host "=== ClearDeed GAP Model Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check Ollama
Write-Host "[1/4] Checking Ollama..." -ForegroundColor Yellow
$response = Invoke-RestMethod -Uri "http://localhost:11434/api/version" -Method GET
Write-Host "  OK Ollama v$($response.version)" -ForegroundColor Green

# Check llama2
Write-Host ""
Write-Host "[2/4] Checking llama2..." -ForegroundColor Yellow
$models = (Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method GET).models
$hasLlama2 = $models | Where-Object { $_.name -like "llama2*" }
if ($hasLlama2) {
    Write-Host "  OK llama2 found" -ForegroundColor Green
}
else {
    Write-Host "  Downloading llama2 (3.8GB)..." -ForegroundColor Yellow
    ollama pull llama2
}

# Create gap-english
Write-Host ""
Write-Host "[3/4] Creating gap-english..." -ForegroundColor Yellow
Set-Location "$PSScriptRoot\..\modelfiles"
ollama create gap-english -f Modelfile.gap-english
Write-Host "  OK gap-english created" -ForegroundColor Green

# Create gap-tamil
Write-Host ""
Write-Host "[4/4] Creating gap-tamil..." -ForegroundColor Yellow
ollama create gap-tamil -f Modelfile.gap-tamil
Write-Host "  OK gap-tamil created" -ForegroundColor Green

# Done
Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Update gap_detection_engine.py to use custom models" -ForegroundColor White
Write-Host "2. Run: ollama list" -ForegroundColor White
Write-Host ""
