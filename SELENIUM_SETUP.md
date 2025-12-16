# Selenium Integration Setup Guide

## Overview
This setup replaces the mock API call with a real selenium-based encumbrance certificate retrieval system.

## Architecture

```
User clicks "Proceed to Analysis"
    ↓
ProcessingScreen initiates download
    ↓
Helper Server (Node.js on port 3001)
    ↓
Launches getEncumbranceCertificate.py (Selenium script)
    ↓
Downloads certificate to D:\Sree\AI Accelerator\Hackathon\cleardeed\downloads
    ↓
Frontend polls every 5 seconds for file
    ↓
Certificate retrieved and sent to analysis
```

## Setup Steps

### 1. Install Helper Server Dependencies
```powershell
cd "D:\Sree\AI Accelerator\Hackathon\cleardeed"
npm install --prefix . express cors
```

Or using the package file:
```powershell
npm install --save express cors
```

### 2. Create the Selenium Script
Create `getEncumbranceCertificate.py` in the root directory with the following structure:

```python
import argparse
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

def download_encumbrance_certificate(district, sro, village, survey_number, subdivision, output_path):
    """
    Selenium script to download encumbrance certificate
    
    Args:
        district: District name
        sro: SRO office name
        village: Village name
        survey_number: Survey number
        subdivision: Subdivision (optional)
        output_path: Path where certificate will be downloaded
    """
    
    # Configure Chrome options for download
    chrome_options = Options()
    prefs = {
        "download.default_directory": output_path,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    chrome_options.add_experimental_option("prefs", prefs)
    chrome_options.add_argument("--start-maximized")
    
    # Initialize driver
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        # Navigate to the government portal
        driver.get("https://encumbrance-portal-url.gov.in")  # Replace with actual URL
        
        # Wait for page to load
        wait = WebDriverWait(driver, 10)
        
        # Fill in the form fields
        # Note: Replace these selectors with actual element IDs/names from the portal
        district_field = wait.until(EC.presence_of_element_located((By.ID, "district")))
        district_field.send_keys(district)
        
        sro_field = driver.find_element(By.ID, "sro")
        sro_field.send_keys(sro)
        
        village_field = driver.find_element(By.ID, "village")
        village_field.send_keys(village)
        
        survey_field = driver.find_element(By.ID, "surveyNumber")
        survey_field.send_keys(survey_number)
        
        if subdivision:
            subdivision_field = driver.find_element(By.ID, "subdivision")
            subdivision_field.send_keys(subdivision)
        
        # Submit form
        submit_button = driver.find_element(By.ID, "submitButton")
        submit_button.click()
        
        # Wait for download to complete
        time.sleep(10)  # Adjust based on actual download time
        
        print(f"Certificate downloaded successfully to {output_path}")
        
    except Exception as e:
        print(f"Error downloading certificate: {str(e)}")
        raise
    
    finally:
        driver.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download Encumbrance Certificate')
    parser.add_argument('--district', required=True, help='District name')
    parser.add_argument('--sro', required=True, help='SRO office name')
    parser.add_argument('--village', required=True, help='Village name')
    parser.add_argument('--survey', required=True, help='Survey number')
    parser.add_argument('--subdivision', default='', help='Subdivision (optional)')
    parser.add_argument('--output', required=True, help='Output directory path')
    
    args = parser.parse_args()
    
    download_encumbrance_certificate(
        args.district,
        args.sro,
        args.village,
        args.survey,
        args.subdivision,
        args.output
    )
```

### 3. Install Python Dependencies
```powershell
pip install selenium webdriver-manager
```

### 4. Create Downloads Directory
```powershell
New-Item -ItemType Directory -Force -Path "D:\Sree\AI Accelerator\Hackathon\cleardeed\downloads"
```

### 5. Start the Helper Server
```powershell
node helper-server.js
```

The server will start on port 3001 and listen for requests from the React app.

### 6. Start the React App
```powershell
npm run dev
```

## Configuration

### Storage Path
The default certificate storage path is:
```
D:\Sree\AI Accelerator\Hackathon\cleardeed\downloads
```

To change this, update:
- `helper-server.js` → `CERTIFICATE_STORAGE_PATH`
- `src/services/api.ts` → `CERTIFICATE_STORAGE_PATH`

### Polling Settings
Default polling configuration:
- **Max Attempts**: 60 (5 minutes total)
- **Poll Interval**: 5000ms (5 seconds)

To adjust, modify the `pollForCertificate` function in `src/services/api.ts`:
```typescript
await pollForCertificate(onStatusChange, signal, 120, 3000); // 6 minutes, 3s interval
```

## API Endpoints

The helper server exposes these endpoints:

### POST /execute-script
Launches the selenium script with property parameters
```json
{
  "params": {
    "district": "Chennai",
    "sro_name": "Mylapore",
    "village": "Nungambakkam",
    "survey_number": "123/4",
    "subdivision": "A"
  }
}
```

### POST /check-certificate
Checks if a certificate PDF has been downloaded
```json
{
  "path": "D:\\Sree\\AI Accelerator\\Hackathon\\cleardeed\\downloads"
}
```

### POST /get-file-base64
Retrieves file content as base64 for analysis
```json
{
  "filePath": "D:\\Sree\\AI Accelerator\\Hackathon\\cleardeed\\downloads\\certificate.pdf"
}
```

### POST /cleanup-certificates
Removes all PDF files from the downloads directory
```json
{}
```

## Workflow

1. **User Input**: User fills property details and uploads documents
2. **Initiate**: User clicks "Proceed to Analysis"
3. **Launch Script**: Helper server spawns Python selenium script
4. **Download**: Script navigates portal and downloads certificate
5. **Poll**: Frontend checks every 5 seconds for downloaded file
6. **Retrieve**: Once found, certificate is included in analysis
7. **Analyze**: All documents sent to Supabase for AI analysis
8. **Results**: User sees analysis results with recommendations

## Troubleshooting

### Script Not Launching
- Ensure Python is in PATH: `python --version`
- Check helper server logs for errors
- Verify script exists: `Test-Path getEncumbranceCertificate.py`

### Certificate Not Downloading
- Check selenium script logs
- Verify Chrome/ChromeDriver compatibility
- Ensure download path has write permissions
- Check government portal availability

### Timeout Issues
- Increase `maxAttempts` in `pollForCertificate`
- Check network connectivity
- Verify portal response time

### File Not Found
- Ensure downloads directory exists and has correct permissions
- Check if antivirus is blocking downloads
- Verify file naming in selenium script

## Production Considerations

1. **Error Handling**: Add retry logic for network failures
2. **Security**: Sanitize inputs to prevent command injection
3. **Logging**: Implement proper logging for debugging
4. **Cleanup**: Schedule periodic cleanup of old certificates
5. **Queue**: Implement job queue for multiple concurrent requests
6. **Monitoring**: Add health checks and performance metrics
