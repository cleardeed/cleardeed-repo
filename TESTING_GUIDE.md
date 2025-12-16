# Testing the Encumbrance Certificate Selenium Script

## Quick Start Guide

### Prerequisites Installation

1. **Install Python Dependencies**
```powershell
pip install selenium webdriver-manager
```

2. **Install ChromeDriver (Automatic)**
The script uses Chrome WebDriver. Install it via webdriver-manager:
```powershell
python -c "from selenium import webdriver; from selenium.webdriver.chrome.service import Service; from webdriver_manager.chrome import ChromeDriverManager; driver = webdriver.Chrome(service=Service(ChromeDriverManager().install())); driver.quit()"
```

3. **Install Node.js Helper Server Dependencies**
```powershell
cd "D:\Sree\AI Accelerator\Hackathon\cleardeed"
npm install express cors
```

4. **Create Downloads Directory**
```powershell
New-Item -ItemType Directory -Force -Path "D:\Sree\AI Accelerator\Hackathon\cleardeed\downloads"
```

---

## Testing the Script Standalone

### Test Command Template
```powershell
python getEncumbranceCertificate.py --zone "Chennai" --district "Chennai" --sro "Mylapore" --village "Nungambakkam" --survey "123/4" --subdivision "A" --output "D:\Sree\AI Accelerator\Hackathon\cleardeed\downloads"
```

### Example Test Cases

**Test 1: Basic Test with All Parameters**
```powershell
python getEncumbranceCertificate.py `
  --zone "Chennai" `
  --district "Chennai" `
  --sro "Ambattur" `
  --village "Ambattur" `
  --survey "45/2B" `
  --subdivision "1" `
  --output "D:\Sree\AI Accelerator\Hackathon\cleardeed\downloads"
```

**Test 2: Without Subdivision (Optional)**
```powershell
python getEncumbranceCertificate.py `
  --zone "Chennai" `
  --district "Kancheepuram" `
  --sro "Tambaram" `
  --village "Tambaram" `
  --survey "101/3" `
  --output "D:\Sree\AI Accelerator\Hackathon\cleardeed\downloads"
```

---

## Testing with Helper Server

### Step 1: Start Helper Server
```powershell
cd "D:\Sree\AI Accelerator\Hackathon\cleardeed"
node helper-server.js
```

Expected output:
```
Helper server running on http://localhost:3001
Certificate storage path: D:\Sree\AI Accelerator\Hackathon\cleardeed\downloads
```

### Step 2: Test Script Launch via API

**Using PowerShell (Invoke-RestMethod)**
```powershell
$body = @{
    params = @{
        zone = "Chennai"
        district = "Chennai"
        sro_name = "Mylapore"
        village = "Nungambakkam"
        survey_number = "123/4"
        subdivision = "A"
    }
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:3001/execute-script" -Method POST -Body $body -ContentType "application/json"
```

**Using curl (if installed)**
```powershell
curl -X POST http://localhost:3001/execute-script `
  -H "Content-Type: application/json" `
  -d '{\"params\": {\"zone\":\"Chennai\",\"district\":\"Chennai\",\"sro_name\":\"Mylapore\",\"village\":\"Nungambakkam\",\"survey_number\":\"123/4\",\"subdivision\":\"A\"}}'
```

### Step 3: Check for Downloaded Certificate
```powershell
$body = @{
    path = "D:\Sree\AI Accelerator\Hackathon\cleardeed\downloads"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:3001/check-certificate" -Method POST -Body $body -ContentType "application/json"
```

---

## Testing Full Integration

### Step 1: Start All Services

**Terminal 1: Helper Server**
```powershell
node helper-server.js
```

**Terminal 2: React App**
```powershell
npm run dev
```

### Step 2: Test Through UI
1. Open browser: `http://localhost:5174`
2. Fill in property details:
   - Zone: Chennai
   - District: Chennai
   - SRO: Mylapore
   - Village: Nungambakkam
   - Survey Number: 123/4
   - Subdivision: A (optional)
3. Upload documents (any PDFs for testing)
4. Click "Proceed to Analysis"
5. Watch the processing screen
6. Certificate should download within 5 minutes

---

## Debugging

### Check Logs

**Script Logs**
```powershell
Get-Content encumbrance_download.log -Tail 50
```

**Helper Server Logs**
Check the console where `node helper-server.js` is running

### View Screenshots on Error
The script automatically saves screenshots on errors:
```powershell
Get-ChildItem "D:\Sree\AI Accelerator\Hackathon\cleardeed\downloads\*.png"
```

### Check Downloads
```powershell
Get-ChildItem "D:\Sree\AI Accelerator\Hackathon\cleardeed\downloads\*.pdf"
```

### Manual Browser Test
Comment out headless mode in script to watch it run:
```python
# In getEncumbranceCertificate.py, line ~73
# chrome_options.add_argument("--headless")  # Comment this line
```

---

## Common Issues & Fixes

### Issue 1: ChromeDriver Version Mismatch
**Error**: `SessionNotCreatedException: Message: session not created: This version of ChromeDriver only supports Chrome version XX`

**Fix**: Update Chrome and reinstall ChromeDriver
```powershell
pip install --upgrade webdriver-manager
```

### Issue 2: Element Not Found
**Error**: `NoSuchElementException` or `TimeoutException`

**Causes**:
- Portal structure changed
- Page loading slowly
- Wrong selectors

**Fix**: 
1. Run in non-headless mode to see what's happening
2. Check the actual HTML element IDs on the portal
3. Update selectors in the script

### Issue 3: Download Not Starting
**Possible Causes**:
- Portal requires CAPTCHA
- Portal requires login
- Network issues

**Fix**:
1. Check if portal is accessible: `https://tnreginet.gov.in/portal/`
2. Verify no CAPTCHA is required for the encumbrance section
3. Add login flow if needed

### Issue 4: Port 3001 Already in Use
**Error**: `Error: listen EADDRINUSE: address already in use :::3001`

**Fix**:
```powershell
# Find process using port 3001
netstat -ano | findstr :3001

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

---

## Customization

### Change Download Timeout
In `src/services/api.ts`, line ~115:
```typescript
const retrievedDocument = await pollForCertificate(
  onStatusChange, 
  signal,
  120,  // Max attempts (change this)
  3000  // Poll interval in ms (change this)
);
```

### Change Storage Path
Update in 3 places:
1. `src/services/api.ts` - `CERTIFICATE_STORAGE_PATH`
2. `helper-server.js` - `CERTIFICATE_STORAGE_PATH`
3. Command line when testing

### Enable Headless Mode
In `getEncumbranceCertificate.py`, line ~73:
```python
chrome_options.add_argument("--headless")  # Uncomment for headless
```

---

## Portal-Specific Notes

### TN RegINet Portal Navigation Path
```
https://tnreginet.gov.in/portal/
  └─> Electronic Services (இ-சேவைகள்)
      └─> Villangan Evidence (வில்லங்கச் சான்று)
          └─> Viewing the details of the Villangan certificate
```

### Form Fields
- **Zone** (மண்டலம்): Dropdown
- **District** (மாவட்டம்): Dropdown (cascading from Zone)
- **Sub-Registrar's Office** (துணை பதிவாளர் அலுவலகம்): Dropdown (cascading from District)
- **Registration Village** (பதிவு கிராமம்): Dropdown (cascading from SRO)
- **Start Date** (தொடக்க தேதி): Date field (Auto-filled: Current Date - 30 years)
- **End Date** (முடிவு தேதி): Date field (Auto-filled: Yesterday)
- **Field Number** (புல எண்): Text input (Survey Number)
- **Subdivision**: Text input (Optional)

### Date Range
- **Start Date**: Automatically set to 30 years before today
- **End Date**: Automatically set to yesterday
- **Rationale**: Encumbrance certificates typically cover past 30 years of property transactions

---

## Success Indicators

✓ Helper server running on port 3001  
✓ Chrome browser opens (if not headless)  
✓ Portal loads successfully  
✓ Navigation through menus works  
✓ Form fields populated correctly  
✓ ADD button clicked  
✓ PDF downloaded to downloads folder  
✓ Frontend polling detects the file  
✓ Processing completes and shows "View Analysis Results"  

---

## Next Steps After Successful Test

1. **Add Error Handling**: Enhance script to handle portal-specific errors
2. **Add CAPTCHA Handling**: If portal adds CAPTCHA, integrate solver
3. **Add Retry Logic**: Implement retries for transient failures
4. **Add Notifications**: Email/SMS when certificate is ready
5. **Queue System**: Handle multiple concurrent requests
6. **Cleanup Automation**: Scheduled cleanup of old certificates
