const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = 3001;

app.use(cors());
app.use(express.json());

// Storage path for downloaded certificates
const CERTIFICATE_STORAGE_PATH = 'D:\\Sree\\AI Accelerator\\Hackathon\\cleardeed\\downloads';

// Ensure downloads directory exists
if (!fs.existsSync(CERTIFICATE_STORAGE_PATH)) {
  fs.mkdirSync(CERTIFICATE_STORAGE_PATH, { recursive: true });
}

// Endpoint to execute the selenium script
app.post('/execute-script', (req, res) => {
  const { params } = req.body;
  
  console.log('Launching selenium script with params:', params);
  
  const args = [
    'getEncumbranceCertificate.py',
    '--zone', params.zone || 'Chennai',
    '--district', params.district,
    '--sro', params.sro_name,
    '--village', params.village,
    '--survey', params.survey_number,
    '--subdivision', params.subdivision || '',
    '--output', CERTIFICATE_STORAGE_PATH
  ];

  // Launch Python script in background
  const pythonProcess = spawn('python', args, {
    detached: true,
    stdio: 'ignore'
  });

  pythonProcess.unref();

  console.log(`Selenium script launched with PID: ${pythonProcess.pid}`);
  
  res.json({ 
    success: true, 
    message: 'Selenium script launched',
    pid: pythonProcess.pid
  });
});

// Endpoint to check if certificate has been downloaded
app.post('/check-certificate', (req, res) => {
  const { path: storagePath } = req.body;
  
  try {
    // Check for PDF files in the downloads directory
    const files = fs.readdirSync(storagePath);
    const pdfFiles = files.filter(file => file.toLowerCase().endsWith('.pdf'));
    
    if (pdfFiles.length > 0) {
      // Get the most recently modified PDF
      const latestFile = pdfFiles
        .map(file => ({
          name: file,
          time: fs.statSync(path.join(storagePath, file)).mtime.getTime()
        }))
        .sort((a, b) => b.time - a.time)[0];
      
      const filePath = path.join(storagePath, latestFile.name);
      const stats = fs.statSync(filePath);
      
      console.log(`Certificate found: ${latestFile.name}`);
      
      res.json({
        fileFound: true,
        fileName: latestFile.name,
        fileSize: stats.size,
        filePath: filePath
      });
    } else {
      res.json({
        fileFound: false
      });
    }
  } catch (error) {
    console.error('Error checking for certificate:', error);
    res.status(500).json({
      fileFound: false,
      error: error.message
    });
  }
});

// Endpoint to get file as base64 for analysis
app.post('/get-file-base64', (req, res) => {
  const { filePath } = req.body;
  
  try {
    const fileBuffer = fs.readFileSync(filePath);
    const base64 = fileBuffer.toString('base64');
    
    res.json({
      success: true,
      base64: base64
    });
  } catch (error) {
    console.error('Error reading file:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Endpoint to clean up old certificates
app.post('/cleanup-certificates', (req, res) => {
  try {
    const files = fs.readdirSync(CERTIFICATE_STORAGE_PATH);
    const pdfFiles = files.filter(file => file.toLowerCase().endsWith('.pdf'));
    
    pdfFiles.forEach(file => {
      fs.unlinkSync(path.join(CERTIFICATE_STORAGE_PATH, file));
    });
    
    console.log(`Cleaned up ${pdfFiles.length} certificate(s)`);
    
    res.json({
      success: true,
      filesDeleted: pdfFiles.length
    });
  } catch (error) {
    console.error('Error cleaning up certificates:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

app.listen(PORT, () => {
  console.log(`Helper server running on http://localhost:${PORT}`);
  console.log(`Certificate storage path: ${CERTIFICATE_STORAGE_PATH}`);
});
