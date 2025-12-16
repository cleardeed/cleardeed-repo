export interface ProcessedDocument {
  id: string;
  name: string;
  source: 'uploaded' | 'retrieved';
  size: number;
  retrievedAt?: string;
}

export interface ApiResponse {
  success: boolean;
  retrievedDocument?: ProcessedDocument;
  error?: string;
}

export interface DocumentAnalysisRequest {
  documents: Array<{
    name: string;
    base64: string;
    size: number;
  }>;
  property_data: {
    district: string;
    sro_name: string;
    village: string;
    survey_number: string;
    subdivision: string;
  };
}

export interface AnalysisFinding {
  id: string;
  category: 'Mandatory' | 'Optional';
  severity: 'High' | 'Medium' | 'Low';
  title: string;
  description: string;
  affected_documents: string[];
}

export interface DocumentAnalysisResponse {
  verdict: 'APPROVED' | 'CONDITIONALLY_APPROVED' | 'REJECTED';
  confidence_score: number;
  findings: AnalysisFinding[];
  total_documents_analyzed: number;
  summary: string;
}

const delay = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

// Path where selenium script will download the encumbrance certificate
const CERTIFICATE_STORAGE_PATH = 'D:\\Sree\\AI Accelerator\\Hackathon\\cleardeed\\downloads';

interface SeleniumScriptParams {
  zone: string;
  district: string;
  sro_name: string;
  village: string;
  survey_number: string;
  subdivision: string;
}

// Launch the selenium script in the background
const launchSeleniumScript = async (params: SeleniumScriptParams): Promise<void> => {
  try {
    const response = await fetch('http://localhost:3001/execute-script', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ params }),
    });

    if (!response.ok) {
      throw new Error('Failed to launch selenium script');
    }
  } catch (error) {
    console.error('Error launching selenium script:', error);
    throw error;
  }
};

// Poll for the downloaded certificate
const pollForCertificate = async (
  onStatusChange: (status: string) => void,
  signal?: AbortSignal,
  maxAttempts: number = 60, // 60 attempts = 5 minutes with 5s intervals
  pollInterval: number = 5000 // 5 seconds
): Promise<ProcessedDocument | null> => {
  let attempts = 0;

  while (attempts < maxAttempts) {
    if (signal?.aborted) throw new Error('Operation cancelled');

    try {
      // Check if file exists in the storage path
      const response = await fetch('http://localhost:3001/check-certificate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: CERTIFICATE_STORAGE_PATH }),
      });

      if (response.ok) {
        const data = await response.json();
        
        if (data.fileFound) {
          // File found, create ProcessedDocument
          return {
            id: `ec-${Date.now()}`,
            name: data.fileName,
            source: 'retrieved',
            size: data.fileSize,
            retrievedAt: new Date().toISOString(),
          };
        }
      }
    } catch (error) {
      console.error('Error polling for certificate:', error);
    }

    attempts++;
    const remainingTime = Math.ceil((maxAttempts - attempts) * pollInterval / 1000);
    onStatusChange(`Waiting for certificate download... (${remainingTime}s remaining)`);
    
    await delay(pollInterval);
  }

  return null; // Timeout
};

export const callExternalApi = async (
  onStatusChange: (status: string) => void,
  propertyData: SeleniumScriptParams,
  signal?: AbortSignal
): Promise<ApiResponse> => {
  try {
    onStatusChange('Initiating encumbrance certificate retrieval...');
    if (signal?.aborted) throw new Error('Operation cancelled');

    // Launch selenium script
    await launchSeleniumScript(propertyData);
    onStatusChange('Selenium script started...');
    await delay(2000);

    if (signal?.aborted) throw new Error('Operation cancelled');

    // Start polling for the certificate
    onStatusChange('Downloading encumbrance certificate...');
    const retrievedDocument = await pollForCertificate(onStatusChange, signal);

    if (!retrievedDocument) {
      return {
        success: false,
        error: 'Timeout: Certificate download took too long. Please try again.',
      };
    }

    onStatusChange('Certificate retrieved successfully!');
    await delay(500);

    return {
      success: true,
      retrievedDocument,
    };
  } catch (error) {
    if (error instanceof Error && error.message === 'Operation cancelled') {
      return {
        success: false,
        error: 'Operation was cancelled',
      };
    }

    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error occurred',
    };
  }
};

export const simulateApiTimeout = (
  ms: number,
  signal?: AbortSignal
): Promise<void> => {
  return new Promise((resolve, reject) => {
    const timeoutId = setTimeout(() => {
      reject(new Error('Request timeout'));
    }, ms);

    signal?.addEventListener('abort', () => {
      clearTimeout(timeoutId);
      reject(new Error('Request aborted'));
    });
  });
};

export const analyzeDocuments = async (
  request: DocumentAnalysisRequest,
  supabaseUrl: string,
  supabaseKey: string
): Promise<DocumentAnalysisResponse> => {
  const apiUrl = `${supabaseUrl}/functions/v1/analyze-documents`;

  const response = await fetch(apiUrl, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${supabaseKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(
      errorData.error || `API error: ${response.status} ${response.statusText}`
    );
  }

  return await response.json();
};
