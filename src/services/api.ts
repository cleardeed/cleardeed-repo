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

const generateMockPDF = (): ProcessedDocument => ({
  id: `doc-${Date.now()}`,
  name: `property-document-${Date.now()}.pdf`,
  source: 'retrieved',
  size: Math.floor(Math.random() * 5000000) + 1000000,
  retrievedAt: new Date().toISOString(),
});

export const callExternalApi = async (
  onStatusChange: (status: string) => void,
  signal?: AbortSignal
): Promise<ApiResponse> => {
  try {
    onStatusChange('Uploading documents...');
    if (signal?.aborted) throw new Error('Operation cancelled');
    await delay(1200);

    onStatusChange('Retrieving additional documents...');
    if (signal?.aborted) throw new Error('Operation cancelled');
    await delay(1500);

    onStatusChange('Analyzing documents with AI...');
    if (signal?.aborted) throw new Error('Operation cancelled');
    await delay(1300);

    const retrievedDocument = generateMockPDF();

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
