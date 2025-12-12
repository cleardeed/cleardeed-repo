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
