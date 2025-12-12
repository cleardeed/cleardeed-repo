import { useEffect, useState, useRef } from 'react';
import { Loader, CheckCircle, AlertCircle, RotateCcw } from 'lucide-react';
import { callExternalApi, ProcessedDocument } from '../services/api';

interface ProcessingScreenProps {
  onComplete: (retrievedDocument: ProcessedDocument) => void;
}

type ProcessingStatus = 'processing' | 'success' | 'error';

export const ProcessingScreen = ({ onComplete }: ProcessingScreenProps) => {
  const [status, setStatus] = useState<ProcessingStatus>('processing');
  const [currentMessage, setCurrentMessage] = useState('Initializing...');
  const [progress, setProgress] = useState(0);
  const [errorMessage, setErrorMessage] = useState('');
  const [retrievedDocument, setRetrievedDocument] =
    useState<ProcessedDocument | null>(null);
  const abortControllerRef = useRef<AbortController>(new AbortController());
  const progressIntervalRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    const processDocuments = async () => {
      try {
        let currentProgress = 0;
        progressIntervalRef.current = setInterval(() => {
          currentProgress = Math.min(currentProgress + Math.random() * 15, 85);
          setProgress(currentProgress);
        }, 300);

        const result = await callExternalApi(
          (msg) => setCurrentMessage(msg),
          abortControllerRef.current.signal
        );

        clearInterval(progressIntervalRef.current);
        setProgress(100);

        if (result.success && result.retrievedDocument) {
          setRetrievedDocument(result.retrievedDocument);
          setStatus('success');
          setCurrentMessage('Processing complete!');

          setTimeout(() => {
            onComplete(result.retrievedDocument!);
          }, 1500);
        } else {
          setStatus('error');
          setErrorMessage(
            result.error || 'Failed to process documents. Please try again.'
          );
        }
      } catch (error) {
        clearInterval(progressIntervalRef.current);
        setStatus('error');
        setErrorMessage(
          error instanceof Error
            ? error.message
            : 'An unexpected error occurred'
        );
      }
    };

    processDocuments();

    return () => {
      clearInterval(progressIntervalRef.current);
      abortControllerRef.current.abort();
    };
  }, [onComplete]);

  const handleRetry = () => {
    setStatus('processing');
    setProgress(0);
    setErrorMessage('');
    setCurrentMessage('Initializing...');
    abortControllerRef.current = new AbortController();

    const processDocuments = async () => {
      try {
        let currentProgress = 0;
        progressIntervalRef.current = setInterval(() => {
          currentProgress = Math.min(currentProgress + Math.random() * 15, 85);
          setProgress(currentProgress);
        }, 300);

        const result = await callExternalApi(
          (msg) => setCurrentMessage(msg),
          abortControllerRef.current.signal
        );

        clearInterval(progressIntervalRef.current);
        setProgress(100);

        if (result.success && result.retrievedDocument) {
          setRetrievedDocument(result.retrievedDocument);
          setStatus('success');
          setCurrentMessage('Processing complete!');

          setTimeout(() => {
            onComplete(result.retrievedDocument!);
          }, 1500);
        } else {
          setStatus('error');
          setErrorMessage(
            result.error || 'Failed to process documents. Please try again.'
          );
        }
      } catch (error) {
        clearInterval(progressIntervalRef.current);
        setStatus('error');
        setErrorMessage(
          error instanceof Error
            ? error.message
            : 'An unexpected error occurred'
        );
      }
    };

    processDocuments();
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-2xl p-12 max-w-md w-full">
        <div className="flex flex-col items-center">
          {status === 'processing' && (
            <>
              <div className="mb-8">
                <Loader className="w-16 h-16 text-blue-600 animate-spin" />
              </div>

              <h2 className="text-2xl font-bold text-gray-800 text-center mb-2">
                Processing Documents
              </h2>
              <p className="text-gray-600 text-center text-sm mb-8">
                This may take a minute
              </p>

              <div className="w-full mb-6">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium text-gray-700">
                    {currentMessage}
                  </span>
                  <span className="text-sm font-semibold text-blue-600">
                    {Math.round(progress)}%
                  </span>
                </div>

                <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-blue-500 to-blue-600 h-full rounded-full transition-all duration-300 ease-out"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>

              <div className="space-y-2 text-xs text-gray-500 w-full">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full" />
                  <span>Uploading documents...</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full" />
                  <span>Retrieving additional documents...</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-gray-300 rounded-full" />
                  <span>Analyzing documents with AI...</span>
                </div>
              </div>
            </>
          )}

          {status === 'success' && (
            <>
              <div className="mb-8">
                <CheckCircle className="w-16 h-16 text-green-600" />
              </div>

              <h2 className="text-2xl font-bold text-gray-800 text-center mb-2">
                Processing Complete
              </h2>
              <p className="text-gray-600 text-center text-sm mb-6">
                Your documents have been successfully processed
              </p>

              {retrievedDocument && (
                <div className="w-full bg-green-50 border border-green-200 rounded-lg p-4 mb-6">
                  <p className="text-sm font-medium text-gray-700 mb-1">
                    Retrieved Document:
                  </p>
                  <p className="text-sm text-green-700 font-semibold truncate">
                    {retrievedDocument.name}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    {Math.round(retrievedDocument.size / 1024 / 1024)}MB
                  </p>
                </div>
              )}
            </>
          )}

          {status === 'error' && (
            <>
              <div className="mb-8">
                <AlertCircle className="w-16 h-16 text-red-600" />
              </div>

              <h2 className="text-2xl font-bold text-gray-800 text-center mb-2">
                Processing Failed
              </h2>
              <p className="text-red-600 text-center text-sm mb-6 font-medium">
                {errorMessage}
              </p>

              <button
                onClick={handleRetry}
                className="flex items-center gap-2 px-6 py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg font-semibold transition-all duration-200 w-full justify-center"
              >
                <RotateCcw className="w-5 h-5" />
                Retry
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
