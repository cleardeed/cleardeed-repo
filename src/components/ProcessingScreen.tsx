import { useEffect, useState, useRef } from 'react';
import { Loader, CheckCircle, AlertCircle, RotateCcw, FileText, ArrowRight } from 'lucide-react';
import { callExternalApi, ProcessedDocument } from '../services/api';
import { PropertyFormData } from '../types/property';
import { UploadedFile } from '../types/upload';
import { cascadingData } from '../utils/dropdownProcessor';

interface ProcessingScreenProps {
  onComplete: (retrievedDocument: ProcessedDocument) => void;
  propertyData: PropertyFormData | null;
  uploadedFiles: UploadedFile[];
}

type ProcessingStatus = 'processing' | 'success' | 'error';

export const ProcessingScreen = ({ onComplete, propertyData, uploadedFiles }: ProcessingScreenProps) => {
  const [status, setStatus] = useState<ProcessingStatus>('processing');
  const [currentMessage, setCurrentMessage] = useState('Initializing...');
  const [progress, setProgress] = useState(0);
  const [errorMessage, setErrorMessage] = useState('');
  const [retrievedDocument, setRetrievedDocument] =
    useState<ProcessedDocument | null>(null);
  const abortControllerRef = useRef<AbortController>(new AbortController());
  const progressIntervalRef = useRef<NodeJS.Timeout>();

  // Helper function to get display name from value
  const getDisplayName = (value: string, type: 'zone' | 'district' | 'sro' | 'village'): string => {
    if (!value || !propertyData) return value;
    
    try {
      switch (type) {
        case 'zone':
          return cascadingData.zones.find(z => z.value === value)?.name || value;
        case 'district':
          const districts = cascadingData.districtsByZone.get(propertyData.zone) || [];
          return districts.find(d => d.value === value)?.name || value;
        case 'sro':
          const sros = cascadingData.srosByDistrict.get(propertyData.district) || [];
          return sros.find(s => s.value === value)?.name || value;
        case 'village':
          const villages = cascadingData.villagesBySro.get(propertyData.sro_name) || [];
          return villages.find(v => v.value === value)?.name || value;
        default:
          return value;
      }
    } catch (error) {
      return value;
    }
  };

  useEffect(() => {
    let isMounted = true;
    
    const processDocuments = async () => {
      try {
        let currentProgress = 0;
        progressIntervalRef.current = setInterval(() => {
          if (isMounted) {
            currentProgress = Math.min(currentProgress + Math.random() * 15, 85);
            setProgress(currentProgress);
          }
        }, 300);

        const result = await callExternalApi(
          (msg) => {
            if (isMounted) setCurrentMessage(msg);
          },
          {
            zone: propertyData?.zone || '',
            district: propertyData?.district || '',
            sro_name: propertyData?.sro_name || '',
            village: propertyData?.village || '',
            survey_number: propertyData?.survey_number || '',
            subdivision: propertyData?.subdivision || '',
          },
          abortControllerRef.current.signal
        );

        if (!isMounted) return;

        clearInterval(progressIntervalRef.current);
        setProgress(100);

        if (result.success && result.retrievedDocument) {
          setRetrievedDocument(result.retrievedDocument);
          setStatus('success');
          setCurrentMessage('Processing complete!');
          // Don't auto-navigate, wait for user confirmation
        } else {
          setStatus('error');
          setErrorMessage(
            result.error || 'Failed to process documents. Please try again.'
          );
        }
      } catch (error) {
        if (!isMounted) return;
        
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
      isMounted = false;
      clearInterval(progressIntervalRef.current);
      // Don't abort in cleanup - let the request complete
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
          {
            zone: propertyData?.zone || '',
            district: propertyData?.district || '',
            sro_name: propertyData?.sro_name || '',
            village: propertyData?.village || '',
            survey_number: propertyData?.survey_number || '',
            subdivision: propertyData?.subdivision || '',
          },
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
    <div className="pt-[100px] px-6 pb-6">
      <div className="grid grid-cols-1 lg:grid-cols-[300px_1fr] gap-4">
        {/* Left Pane - Property Summary */}
        <div className="bg-white rounded-xl shadow-lg p-4 space-y-4">
          {/* Property Details Section */}
          <div>
            <h2 className="text-base font-bold text-gray-800 mb-3 pb-2 border-b border-gray-200">
              Property Details
            </h2>
            {propertyData ? (
              <div className="space-y-2 text-sm">
                <div className="flex">
                  <span className="font-semibold text-gray-600 w-24 flex-shrink-0">Zone:</span>
                  <span className="text-gray-800">{getDisplayName(propertyData.zone, 'zone') || '-'}</span>
                </div>
                <div className="flex">
                  <span className="font-semibold text-gray-600 w-24 flex-shrink-0">District:</span>
                  <span className="text-gray-800">{getDisplayName(propertyData.district, 'district') || '-'}</span>
                </div>
                <div className="flex">
                  <span className="font-semibold text-gray-600 w-24 flex-shrink-0">SRO:</span>
                  <span className="text-gray-800">{getDisplayName(propertyData.sro_name, 'sro') || '-'}</span>
                </div>
                <div className="flex">
                  <span className="font-semibold text-gray-600 w-24 flex-shrink-0">Village:</span>
                  <span className="text-gray-800">{getDisplayName(propertyData.village, 'village') || '-'}</span>
                </div>
                <div className="flex">
                  <span className="font-semibold text-gray-600 w-24 flex-shrink-0">Survey #:</span>
                  <span className="text-gray-800">{propertyData.survey_number || '-'}</span>
                </div>
                {propertyData.subdivision && (
                  <div className="flex">
                    <span className="font-semibold text-gray-600 w-24 flex-shrink-0">Subdivision:</span>
                    <span className="text-gray-800">{propertyData.subdivision}</span>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-sm text-gray-500">No property details available</p>
            )}
          </div>

          {/* Uploaded Documents Section */}
          <div>
            <h2 className="text-base font-bold text-gray-800 mb-3 pb-2 border-b border-gray-200">
              Documents ({uploadedFiles.length})
            </h2>
            {uploadedFiles.length > 0 ? (
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {uploadedFiles.map((file) => (
                  <div key={file.id} className="flex items-start gap-2 text-sm">
                    <FileText className="w-4 h-4 text-blue-500 flex-shrink-0 mt-0.5" />
                    <div className="flex-1 min-w-0">
                      <p className="text-gray-800 truncate" title={file.name}>
                        {file.name}
                      </p>
                      <p className="text-xs text-gray-500">
                        {(file.size / (1024 * 1024)).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-gray-500">No documents uploaded</p>
            )}
          </div>
        </div>

        {/* Right Pane - Processing Progress */}
        <div className="bg-white rounded-xl shadow-lg p-8 flex items-center justify-center">
          <div className="flex flex-col items-center max-w-md w-full">
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

              <div className="space-y-3 text-sm w-full">
                <div className="flex items-center gap-3">
                  {progress < 33 ? (
                    <Loader className="w-5 h-5 text-blue-500 animate-spin flex-shrink-0" />
                  ) : (
                    <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0" />
                  )}
                  <div className="flex-1">
                    <p className={`font-medium ${progress < 33 ? 'text-blue-600' : 'text-green-600'}`}>
                      Fetching Encumbrance Certificate
                    </p>
                    <p className="text-xs text-gray-500 mt-0.5">
                      Retrieving property records and documents
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center gap-3">
                  {progress < 33 ? (
                    <div className="w-5 h-5 border-2 border-gray-300 rounded-full flex-shrink-0" />
                  ) : progress < 66 ? (
                    <Loader className="w-5 h-5 text-blue-500 animate-spin flex-shrink-0" />
                  ) : (
                    <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0" />
                  )}
                  <div className="flex-1">
                    <p className={`font-medium ${progress < 33 ? 'text-gray-400' : progress < 66 ? 'text-blue-600' : 'text-green-600'}`}>
                      Analyzing Documents
                    </p>
                    <p className="text-xs text-gray-500 mt-0.5">
                      AI-powered document verification and analysis
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center gap-3">
                  {progress < 66 ? (
                    <div className="w-5 h-5 border-2 border-gray-300 rounded-full flex-shrink-0" />
                  ) : progress < 100 ? (
                    <Loader className="w-5 h-5 text-blue-500 animate-spin flex-shrink-0" />
                  ) : (
                    <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0" />
                  )}
                  <div className="flex-1">
                    <p className={`font-medium ${progress < 66 ? 'text-gray-400' : progress < 100 ? 'text-blue-600' : 'text-green-600'}`}>
                      Preparing Recommendations
                    </p>
                    <p className="text-xs text-gray-500 mt-0.5">
                      Generating detailed findings and verdict
                    </p>
                  </div>
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
                Analysis Complete!
              </h2>
              <p className="text-gray-600 text-center text-sm mb-6">
                Your documents have been successfully analyzed
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

              <button
                onClick={() => retrievedDocument && onComplete(retrievedDocument)}
                className="w-full px-6 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg font-semibold transition-all duration-200 shadow-md hover:shadow-lg flex items-center justify-center gap-2"
              >
                View Analysis Results
                <ArrowRight className="w-5 h-5" />
              </button>
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
    </div>
  );
};
