import { useEffect } from 'react';
import { ArrowRight, ArrowLeft, CheckCircle } from 'lucide-react';
import { DropZone } from './components/DropZone';
import { FileList } from './components/FileList';
import { PropertyDetailsForm } from './components/PropertyDetailsForm';
import { ProcessingScreen } from './components/ProcessingScreen';
import { AnalysisResults } from './components/AnalysisResults';
import { StepIndicator } from './components/StepIndicator';
import { useAppContext } from './context/AppContext';
import { MAX_FILES, validateFile } from './types/upload';
import { UploadedFile } from './types/upload';
import { analyzeDocuments, ProcessedDocument } from './services/api';

function AppContent() {
  const {
    currentStep,
    uploadedFiles,
    propertyData,
    allDocuments,
    analysisResults,
    analysisError,
    globalError,

    setCurrentStep,
    addUploadedFiles,
    removeUploadedFile,
    clearGlobalError,
    setPropertyData,
    setAllDocuments,
    setAnalysisResults,
    setAnalysisError,
    setGlobalError,
    startNewAnalysis,
    saveSessionData,
    restoreSessionData,
  } = useAppContext();

  useEffect(() => {
    restoreSessionData();
  }, [restoreSessionData]);

  const handleFilesSelected = (files: File[]) => {
    clearGlobalError();

    const remainingSlots = MAX_FILES - uploadedFiles.length;

    if (files.length > remainingSlots) {
      setGlobalError(
        `You can only upload ${remainingSlots} more file${
          remainingSlots === 1 ? '' : 's'
        }. Maximum ${MAX_FILES} files allowed.`
      );
      return;
    }

    const newFiles: UploadedFile[] = files.map((file) => ({
      id: `${file.name}-${Date.now()}-${Math.random()}`,
      file,
      name: file.name,
      size: file.size,
      error: validateFile(file) || undefined,
    }));

    addUploadedFiles(newFiles);
  };

  const handleRemoveFile = (fileId: string) => {
    removeUploadedFile(fileId);
    clearGlobalError();
  };

  const validFiles = uploadedFiles.filter((f) => !f.error);
  const isMaxReached = uploadedFiles.length >= MAX_FILES;
  const canProceedUpload = validFiles.length > 0;

  const handlePropertySubmit = (data: typeof propertyData) => {
    setPropertyData(data);
    saveSessionData();
    setCurrentStep('processing');
  };

  const handleBackToUpload = () => {
    setCurrentStep('upload');
  };

  const handleProceedToDetails = () => {
    saveSessionData();
    setCurrentStep('details');
  };

  const handleProcessingComplete = async (retrievedDocument: ProcessedDocument) => {
    const uploadedDocs: ProcessedDocument[] = validFiles.map((f) => ({
      id: f.id,
      name: f.name,
      source: 'uploaded' as const,
      size: f.size,
    }));

    const allDocs = [...uploadedDocs, retrievedDocument];
    setAllDocuments(allDocs);
    await startAnalysis(allDocs, propertyData!);
  };

  const startAnalysis = async (
    documents: ProcessedDocument[],
    property: typeof propertyData
  ) => {
    if (!property) return;

    try {
      setAnalysisError('');

      const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
      const supabaseKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

      if (!supabaseUrl || !supabaseKey) {
        throw new Error('Supabase configuration missing');
      }

      const analysisRequest = {
        documents: documents.map((doc) => ({
          name: doc.name,
          base64: 'mock_base64_data',
          size: doc.size,
        })),
        property_data: property,
      };

      const results = await analyzeDocuments(
        analysisRequest,
        supabaseUrl,
        supabaseKey
      );

      setAnalysisResults(results);
      setCurrentStep('results');
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Analysis failed';
      setAnalysisError(message);
      setCurrentStep('results');
    }
  };

  const handleStepClick = (step: typeof currentStep) => {
    if (step === 'upload') {
      setCurrentStep('upload');
    } else if (step === 'details' && validFiles.length > 0) {
      setCurrentStep('details');
    }
  };

  const isStepDisabled = (step: typeof currentStep): boolean => {
    if (step === 'details' && validFiles.length === 0) return true;
    if (step === 'processing' || step === 'results') return true;
    return false;
  };

  if (currentStep === 'results') {
    if (analysisError) {
      return (
        <>
          <StepIndicator currentStep="results" isDisabled={isStepDisabled} />
          <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center p-4">
            <div className="bg-white rounded-xl shadow-lg p-8 max-w-md w-full text-center">
              <h2 className="text-2xl font-bold text-red-700 mb-2">
                Analysis Failed
              </h2>
              <p className="text-gray-600 mb-6">{analysisError}</p>
              <button
                onClick={startNewAnalysis}
                className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold transition-colors"
              >
                Start Over
              </button>
            </div>
          </div>
        </>
      );
    }

    if (analysisResults && propertyData && allDocuments) {
      const documentNames = allDocuments.map((doc) => doc.name);
      return (
        <>
          <StepIndicator currentStep="results" isDisabled={isStepDisabled} />
          <AnalysisResults
            results={analysisResults}
            propertyData={propertyData}
            documentNames={documentNames}
            onStartNew={startNewAnalysis}
          />
        </>
      );
    }

    return (
      <>
        <StepIndicator currentStep="results" isDisabled={isStepDisabled} />
        <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center">
          <div className="text-center">
            <p className="text-gray-600">Analyzing documents...</p>
          </div>
        </div>
      </>
    );
  }

  if (currentStep === 'processing') {
    return (
      <>
        <StepIndicator currentStep="processing" isDisabled={isStepDisabled} />
        <ProcessingScreen onComplete={handleProcessingComplete} />
      </>
    );
  }

  if (currentStep === 'details') {
    return (
      <>
        <StepIndicator currentStep="details" isDisabled={isStepDisabled} />
        <PropertyDetailsForm
          onBack={handleBackToUpload}
          onSubmit={handlePropertySubmit}
          initialData={propertyData || undefined}
        />
      </>
    );
  }

  return (
    <>
      <StepIndicator currentStep="upload" isDisabled={isStepDisabled} />
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
        <div className="max-w-4xl mx-auto px-4 py-12">
          <div className="mb-8">
            <h1 className="text-4xl font-bold text-gray-800 mb-2">
              Upload Documents for Validation
            </h1>
            <p className="text-gray-600">
              Upload your property documents for AI-powered verification and analysis
            </p>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-8 mb-6">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-gray-700">
                  Files Uploaded:
                </span>
                <span
                  className={`text-lg font-bold ${
                    uploadedFiles.length === MAX_FILES
                      ? 'text-orange-600'
                      : 'text-blue-600'
                  }`}
                >
                  {uploadedFiles.length}/{MAX_FILES}
                </span>
              </div>

              {validFiles.length > 0 && (
                <div className="flex items-center gap-2 text-green-600">
                  <CheckCircle className="w-5 h-5" />
                  <span className="text-sm font-medium">
                    {validFiles.length} valid file{validFiles.length !== 1 ? 's' : ''}
                  </span>
                </div>
              )}
            </div>

            {globalError && (
              <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-sm text-red-700">{globalError}</p>
              </div>
            )}

            <DropZone
              onFilesSelected={handleFilesSelected}
              disabled={isMaxReached}
            />

            {isMaxReached && (
              <p className="mt-4 text-sm text-orange-600 text-center font-medium">
                Maximum file limit reached. Remove files to upload more.
              </p>
            )}
          </div>

          {uploadedFiles.length > 0 && (
            <div className="bg-white rounded-xl shadow-lg p-8 mb-6">
              <FileList files={uploadedFiles} onRemove={handleRemoveFile} />
            </div>
          )}

          <div className="flex justify-end">
            <button
              onClick={handleProceedToDetails}
              disabled={!canProceedUpload}
              className={`
                flex items-center gap-2 px-8 py-3 rounded-lg font-semibold
                transition-all duration-200 shadow-md
                ${
                  canProceedUpload
                    ? 'bg-blue-600 hover:bg-blue-700 text-white hover:shadow-lg hover:scale-105 active:scale-100'
                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                }
              `}
            >
              Next
              <ArrowRight className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>
    </>
  );
}

function App() {
  return <AppContent />;
}

export default App;
