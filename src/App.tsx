import { useState } from 'react';
import { ArrowRight, CheckCircle } from 'lucide-react';
import { DropZone } from './components/DropZone';
import { FileList } from './components/FileList';
import { PropertyDetailsForm } from './components/PropertyDetailsForm';
import {
  UploadedFile,
  MAX_FILES,
  validateFile,
} from './types/upload';
import { PropertyFormData } from './types/property';

type AppStep = 'upload' | 'details';

function App() {
  const [currentStep, setCurrentStep] = useState<AppStep>('upload');
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [globalError, setGlobalError] = useState<string>('');

  const handleFilesSelected = (files: File[]) => {
    setGlobalError('');

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

    setUploadedFiles((prev) => [...prev, ...newFiles]);
  };

  const handleRemoveFile = (fileId: string) => {
    setUploadedFiles((prev) => prev.filter((f) => f.id !== fileId));
    setGlobalError('');
  };

  const validFiles = uploadedFiles.filter((f) => !f.error);
  const isMaxReached = uploadedFiles.length >= MAX_FILES;
  const canProceed = validFiles.length > 0;

  const handlePropertySubmit = (data: PropertyFormData) => {
    console.log('Property details submitted:', {
      files: uploadedFiles.filter((f) => !f.error).map((f) => f.name),
      propertyData: data,
    });
    alert('Property details submitted successfully! Check console for details.');
  };

  const handleBackToUpload = () => {
    setCurrentStep('upload');
  };

  const handleProceedToDetails = () => {
    setCurrentStep('details');
  };

  if (currentStep === 'details') {
    return (
      <PropertyDetailsForm
        onBack={handleBackToUpload}
        onSubmit={handlePropertySubmit}
      />
    );
  }

  return (
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
            disabled={!canProceed}
            className={`
              flex items-center gap-2 px-8 py-3 rounded-lg font-semibold
              transition-all duration-200 shadow-md
              ${
                canProceed
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
  );
}

export default App;
