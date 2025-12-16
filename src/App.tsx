import { useEffect, useState } from 'react';
import { ArrowRight, ArrowLeft, CheckCircle, FileText, Archive, FolderOpen, Search } from 'lucide-react';
import { DropZone } from './components/DropZone';
import { FileList } from './components/FileList';
import { CategorizedUpload } from './components/CategorizedUpload';
import { PropertyDetailsForm } from './components/PropertyDetailsForm';
import { ProcessingScreen } from './components/ProcessingScreen';
import { AnalysisResults } from './components/AnalysisResults';
import { StepIndicator } from './components/StepIndicator';
import { useAppContext } from './context/AppContext';
import { MAX_FILES, validateFile } from './types/upload';
import { UploadedFile } from './types/upload';
import { ProcessedDocument } from './services/api';
import { getAllVillages, villageSearchMap } from './utils/dropdownProcessor';

// Backend API configuration
const BACKEND_URL = 'http://localhost:8000';

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

  const [activeMenuItem, setActiveMenuItem] = useState('property-analysis');
  const [villageSearchText, setVillageSearchText] = useState('');
  const [showVillageSuggestions, setShowVillageSuggestions] = useState(false);
  const [allVillages] = useState(() => getAllVillages());
  const [categorizedFiles, setCategorizedFiles] = useState<any[]>([]);

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
  const validCategorizedFiles = categorizedFiles.filter((f) => !f.error);
  const hasPropertyDocument = validCategorizedFiles.some((f) => f.category === 'property-document');
  const canProceedUpload = hasPropertyDocument;

  const handlePropertySubmit = (data: typeof propertyData) => {
    console.log('Property data updated:', data);
    setPropertyData(data);
  };

  const handleVillageSearch = (searchValue: string) => {
    setVillageSearchText(searchValue);
    setShowVillageSuggestions(searchValue.length > 0);
  };

  const handleVillageSelect = (villageValue: string) => {
    const hierarchy = villageSearchMap.get(villageValue);
    if (hierarchy) {
      setPropertyData({
        zone: hierarchy.zone.value,
        district: hierarchy.district.value,
        sro_name: hierarchy.sro.value,
        village: hierarchy.village.value,
        survey_number: propertyData?.survey_number || '',
        subdivision: propertyData?.subdivision || '',
      });
      setVillageSearchText(hierarchy.village.name);
      setShowVillageSuggestions(false);
    }
  };

  const filteredVillages = villageSearchText.length > 0
    ? allVillages.filter(v => 
        v.name.toLowerCase().includes(villageSearchText.toLowerCase())
      ).slice(0, 10)
    : [];

  const handleBackToUpload = () => {
    setCurrentStep('upload');
  };

  const handleProceedToAnalysis = async () => {
    if (propertyData) {
      saveSessionData();
      
      console.log('=== PROCEED TO ANALYSIS ===');
      console.log('Categorized files count:', validCategorizedFiles.length);
      console.log('Files:', validCategorizedFiles);
      
      // If files are already uploaded, skip the processing screen (Selenium download)
      // and go directly to analysis
      if (validCategorizedFiles.length > 0) {
        console.log('✅ Files detected - going directly to backend analysis');
        
        // Move to analysis page immediately
        setCurrentStep('analysis');
        
        const uploadedDocs: ProcessedDocument[] = validCategorizedFiles.map((f) => ({
          id: f.id,
          name: f.name,
          source: 'uploaded' as const,
          size: f.size,
        }));
        
        setAllDocuments(uploadedDocs);
        await startAnalysis(uploadedDocs, propertyData);
      } else {
        // No files uploaded, need to retrieve EC via Selenium
        console.log('⚠️ No files - showing Selenium processing screen');
        setCurrentStep('processing');
      }
    }
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
      console.log('🚀 Starting analysis...');
      setAnalysisError('');

      // Step 1: Upload all PDF files to backend (without immediate processing)
      console.log('📤 Step 1: Uploading documents to backend...');
      console.log('📋 Processing the documents - please wait...');
      
      const uploadedDocIds: string[] = [];
      
      for (const doc of documents) {
        console.log(`Processing document: ${doc.name}, source: ${doc.source}`);
        if (doc.source === 'uploaded') {
          // Find the actual file from categorizedFiles
          const fileItem = categorizedFiles.find(f => f.name === doc.name);
          console.log(`File item found:`, fileItem ? 'YES' : 'NO');
          if (fileItem && fileItem.file) {
            console.log(`Uploading ${doc.name}...`);
            const formData = new FormData();
            formData.append('file', fileItem.file);
            formData.append('document_type', fileItem.category || 'property-document');
            formData.append('process_immediately', 'false');  // Process in background

            const uploadRes = await fetch(`${BACKEND_URL}/api/documents/upload`, {
              method: 'POST',
              body: formData,
            });

            console.log(`Upload response status: ${uploadRes.status}`);
            if (!uploadRes.ok) {
              const errorText = await uploadRes.text();
              console.error(`Upload failed:`, errorText);
              throw new Error(`Failed to upload ${doc.name}`);
            }

            const uploadData = await uploadRes.json();
            console.log(`✅ Uploaded successfully, doc_id: ${uploadData.document_id}`);
            uploadedDocIds.push(uploadData.document_id);
          }
        }
      }

      console.log(`Total uploaded: ${uploadedDocIds.length} documents`);
      if (uploadedDocIds.length === 0) {
        throw new Error('No documents were uploaded successfully');
      }

      // Step 2: Generate embeddings for all documents
      console.log('🔢 Step 2: Generating embeddings...');
      for (const docId of uploadedDocIds) {
        console.log(`Generating embeddings for ${docId}...`);
        const embeddingRes = await fetch(`${BACKEND_URL}/api/embeddings/generate/${docId}`, {
          method: 'POST',
        });

        console.log(`Embeddings response status: ${embeddingRes.status}`);
        if (!embeddingRes.ok) {
          console.warn(`Failed to generate embeddings for ${docId}`);
        } else {
          console.log(`✅ Embeddings generated for ${docId}`);
        }
      }

      // Step 3: Run GAP analysis on the first document
      console.log('🔍 Step 3: Running GAP analysis...');
      const primaryDocId = uploadedDocIds[0];
      console.log(`Primary document: ${primaryDocId}`);
      const gapRes = await fetch(`${BACKEND_URL}/api/documents/${primaryDocId}/gap-analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          gap_prompt: 'Analyze this property document for gaps in Grantor, Attestation, and Property information',
          top_k: 10,
        }),
      });

      if (!gapRes.ok) {
        const errorData = await gapRes.json();
        throw new Error(errorData.error || 'GAP analysis failed');
      }

      const gapResults = await gapRes.json();
      
      // Format results for display
      const formattedResults = {
        gaps: gapResults.gaps || [],
        summary: gapResults.summary || 'Analysis complete',
        document_id: primaryDocId,
      };

      setAnalysisResults(formattedResults);
      setCurrentStep('results');
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Analysis failed';
      console.error('Analysis error:', error);
      setAnalysisError(message);
      setCurrentStep('results');
    }
  };

  const handleStepClick = (step: typeof currentStep) => {
    // Allow navigation to upload step always
    if (step === 'upload') {
      setCurrentStep('upload');
      return;
    }
    
    // Allow navigation to results if analysis has been completed
    if (step === 'results' && (analysisResults || analysisError)) {
      setCurrentStep('results');
      return;
    }
    
    // Don't allow direct navigation to processing step
    // It should only be accessed by clicking "Proceed to Analysis"
  };

  const isStepDisabled = (step: typeof currentStep): boolean => {
    // Processing step cannot be clicked directly
    if (step === 'processing') return true;
    
    // Results step is disabled if there are no results yet
    if (step === 'results' && !analysisResults && !analysisError) return true;
    
    return false;
  };

  const TopNavBar = () => (
    <nav className="bg-[#005f73] shadow-md border-b border-[#004d5c] fixed top-0 left-0 right-0 z-50">
      <div className="px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <h1 className="text-2xl font-bold text-white">ClearDeed</h1>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setActiveMenuItem('property-analysis')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-sm transition-colors ${
                activeMenuItem === 'property-analysis'
                  ? 'bg-[#0a9396] text-white border-b-2 border-white'
                  : 'text-gray-200 hover:bg-[#007f8c] hover:text-white'
              }`}
            >
              <FileText className="w-4 h-4" />
              Property Analysis
            </button>
            <button
              onClick={() => setActiveMenuItem('archived-analysis')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-sm transition-colors ${
                activeMenuItem === 'archived-analysis'
                  ? 'bg-[#0a9396] text-white border-b-2 border-white'
                  : 'text-gray-200 hover:bg-[#007f8c] hover:text-white'
              }`}
            >
              <Archive className="w-4 h-4" />
              Archived Analysis
            </button>
            <button
              onClick={() => setActiveMenuItem('e-documents')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-sm transition-colors ${
                activeMenuItem === 'e-documents'
                  ? 'bg-[#0a9396] text-white border-b-2 border-white'
                  : 'text-gray-200 hover:bg-[#007f8c] hover:text-white'
              }`}
            >
              <FolderOpen className="w-4 h-4" />
              E-Documents
            </button>
          </div>
        </div>
      </div>
    </nav>
  );

  if (currentStep === 'results') {
    if (analysisError) {
      return (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
          <TopNavBar />
          <StepIndicator currentStep="results" onStepClick={handleStepClick} isDisabled={isStepDisabled} />
          <div className="flex items-center justify-center p-4 h-[calc(100vh-145px)]">
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
        </div>
      );
    }

    if (analysisResults && propertyData && allDocuments) {
      const documentNames = allDocuments.map((doc) => doc.name);
      return (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
          <TopNavBar />
          <StepIndicator currentStep="results" onStepClick={handleStepClick} isDisabled={isStepDisabled} />
          <AnalysisResults
            results={analysisResults}
            propertyData={propertyData}
            documentNames={documentNames}
            onStartNew={startNewAnalysis}
          />
        </div>
      );
    }

    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
        <TopNavBar />
        <StepIndicator currentStep="results" onStepClick={handleStepClick} isDisabled={isStepDisabled} />
        <div className="flex items-center justify-center h-[calc(100vh-145px)]">
          <div className="text-center">
            <p className="text-gray-600">Analyzing documents...</p>
          </div>
        </div>
      </div>
    );
  }

  if (currentStep === 'processing') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
        <TopNavBar />
        <StepIndicator currentStep="processing" onStepClick={handleStepClick} isDisabled={isStepDisabled} />
        <ProcessingScreen 
          onComplete={handleProcessingComplete} 
          propertyData={propertyData}
          uploadedFiles={uploadedFiles}
        />
      </div>
    );
  }

  if (currentStep === 'analysis') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
        <TopNavBar />
        <StepIndicator currentStep="results" onStepClick={handleStepClick} isDisabled={isStepDisabled} />
        <div className="flex items-center justify-center h-[calc(100vh-145px)]">
          <div className="text-center">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
            <p className="text-lg font-semibold text-gray-800">Analyzing documents...</p>
            <p className="text-sm text-gray-600 mt-2">Uploading PDFs, generating embeddings, and running GAP analysis</p>
          </div>
        </div>
      </div>
    );
  }



  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <TopNavBar />
      <StepIndicator currentStep="upload" onStepClick={handleStepClick} isDisabled={isStepDisabled} />
      <div className="pt-[100px]">
        <div className="px-6 pb-2">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-800 mb-1">
                Property Details
              </h1>
              <p className="text-gray-600 text-sm">
                Enter property information and upload documents for AI-powered verification
              </p>
            </div>
            <button
              onClick={() => {
                console.log('Button clicked. State:', {
                  canProceedUpload,
                  propertyData,
                  zone: propertyData?.zone,
                  district: propertyData?.district,
                  sro_name: propertyData?.sro_name,
                  village: propertyData?.village,
                  survey_number: propertyData?.survey_number
                });
                handleProceedToAnalysis();
              }}
              disabled={
                !canProceedUpload || 
                !propertyData?.zone || 
                !propertyData?.district || 
                !propertyData?.sro_name || 
                !propertyData?.village ||
                !propertyData?.survey_number
              }
              className={`
                flex items-center gap-2 px-6 py-2.5 rounded-lg font-semibold text-sm
                transition-all duration-200 shadow-md whitespace-nowrap
                ${
                  canProceedUpload && 
                  propertyData?.zone && 
                  propertyData?.district && 
                  propertyData?.sro_name && 
                  propertyData?.village &&
                  propertyData?.survey_number
                    ? 'bg-blue-600 hover:bg-blue-700 text-white hover:shadow-lg hover:scale-105 active:scale-100'
                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                }
              `}
            >
              Proceed to Analysis
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="px-6 pb-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Left Pane - Property Details Form */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <div className="pb-3 border-b border-gray-200 mb-4">
                <h2 className="text-base font-bold text-gray-800">Property Details</h2>
              </div>

              {/* Village Search */}
              <div className="mb-4 relative">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Quick Search by Village Name
                </label>
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type="text"
                    value={villageSearchText}
                    onChange={(e) => handleVillageSearch(e.target.value)}
                    onFocus={() => villageSearchText.length > 0 && setShowVillageSuggestions(true)}
                    onBlur={() => setTimeout(() => setShowVillageSuggestions(false), 200)}
                    placeholder="Type village name..."
                    className="w-full pl-10 pr-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition"
                  />
                </div>
                {showVillageSuggestions && filteredVillages.length > 0 && (
                  <div className="absolute z-10 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-60 overflow-y-auto">
                    {filteredVillages.map((village) => {
                      const hierarchy = villageSearchMap.get(village.value);
                      return (
                        <div
                          key={village.value}
                          onClick={() => handleVillageSelect(village.value)}
                          className="px-3 py-2 hover:bg-blue-50 cursor-pointer border-b border-gray-100 last:border-b-0"
                        >
                          <div className="text-sm font-medium text-gray-800">{village.name}</div>
                          {hierarchy && (
                            <div className="text-xs text-gray-500 mt-0.5">
                              {hierarchy.zone.name} → {hierarchy.district.name} → {hierarchy.sro.name}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              <PropertyDetailsForm
                onBack={() => {}}
                onSubmit={handlePropertySubmit}
                initialData={propertyData || undefined}
                hideButtons={true}
              />
            </div>

            {/* Right Pane - Categorized Document Upload */}
            <div className="flex flex-col gap-3">
              <div className="bg-gray-50 rounded-xl shadow-lg p-4">
                <div className="mb-3">
                  <h2 className="text-base font-bold text-gray-800">Document Upload</h2>
                  <p className="text-xs text-gray-600 mt-1">
                    Upload property-related documents for AI analysis
                  </p>
                </div>

                <CategorizedUpload
                  uploadedFiles={categorizedFiles}
                  onFilesChange={setCategorizedFiles}
                />

                {globalError && (
                  <div className="mt-3 p-2 bg-red-50 border border-red-200 rounded text-xs text-red-700">
                    {globalError}
                  </div>
                )}
              </div>

              {/* Summary of uploaded files */}
              <div className="bg-white rounded-xl shadow-lg p-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-base font-bold text-gray-800">Upload Summary</h2>
                  {validCategorizedFiles.length > 0 && (
                    <div className="flex items-center gap-2 text-green-600">
                      <CheckCircle className="w-4 h-4" />
                      <span className="text-sm font-medium">{validCategorizedFiles.length} file(s) ready</span>
                    </div>
                  )}
                </div>
                <div className="mt-3 space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Property Document:</span>
                    <span className={`font-medium ${categorizedFiles.filter(f => f.category === 'property-document' && !f.error).length > 0 ? 'text-green-600' : 'text-gray-400'}`}>
                      {categorizedFiles.filter(f => f.category === 'property-document' && !f.error).length}/1
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Encumbrance Certificate:</span>
                    <span className={`font-medium ${categorizedFiles.filter(f => f.category === 'encumbrance-certificate' && !f.error).length > 0 ? 'text-green-600' : 'text-gray-400'}`}>
                      {categorizedFiles.filter(f => f.category === 'encumbrance-certificate' && !f.error).length}/1
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Other Documents:</span>
                    <span className={`font-medium ${categorizedFiles.filter(f => f.category === 'other-documents' && !f.error).length > 0 ? 'text-green-600' : 'text-gray-400'}`}>
                      {categorizedFiles.filter(f => f.category === 'other-documents' && !f.error).length}/8
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function App() {
  return <AppContent />;
}

export default App;
