import { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import { UploadedFile } from '../types/upload';
import { PropertyFormData } from '../types/property';
import { DocumentAnalysisResponse, ProcessedDocument } from '../services/api';

export type AppStep = 'upload' | 'processing' | 'results';

interface AppContextType {
  currentStep: AppStep;
  uploadedFiles: UploadedFile[];
  propertyData: PropertyFormData | null;
  allDocuments: ProcessedDocument[];
  analysisResults: DocumentAnalysisResponse | null;
  analysisError: string;
  globalError: string;

  setCurrentStep: (step: AppStep) => void;
  setUploadedFiles: (files: UploadedFile[]) => void;
  addUploadedFiles: (files: UploadedFile[]) => void;
  removeUploadedFile: (fileId: string) => void;
  clearUploadedFiles: () => void;

  setPropertyData: (data: PropertyFormData | null) => void;
  setAllDocuments: (docs: ProcessedDocument[]) => void;
  setAnalysisResults: (results: DocumentAnalysisResponse | null) => void;
  setAnalysisError: (error: string) => void;
  setGlobalError: (error: string) => void;
  clearGlobalError: () => void;

  startNewAnalysis: () => void;
  saveSessionData: () => void;
  restoreSessionData: () => void;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

const STORAGE_KEY = 'analysis_session';

export const AppProvider = ({ children }: { children: ReactNode }) => {
  const [currentStep, setCurrentStep] = useState<AppStep>('upload');
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [propertyData, setPropertyData] = useState<PropertyFormData | null>(null);
  const [allDocuments, setAllDocuments] = useState<ProcessedDocument[]>([]);
  const [analysisResults, setAnalysisResults] = useState<DocumentAnalysisResponse | null>(null);
  const [analysisError, setAnalysisError] = useState<string>('');
  const [globalError, setGlobalError] = useState<string>('');

  const addUploadedFiles = useCallback((files: UploadedFile[]) => {
    setUploadedFiles((prev) => [...prev, ...files]);
  }, []);

  const removeUploadedFile = useCallback((fileId: string) => {
    setUploadedFiles((prev) => prev.filter((f) => f.id !== fileId));
  }, []);

  const clearUploadedFiles = useCallback(() => {
    setUploadedFiles([]);
  }, []);

  const clearGlobalError = useCallback(() => {
    setGlobalError('');
  }, []);

  const saveSessionData = useCallback(() => {
    const sessionData = {
      uploadedFiles,
      propertyData,
      currentStep,
    };
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(sessionData));
  }, [uploadedFiles, propertyData, currentStep]);

  const restoreSessionData = useCallback(() => {
    const stored = sessionStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        const sessionData = JSON.parse(stored);
        if (sessionData.uploadedFiles) {
          setUploadedFiles(sessionData.uploadedFiles);
        }
        if (sessionData.propertyData) {
          setPropertyData(sessionData.propertyData);
        }
        if (sessionData.currentStep) {
          setCurrentStep(sessionData.currentStep);
        }
      } catch (error) {
        console.error('Failed to restore session data:', error);
      }
    }
  }, []);

  const startNewAnalysis = useCallback(() => {
    setCurrentStep('upload');
    setUploadedFiles([]);
    setPropertyData(null);
    setAllDocuments([]);
    setAnalysisResults(null);
    setAnalysisError('');
    setGlobalError('');
    sessionStorage.removeItem(STORAGE_KEY);
  }, []);

  const value: AppContextType = {
    currentStep,
    uploadedFiles,
    propertyData,
    allDocuments,
    analysisResults,
    analysisError,
    globalError,

    setCurrentStep,
    setUploadedFiles,
    addUploadedFiles,
    removeUploadedFile,
    clearUploadedFiles,

    setPropertyData,
    setAllDocuments,
    setAnalysisResults,
    setAnalysisError,
    setGlobalError,
    clearGlobalError,

    startNewAnalysis,
    saveSessionData,
    restoreSessionData,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

export const useAppContext = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useAppContext must be used within AppProvider');
  }
  return context;
};
