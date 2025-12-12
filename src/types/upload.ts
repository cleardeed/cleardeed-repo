export interface UploadedFile {
  id: string;
  file: File;
  name: string;
  size: number;
  error?: string;
}

export const MAX_FILES = 10;
export const MAX_FILE_SIZE = 10 * 1024 * 1024;
export const ALLOWED_FILE_TYPE = 'application/pdf';

export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
};

export const validateFile = (file: File): string | null => {
  if (file.type !== ALLOWED_FILE_TYPE) {
    return 'Only PDF files are allowed';
  }
  if (file.size > MAX_FILE_SIZE) {
    return `File size must be less than ${formatFileSize(MAX_FILE_SIZE)}`;
  }
  return null;
};
