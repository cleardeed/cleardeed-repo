import { FileText, X, AlertCircle } from 'lucide-react';
import { UploadedFile, formatFileSize } from '../types/upload';

interface FileListProps {
  files: UploadedFile[];
  onRemove: (fileId: string) => void;
}

export const FileList = ({ files, onRemove }: FileListProps) => {
  if (files.length === 0) {
    return null;
  }

  return (
    <div className="w-full">
      <h3 className="text-lg font-semibold text-gray-700 mb-4">
        Uploaded Files
      </h3>
      <div className="space-y-2">
        {files.map((uploadedFile) => (
          <div
            key={uploadedFile.id}
            className={`
              flex items-center justify-between p-4 rounded-lg border
              transition-all duration-200
              ${
                uploadedFile.error
                  ? 'bg-red-50 border-red-300'
                  : 'bg-white border-gray-200 hover:border-gray-300 hover:shadow-sm'
              }
            `}
          >
            <div className="flex items-center gap-3 flex-1 min-w-0">
              {uploadedFile.error ? (
                <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
              ) : (
                <FileText className="w-5 h-5 text-blue-500 flex-shrink-0" />
              )}

              <div className="flex-1 min-w-0">
                <p
                  className={`text-sm font-medium truncate ${
                    uploadedFile.error ? 'text-red-700' : 'text-gray-700'
                  }`}
                  title={uploadedFile.name}
                >
                  {uploadedFile.name}
                </p>
                {uploadedFile.error ? (
                  <p className="text-xs text-red-600 mt-1">
                    {uploadedFile.error}
                  </p>
                ) : (
                  <p className="text-xs text-gray-500 mt-1">
                    {formatFileSize(uploadedFile.size)}
                  </p>
                )}
              </div>
            </div>

            <button
              onClick={() => onRemove(uploadedFile.id)}
              className="ml-4 p-1 rounded-full hover:bg-gray-200 transition-colors flex-shrink-0"
              aria-label={`Remove ${uploadedFile.name}`}
            >
              <X className="w-5 h-5 text-gray-500" />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};
