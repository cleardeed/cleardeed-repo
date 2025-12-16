import { useState } from 'react';
import { Upload, FileText, X, AlertCircle, CheckCircle } from 'lucide-react';

interface DocumentCategory {
  id: string;
  label: string;
  description: string;
  maxFiles: number;
  required: boolean;
}

interface CategorizedFile {
  id: string;
  file: File;
  category: string;
  name: string;
  size: number;
  error?: string;
}

interface CategorizedUploadProps {
  onFilesChange: (files: CategorizedFile[]) => void;
  uploadedFiles: CategorizedFile[];
}

const categories: DocumentCategory[] = [
  {
    id: 'property-document',
    label: 'Latest Property Document',
    description: 'Sale deed, gift deed, or latest ownership document',
    maxFiles: 1,
    required: true,
  },
  {
    id: 'encumbrance-certificate',
    label: 'Encumbrance Certificate (30 years)',
    description: 'EC for the last 30 years from registration office',
    maxFiles: 1,
    required: false,
  },
  {
    id: 'other-documents',
    label: 'Other Property Documents',
    description: 'Patta, tax receipts, survey documents, etc.',
    maxFiles: 8,
    required: false,
  },
];

export const CategorizedUpload = ({ onFilesChange, uploadedFiles }: CategorizedUploadProps) => {
  const [draggedCategory, setDraggedCategory] = useState<string | null>(null);

  const handleFileSelect = (category: string, files: FileList | null) => {
    if (!files) return;

    const categoryFiles = uploadedFiles.filter((f) => f.category === category);
    const categoryConfig = categories.find((c) => c.id === category);
    
    if (!categoryConfig) return;

    const remainingSlots = categoryConfig.maxFiles - categoryFiles.length;
    const filesToAdd = Array.from(files).slice(0, remainingSlots);

    const newFiles: CategorizedFile[] = filesToAdd.map((file) => {
      let error: string | undefined;

      if (file.type !== 'application/pdf') {
        error = 'Only PDF files are allowed';
      } else if (file.size > 50 * 1024 * 1024) {
        error = 'File size must be less than 50MB';
      }

      return {
        id: `${category}-${file.name}-${Date.now()}-${Math.random()}`,
        file,
        category,
        name: file.name,
        size: file.size,
        error,
      };
    });

    onFilesChange([...uploadedFiles, ...newFiles]);
  };

  const handleRemoveFile = (fileId: string) => {
    onFilesChange(uploadedFiles.filter((f) => f.id !== fileId));
  };

  const handleDragOver = (e: React.DragEvent, category: string) => {
    e.preventDefault();
    setDraggedCategory(category);
  };

  const handleDragLeave = () => {
    setDraggedCategory(null);
  };

  const handleDrop = (e: React.DragEvent, category: string) => {
    e.preventDefault();
    setDraggedCategory(null);
    handleFileSelect(category, e.dataTransfer.files);
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <div className="space-y-3">
      {categories.map((category) => {
        const categoryFiles = uploadedFiles.filter((f) => f.category === category.id);
        const canUploadMore = categoryFiles.length < category.maxFiles;
        const isDragging = draggedCategory === category.id;

        return (
          <div key={category.id} className="bg-white rounded-lg shadow p-3">
            <div className="flex items-center gap-3">
              {/* Label Section */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <h3 className="text-sm font-semibold text-gray-800">
                    {category.label}
                  </h3>
                  {category.required && (
                    <span className="text-xs text-red-500 font-medium">*</span>
                  )}
                </div>
                <p className="text-xs text-gray-500 mt-0.5 truncate">{category.description}</p>
              </div>

              {/* Upload Area - Slightly Larger */}
              {canUploadMore && (
                <div
                  onDragOver={(e) => handleDragOver(e, category.id)}
                  onDragLeave={handleDragLeave}
                  onDrop={(e) => handleDrop(e, category.id)}
                  className={`
                    relative border-2 border-dashed rounded-lg px-6 py-0.5 cursor-pointer
                    transition-all duration-200 flex-shrink-0 min-w-[250px]
                    ${
                      isDragging
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
                    }
                  `}
                >
                  <input
                    type="file"
                    accept=".pdf"
                    multiple={category.maxFiles > 1}
                    onChange={(e) => handleFileSelect(category.id, e.target.files)}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  />
                  <div className="flex flex-col items-center gap-1">
                    <Upload className="w-5 h-5 text-gray-400" />
                    <span className="text-xs text-gray-600 font-medium whitespace-nowrap">
                      Drop or Click
                    </span>
                    <span className="text-xs text-gray-500 font-medium">
                      {categoryFiles.length}/{category.maxFiles}
                    </span>
                  </div>
                </div>
              )}

              {/* Counter when upload is full */}
              {!canUploadMore && (
                <div className="flex items-center gap-2 px-3 py-2 bg-green-50 rounded-lg flex-shrink-0">
                  <CheckCircle className="w-4 h-4 text-green-600" />
                  <span className="text-xs text-green-700 font-medium whitespace-nowrap">
                    {categoryFiles.length}/{category.maxFiles}
                  </span>
                </div>
              )}
            </div>

            {/* Uploaded Files List - More Compact */}
            {categoryFiles.length > 0 && (
              <div className="mt-2 space-y-1">
                {categoryFiles.map((file) => (
                  <div
                    key={file.id}
                    className={`
                      flex items-center gap-2 p-1.5 rounded border text-xs
                      ${file.error ? 'bg-red-50 border-red-200' : 'bg-gray-50 border-gray-200'}
                    `}
                  >
                    {file.error ? (
                      <AlertCircle className="w-3 h-3 text-red-500 flex-shrink-0" />
                    ) : (
                      <CheckCircle className="w-3 h-3 text-green-500 flex-shrink-0" />
                    )}
                    <FileText className="w-3 h-3 text-gray-600 flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-gray-800 truncate">
                        {file.name}
                      </p>
                      {file.error && (
                        <span className="text-red-600">• {file.error}</span>
                      )}
                    </div>
                    <span className="text-gray-500 flex-shrink-0">
                      {formatFileSize(file.size)}
                    </span>
                    <button
                      onClick={() => handleRemoveFile(file.id)}
                      className="p-0.5 hover:bg-gray-200 rounded transition-colors flex-shrink-0"
                      title="Remove file"
                    >
                      <X className="w-3 h-3 text-gray-600" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};
