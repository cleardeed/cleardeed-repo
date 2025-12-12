import { useRef, useState } from 'react';
import { Upload, FileText } from 'lucide-react';

interface DropZoneProps {
  onFilesSelected: (files: File[]) => void;
  disabled: boolean;
}

export const DropZone = ({ onFilesSelected, disabled }: DropZoneProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    if (disabled) return;

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      onFilesSelected(files);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      onFilesSelected(files);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleClick = () => {
    if (!disabled) {
      fileInputRef.current?.click();
    }
  };

  return (
    <div
      onClick={handleClick}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`
        relative border-2 border-dashed rounded-lg p-12 text-center cursor-pointer
        transition-all duration-200 ease-in-out
        ${
          isDragging
            ? 'border-blue-500 bg-blue-50 scale-[1.02]'
            : disabled
            ? 'border-gray-300 bg-gray-50 cursor-not-allowed opacity-60'
            : 'border-gray-300 bg-white hover:border-blue-400 hover:bg-blue-50'
        }
      `}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,application/pdf"
        multiple
        onChange={handleFileInput}
        disabled={disabled}
        className="hidden"
      />

      <div className="flex flex-col items-center gap-4">
        {isDragging ? (
          <FileText className="w-16 h-16 text-blue-500 animate-pulse" />
        ) : (
          <Upload className="w-16 h-16 text-gray-400" />
        )}

        <div>
          <p className="text-lg font-semibold text-gray-700 mb-2">
            {isDragging ? 'Drop files here' : 'Drag and drop PDF files here'}
          </p>
          <p className="text-sm text-gray-500 mb-4">
            or click to browse your files
          </p>
          <div className="flex flex-col gap-1 text-xs text-gray-400">
            <span>Maximum 10 files</span>
            <span>PDF only, up to 10MB per file</span>
          </div>
        </div>
      </div>
    </div>
  );
};
