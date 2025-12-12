import { Upload, FileText, Loader, CheckCircle } from 'lucide-react';
import { AppStep } from '../context/AppContext';

interface StepIndicatorProps {
  currentStep: AppStep;
  onStepClick?: (step: AppStep) => void;
  isDisabled?: (step: AppStep) => boolean;
}

const steps: { id: AppStep; label: string; icon: React.ReactNode }[] = [
  { id: 'upload', label: 'Upload', icon: <Upload className="w-5 h-5" /> },
  { id: 'details', label: 'Details', icon: <FileText className="w-5 h-5" /> },
  { id: 'processing', label: 'Analyzing', icon: <Loader className="w-5 h-5" /> },
  { id: 'results', label: 'Results', icon: <CheckCircle className="w-5 h-5" /> },
];

export const StepIndicator = ({
  currentStep,
  onStepClick,
  isDisabled,
}: StepIndicatorProps) => {
  const currentStepIndex = steps.findIndex((s) => s.id === currentStep);

  return (
    <div className="bg-white border-b border-gray-200 sticky top-0 z-40 print:hidden">
      <div className="max-w-5xl mx-auto px-4 py-6 md:py-8">
        <div className="flex items-center justify-between relative">
          {steps.map((step, index) => {
            const isActive = step.id === currentStep;
            const isCompleted = index < currentStepIndex;
            const disabled = isDisabled?.(step.id) ?? false;
            const canClick = isCompleted || isActive || !disabled;

            return (
              <div key={step.id} className="flex items-center flex-1">
                <button
                  onClick={() => canClick && onStepClick?.(step.id)}
                  disabled={disabled}
                  className={`flex items-center gap-3 relative z-10 transition-all ${
                    disabled && !isActive ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'
                  }`}
                >
                  <div
                    className={`w-10 h-10 md:w-12 md:h-12 rounded-full flex items-center justify-center font-bold text-sm md:text-base transition-all ${
                      isActive
                        ? 'bg-blue-600 text-white shadow-lg'
                        : isCompleted
                          ? 'bg-green-600 text-white'
                          : 'bg-gray-200 text-gray-600'
                    }`}
                  >
                    {isCompleted ? (
                      <CheckCircle className="w-6 h-6 md:w-7 md:h-7" />
                    ) : (
                      <span>{index + 1}</span>
                    )}
                  </div>
                  <div className="hidden sm:block text-left">
                    <p
                      className={`text-xs md:text-sm font-semibold transition-colors ${
                        isActive
                          ? 'text-blue-600'
                          : isCompleted
                            ? 'text-green-600'
                            : 'text-gray-500'
                      }`}
                    >
                      {step.label}
                    </p>
                  </div>
                </button>

                {index < steps.length - 1 && (
                  <div
                    className={`flex-1 h-1 md:h-1.5 mx-2 md:mx-4 rounded-full transition-colors ${
                      isCompleted ? 'bg-green-600' : 'bg-gray-200'
                    }`}
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
