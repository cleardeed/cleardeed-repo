import {
  CheckCircle,
  AlertCircle,
  XCircle,
  ChevronDown,
  Download,
} from 'lucide-react';
import { DocumentAnalysisResponse } from '../services/api';
import { useState } from 'react';

interface AnalysisResultsProps {
  results: DocumentAnalysisResponse;
  onDownload?: () => void;
}

const verdictConfig = {
  APPROVED: {
    color: 'text-green-700',
    bgColor: 'bg-green-50',
    borderColor: 'border-green-200',
    icon: CheckCircle,
  },
  CONDITIONALLY_APPROVED: {
    color: 'text-amber-700',
    bgColor: 'bg-amber-50',
    borderColor: 'border-amber-200',
    icon: AlertCircle,
  },
  REJECTED: {
    color: 'text-red-700',
    bgColor: 'bg-red-50',
    borderColor: 'border-red-200',
    icon: XCircle,
  },
};

const severityConfig = {
  High: { color: 'bg-red-100 text-red-800', icon: 'H' },
  Medium: { color: 'bg-amber-100 text-amber-800', icon: 'M' },
  Low: { color: 'bg-blue-100 text-blue-800', icon: 'L' },
};

export const AnalysisResults = ({
  results,
  onDownload,
}: AnalysisResultsProps) => {
  const [expandedFindings, setExpandedFindings] = useState<Set<string>>(
    new Set()
  );

  const config = verdictConfig[results.verdict];
  const VerdictIcon = config.icon;

  const toggleFinding = (findingId: string) => {
    const newExpanded = new Set(expandedFindings);
    if (newExpanded.has(findingId)) {
      newExpanded.delete(findingId);
    } else {
      newExpanded.add(findingId);
    }
    setExpandedFindings(newExpanded);
  };

  const mandatoryFindings = results.findings.filter(
    (f) => f.category === 'Mandatory'
  );
  const optionalFindings = results.findings.filter(
    (f) => f.category === 'Optional'
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <div className="max-w-4xl mx-auto px-4 py-12">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            Analysis Results
          </h1>
          <p className="text-gray-600">
            Document validation and AI analysis report
          </p>
        </div>

        <div
          className={`rounded-xl shadow-lg p-8 mb-6 border-2 ${config.bgColor} ${config.borderColor}`}
        >
          <div className="flex items-center gap-4 mb-6">
            <VerdictIcon className={`w-12 h-12 ${config.color}`} />
            <div>
              <h2 className={`text-3xl font-bold ${config.color}`}>
                {results.verdict.replace(/_/g, ' ')}
              </h2>
              <p className="text-gray-600 mt-1">{results.summary}</p>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="bg-white rounded-lg p-4">
              <p className="text-sm text-gray-600 mb-1">Confidence Score</p>
              <p className="text-2xl font-bold text-blue-600">
                {results.confidence_score}%
              </p>
            </div>

            <div className="bg-white rounded-lg p-4">
              <p className="text-sm text-gray-600 mb-1">Documents Analyzed</p>
              <p className="text-2xl font-bold text-gray-800">
                {results.total_documents_analyzed}
              </p>
            </div>

            <div className="bg-white rounded-lg p-4">
              <p className="text-sm text-gray-600 mb-1">Findings</p>
              <p className="text-2xl font-bold text-gray-800">
                {results.findings.length}
              </p>
            </div>
          </div>

          {onDownload && (
            <button
              onClick={onDownload}
              className="flex items-center gap-2 px-6 py-2 rounded-lg bg-white hover:bg-gray-50 text-gray-700 font-semibold border border-gray-300 transition-colors"
            >
              <Download className="w-5 h-5" />
              Download Report
            </button>
          )}
        </div>

        {mandatoryFindings.length > 0 && (
          <div className="bg-white rounded-xl shadow-lg p-8 mb-6">
            <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <AlertCircle className="w-6 h-6 text-red-600" />
              Mandatory Findings
            </h3>

            <div className="space-y-3">
              {mandatoryFindings.map((finding) => (
                <div
                  key={finding.id}
                  className="border border-gray-200 rounded-lg overflow-hidden"
                >
                  <button
                    onClick={() => toggleFinding(finding.id)}
                    className="w-full p-4 hover:bg-gray-50 transition-colors flex items-start justify-between"
                  >
                    <div className="flex items-start gap-3 flex-1 text-left">
                      <div
                        className={`px-2 py-1 rounded text-xs font-semibold flex-shrink-0 mt-0.5 ${
                          severityConfig[finding.severity].color
                        }`}
                      >
                        {finding.severity}
                      </div>

                      <div className="flex-1 min-w-0">
                        <p className="font-semibold text-gray-800">
                          {finding.title}
                        </p>
                        <p className="text-sm text-gray-500 mt-1">
                          Affects:{' '}
                          {finding.affected_documents.join(', ') || 'All documents'}
                        </p>
                      </div>
                    </div>

                    <ChevronDown
                      className={`w-5 h-5 text-gray-400 flex-shrink-0 transition-transform ${
                        expandedFindings.has(finding.id) ? 'rotate-180' : ''
                      }`}
                    />
                  </button>

                  {expandedFindings.has(finding.id) && (
                    <div className="px-4 py-3 bg-gray-50 border-t border-gray-200">
                      <p className="text-sm text-gray-700">
                        {finding.description}
                      </p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {optionalFindings.length > 0 && (
          <div className="bg-white rounded-xl shadow-lg p-8">
            <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <AlertCircle className="w-6 h-6 text-amber-600" />
              Optional Findings
            </h3>

            <div className="space-y-3">
              {optionalFindings.map((finding) => (
                <div
                  key={finding.id}
                  className="border border-gray-200 rounded-lg overflow-hidden"
                >
                  <button
                    onClick={() => toggleFinding(finding.id)}
                    className="w-full p-4 hover:bg-gray-50 transition-colors flex items-start justify-between"
                  >
                    <div className="flex items-start gap-3 flex-1 text-left">
                      <div
                        className={`px-2 py-1 rounded text-xs font-semibold flex-shrink-0 mt-0.5 ${
                          severityConfig[finding.severity].color
                        }`}
                      >
                        {finding.severity}
                      </div>

                      <div className="flex-1 min-w-0">
                        <p className="font-semibold text-gray-800">
                          {finding.title}
                        </p>
                        <p className="text-sm text-gray-500 mt-1">
                          Affects:{' '}
                          {finding.affected_documents.join(', ') || 'All documents'}
                        </p>
                      </div>
                    </div>

                    <ChevronDown
                      className={`w-5 h-5 text-gray-400 flex-shrink-0 transition-transform ${
                        expandedFindings.has(finding.id) ? 'rotate-180' : ''
                      }`}
                    />
                  </button>

                  {expandedFindings.has(finding.id) && (
                    <div className="px-4 py-3 bg-gray-50 border-t border-gray-200">
                      <p className="text-sm text-gray-700">
                        {finding.description}
                      </p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {results.findings.length === 0 && (
          <div className="bg-white rounded-xl shadow-lg p-8 text-center">
            <CheckCircle className="w-16 h-16 text-green-600 mx-auto mb-4" />
            <h3 className="text-xl font-bold text-gray-800">
              No issues found
            </h3>
            <p className="text-gray-600 mt-2">
              All documents have been validated successfully
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
