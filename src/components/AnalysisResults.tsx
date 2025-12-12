import {
  CheckCircle,
  AlertCircle,
  XCircle,
  ChevronDown,
  Download,
  FileText,
  RefreshCw,
  TrendingUp,
  AlertTriangle,
} from 'lucide-react';
import { DocumentAnalysisResponse, AnalysisFinding } from '../services/api';
import { useState } from 'react';

interface AnalysisResultsProps {
  results: DocumentAnalysisResponse;
  onDownload?: () => void;
  onStartNew?: () => void;
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
  High: { color: 'bg-red-100 text-red-800 border-red-200', priority: 1 },
  Medium: { color: 'bg-amber-100 text-amber-800 border-amber-200', priority: 2 },
  Low: { color: 'bg-blue-100 text-blue-800 border-blue-200', priority: 3 },
};

type FindingTab = 'mandatory' | 'optional';

const sortFindingsBySeverity = (findings: AnalysisFinding[]): AnalysisFinding[] => {
  return [...findings].sort((a, b) => {
    return severityConfig[a.severity].priority - severityConfig[b.severity].priority;
  });
};

export const AnalysisResults = ({
  results,
  onDownload,
  onStartNew,
}: AnalysisResultsProps) => {
  const [expandedFindings, setExpandedFindings] = useState<Set<string>>(new Set());
  const [activeTab, setActiveTab] = useState<FindingTab>('mandatory');

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

  const mandatoryFindings = sortFindingsBySeverity(
    results.findings.filter((f) => f.category === 'Mandatory')
  );
  const optionalFindings = sortFindingsBySeverity(
    results.findings.filter((f) => f.category === 'Optional')
  );

  const displayedFindings = activeTab === 'mandatory' ? mandatoryFindings : optionalFindings;

  const FindingCard = ({ finding }: { finding: AnalysisFinding }) => (
    <div className="border border-gray-200 rounded-lg overflow-hidden hover:shadow-md transition-shadow">
      <button
        onClick={() => toggleFinding(finding.id)}
        className="w-full p-4 hover:bg-gray-50 transition-colors flex items-start justify-between"
      >
        <div className="flex items-start gap-3 flex-1 text-left">
          <div
            className={`px-3 py-1 rounded-md text-xs font-bold flex-shrink-0 mt-0.5 border ${
              severityConfig[finding.severity].color
            }`}
          >
            {finding.severity}
          </div>

          <div className="flex-1 min-w-0">
            <p className="font-bold text-gray-800 text-base leading-tight">
              {finding.title}
            </p>
            <p className="text-sm text-gray-500 mt-2 flex items-center gap-1">
              <FileText className="w-4 h-4" />
              {finding.affected_documents.length > 0
                ? finding.affected_documents.join(', ')
                : 'All documents'}
            </p>
          </div>
        </div>

        <ChevronDown
          className={`w-5 h-5 text-gray-400 flex-shrink-0 transition-transform ml-2 ${
            expandedFindings.has(finding.id) ? 'rotate-180' : ''
          }`}
        />
      </button>

      {expandedFindings.has(finding.id) && (
        <div className="px-4 py-4 bg-gray-50 border-t border-gray-200">
          <p className="text-sm text-gray-700 leading-relaxed">{finding.description}</p>
        </div>
      )}
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 print:bg-white">
      <div className="max-w-5xl mx-auto px-4 py-8 md:py-12">
        <div className="mb-8 print:mb-6">
          <h1 className="text-3xl md:text-4xl font-bold text-gray-800 mb-2">
            Analysis Results
          </h1>
          <p className="text-gray-600 text-sm md:text-base">
            Document validation and AI analysis report
          </p>
        </div>

        <div
          className={`rounded-xl shadow-lg p-6 md:p-8 mb-6 border-2 ${config.bgColor} ${config.borderColor} print:shadow-none print:border`}
        >
          <div className="flex flex-col sm:flex-row sm:items-center gap-4 mb-6">
            <VerdictIcon className={`w-12 h-12 md:w-16 md:h-16 ${config.color} flex-shrink-0`} />
            <div className="flex-1">
              <h2 className={`text-2xl md:text-3xl font-bold ${config.color} mb-1`}>
                {results.verdict.replace(/_/g, ' ')}
              </h2>
              <p className="text-gray-700 text-sm md:text-base leading-relaxed">
                {results.summary}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4">
            <div className="bg-white rounded-lg p-4 shadow-sm">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="w-4 h-4 text-blue-600" />
                <p className="text-xs font-semibold text-gray-600 uppercase tracking-wide">
                  Confidence
                </p>
              </div>
              <p className="text-2xl md:text-3xl font-bold text-blue-600">
                {results.confidence_score}%
              </p>
            </div>

            <div className="bg-white rounded-lg p-4 shadow-sm">
              <div className="flex items-center gap-2 mb-2">
                <FileText className="w-4 h-4 text-gray-600" />
                <p className="text-xs font-semibold text-gray-600 uppercase tracking-wide">
                  Documents
                </p>
              </div>
              <p className="text-2xl md:text-3xl font-bold text-gray-800">
                {results.total_documents_analyzed}
              </p>
            </div>

            <div className="bg-white rounded-lg p-4 shadow-sm">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="w-4 h-4 text-red-600" />
                <p className="text-xs font-semibold text-gray-600 uppercase tracking-wide">
                  Mandatory
                </p>
              </div>
              <p className="text-2xl md:text-3xl font-bold text-red-600">
                {mandatoryFindings.length}
              </p>
            </div>

            <div className="bg-white rounded-lg p-4 shadow-sm">
              <div className="flex items-center gap-2 mb-2">
                <AlertCircle className="w-4 h-4 text-amber-600" />
                <p className="text-xs font-semibold text-gray-600 uppercase tracking-wide">
                  Optional
                </p>
              </div>
              <p className="text-2xl md:text-3xl font-bold text-amber-600">
                {optionalFindings.length}
              </p>
            </div>
          </div>
        </div>

        {results.findings.length > 0 ? (
          <div className="bg-white rounded-xl shadow-lg overflow-hidden mb-6 print:shadow-none print:border">
            <div className="border-b border-gray-200 print:hidden">
              <div className="flex">
                <button
                  onClick={() => setActiveTab('mandatory')}
                  className={`flex-1 px-6 py-4 font-bold text-sm md:text-base transition-all ${
                    activeTab === 'mandatory'
                      ? 'text-red-700 border-b-2 border-red-600 bg-red-50'
                      : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  <div className="flex items-center justify-center gap-2">
                    <AlertTriangle className="w-5 h-5" />
                    <span>Mandatory Findings</span>
                    <span
                      className={`px-2 py-0.5 rounded-full text-xs font-bold ${
                        activeTab === 'mandatory'
                          ? 'bg-red-600 text-white'
                          : 'bg-gray-200 text-gray-600'
                      }`}
                    >
                      {mandatoryFindings.length}
                    </span>
                  </div>
                </button>

                <button
                  onClick={() => setActiveTab('optional')}
                  className={`flex-1 px-6 py-4 font-bold text-sm md:text-base transition-all ${
                    activeTab === 'optional'
                      ? 'text-amber-700 border-b-2 border-amber-600 bg-amber-50'
                      : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  <div className="flex items-center justify-center gap-2">
                    <AlertCircle className="w-5 h-5" />
                    <span>Optional Findings</span>
                    <span
                      className={`px-2 py-0.5 rounded-full text-xs font-bold ${
                        activeTab === 'optional'
                          ? 'bg-amber-600 text-white'
                          : 'bg-gray-200 text-gray-600'
                      }`}
                    >
                      {optionalFindings.length}
                    </span>
                  </div>
                </button>
              </div>
            </div>

            <div className="p-6 md:p-8">
              {displayedFindings.length > 0 ? (
                <div className="space-y-3">
                  {displayedFindings.map((finding) => (
                    <FindingCard key={finding.id} finding={finding} />
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <CheckCircle className="w-12 h-12 text-green-600 mx-auto mb-3" />
                  <p className="text-gray-600 font-medium">
                    No {activeTab} findings detected
                  </p>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="bg-white rounded-xl shadow-lg p-8 text-center mb-6 print:shadow-none print:border">
            <CheckCircle className="w-16 h-16 text-green-600 mx-auto mb-4" />
            <h3 className="text-xl font-bold text-gray-800 mb-2">No issues found</h3>
            <p className="text-gray-600">
              All documents have been validated successfully
            </p>
          </div>
        )}

        <div className="flex flex-col sm:flex-row gap-4 print:hidden">
          {onDownload && (
            <button
              onClick={onDownload}
              className="flex items-center justify-center gap-2 px-6 py-3 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-semibold transition-all shadow-md hover:shadow-lg"
            >
              <Download className="w-5 h-5" />
              Export Report
            </button>
          )}

          {onStartNew && (
            <button
              onClick={onStartNew}
              className="flex items-center justify-center gap-2 px-6 py-3 rounded-lg bg-white hover:bg-gray-50 text-gray-700 font-semibold border-2 border-gray-300 transition-all hover:border-gray-400"
            >
              <RefreshCw className="w-5 h-5" />
              Start New Analysis
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
