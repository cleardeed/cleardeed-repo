import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import { DocumentAnalysisResponse } from './api';
import { PropertyFormData } from '../types/property';

interface ReportData {
  propertyData: PropertyFormData;
  analysisResults: DocumentAnalysisResponse;
  documentNames: string[];
}

export const generatePDFReport = async (data: ReportData): Promise<void> => {
  const element = document.getElementById('pdf-report-content');
  if (!element) {
    throw new Error('Report element not found');
  }

  const canvas = await html2canvas(element, {
    scale: 2,
    useCORS: true,
    allowTaint: true,
    backgroundColor: '#ffffff',
  });

  const imgData = canvas.toDataURL('image/png');
  const pdf = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: 'a4',
  });

  const imgWidth = 210;
  const imgHeight = (canvas.height * imgWidth) / canvas.width;
  let heightLeft = imgHeight;
  let position = 0;

  pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
  heightLeft -= 297;

  while (heightLeft >= 0) {
    position = heightLeft - imgHeight;
    pdf.addPage();
    pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
    heightLeft -= 297;
  }

  const filename = `validation-report-${new Date().toISOString().split('T')[0]}.pdf`;
  pdf.save(filename);
};

export const exportReportAsHTML = (data: ReportData): string => {
  const { propertyData, analysisResults, documentNames } = data;
  const timestamp = new Date().toLocaleString();

  const verdictColor = {
    APPROVED: '#10b981',
    CONDITIONALLY_APPROVED: '#f59e0b',
    REJECTED: '#ef4444',
  }[analysisResults.verdict];

  const severityColor = {
    High: '#ef4444',
    Medium: '#f59e0b',
    Low: '#3b82f6',
  };

  return `
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Property Validation Report</title>
      <style>
        * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }
        body {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
          line-height: 1.6;
          color: #1f2937;
          background: white;
        }
        .container {
          max-width: 900px;
          margin: 0 auto;
          padding: 40px 20px;
        }
        .header {
          text-align: center;
          margin-bottom: 40px;
          border-bottom: 3px solid #e5e7eb;
          padding-bottom: 30px;
        }
        .header h1 {
          font-size: 32px;
          margin-bottom: 10px;
          color: #111827;
        }
        .header p {
          color: #6b7280;
          font-size: 14px;
        }
        .verdict-section {
          background: ${verdictColor}15;
          border: 2px solid ${verdictColor};
          border-radius: 12px;
          padding: 30px;
          margin-bottom: 30px;
          text-align: center;
        }
        .verdict-text {
          font-size: 28px;
          font-weight: bold;
          color: ${verdictColor};
          margin-bottom: 10px;
        }
        .verdict-summary {
          color: #374151;
          font-size: 16px;
          margin-bottom: 15px;
        }
        .confidence {
          font-size: 20px;
          font-weight: 600;
          color: ${verdictColor};
        }
        .property-details {
          margin-bottom: 30px;
          background: #f9fafb;
          padding: 20px;
          border-radius: 8px;
        }
        .property-details h3 {
          color: #1f2937;
          margin-bottom: 15px;
          font-size: 18px;
        }
        .detail-row {
          display: flex;
          margin-bottom: 10px;
        }
        .detail-label {
          font-weight: 600;
          color: #4b5563;
          width: 150px;
          min-width: 150px;
        }
        .detail-value {
          color: #6b7280;
        }
        .metrics {
          display: grid;
          grid-template-columns: 1fr 1fr 1fr 1fr;
          gap: 15px;
          margin-bottom: 30px;
        }
        .metric-card {
          background: white;
          border: 1px solid #e5e7eb;
          padding: 15px;
          border-radius: 8px;
          text-align: center;
        }
        .metric-label {
          font-size: 12px;
          color: #6b7280;
          text-transform: uppercase;
          font-weight: 600;
          margin-bottom: 5px;
        }
        .metric-value {
          font-size: 24px;
          font-weight: bold;
          color: #1f2937;
        }
        .findings-section {
          margin-top: 30px;
        }
        .findings-section h3 {
          color: #1f2937;
          margin-bottom: 15px;
          font-size: 18px;
          border-bottom: 2px solid #e5e7eb;
          padding-bottom: 10px;
        }
        .finding-item {
          margin-bottom: 20px;
          border-left: 4px solid #3b82f6;
          padding: 15px;
          background: #f9fafb;
          page-break-inside: avoid;
        }
        .finding-item.high {
          border-left-color: #ef4444;
        }
        .finding-item.medium {
          border-left-color: #f59e0b;
        }
        .finding-item.low {
          border-left-color: #3b82f6;
        }
        .finding-header {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-bottom: 8px;
        }
        .severity-badge {
          display: inline-block;
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 12px;
          font-weight: bold;
          color: white;
        }
        .severity-badge.high {
          background-color: #ef4444;
        }
        .severity-badge.medium {
          background-color: #f59e0b;
        }
        .severity-badge.low {
          background-color: #3b82f6;
        }
        .finding-title {
          font-size: 16px;
          font-weight: 600;
          color: #1f2937;
          margin: 0;
        }
        .finding-description {
          color: #4b5563;
          margin: 8px 0;
          font-size: 14px;
        }
        .finding-docs {
          color: #6b7280;
          font-size: 13px;
          margin-top: 8px;
        }
        .documents-checklist {
          margin-top: 30px;
          background: #f9fafb;
          padding: 20px;
          border-radius: 8px;
        }
        .documents-checklist h3 {
          color: #1f2937;
          margin-bottom: 15px;
          font-size: 18px;
        }
        .doc-item {
          display: flex;
          align-items: center;
          margin-bottom: 8px;
          font-size: 14px;
        }
        .doc-checkbox {
          width: 20px;
          height: 20px;
          margin-right: 10px;
          border: 2px solid #d1d5db;
          border-radius: 4px;
          background: white;
        }
        .footer {
          margin-top: 40px;
          padding-top: 20px;
          border-top: 1px solid #e5e7eb;
          text-align: center;
          color: #6b7280;
          font-size: 12px;
        }
        @media print {
          body {
            background: white;
          }
          .container {
            padding: 20px;
          }
        }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>Property Validation Report</h1>
          <p>Document analysis and validation results</p>
        </div>

        <div class="verdict-section">
          <div class="verdict-text">${analysisResults.verdict.replace(/_/g, ' ')}</div>
          <div class="verdict-summary">${analysisResults.summary}</div>
          <div class="confidence">Confidence Score: ${analysisResults.confidence_score}%</div>
        </div>

        <div class="property-details">
          <h3>Property Details</h3>
          <div class="detail-row">
            <div class="detail-label">Property Address:</div>
            <div class="detail-value">${propertyData.address}</div>
          </div>
          <div class="detail-row">
            <div class="detail-label">Property Type:</div>
            <div class="detail-value">${propertyData.propertyType}</div>
          </div>
          <div class="detail-row">
            <div class="detail-label">Year Built:</div>
            <div class="detail-value">${propertyData.yearBuilt}</div>
          </div>
          <div class="detail-row">
            <div class="detail-label">Square Footage:</div>
            <div class="detail-value">${propertyData.squareFootage} sq ft</div>
          </div>
          <div class="detail-row">
            <div class="detail-label">Number of Bedrooms:</div>
            <div class="detail-value">${propertyData.bedrooms}</div>
          </div>
          <div class="detail-row">
            <div class="detail-label">Number of Bathrooms:</div>
            <div class="detail-value">${propertyData.bathrooms}</div>
          </div>
        </div>

        <div class="metrics">
          <div class="metric-card">
            <div class="metric-label">Documents</div>
            <div class="metric-value">${analysisResults.total_documents_analyzed}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Mandatory</div>
            <div class="metric-value">${analysisResults.findings.filter(f => f.category === 'Mandatory').length}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Optional</div>
            <div class="metric-value">${analysisResults.findings.filter(f => f.category === 'Optional').length}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Confidence</div>
            <div class="metric-value">${analysisResults.confidence_score}%</div>
          </div>
        </div>

        ${
          analysisResults.findings.length > 0
            ? `
          <div class="findings-section">
            <h3>Findings Summary</h3>
            ${analysisResults.findings
              .map(
                (finding) => `
              <div class="finding-item ${finding.severity.toLowerCase()}">
                <div class="finding-header">
                  <span class="severity-badge ${finding.severity.toLowerCase()}">${finding.severity}</span>
                  <p class="finding-title">${finding.title}</p>
                </div>
                <div class="finding-description">${finding.description}</div>
                <div class="finding-docs">
                  <strong>Category:</strong> ${finding.category}<br>
                  <strong>Affected Documents:</strong> ${
                    finding.affected_documents.length > 0
                      ? finding.affected_documents.join(', ')
                      : 'All documents'
                  }
                </div>
              </div>
            `
              )
              .join('')}
          </div>
        `
            : ''
        }

        <div class="documents-checklist">
          <h3>Documents Analyzed</h3>
          ${documentNames.map((name) => `<div class="doc-item"><div class="doc-checkbox"></div>${name}</div>`).join('')}
        </div>

        <div class="footer">
          <p>Report generated on ${timestamp}</p>
          <p>This report contains confidential information and should be treated accordingly.</p>
        </div>
      </div>
    </body>
    </html>
  `;
};
