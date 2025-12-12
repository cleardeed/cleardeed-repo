import "jsr:@supabase/functions-js/edge-runtime.d.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
  "Access-Control-Allow-Headers":
    "Content-Type, Authorization, X-Client-Info, Apikey",
};

interface DocumentData {
  name: string;
  base64: string;
  size: number;
}

interface AnalysisRequest {
  documents: DocumentData[];
  property_data: {
    district: string;
    sro_name: string;
    village: string;
    survey_number: string;
    subdivision: string;
  };
}

interface AnalysisFinding {
  id: string;
  category: "Mandatory" | "Optional";
  severity: "High" | "Medium" | "Low";
  title: string;
  description: string;
  affected_documents: string[];
}

interface AnalysisResponse {
  verdict: "APPROVED" | "CONDITIONALLY_APPROVED" | "REJECTED";
  confidence_score: number;
  findings: AnalysisFinding[];
  total_documents_analyzed: number;
  summary: string;
}

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response(null, {
      status: 200,
      headers: corsHeaders,
    });
  }

  try {
    const body: AnalysisRequest = await req.json();

    const anthropicApiKey = Deno.env.get("ANTHROPIC_API_KEY");
    if (!anthropicApiKey) {
      throw new Error("ANTHROPIC_API_KEY not configured");
    }

    const documentSummaries = body.documents
      .map((doc, idx) => `Document ${idx + 1}: ${doc.name} (${doc.size} bytes)`)
      .join("\n");

    const prompt = `You are an AI expert in property document validation. Analyze the provided property documents and return a structured JSON response.

Property Details:
- District: ${body.property_data.district}
- SRO: ${body.property_data.sro_name}
- Village: ${body.property_data.village}
- Survey Number: ${body.property_data.survey_number}
- Subdivision: ${body.property_data.subdivision || "N/A"}

Documents Provided:
${documentSummaries}

Please validate the documents and return a JSON response with this exact structure:
{
  "verdict": "APPROVED" | "CONDITIONALLY_APPROVED" | "REJECTED",
  "confidence_score": number (0-100),
  "summary": "Brief summary of the analysis",
  "findings": [
    {
      "id": "finding_1",
      "category": "Mandatory" | "Optional",
      "severity": "High" | "Medium" | "Low",
      "title": "Finding title",
      "description": "Detailed description",
      "affected_documents": ["document_name"]
    }
  ],
  "total_documents_analyzed": number
}

Validation criteria:
1. Check for presence of critical documents (deed, title deed, property certificate)
2. Verify property details consistency
3. Check for required signatures and stamps
4. Identify any missing or damaged pages
5. Flag any data inconsistencies

Return ONLY valid JSON, no additional text.`;

    const anthropicResponse = await fetch(
      "https://api.anthropic.com/v1/messages",
      {
        method: "POST",
        headers: {
          "x-api-key": anthropicApiKey,
          "anthropic-version": "2023-06-01",
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: "claude-3-5-sonnet-20241022",
          max_tokens: 2000,
          messages: [
            {
              role: "user",
              content: prompt,
            },
          ],
        }),
      }
    );

    if (!anthropicResponse.ok) {
      const errorData = await anthropicResponse.json();
      throw new Error(
        `Anthropic API error: ${anthropicResponse.status} - ${JSON.stringify(errorData)}`
      );
    }

    const anthropicData = await anthropicResponse.json();
    const responseText =
      anthropicData.content[0]?.type === "text"
        ? anthropicData.content[0].text
        : "";

    const jsonMatch = responseText.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error("No valid JSON found in Claude response");
    }

    const analysisResult: AnalysisResponse = JSON.parse(jsonMatch[0]);

    return new Response(JSON.stringify(analysisResult), {
      headers: {
        ...corsHeaders,
        "Content-Type": "application/json",
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";

    return new Response(
      JSON.stringify({
        error: message,
      }),
      {
        status: 400,
        headers: {
          ...corsHeaders,
          "Content-Type": "application/json",
        },
      }
    );
  }
});
