import type { RetrievalDecision, RetrievalStatus } from "./types";

export interface DocumentResult {
  document_id: string | null;
  filename: string | null;
  source: string | null;
  trust_score: number | null;
  security_status: RetrievalStatus;
  decision: string;
  flag_reason: string | null;
  quarantine_status: string | null;
  status?: string;
  processing_status?: "queued" | "processing" | "processed" | "failed";
  processing_error?: string | null;
  uploaded_at?: string;
  raw_sha256?: string;
  text_sha256?: string | null;
  policy_version?: string | null;
  metadata?: { source_type: string; uploader_id: string; uploaded_at: string; sanitization_applied?: boolean };
  chunks?: { chunk_id: string; reference: ChunkReference }[];
}

export interface ChunkReference {
  page: number | null;
  chunk_index: number;
  line_start: number | null;
  line_end: number | null;
}

export interface ApiRetrievedDocument extends DocumentResult {
  result_id: string;
  inclusion: RetrievalDecision;
  excerpt: string | null;
  similarity: number | null;
  effective_similarity: number | null;
  chunk_id?: string;
  reference?: ChunkReference;
}

export interface QueryResult {
  retrieval_event_id: string;
  query: string | null;
  timestamp: string | null;
  origin: "file_pipeline" | "legacy_log";
  retrieved_documents: ApiRetrievedDocument[];
  trusted_documents: ApiRetrievedDocument[];
  excluded_documents: ApiRetrievedDocument[];
  trust_scores: Record<string, number | null>;
  security_decisions: { document_id: string | null; decision: string; inclusion: RetrievalDecision }[];
  retrieval_trace: ApiRetrievedDocument[];
}

export interface ProcessingError {
  filename: string;
  source: string;
  error: string;
}

export interface FlaggedEvent {
  event_type?: "ingestion" | "retrieval";
  event_id: string;
  high_severity_flag?: boolean;
  request_id: string | null;
  timestamp: string | null;
  query: string | null;
  document_id: string | null;
  trust_score: number | null;
  status: string | null;
  flag_reason: string | null;
  source?: string | null;
  quarantine_status?: string | null;
  source_document?: { document_id?: string | null; found?: boolean };
}

export interface EvaluationResult {
  source: string;
  timestamp: string | null;
  total_documents_tested: number;
  processed_documents: number | null;
  correctly_classified: number | null;
  false_positives: number | null;
  missed_malicious_documents: number | null;
  accuracy: number | null;
  quarantine_exclusion_passed: boolean | null;
  all_low_trust_test_passed: boolean | null;
  individual_test_results: {
    filename: string;
    document_id: string | null;
    expected: string | null;
    trust_score: number | null;
    decision: string | null;
    correct: boolean | null;
    entered_context: boolean | null;
    flags: string[];
    error?: string;
  }[];
}
