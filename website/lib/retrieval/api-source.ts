import type { DocumentResult, EvaluationResult, FlaggedEvent, ProcessingError, QueryResult } from "./api-types";
import type { RetrievalEvent } from "./types";
import { validTrustScore } from "./selectors";

const baseUrl = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

async function sendRequest<T>(path: string, init?: RequestInit): Promise<{ data: T; errors?: ProcessingError[] }> {
  let response: Response;
  try {
    response = await fetch(`${baseUrl}${path}`, { ...init, cache: "no-store", signal: AbortSignal.timeout(60000) });
  } catch {
    throw new Error("Cannot reach the backend. Check that it is running and the API address is correct, then retry.");
  }
  let body;
  try { body = await response.json(); } catch { throw new Error("The backend returned an invalid response."); }
  if (!response.ok) throw new Error(typeof body?.error === "string" ? body.error : `Request failed (${response.status}).`);
  if (!body || !("data" in body)) throw new Error("The backend response is missing data.");
  return body;
}

const pendingReads = new Map<string, Promise<unknown>>();

function request<T>(path: string, init?: RequestInit): Promise<{ data: T; errors?: ProcessingError[] }> {
  if (init?.method && init.method !== "GET") return sendRequest<T>(path, init);
  // Share simultaneous reads (including Strict Mode mounts), but never cache stale results.
  const existing = pendingReads.get(path);
  if (existing) return existing as Promise<{ data: T; errors?: ProcessingError[] }>;
  const pending = sendRequest<T>(path, init).finally(() => pendingReads.delete(path));
  pendingReads.set(path, pending);
  return pending;
}

export function toRetrievalEvent(record: QueryResult): RetrievalEvent {
  if (!record || typeof record.retrieval_event_id !== "string" || !Array.isArray(record.retrieved_documents)) {
    throw new Error("The backend returned an invalid retrieval event.");
  }
  return {
    id: record.retrieval_event_id,
    query: record.query ?? "",
    occurredAt: record.timestamp ?? "",
    mode: record.origin === "file_pipeline" ? "security" : "legacy",
    origin: "backend",
    scenarioLabel: record.origin === "file_pipeline" ? "Document retrieval" : "Historical log",
    documents: record.retrieved_documents.map((document, index) => ({
      id: document.result_id || `${document.document_id || "unknown"}-${index}`,
      documentId: document.document_id,
      title: document.filename,
      source: document.source ? { id: document.source, name: document.source } : null,
      trustScore: validTrustScore(document.trust_score) ? document.trust_score : null,
      status: ["trusted", "flagged", "blocked"].includes(document.security_status) ? document.security_status : "unknown",
      decision: ["allowed", "excluded"].includes(document.inclusion) ? document.inclusion : "unknown",
      flagReason: document.flag_reason ?? null,
      quarantineStatus: document.quarantine_status ?? null,
      similarity: document.similarity,
      excerpt: document.excerpt ?? null,
    })),
    answer: { status: "unavailable", text: null, citationIds: [] },
  };
}

export const api = {
  async listEvents() {
    const { data } = await request<QueryResult[]>("/api/retrieval-events");
    if (!Array.isArray(data)) throw new Error("The backend returned invalid retrieval history.");
    return data.map(toRetrievalEvent);
  },
  async query(query: string, top_k: number) {
    return toRetrievalEvent((await request<QueryResult>("/api/query", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query, top_k }),
    })).data);
  },
  documents: () => request<DocumentResult[]>("/api/documents"),
  flaggedEvents: () => request<FlaggedEvent[]>("/api/flagged-events"),
  evaluation: () => request<EvaluationResult | null>("/api/evaluation-results"),
  async upload(file: File) {
    return (await request<DocumentResult>(`/api/documents/upload?filename=${encodeURIComponent(file.name)}`, {
      method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: file,
    })).data;
  },
};
