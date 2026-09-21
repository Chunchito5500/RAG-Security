/** Normalized trust score. Data adapters must validate finite values in [0, 1]. */
export type TrustScore = number | null;
export type FlagReason = string | null;
export type RetrievalStatus = "trusted" | "flagged" | "blocked" | "unknown";
export type RetrievalDecision = "allowed" | "excluded" | "unknown";
export type RetrievalMode = "strict" | "balanced" | "observe";

export interface SourceMetadata {
  id?: string | null;
  name?: string | null;
  chunk?: string | null;
}

export interface RetrievedDocument {
  /** Stable retrieval-result identity, distinct from the source document ID. */
  id: string;
  documentId?: string | null;
  title?: string | null;
  source?: SourceMetadata | null;
  excerpt?: string | null;
  trustScore: TrustScore;
  status: RetrievalStatus;
  flagReason: FlagReason;
  /** Eligibility for answer context; being allowed does not imply being cited. */
  decision: RetrievalDecision;
}

export interface RetrievalAnswer {
  status: "available" | "withheld" | "unavailable";
  text: string | null;
  /** References RetrievedDocument.id within this event, never array indexes. */
  citationIds: string[];
}

export interface RetrievalEvent {
  id: string;
  query: string;
  occurredAt: string;
  mode: RetrievalMode;
  origin: "demo" | "backend";
  scenarioLabel?: string;
  documents: RetrievedDocument[];
  answer: RetrievalAnswer;
}

/** Frontend read boundary. A future adapter maps backend data to these types. */
export interface RetrievalDataSource {
  listEvents(): Promise<RetrievalEvent[]>;
}

export interface DemoScenario {
  id: string;
  label: string;
  description: string;
  query: string;
  documents: RetrievedDocument[];
  answer: RetrievalAnswer;
}

export type DemoDataset = "scenarios" | "trusted-only" | "empty" | "error";
