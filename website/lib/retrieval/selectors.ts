import type { RetrievedDocument, RetrievalEvent, TrustScore } from "./types";

export function validTrustScore(score: TrustScore): score is number {
  return typeof score === "number" && Number.isFinite(score) && score >= 0 && score <= 1;
}

export function formatTrustScore(score: TrustScore): string {
  return validTrustScore(score) ? score.toFixed(2) : "Not available";
}

export function documentTitle(document: RetrievedDocument): string {
  return document.title?.trim() || "Untitled document";
}

export function isFlaggedDocument(document: RetrievedDocument): boolean {
  return document.status === "flagged" || document.status === "blocked";
}

export function isFlaggedEvent(event: RetrievalEvent): boolean {
  return event.documents.some(isFlaggedDocument);
}
