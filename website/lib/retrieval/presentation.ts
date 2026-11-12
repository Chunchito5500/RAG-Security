import type { DocumentResult } from "./api-types";
import type { RetrievalEvent, RetrievalStatus } from "./types";

export function eventTime(value?: string | null) {
  if (!value) return "Time not recorded";
  // Old records have no timezone; do not imply that one was recorded.
  if (!/[zZ]|[+-]\d{2}:\d{2}$/.test(value)) return `${value} (timezone not recorded)`;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Time not recorded" : date.toLocaleString("en-US", { dateStyle: "medium", timeStyle: "short" });
}

export function isQuarantined(document: DocumentResult) {
  return document.decision === "quarantine" || document.quarantine_status === "flagged";
}

export function eligibility(decision?: string | null) {
  if (decision?.toLowerCase() === "quarantine") return "Excluded from context";
  if (decision?.toLowerCase() === "warn") return "Eligible with warning";
  if (decision?.toLowerCase() === "allow") return "Eligible for retrieval";
  return "Eligibility not recorded";
}

export function recordedStatus(decision?: string | null): RetrievalStatus {
  return ({ allow: "trusted", warn: "flagged", quarantine: "blocked" } as const)[decision?.toLowerCase() as "allow" | "warn" | "quarantine"] ?? "unknown";
}

export function severity(reason?: string | null, highSeverity?: boolean) {
  if (highSeverity || reason?.includes("HIGH:")) return "High";
  if (reason?.includes("MEDIUM:")) return "Medium";
  return "Not recorded";
}

/** Translate recorded rule labels only; never scan text or recalculate a decision. */
export function readableReason(reason?: string | null) {
  if (!reason?.trim()) return "No reason recorded.";
  const explanations = reason.split("; ").map((rule) => {
    if (rule === "INTEGRITY_FAIL") return "Document integrity check failed.";
    if (rule.startsWith("HIGH:")) {
      if (rule.includes("summarizing")) return "High-severity attempt to influence the retrieved response detected.";
      if (rule.includes("mention|reveal")) return "High-severity instruction to conceal document behavior detected.";
      if (rule.includes("open|fetch")) return "High-severity instruction to invoke a tool or external action detected.";
      return "High-severity instruction override pattern detected.";
    }
    if (rule.startsWith("MEDIUM:")) return "A medium-severity suspicious content pattern was detected.";
    return rule;
  });
  return [...new Set(explanations)].join(" ");
}

export function eventCounts(event: RetrievalEvent) {
  return {
    retrieved: event.documents.length,
    allowed: event.documents.filter(d => d.decision === "allowed").length,
    flagged: event.documents.filter(d => d.status === "flagged").length,
    blocked: event.documents.filter(d => d.status === "blocked").length,
  };
}

export function eventStatus(event: RetrievalEvent) {
  const counts = eventCounts(event);
  if (!counts.retrieved) return "No results";
  if (counts.blocked) return "Blocked sources";
  if (counts.flagged) return "Warnings";
  if (event.documents.some(d => d.status === "unknown")) return "Incomplete metadata";
  return "No warnings";
}
