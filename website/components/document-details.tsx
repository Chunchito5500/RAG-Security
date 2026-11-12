import type { DocumentResult } from "@/lib/retrieval/api-types";
import { eligibility, isQuarantined, readableReason } from "@/lib/retrieval/presentation";
import { formatTrustScore, validTrustScore } from "@/lib/retrieval/selectors";
import { Panel, StatusBadge } from "./retrieval-ui";

export function DetectionDetails({ reason }: { reason?: string | null }) {
  return <div><p className="text-sm leading-6 text-slate-600">{readableReason(reason)}</p>{reason && <details className="mt-3 text-xs"><summary className="font-medium text-slate-600">Technical detection details</summary><p className="mt-2 break-all rounded bg-slate-50 p-3 font-mono leading-6">{reason}</p></details>}</div>;
}

export function DocumentDetails({ document }: { document: DocumentResult }) {
  return <Panel title="Document details" description={document.filename || document.document_id || "Document metadata unavailable"}>
    <dl className="grid gap-4 text-sm sm:grid-cols-2 lg:grid-cols-3">
      <div><dt className="text-xs text-slate-500">Document ID</dt><dd className="mt-1 break-all font-mono text-xs">{document.document_id || "Not recorded"}</dd></div>
      <div><dt className="text-xs text-slate-500">Trust score</dt><dd className="mt-1">{formatTrustScore(document.trust_score)}{validTrustScore(document.trust_score) ? " / 1.00" : ""}</dd></div>
      <div><dt className="text-xs text-slate-500">Security status</dt><dd className="mt-1"><StatusBadge status={document.security_status} /></dd></div>
      <div><dt className="text-xs text-slate-500">Quarantine</dt><dd className="mt-1">{isQuarantined(document) ? "Quarantined" : document.quarantine_status === "none" ? "Not quarantined" : "Not recorded"}</dd></div>
      <div><dt className="text-xs text-slate-500">Retrieval eligibility</dt><dd className="mt-1">{eligibility(document.decision)}</dd></div>
      <div><dt className="text-xs text-slate-500">Source</dt><dd className="mt-1 break-all text-xs">{document.source || "Source metadata was not recorded."}</dd></div>
    </dl>
    <div className="mt-5 border-t pt-4"><DetectionDetails reason={document.flag_reason} /></div>
  </Panel>;
}
