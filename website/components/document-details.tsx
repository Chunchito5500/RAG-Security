import type { DocumentResult } from "@/lib/retrieval/api-types";
import { eligibility, eventTime, isQuarantined, readableReason } from "@/lib/retrieval/presentation";
import { formatTrustScore, validTrustScore } from "@/lib/retrieval/selectors";
import { Panel, StatusBadge } from "./retrieval-ui";

export function DetectionDetails({ reason }: { reason?: string | null }) {
  return <div><p className="text-sm leading-6 text-slate-600">{readableReason(reason)}</p>{reason && <details className="mt-3 text-xs"><summary className="font-medium text-slate-600">Technical detection details</summary><p className="mt-2 break-all rounded bg-slate-50 p-3 font-mono leading-6">{reason}</p></details>}</div>;
}

export function DocumentDetails({ document }: { document: DocumentResult }) {
  return <Panel title="Document details" description={document.filename || document.document_id || "Document metadata unavailable"}>
    <dl className="grid gap-4 text-sm sm:grid-cols-2 lg:grid-cols-3">
      <div><dt className="text-xs text-slate-500">Processing</dt><dd className="mt-1">{document.processing_status || "processed"}</dd></div>
      <div><dt className="text-xs text-slate-500">Uploaded</dt><dd className="mt-1">{eventTime(document.uploaded_at)}</dd></div>
      <div><dt className="text-xs text-slate-500">Recorded provenance</dt><dd className="mt-1">{document.metadata?.source_type || "Unknown"} · {document.metadata?.uploader_id || "Not recorded"}</dd></div>
      <div><dt className="text-xs text-slate-500">Document ID</dt><dd className="mt-1 break-all font-mono text-xs">{document.document_id || "Not recorded"}</dd></div>
      <div><dt className="text-xs text-slate-500">Trust score</dt><dd className="mt-1">{formatTrustScore(document.trust_score)}{validTrustScore(document.trust_score) ? " / 1.00" : ""}</dd></div>
      <div><dt className="text-xs text-slate-500">Security status</dt><dd className="mt-1"><StatusBadge status={document.security_status} /></dd></div>
      <div><dt className="text-xs text-slate-500">Quarantine</dt><dd className="mt-1">{isQuarantined(document) ? "Quarantined" : document.quarantine_status === "none" ? "Not quarantined" : "Not recorded"}</dd></div>
      <div><dt className="text-xs text-slate-500">Retrieval eligibility</dt><dd className="mt-1">{eligibility(document.decision)}</dd></div>
      <div><dt className="text-xs text-slate-500">Source</dt><dd className="mt-1 break-all text-xs">{document.source || "Source metadata was not recorded."}</dd></div>
    </dl>
    {document.processing_error && <p role="alert" className="mt-4 text-sm text-red-700">{document.processing_error}</p>}
    {document.raw_sha256 && <details className="mt-4 text-xs"><summary>Integrity and policy metadata</summary><dl className="mt-3 space-y-2 break-all"><div><dt>Original file SHA-256</dt><dd>{document.raw_sha256}</dd></div><div><dt>Extracted text SHA-256</dt><dd>{document.text_sha256 || "Pending"}</dd></div><div><dt>Security policy</dt><dd>{document.policy_version || "Pending"}</dd></div></dl><p className="mt-2">Hashes record content consistency. They do not authenticate the original publisher. Documents are not sanitized.</p></details>}
    <div className="mt-5 border-t pt-4"><DetectionDetails reason={document.flag_reason} /></div>
  </Panel>;
}
