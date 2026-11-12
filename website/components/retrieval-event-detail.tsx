import type { RetrievalEvent } from "@/lib/retrieval/types";
import Link from "next/link";
import { documentTitle, formatTrustScore, isFlaggedDocument } from "@/lib/retrieval/selectors";
import { eventCounts, eventTime, readableReason } from "@/lib/retrieval/presentation";
import { DetectionDetails } from "./document-details";
import { DecisionLabel, EmptyState, MetricCard, Panel, SecurityExplanation, StatusBadge } from "./retrieval-ui";

export function RetrievalEventDetail({ event }: { event: RetrievalEvent }) {
  const counts = eventCounts(event);
  const flagged = event.documents.filter(isFlaggedDocument);
  const passages = event.documents.filter(d => d.decision === "allowed" && d.status !== "blocked" && d.status !== "unknown" && d.excerpt);
  return <div className="min-w-0 space-y-5" id="event-details">
    <Panel title="Retrieval summary" description={event.query || "Query text was not recorded."}>
      <p className="mb-4 break-all text-xs text-slate-500">{eventTime(event.occurredAt)} · {event.id}{event.mode === "legacy" ? " · Historical record" : ""}</p>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">{Object.entries({ Retrieved: counts.retrieved, Allowed: counts.allowed, Flagged: counts.flagged, Blocked: counts.blocked }).map(([label, value]) => <MetricCard key={label} label={label} value={value} />)}</div>
    </Panel>
    <Panel title="Trusted passages" description="Eligible source text, not an AI-generated answer. Warned sources retain their warning.">
      {passages.length ? <div className="divide-y divide-slate-200">{passages.map((d, index) => <details key={d.id} open={index === 0} className="py-3 first:pt-0">
        <summary className="text-sm font-semibold text-slate-800">{documentTitle(d)} <span className="ml-2"><StatusBadge status={d.status} /></span></summary>
        <p className="mt-2 break-all text-xs text-slate-500">{d.source?.name || "Source metadata not recorded"}</p>
        {d.reference && <p className="mt-2 text-xs text-slate-600">Chunk {d.reference.chunk_index} · {d.reference.page ? `Page ${d.reference.page}` : `Lines ${d.reference.line_start}–${d.reference.line_end}`} · Trust {formatTrustScore(d.trustScore)} · Retrieval score {d.effectiveSimilarity?.toFixed(3) ?? "Not recorded"}</p>}
        {d.documentId && <Link className="mt-2 inline-block text-xs underline" href={`/documents?document=${encodeURIComponent(d.documentId)}`}>View source document</Link>}
        <p className="mt-3 max-h-80 overflow-y-auto whitespace-pre-wrap break-words text-sm leading-7 text-slate-700">{d.excerpt}</p>
        {d.status === "flagged" && <p className="mt-3 text-xs text-amber-800">Warning: {readableReason(d.flagReason)}</p>}
      </details>)}</div> : <EmptyState>No trusted passages are available. {event.documents.some(d => d.decision === "allowed") ? "Source text was not recorded for these results." : "No retrieved source with available text was eligible."}</EmptyState>}
    </Panel>
    <Panel title="Retrieved documents">
      {event.documents.length ? <div className="table-scroll" tabIndex={0} role="region" aria-label="Retrieved documents"><table className="data-table"><thead><tr>{["Document", "Trust score", "Security status", "Context decision", "Details"].map(label => <th scope="col" key={label}>{label}</th>)}</tr></thead><tbody>
        {event.documents.map(d => <tr key={d.id}><td className="name-cell font-medium">{documentTitle(d)}</td><td>{formatTrustScore(d.trustScore)}</td><td><StatusBadge status={d.status} /></td><td><DecisionLabel decision={d.decision} /></td><td className="min-w-40 max-w-sm"><details><summary className="text-xs font-medium text-slate-700">View details<span className="sr-only"> for {documentTitle(d)}</span></summary><dl className="my-3 space-y-2 break-all text-xs"><div><dt className="text-slate-500">Document ID</dt><dd>{d.documentId || "Not recorded"}</dd></div><div><dt className="text-slate-500">Source</dt><dd>{d.source?.name || "Not recorded"}</dd></div><div><dt className="text-slate-500">Similarity</dt><dd>{d.similarity ?? "Not recorded"}</dd></div><div><dt className="text-slate-500">Quarantine record</dt><dd>{d.quarantineStatus === "flagged" ? "Quarantined" : d.quarantineStatus === "none" ? "Not quarantined" : "Not recorded"}</dd></div></dl><DetectionDetails reason={d.flagReason} /></details></td></tr>)}
      </tbody></table></div> : <EmptyState>No documents were retrieved for this event.</EmptyState>}
    </Panel>
    <Panel title="Security decisions">
      <SecurityExplanation />
      {flagged.length ? <ul className="mt-4 divide-y divide-slate-100">{flagged.map(d => <li key={d.id} className="py-3"><div className="flex flex-wrap items-center gap-2 text-sm"><span className="break-all font-medium">{documentTitle(d)}</span><StatusBadge status={d.status} /><DecisionLabel decision={d.decision} /></div><p className="mt-2 text-sm leading-6 text-slate-600">{readableReason(d.flagReason)}</p></li>)}</ul> : <p className="mt-4 text-sm text-slate-500">No flagged or blocked sources were recorded.</p>}
    </Panel>
    <Panel title="Retrieval trace">
      <ol aria-label="Retrieval stages" className="flex flex-wrap gap-2">{["Query", "Retrieval", "Trust Evaluation", "Filtering", "Trusted Passages"].map((step, index) => <li key={step} className="rounded border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600"><span className="mr-2 font-semibold text-slate-900">{index + 1}</span>{step}{index < 4 && <span className="ml-2" aria-hidden="true">→</span>}</li>)}</ol>
      <p className="mt-3 text-xs leading-6 text-slate-500">{counts.retrieved} candidates recorded · {counts.allowed} allowed into context · {event.documents.filter(d => d.decision === "excluded").length} excluded · {event.documents.filter(d => d.decision === "unknown").length} decisions not recorded.</p>
    </Panel>
  </div>;
}
