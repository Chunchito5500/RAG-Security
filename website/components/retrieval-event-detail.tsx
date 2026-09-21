import type { RetrievalEvent } from "@/lib/retrieval/types";
import { documentTitle, formatTrustScore, isFlaggedDocument, validTrustScore } from "@/lib/retrieval/selectors";
import { DecisionLabel, Panel, StatusBadge } from "./retrieval-ui";

function resultAnchor(eventId: string, resultId: string) {
  return `result-${encodeURIComponent(eventId)}-${encodeURIComponent(resultId)}`;
}

export function RetrievalEventDetail({ event }: { event: RetrievalEvent }) {
  const flagged = event.documents.filter(isFlaggedDocument);
  const allowed = event.documents.filter((document) => document.decision === "allowed").length;
  const excluded = event.documents.filter((document) => document.decision === "excluded").length;
  const citations = event.documents.filter((document) => event.answer.citationIds.includes(document.id) && document.decision === "allowed");

  return <div className="min-w-0 space-y-4" id="event-details">
    <Panel title="Selected retrieval event" description={event.id}>
      <p className="text-sm text-slate-500">{event.scenarioLabel} · {event.mode} mode</p>
      <h3 className="mt-4 text-sm font-semibold">Retrieval trace</h3>
      <ol aria-label="Retrieval trace" className="mt-3 flex flex-wrap gap-2 text-xs font-medium text-slate-700">
        {["Query", "Retrieved documents", "Trust decision", "Final answer"].map((step, index) => <li key={step} className="border bg-slate-50 px-2 py-2">{index + 1}. {step}{index < 3 ? " →" : ""}</li>)}
      </ol>
      <h3 className="mt-4 text-sm font-semibold">Query</h3>
      <p className="mt-2 break-words border-l-2 border-slate-900 pl-3 text-sm leading-6">{event.query || "Query text unavailable."}</p>
    </Panel>

    {flagged.length > 0 && <Panel title="Flagged source review" description="Select a source to inspect its document.">
      <ul className="space-y-3">
        {flagged.map((document) => <li key={document.id} className="border border-amber-200 bg-amber-50 p-3 text-sm">
          <a className="break-words font-semibold underline underline-offset-2" href={`#${resultAnchor(event.id, document.id)}`}>{documentTitle(document)}</a>
          <p className="mt-1 break-words text-xs text-slate-600">Document: {document.documentId?.trim() || "Not provided"} · Source: {document.source?.id?.trim() || "Not provided"}</p>
          <div className="mt-2 flex flex-wrap items-center gap-2"><StatusBadge status={document.status} /><span>Trust: {formatTrustScore(document.trustScore)}</span><span>·</span><DecisionLabel decision={document.decision} /></div>
          <p className="mt-2 leading-6"><strong>Flag reason:</strong> {document.flagReason?.trim() || "No flag reason provided."}</p>
        </li>)}
      </ul>
    </Panel>}

    <Panel id="documents" title="Retrieved documents" description="Trust scores and inclusion decisions.">
      <p className="mb-4 text-sm text-slate-600">{event.documents.length} retrieved · {allowed} allowed · {excluded} excluded{event.documents.length - allowed - excluded > 0 ? ` · ${event.documents.length - allowed - excluded} decisions unavailable` : ""}</p>
      {event.documents.length === 0 ? <p className="border border-dashed p-5 text-sm text-slate-600">No documents were retrieved for this event.</p> : <div className="space-y-4">
        {event.documents.map((document) => <article id={resultAnchor(event.id, document.id)} key={document.id} tabIndex={-1} className="scroll-mt-4 border border-slate-200 bg-white p-3 target:border-asu-maroon target:ring-2 target:ring-asu-maroon">
          <div className="flex flex-wrap items-start justify-between gap-2">
            <h3 className="min-w-0 flex-1 break-words text-sm font-semibold">{documentTitle(document)}</h3>
            <StatusBadge status={document.status} />
          </div>
          <dl className="mt-3 grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-2 text-xs">
            <dt className="text-slate-500">Document ID</dt><dd className="break-all font-mono">{document.documentId?.trim() || "Not provided"}</dd>
            <dt className="text-slate-500">Source ID</dt><dd className="break-all font-mono">{document.source?.id?.trim() || "Not provided"}</dd>
            <dt className="text-slate-500">Source</dt><dd className="break-words">{document.source?.name?.trim() || "Source metadata unavailable"}{document.source?.chunk ? ` / ${document.source.chunk}` : ""}</dd>
            <dt className="text-slate-500">Retrieval item</dt><dd className="break-all font-mono">{document.id}</dd>
          </dl>
          <div className="mt-4 flex flex-wrap items-center justify-between gap-2 text-sm">
            <span>Trust score: <strong>{formatTrustScore(document.trustScore)}</strong>{validTrustScore(document.trustScore) ? " / 1.00" : ""}</span>
            <DecisionLabel decision={document.decision} />
          </div>
          {validTrustScore(document.trustScore) && <meter className="mt-2 h-2 w-full" min={0} max={1} value={document.trustScore} aria-label={`Trust score for ${documentTitle(document)}`}>{document.trustScore}</meter>}
          {(isFlaggedDocument(document) || document.flagReason) && <p className="mt-3 border-l-2 border-amber-400 pl-3 text-sm leading-6"><strong>Flag reason:</strong> {document.flagReason?.trim() || "No flag reason provided."}</p>}
          <p className="mt-3 text-sm leading-6 text-slate-600">{document.excerpt?.trim() || "Document excerpt unavailable."}</p>
          {citations.some((citation) => citation.id === document.id) && <p className="mt-2 text-xs font-medium text-emerald-800">Cited in answer</p>}
        </article>)}
      </div>}
    </Panel>

    <Panel title="Final answer">
      <p className="text-sm font-semibold">{event.answer.status === "available" ? "Answer available" : event.answer.status === "withheld" ? "Answer withheld" : "Answer unavailable"}</p>
      {flagged.some((document) => document.decision === "allowed") && <p className="mt-3 border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">Flagged evidence was allowed into context.</p>}
      <p className="mt-3 whitespace-pre-wrap text-sm leading-7 text-slate-700">{event.answer.text?.trim() || "No answer was supplied for this event."}</p>
      {citations.length > 0 ? <ul aria-label="Answer citations" className="mt-4 space-y-2 text-sm">
        {citations.map((document) => <li key={document.id}><a className="break-words font-medium underline underline-offset-2" href={`#${resultAnchor(event.id, document.id)}`}>{documentTitle(document)} · {document.documentId?.trim() || "Document ID unavailable"}</a></li>)}
      </ul> : <p className="mt-3 text-xs text-slate-500">No answer citations available.</p>}
    </Panel>
  </div>;
}
