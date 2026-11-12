"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { api } from "@/lib/retrieval/api-source";
import { useApi } from "@/lib/retrieval/use-api";
import { eligibility, eventTime, isQuarantined, readableReason, recordedStatus, severity } from "@/lib/retrieval/presentation";
import { formatTrustScore } from "@/lib/retrieval/selectors";
import { DetectionDetails, DocumentDetails } from "./document-details";
import { buttonClass, EmptyState, LoadState, Panel, primaryButtonClass, SecurityExplanation, StatusBadge } from "./retrieval-ui";

function RelatedEvents({ documentId }: { documentId: string | null }) {
  const resource = useApi(api.listEvents);
  const related = resource.data?.filter(event => event.documents.some(d => d.documentId === documentId)) ?? [];
  return <Panel title="Related retrieval events"><LoadState {...resource} />{!resource.loading && !resource.error && (related.length ? <ul className="max-h-72 space-y-3 overflow-y-auto">{related.map(event => <li key={event.id}><Link className="break-words text-sm font-medium underline underline-offset-4" href={`/activity?event=${encodeURIComponent(event.id)}`}>{event.query || event.id}</Link><p className="mt-1 text-xs text-slate-500">{eventTime(event.occurredAt)}</p></li>)}</ul> : <EmptyState>No related retrieval events are recorded. A document can be quarantined during ingestion before it appears in a query.</EmptyState>)}</Panel>;
}

function SecurityDocuments({ tab, selectedId, source }: { tab: string; selectedId: string | null; source: string | null }) {
  const resource = useApi(api.documents);
  const documents = resource.data?.data.filter(d => tab === "quarantined" ? isQuarantined(d) : d.security_status === "flagged" && !isQuarantined(d)) ?? [];
  const selected = documents.find(d => d.document_id === selectedId && (!source || d.source === source));
  return <>
    <Panel title={tab === "quarantined" ? "Quarantined documents" : "Flagged documents"} description={tab === "quarantined" ? "Documents excluded by the recorded security decision." : "Documents with warnings. These may remain eligible for retrieval context."} action={<button className={buttonClass} disabled={resource.loading} onClick={resource.refresh}>Refresh</button>}>
      <LoadState {...resource} />{!resource.loading && !resource.error && (selectedId ? <Link href={`/security?tab=${tab}`} className="text-sm underline underline-offset-4">← Back to documents</Link> : documents.length ? <div className="table-scroll" role="region" aria-label="Security documents" tabIndex={0}><table className="data-table"><thead><tr>{["Document", "Trust score", "Severity", "Security status", "Flag reason", "Context eligibility"].map(label => <th scope="col" key={label}>{label}</th>)}</tr></thead><tbody>{documents.map((d, index) => <tr key={`${d.source}-${index}`}><td className="name-cell"><Link href={`/security?tab=${tab}&document=${encodeURIComponent(d.document_id ?? "")}&source=${encodeURIComponent(d.source ?? "")}`} className="font-medium underline underline-offset-4">{d.filename || d.document_id || "Not recorded"}</Link></td><td>{formatTrustScore(d.trust_score)}</td><td className="whitespace-nowrap text-xs">{severity(d.flag_reason)}</td><td><StatusBadge status={d.security_status} /></td><td className="min-w-52 max-w-md text-xs leading-5">{readableReason(d.flag_reason)}</td><td className="text-xs">{eligibility(d.decision)}</td></tr>)}</tbody></table></div> : <EmptyState>No {tab === "quarantined" ? "quarantined" : "flagged"} documents in the current library.</EmptyState>)}
    </Panel>
    {!resource.loading && !resource.error && selectedId && (selected ? <><DocumentDetails document={selected} /><RelatedEvents documentId={selected.document_id} /></> : <EmptyState>This document is not available in this review category.</EmptyState>)}
  </>;
}

function SecurityEvents({ selectedId }: { selectedId: string | null }) {
  const resource = useApi(api.flaggedEvents);
  const selected = resource.data?.data.find(event => event.event_id === selectedId);
  return <>
    <Panel title="All security events" description="Warning and quarantine records created during retrieval. This log may be empty even when ingestion has quarantined documents." action={<button className={buttonClass} onClick={resource.refresh} disabled={resource.loading}>Refresh</button>}>
      <LoadState {...resource} />{!resource.loading && !resource.error && (selectedId ? <Link href="/security?tab=events" className="text-sm underline underline-offset-4">← All security events</Link> : resource.data?.data.length ? <div className="table-scroll" role="region" aria-label="Security events" tabIndex={0}><table className="data-table"><thead><tr>{["Document", "Trust score", "Severity", "Security status", "Flag reason", "Context eligibility"].map(label => <th scope="col" key={label}>{label}</th>)}</tr></thead><tbody>{resource.data.data.map(event => <tr key={event.event_id}><td className="name-cell"><Link className="font-medium underline underline-offset-4" href={`/security?tab=events&record=${encodeURIComponent(event.event_id)}`}>{event.document_id || "Not recorded"}</Link><p className="mt-1 text-xs text-slate-500">{eventTime(event.timestamp)}</p></td><td>{formatTrustScore(event.trust_score)}</td><td className="whitespace-nowrap text-xs">{severity(event.flag_reason, event.high_severity_flag)}</td><td><StatusBadge status={recordedStatus(event.status)} /></td><td className="min-w-52 max-w-md text-xs leading-5">{readableReason(event.flag_reason)}</td><td className="text-xs">{eligibility(event.status)}</td></tr>)}</tbody></table></div> : <EmptyState>No security events have been recorded. Review the Quarantined tab for current document decisions.</EmptyState>)}
    </Panel>
    {!resource.loading && !resource.error && selectedId && (selected ? <Panel title="Security decision details" description={selected.document_id || "Document identifier not recorded"}>
      <dl className="grid gap-4 text-sm sm:grid-cols-2"><div><dt className="text-xs text-slate-500">Query</dt><dd className="mt-1 break-words">{selected.query || "Not recorded"}</dd></div><div><dt className="text-xs text-slate-500">Timestamp</dt><dd className="mt-1">{eventTime(selected.timestamp)}</dd></div><div><dt className="text-xs text-slate-500">Trust score</dt><dd className="mt-1">{formatTrustScore(selected.trust_score)}</dd></div><div><dt className="text-xs text-slate-500">Security status</dt><dd className="mt-1"><StatusBadge status={recordedStatus(selected.status)} /></dd></div><div><dt className="text-xs text-slate-500">Source</dt><dd className="mt-1 break-all">{selected.source || "Source metadata not recorded"}</dd></div><div><dt className="text-xs text-slate-500">Quarantine</dt><dd className="mt-1">{selected.quarantine_status === "flagged" || selected.status?.toLowerCase() === "quarantine" ? "Quarantined" : selected.quarantine_status === "none" ? "Not quarantined" : "Not recorded"}</dd></div><div><dt className="text-xs text-slate-500">Context eligibility</dt><dd className="mt-1">{eligibility(selected.status)}</dd></div></dl>
      <div className="mt-5 border-t pt-4"><DetectionDetails reason={selected.flag_reason} /></div>
      {selected.request_id ? <Link className="mt-4 inline-block text-sm font-medium underline underline-offset-4" href={`/activity?event=${encodeURIComponent(selected.request_id)}`}>View related retrieval and actual context decision →</Link> : <p className="mt-4 text-xs text-slate-500">No related retrieval event was recorded.</p>}
    </Panel> : <EmptyState>This security event is not available.</EmptyState>)}
  </>;
}

export function SecurityWorkspace() {
  const params = useSearchParams();
  const tab = ["flagged", "quarantined", "events"].includes(params.get("tab") ?? "") ? params.get("tab")! : "flagged";
  return <>
    <nav aria-label="Security review categories" className="flex flex-wrap gap-2">{[["flagged", "Flagged"], ["quarantined", "Quarantined"], ["events", "All Security Events"]].map(([value, label]) => <Link key={value} href={`/security?tab=${value}`} aria-current={tab === value ? "page" : undefined} className={tab === value ? primaryButtonClass : buttonClass}>{label}</Link>)}</nav>
    <SecurityExplanation />
    {tab === "events" ? <SecurityEvents selectedId={params.get("record")} /> : <SecurityDocuments tab={tab} selectedId={params.get("document")} source={params.get("source")} />}
  </>;
}
