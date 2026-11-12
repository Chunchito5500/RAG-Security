"use client";

import Link from "next/link";
import { api } from "@/lib/retrieval/api-source";
import { useApi } from "@/lib/retrieval/use-api";
import { isQuarantined, readableReason } from "@/lib/retrieval/presentation";
import { ActivityTable } from "./activity-table";
import { EmptyState, LoadState, MetricCard, Panel, StatusBadge } from "./retrieval-ui";

export function RagShieldDashboard() {
  const docs = useApi(api.documents);
  const events = useApi(api.listEvents);
  const evaluation = useApi(api.evaluation);
  const documents = docs.data?.data;
  const decisions = documents?.filter(d => d.security_status === "flagged" || isQuarantined(d)).slice(-5).reverse() ?? [];
  const metric = (loading: boolean, error: string, value?: number | string | null) => loading ? "…" : error ? "—" : value ?? "—";
  return <>
    <p className="text-sm text-slate-500">A concise view of your document collection and retrieval security.</p>
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <MetricCard label="Total Documents" value={metric(docs.loading, docs.error, documents?.length)} note="Current document library" />
      <MetricCard label="Quarantined Documents" value={metric(docs.loading, docs.error, documents?.filter(isQuarantined).length)} note="Excluded by security controls" />
      <MetricCard label="Retrieval Events" value={metric(events.loading, events.error, events.data?.length)} note="Recorded retrieval requests" />
      <MetricCard label="Evaluation Accuracy" value={metric(evaluation.loading, evaluation.error, evaluation.data?.data?.accuracy != null ? `${evaluation.data.data.accuracy}%` : null)} note="Current saved test dataset only" />
    </div>
    {evaluation.error && <LoadState {...evaluation} />}
    {!evaluation.loading && !evaluation.error && !evaluation.data?.data && <p className="text-xs text-slate-500">No saved evaluation is available. <Link href="/evaluation" className="underline">View evaluation</Link></p>}
    <Panel title="Recent retrieval activity" action={<Link href="/activity" className="text-xs font-medium text-slate-700 underline underline-offset-4">View all activity →</Link>}>
      <LoadState {...events} />{!events.loading && !events.error && <ActivityTable events={(events.data ?? []).slice(0, 5)} compact />}
    </Panel>
    <div className="grid min-w-0 gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)]">
      <Panel title="Security overview" description="Current document decisions, separate from recorded retrieval events.">
        <LoadState {...docs} />{!docs.loading && !docs.error && (documents?.length ? <div className="space-y-4">{(["trusted", "flagged", "blocked"] as const).map(status => <div key={status} className="flex items-center justify-between border-b border-slate-100 pb-3"><StatusBadge status={status} /><span className="font-semibold tabular-nums">{documents.filter(d => d.security_status === status).length}</span></div>)}{documents.some(d => d.security_status === "unknown") && <p className="text-xs text-slate-500">{documents.filter(d => d.security_status === "unknown").length} documents have no recorded status.</p>}<Link href="/documents" className="inline-block text-xs font-medium underline underline-offset-4">View document library →</Link></div> : <EmptyState>No documents have been loaded.</EmptyState>)}
        {!!docs.data?.errors?.length && !docs.loading && <p className="mt-3 text-xs text-amber-800">{docs.data.errors.length} files could not be processed. See Documents for details.</p>}
      </Panel>
      <Panel title="Recent security decisions" description="Latest available entries in library order; decision timestamps are not recorded." action={<Link href="/security" className="text-xs font-medium text-slate-700 underline underline-offset-4">Security Review →</Link>}>
        {docs.loading ? <p role="status" className="text-sm text-slate-500">Loading decisions…</p> : docs.error ? <p className="text-sm text-slate-500">Document decisions are unavailable.</p> : decisions.length ? <ul className="divide-y divide-slate-100">{decisions.map((d, index) => <li key={`${d.source}-${index}`} className="py-3 first:pt-0"><div className="flex flex-wrap items-center justify-between gap-2"><Link href={`/security?tab=${isQuarantined(d) ? "quarantined" : "flagged"}&document=${encodeURIComponent(d.document_id ?? "")}&source=${encodeURIComponent(d.source ?? "")}`} className="break-all text-sm font-medium text-slate-900 underline decoration-slate-300 underline-offset-4">{d.filename || d.document_id || "Document not recorded"}</Link><StatusBadge status={d.security_status} /></div><p className="mt-1 text-xs leading-5 text-slate-500">{readableReason(d.flag_reason)}</p></li>)}</ul> : <EmptyState>No warning or quarantine decisions in the library.</EmptyState>}
      </Panel>
    </div>
  </>;
}
