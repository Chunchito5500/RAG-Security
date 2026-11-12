import type { ReactNode } from "react";
import type { RetrievalDecision, RetrievalStatus } from "@/lib/retrieval/types";

export const buttonClass = "inline-flex items-center justify-center rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50";
export const primaryButtonClass = "inline-flex items-center justify-center rounded-md border border-slate-900 bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50";
export const inputClass = "w-full rounded-md border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-800 disabled:opacity-50";

export function Panel({ title, description, children, id, action }: { title: string; description?: string; children: ReactNode; id?: string; action?: ReactNode }) {
  return <section id={id} className="min-w-0 scroll-mt-6 rounded-lg border border-slate-200 bg-white p-4 sm:p-5">
    <div className="flex flex-wrap items-center justify-between gap-3"><h2 className="text-sm font-semibold text-slate-900">{title}</h2>{action}</div>
    {description && <p className="mt-1.5 max-w-3xl text-sm leading-6 text-slate-500">{description}</p>}
    <div className="mt-4 min-w-0">{children}</div>
  </section>;
}

export function StatusBadge({ status }: { status: RetrievalStatus }) {
  const styles = {
    trusted: "border-emerald-200 bg-emerald-50 text-emerald-800",
    flagged: "border-amber-200 bg-amber-50 text-amber-900",
    blocked: "border-red-200 bg-red-50 text-red-800",
    unknown: "border-slate-200 bg-slate-50 text-slate-600",
  };
  return <span className={`inline-flex whitespace-nowrap rounded border px-2 py-0.5 text-xs font-medium capitalize ${styles[status] ?? styles.unknown}`}>{status === "unknown" ? "Not recorded" : status}</span>;
}

export function DecisionLabel({ decision }: { decision: RetrievalDecision }) {
  return <span className="text-xs font-medium">{decision === "allowed" ? "Allowed into context" : decision === "excluded" ? "Excluded from context" : "Decision not recorded"}</span>;
}

export function MetricCard({ label, value, note }: { label: string; value: ReactNode; note?: string }) {
  return <div className="min-w-0 rounded-lg border border-slate-200 bg-white p-4"><p className="text-xs font-medium text-slate-500">{label}</p><p className="mt-2 text-2xl font-semibold tracking-tight text-slate-900">{value ?? "—"}</p>{note && <p className="mt-2 text-xs text-slate-500">{note}</p>}</div>;
}

export function LoadState({ loading, error, refresh }: { loading: boolean; error: string; refresh: () => void }) {
  if (loading) return <p role="status" className="py-5 text-sm text-slate-500">Loading…</p>;
  if (error) return <div role="alert" className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-800"><p>{error}</p><button className={`${buttonClass} mt-3`} onClick={refresh}>Retry</button></div>;
  return null;
}

export function EmptyState({ children }: { children: ReactNode }) {
  return <p className="rounded-md border border-dashed border-slate-200 p-5 text-sm leading-6 text-slate-500">{children}</p>;
}

export function SecurityExplanation() {
  return <details className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm"><summary className="cursor-pointer font-medium text-slate-700">Understanding security decisions</summary>
    <dl className="mt-3 grid gap-3 text-xs leading-6 sm:grid-cols-3"><div><dt className="font-semibold text-emerald-800">Trusted</dt><dd>Passed security checks and eligible for retrieval.</dd></div><div><dt className="font-semibold text-amber-800">Flagged</dt><dd>A warning is present. The document may still be eligible for context.</dd></div><div><dt className="font-semibold text-red-800">Blocked</dt><dd>The security decision excludes the document from trusted context.</dd></div></dl>
    <p className="mt-3 text-xs leading-6 text-slate-600">Trust score and quarantine are separate. A high-severity detection can quarantine a document even with a high numerical score. Context decisions show whether a candidate was selected; an eligible document is not necessarily selected.</p>
  </details>;
}
