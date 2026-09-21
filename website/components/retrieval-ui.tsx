import type { ReactNode } from "react";
import type { RetrievalDecision, RetrievalStatus } from "@/lib/retrieval/types";

export const buttonClass = "border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition hover:border-slate-500 disabled:cursor-not-allowed disabled:opacity-50";
export const primaryButtonClass = "border border-slate-900 bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50";

export function Panel({ title, description, children, id }: { title: string; description?: string; children: ReactNode; id?: string }) {
  return <section id={id} className="min-w-0 scroll-mt-4 border bg-white p-4">
    <h2 className="text-base font-semibold text-slate-900">{title}</h2>
    {description && <p className="mt-1 break-words text-sm text-slate-600">{description}</p>}
    <div className="mt-4">{children}</div>
  </section>;
}

export function StatusBadge({ status }: { status: RetrievalStatus }) {
  const styles = {
    trusted: "border-emerald-200 bg-emerald-50 text-emerald-800",
    flagged: "border-amber-200 bg-amber-50 text-amber-900",
    blocked: "border-red-200 bg-red-50 text-red-800",
    unknown: "border-slate-200 bg-slate-50 text-slate-700",
  };
  return <span className={`inline-flex border px-2 py-1 text-xs font-semibold capitalize ${styles[status]}`}>{status === "unknown" ? "Status unavailable" : status}</span>;
}

export function DecisionLabel({ decision }: { decision: RetrievalDecision }) {
  return <span className="font-semibold">{decision === "allowed" ? "Allowed into context" : decision === "excluded" ? "Excluded from context" : "Decision unavailable"}</span>;
}
