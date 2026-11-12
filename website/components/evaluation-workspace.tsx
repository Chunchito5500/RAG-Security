"use client";

import { api } from "@/lib/retrieval/api-source";
import type { EvaluationResult } from "@/lib/retrieval/api-types";
import { useApi } from "@/lib/retrieval/use-api";
import { eventTime, recordedStatus } from "@/lib/retrieval/presentation";
import { formatTrustScore } from "@/lib/retrieval/selectors";
import { buttonClass, EmptyState, LoadState, MetricCard, Panel, StatusBadge } from "./retrieval-ui";

export function EvaluationResults({ result }: { result: EvaluationResult }) {
  const metrics = { "Documents Tested": result.total_documents_tested, "Correct Classifications": result.correctly_classified, Accuracy: result.accuracy == null ? null : `${result.accuracy}%`, "False Positives": result.false_positives, "Missed Malicious Documents": result.missed_malicious_documents };
  return <>
    <p className="text-xs text-slate-500">Saved dataset: {result.source} · {eventTime(result.timestamp)}</p>
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">{Object.entries(metrics).map(([label, value]) => <MetricCard key={label} label={label} value={value} note={value == null ? "Not available" : undefined} />)}</div>
    <div className="grid gap-4 sm:grid-cols-2">{[["Quarantine Exclusion Result", result.quarantine_exclusion_passed], ["All-Quarantined Retrieval Result", result.all_low_trust_test_passed]].map(([label, passed]) => <div key={String(label)} className="rounded-lg border border-slate-200 bg-white p-4"><p className="text-xs font-medium text-slate-500">{label}</p><p className={`mt-2 text-sm font-semibold ${passed === true ? "text-emerald-800" : passed === false ? "text-red-800" : "text-slate-500"}`}>{passed === true ? "Passed" : passed === false ? "Failed" : "Not recorded"}</p></div>)}</div>
    <Panel title="Individual test results">
      {result.individual_test_results.length ? <div className="table-scroll" role="region" aria-label="Evaluation results" tabIndex={0}><table className="data-table"><thead><tr>{["Document", "Expected classification", "Trust score", "Security decision", "Correct", "Entered context"].map(label => <th key={label} scope="col">{label}</th>)}</tr></thead><tbody>{result.individual_test_results.map((row, index) => <tr key={`${row.filename}-${index}`}><td className="name-cell font-medium">{row.filename || row.document_id || "Not recorded"}{row.error && <p className="mt-1 text-xs text-red-700">{row.error}</p>}</td><td className="capitalize">{row.expected || "Not recorded"}</td><td>{formatTrustScore(row.trust_score)}</td><td><StatusBadge status={recordedStatus(row.decision)} /><span className="ml-2 text-xs">{row.decision || ""}</span></td><td className={row.correct === true ? "text-emerald-800" : row.correct === false ? "text-red-800" : ""}>{row.correct == null ? "Not recorded" : row.correct ? "Yes" : "No"}</td><td>{row.entered_context == null ? "Not recorded" : row.entered_context ? "Yes" : "No"}</td></tr>)}</tbody></table></div> : <EmptyState>No individual test results are available.</EmptyState>}
    </Panel>
  </>;
}

export function EvaluationWorkspace() {
  const resource = useApi(api.evaluation);
  return <>
    <Panel title="Saved security evaluation" description="Results from the current saved test dataset. These metrics do not measure every uploaded document or imply universal detection accuracy." action={<button className={buttonClass} disabled={resource.loading} onClick={resource.refresh}>Refresh results</button>}>
      <p className="text-xs text-slate-500">Refreshing reads the saved results; it does not rerun validation.</p><LoadState {...resource} />
    </Panel>
    {!resource.loading && !resource.error && (resource.data?.data ? <EvaluationResults result={resource.data.data} /> : <EmptyState>No saved evaluation results are available.</EmptyState>)}
  </>;
}
