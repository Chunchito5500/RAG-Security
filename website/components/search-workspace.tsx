"use client";

import { useState } from "react";
import { api } from "@/lib/retrieval/api-source";
import type { RetrievalEvent } from "@/lib/retrieval/types";
import { RetrievalEventDetail } from "./retrieval-event-detail";
import { EmptyState, inputClass, Panel, primaryButtonClass } from "./retrieval-ui";

export function SearchWorkspace() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(3);
  const [event, setEvent] = useState<RetrievalEvent | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  async function runQuery() {
    if (running) return;
    setError("");
    if (!query.trim()) { setError("Enter a question."); return; }
    setRunning(true);
    setEvent(null);
    try { setEvent(await api.query(query.trim(), topK)); }
    catch (failure) { setError(failure instanceof Error ? failure.message : "Unable to run retrieval."); }
    finally { setRunning(false); }
  }
  return <>
    <Panel title="Search your documents" description="Retrieve source passages and inspect the security decisions behind each result.">
      <form onSubmit={e => { e.preventDefault(); void runQuery(); }} className="space-y-4">
        <label className="block text-sm font-medium text-slate-700" htmlFor="query">Question</label>
        <textarea id="query" maxLength={2000} rows={3} disabled={running} value={query} onChange={e => { setQuery(e.target.value); setError(""); }} placeholder="What information is contained in these documents?" className={inputClass} aria-invalid={!!error} aria-describedby={error ? "query-error" : undefined} />
        <div className="flex flex-wrap items-end gap-3"><label className="text-xs font-medium text-slate-600">Candidate retrieval count<select className={`${inputClass} mt-1 min-w-32 pr-9`} value={topK} disabled={running} onChange={e => setTopK(Number(e.target.value))}>{[3, 5, 10, 20, 50].map(n => <option key={n}>{n}</option>)}</select></label><button type="submit" disabled={running} className={primaryButtonClass}>{running ? "Retrieving…" : "Run Retrieval"}</button></div>
        {error && <p id="query-error" role="alert" className="text-sm text-red-700">{error}</p>}
      </form>
    </Panel>
    <div aria-live="polite">{running ? <p role="status" className="text-sm text-slate-500">Evaluating retrieved candidates…</p> : event ? <RetrievalEventDetail event={event} /> : !error && <EmptyState>Your retrieval results will appear here after you run a query.</EmptyState>}</div>
  </>;
}
