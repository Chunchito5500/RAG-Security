"use client";

import { useEffect, useRef, useState } from "react";
import { createDemoDataSource, demoModes, runDemoScenario } from "@/lib/retrieval/demo-source";
import { demoIngestionQueue, demoScenarios } from "@/lib/retrieval/mock-scenarios";
import { isFlaggedDocument, isFlaggedEvent } from "@/lib/retrieval/selectors";
import type { DemoDataset, RetrievalEvent, RetrievalMode } from "@/lib/retrieval/types";
import { RetrievalEventDetail } from "./retrieval-event-detail";
import { buttonClass, Panel, primaryButtonClass } from "./retrieval-ui";

type LoadState = "loading" | "ready" | "error";

function eventTime(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Time unavailable" : date.toLocaleString("en-US", { dateStyle: "medium", timeStyle: "short" });
}

export function RagShieldDashboard() {
  const [mode, setMode] = useState<RetrievalMode>("strict");
  const [scenarioId, setScenarioId] = useState(demoScenarios[0].id);
  const [query, setQuery] = useState(demoScenarios[0].query);
  const [dataset, setDataset] = useState<DemoDataset>("scenarios");
  const [events, setEvents] = useState<RetrievalEvent[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [flaggedOnly, setFlaggedOnly] = useState(false);
  const [loadState, setLoadState] = useState<LoadState>("loading");
  const [loadError, setLoadError] = useState("");
  const [reload, setReload] = useState(0);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState("");
  const [lastRun, setLastRun] = useState("Ready.");
  const requestVersion = useRef(0);
  const activeScenario = demoScenarios.find((scenario) => scenario.id === scenarioId) ?? demoScenarios[0];

  useEffect(() => {
    const version = ++requestVersion.current;
    setLoadState("loading");
    setEvents([]);
    setSelectedId(null);
    setLoadError("");
    setRunError("");
    setRunning(false);
    setLastRun("Ready.");
    // Replace this factory with a backend adapter implementing RetrievalDataSource.
    createDemoDataSource(dataset).listEvents().then((records) => {
      if (version !== requestVersion.current) return;
      setEvents(records);
      setLoadState("ready");
    }).catch((error: unknown) => {
      if (version !== requestVersion.current) return;
      setLoadError(error instanceof Error ? error.message : "Unable to load retrieval events.");
      setLoadState("error");
    });
    return () => { requestVersion.current = version + 1; };
  }, [dataset, reload]);

  const visibleEvents = flaggedOnly ? events.filter(isFlaggedEvent) : events;
  const selectedEvent = visibleEvents.find((event) => event.id === selectedId) ?? visibleEvents[0];
  const flaggedCount = events.filter(isFlaggedEvent).length;
  const busy = running || loadState === "loading";

  async function runSimulation() {
    if (busy || loadState !== "ready") return;
    setRunError("");
    // Do not silently return an unrelated canned answer for arbitrary user input.
    const matchedScenario = demoScenarios.find((scenario) => scenario.query.trim().toLowerCase() === query.trim().toLowerCase());
    if (!matchedScenario) {
      setRunError(query.trim() ? "This question is unavailable. Select a query preset." : "Enter a question or select a query preset.");
      return;
    }
    const version = requestVersion.current;
    setRunning(true);
    try {
      const record = await runDemoScenario(matchedScenario.id, mode);
      if (version !== requestVersion.current) return;
      setEvents((previous) => [record, ...previous]);
      setSelectedId(record.id);
      setFlaggedOnly(false);
      setScenarioId(matchedScenario.id);
      setLastRun(`${matchedScenario.label} · ${mode} mode. Event added.`);
    } catch (error: unknown) {
      if (version === requestVersion.current) setRunError(error instanceof Error ? error.message : "Unable to run retrieval. Please try again.");
    } finally {
      if (version === requestVersion.current) setRunning(false);
    }
  }

  return <div className="pb-10">
    <header className="border-b bg-white">
      <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-4 sm:px-6 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex items-center gap-4">
          <div className="grid h-10 w-10 place-items-center rounded bg-slate-900 text-xs font-bold tracking-[0.24em] text-white">RS</div>
          <div><p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Retrieval Poisoning Defense</p><h1 className="text-lg font-semibold text-slate-900">Retrieval Security Dashboard</h1></div>
        </div>
        <nav aria-label="Main navigation" className="flex flex-wrap items-center gap-4 text-sm text-slate-600">
          <a className="hover:text-slate-900" href="#dashboard" onClick={() => setFlaggedOnly(false)}>Dashboard</a>
          <a className="hover:text-slate-900" href="#retrieval-events" onClick={() => setFlaggedOnly(true)}>Flagged review</a>
          <a className="hover:text-slate-900" href={loadState === "ready" && selectedEvent ? "#documents" : "#retrieval-events"}>Docs</a>
          <a className="hover:text-slate-900" href="#queue">Queue</a>
        </nav>
      </div>
    </header>

    <div id="dashboard" className="mx-auto max-w-6xl space-y-6 px-4 py-6 sm:px-6">
      <Panel title="Query">
        <div className="grid gap-5 lg:grid-cols-2">
          <div>
            <p className="mb-2 text-sm font-medium text-slate-700">Query presets</p>
            <div className="flex flex-wrap gap-2">
              {demoScenarios.map((scenario) => <button type="button" key={scenario.id} disabled={busy} aria-pressed={scenario.id === scenarioId} className={scenario.id === scenarioId ? primaryButtonClass : buttonClass} onClick={() => {
                setScenarioId(scenario.id); setQuery(scenario.query); setRunError("");
                setLastRun(`Selected: ${scenario.label}.`);
              }}>{scenario.label}</button>)}
            </div>
            <p className="mt-3 text-sm leading-6 text-slate-600">{activeScenario.description}</p>
            <label className="mt-4 block text-sm font-medium text-slate-700">Retrieval mode
              <select value={mode} disabled={busy} onChange={(event) => setMode(event.target.value as RetrievalMode)} className="mt-2 w-full border border-slate-300 bg-white py-3 pl-3 pr-10 text-sm">
                {demoModes.map((entry) => <option key={entry.id} value={entry.id}>{entry.label} — {entry.description}</option>)}
              </select>
            </label>
          </div>
          <form onSubmit={(event) => { event.preventDefault(); void runSimulation(); }}>
            <label htmlFor="query" className="mb-2 block text-sm font-medium text-slate-700">Question</label>
            <textarea id="query" value={query} disabled={busy} onChange={(event) => { setQuery(event.target.value); setRunError(""); }} aria-describedby={runError ? "query-error" : undefined} aria-invalid={!!runError} className="min-h-[120px] w-full border border-slate-300 bg-white px-3 py-3 text-sm leading-6 text-slate-700" />
            {runError && <p id="query-error" role="alert" className="mt-3 text-sm text-red-700">{runError}</p>}
            <button type="submit" disabled={busy || loadState !== "ready"} className={`${primaryButtonClass} mt-4`}>{running ? "Retrieving…" : "Run retrieval"}</button>
            <p role="status" className="mt-3 text-sm text-slate-500">{lastRun}</p>
          </form>
        </div>
      </Panel>

      <section id="retrieval-events" aria-labelledby="events-heading" className="scroll-mt-4 space-y-4">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div><h2 id="events-heading" className="text-lg font-semibold">Retrieval events</h2><p className="mt-1 text-sm text-slate-600">Inspect queries, sources, and trust decisions.</p></div>
          <label className="text-xs font-medium text-slate-600">Record set (resets history)
            <select value={dataset} disabled={running} onChange={(event) => setDataset(event.target.value as DemoDataset)} className="mt-1 block w-full border border-slate-300 bg-white py-2 pl-3 pr-10 text-sm">
              <option value="scenarios">All records</option><option value="trusted-only">Trusted only</option><option value="empty">No retrieval records</option><option value="error">Loading failure</option>
            </select>
          </label>
        </div>
        <div className="flex flex-wrap gap-2" role="group" aria-label="Filter retrieval events">
          <button type="button" aria-pressed={!flaggedOnly} onClick={() => setFlaggedOnly(false)} className={!flaggedOnly ? primaryButtonClass : buttonClass}>All events{loadState === "ready" ? ` (${events.length})` : ""}</button>
          <button type="button" aria-pressed={flaggedOnly} onClick={() => setFlaggedOnly(true)} className={flaggedOnly ? primaryButtonClass : buttonClass}>Flagged events{loadState === "ready" ? ` (${flaggedCount})` : ""}</button>
        </div>
        <p className="text-xs text-slate-500">Flagged events include blocked documents.</p>

        {loadState === "loading" && <div role="status" className="border bg-white p-8 text-sm text-slate-600">Loading retrieval events…</div>}
        {loadState === "error" && <div role="alert" className="border border-red-200 bg-white p-5"><h3 className="font-semibold text-red-800">Unable to load retrieval events</h3><p className="mt-2 text-sm text-slate-600">{loadError}</p><button type="button" className={`${buttonClass} mt-4`} onClick={() => setReload((value) => value + 1)}>Retry loading</button></div>}
        {loadState === "ready" && visibleEvents.length === 0 && <div role="status" className="border border-dashed border-slate-300 bg-white p-8">
          <h3 className="font-semibold">{events.length === 0 ? "No retrieval records yet" : "No flagged retrieval events"}</h3>
          <p className="mt-2 text-sm text-slate-600">{events.length === 0 ? "Run a query or choose another record set." : "No flagged or blocked documents found."}</p>
          {events.length > 0 && <button type="button" className={`${buttonClass} mt-4`} onClick={() => setFlaggedOnly(false)}>View all events</button>}
        </div>}
        {loadState === "ready" && selectedEvent && <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1.6fr)]">
          <div className="min-w-0 border bg-white p-4">
            <h3 className="mb-3 text-sm font-semibold">{flaggedOnly ? "Flagged retrieval review" : "Retrieval events"}</h3>
            <ul className="max-h-[650px] space-y-3 overflow-y-auto">
              {visibleEvents.map((event) => {
                const count = event.documents.filter(isFlaggedDocument).length;
                return <li key={event.id}><button type="button" aria-pressed={selectedEvent.id === event.id} aria-controls="event-details" onClick={() => setSelectedId(event.id)} className={`w-full border p-3 text-left transition ${selectedEvent.id === event.id ? "border-slate-900 bg-slate-50 ring-1 ring-slate-900" : "border-slate-200 hover:border-slate-500"}`}>
                  <span className="block text-xs font-semibold text-slate-500">{event.scenarioLabel || "Retrieval event"} · {event.mode}</span>
                  <span className="mt-2 block break-words text-sm font-medium leading-6">{event.query || "Query text unavailable"}</span>
                  <span className="mt-2 block break-all font-mono text-xs text-slate-500">{event.id}</span>
                  <span className="mt-1 block text-xs text-slate-500">{eventTime(event.occurredAt)}</span>
                  <span className="mt-3 flex flex-wrap gap-2 text-xs"><span>{event.documents.length} documents</span><span className={count > 0 ? "font-semibold text-amber-800" : "text-emerald-800"}>{count > 0 ? `${count} flagged / blocked` : "No flags"}</span></span>
                </button></li>;
              })}
            </ul>
          </div>
          <RetrievalEventDetail event={selectedEvent} />
        </div>}
      </section>

      <Panel id="queue" title="Ingestion queue">
        <div className="grid gap-3 sm:grid-cols-2">
          {demoIngestionQueue.map((item) => <div key={item.id} className="min-w-0 border border-slate-200 p-3"><div className="flex flex-wrap items-start justify-between gap-2"><h3 className="min-w-0 break-all text-sm font-semibold">{item.name}</h3><span className="text-xs font-medium text-slate-500">{item.stage}</span></div><p className="mt-2 text-sm leading-6 text-slate-600">{item.note}</p></div>)}
        </div>
      </Panel>
    </div>
  </div>;
}
