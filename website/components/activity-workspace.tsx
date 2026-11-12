"use client";

import Link from "next/link";
import { useState } from "react";
import { useSearchParams } from "next/navigation";
import { api } from "@/lib/retrieval/api-source";
import { useApi } from "@/lib/retrieval/use-api";
import { isFlaggedEvent } from "@/lib/retrieval/selectors";
import { ActivityTable } from "./activity-table";
import { RetrievalEventDetail } from "./retrieval-event-detail";
import { buttonClass, EmptyState, LoadState, Panel, primaryButtonClass } from "./retrieval-ui";

export function ActivityWorkspace() {
  const resource = useApi(api.listEvents);
  const [flagged, setFlagged] = useState(false);
  const selectedId = useSearchParams().get("event");
  const selected = resource.data?.find(event => event.id === selectedId);
  return <>
    <Panel title={selectedId ? "Retrieval event" : "Retrieval history"} description="Review recorded queries, source decisions, and available passages." action={<button onClick={resource.refresh} disabled={resource.loading} className={buttonClass}>Refresh</button>}>
      {selectedId && <Link href="/activity" className="mb-4 inline-block text-sm font-medium text-slate-700 underline underline-offset-4">← All activity</Link>}
      <LoadState {...resource} />
      {!resource.loading && !resource.error && (selectedId ? !selected && <EmptyState>This event is not available in the current retrieval history.</EmptyState> : <>
        <div className="mb-4 flex gap-2" role="group" aria-label="Filter retrieval events"><button className={!flagged ? primaryButtonClass : buttonClass} aria-pressed={!flagged} onClick={() => setFlagged(false)}>All Events</button><button className={flagged ? primaryButtonClass : buttonClass} aria-pressed={flagged} onClick={() => setFlagged(true)}>Flagged Events</button></div>
        <ActivityTable events={(resource.data ?? []).filter(event => !flagged || isFlaggedEvent(event))} flaggedOnly={flagged} />
      </>)}
    </Panel>
    {!resource.loading && !resource.error && selected && <RetrievalEventDetail event={selected} />}
  </>;
}
