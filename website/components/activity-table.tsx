import Link from "next/link";
import type { RetrievalEvent } from "@/lib/retrieval/types";
import { eventCounts, eventStatus, eventTime } from "@/lib/retrieval/presentation";
import { EmptyState } from "./retrieval-ui";

export function ActivityTable({ events, compact = false, flaggedOnly = false }: { events: RetrievalEvent[]; compact?: boolean; flaggedOnly?: boolean }) {
  if (!events.length) return <EmptyState>{flaggedOnly ? "No flagged retrieval events are recorded." : "No retrieval events yet. Run a query in Search to create one."}</EmptyState>;
  return <div className="table-scroll" tabIndex={0} role="region" aria-label="Retrieval activity"><table className="data-table"><thead><tr><th scope="col">Query</th><th scope="col">Timestamp</th><th scope="col">Retrieved</th>{!compact && <th scope="col">Allowed</th>}<th scope="col">Blocked</th><th scope="col">Status</th></tr></thead><tbody>
    {events.map(event => { const counts = eventCounts(event); return <tr key={event.id}><td className="name-cell"><Link className="font-medium text-slate-900 underline decoration-slate-300 underline-offset-4" href={`/activity?event=${encodeURIComponent(event.id)}`}>{event.query || "Query not recorded"}</Link></td><td className="min-w-36 text-xs">{eventTime(event.occurredAt)}</td><td>{counts.retrieved}</td>{!compact && <td>{counts.allowed}</td>}<td>{counts.blocked}</td><td><span className={`whitespace-nowrap text-xs ${counts.blocked ? "text-red-800" : counts.flagged ? "text-amber-800" : "text-slate-600"}`}>{eventStatus(event)}</span></td></tr>; })}
  </tbody></table></div>;
}
