"use client";

import Link from "next/link";
import { useState } from "react";
import { useSearchParams } from "next/navigation";
import { api } from "@/lib/retrieval/api-source";
import { useApi } from "@/lib/retrieval/use-api";
import { isQuarantined } from "@/lib/retrieval/presentation";
import { formatTrustScore } from "@/lib/retrieval/selectors";
import { DocumentUpload } from "./document-panels";
import { DocumentDetails } from "./document-details";
import { buttonClass, EmptyState, inputClass, LoadState, Panel, primaryButtonClass, StatusBadge } from "./retrieval-ui";

export function DocumentsWorkspace() {
  const resource = useApi(api.documents);
  const [filter, setFilter] = useState("All");
  const [search, setSearch] = useState("");
  const [uploadOpen, setUploadOpen] = useState(false);
  const params = useSearchParams();
  const selectedId = params.get("document");
  const selectedSource = params.get("source");
  const selected = resource.data?.data.find(d => d.document_id === selectedId && (!selectedSource || d.source === selectedSource));
  const documents = (resource.data?.data ?? []).filter(d => (!search || (d.filename || d.document_id || "").toLowerCase().includes(search.toLowerCase())) && (filter === "All" || (filter === "Quarantined" ? isQuarantined(d) : d.security_status === filter.toLowerCase())));
  return <>
    <div className="flex flex-wrap items-center justify-between gap-3"><p className="text-sm text-slate-500">Manage and inspect your document collection.</p><button className={primaryButtonClass} aria-expanded={uploadOpen} aria-controls="upload-panel" onClick={() => setUploadOpen(!uploadOpen)}>{uploadOpen ? "Close upload" : "Upload Document"}</button></div>
    {uploadOpen && <DocumentUpload onUpload={resource.refresh} />}
    {selectedId ? <><Link href="/documents" className="inline-block text-sm underline underline-offset-4">← Document library</Link><LoadState {...resource} />{!resource.loading && !resource.error && (selected ? <DocumentDetails document={selected} /> : <EmptyState>This document is not available in the current library.</EmptyState>)}</> : <Panel title="Document library" action={<button className={buttonClass} onClick={resource.refresh} disabled={resource.loading}>Refresh</button>}>
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3"><div role="group" aria-label="Filter documents" className="flex flex-wrap gap-2">{["All", "Trusted", "Flagged", "Quarantined"].map(value => <button key={value} aria-pressed={filter === value} onClick={() => setFilter(value)} className={filter === value ? primaryButtonClass : buttonClass}>{value}</button>)}</div><label className="w-full text-xs font-medium text-slate-500 sm:w-64">Search filenames<input type="search" className={`${inputClass} mt-1`} value={search} onChange={e => setSearch(e.target.value)} placeholder="Filter by filename" /></label></div>
      <LoadState {...resource} />{!resource.loading && !resource.error && (documents.length ? <div className="table-scroll" tabIndex={0} role="region" aria-label="Document library"><table className="data-table"><thead><tr>{["Filename", "Type", "Trust score", "Security status", "Quarantine", "Source", "Actions"].map(label => <th scope="col" key={label}>{label}</th>)}</tr></thead><tbody>{documents.map((d, index) => <tr key={`${d.source}-${index}`}><td className="name-cell font-medium">{d.filename || d.document_id || "Not recorded"}</td><td>{/\.(pdf|txt)$/i.exec(d.filename ?? "")?.[1].toUpperCase() || "—"}</td><td>{formatTrustScore(d.trust_score)}</td><td><StatusBadge status={d.security_status} /></td><td className="text-xs">{isQuarantined(d) ? "Quarantined" : d.quarantine_status === "none" ? "None" : "Not recorded"}</td><td className="name-cell text-xs">{d.source || "Not recorded"}</td><td>{d.document_id ? <Link className="whitespace-nowrap text-xs font-medium underline underline-offset-4" href={`/documents?document=${encodeURIComponent(d.document_id)}&source=${encodeURIComponent(d.source ?? "")}`}>View details<span className="sr-only"> for {d.filename || d.document_id}</span></Link> : "Unavailable"}</td></tr>)}</tbody></table></div> : <EmptyState>{resource.data?.data.length ? "No documents match these filters." : "No documents loaded. Upload a PDF or TXT file to begin."}</EmptyState>)}
      {!resource.loading && resource.data?.errors?.map(error => <p role="alert" key={error.source} className="mt-3 text-sm text-red-700">{error.filename}: {error.error}</p>)}
    </Panel>}
  </>;
}
