"use client";

import { useState } from "react";
import type { DocumentResult } from "@/lib/retrieval/api-types";
import { api } from "@/lib/retrieval/api-source";
import { formatTrustScore } from "@/lib/retrieval/selectors";
import { readableReason } from "@/lib/retrieval/presentation";
import { inputClass, Panel, primaryButtonClass, StatusBadge } from "./retrieval-ui";

export function DocumentUpload({ onUpload }: { onUpload: () => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<DocumentResult | null>(null);
  async function upload() {
    if (!file || uploading) return;
    setError(""); setResult(null);
    if (!/\.(pdf|txt)$/i.test(file.name) || file.size === 0 || file.size > 10 * 1024 * 1024) {
      setError("Choose a nonempty PDF or UTF-8 TXT file up to 10 MB."); return;
    }
    setUploading(true);
    try { setResult(await api.upload(file)); onUpload(); }
    catch (failure) { setError(failure instanceof Error ? failure.message : "Unable to upload document."); }
    finally { setUploading(false); }
  }
  return <Panel id="upload-panel" title="Upload a document" description="PDF or UTF-8 TXT · Up to 10 MB · Security checks run automatically.">
    <form onSubmit={e => { e.preventDefault(); void upload(); }} className="space-y-4"><label className="block text-sm font-medium">File<input className={`${inputClass} mt-2 max-w-lg`} type="file" accept=".pdf,.txt" disabled={uploading} onChange={e => { setFile(e.target.files?.[0] ?? null); setResult(null); setError(""); }} /></label><button type="submit" disabled={!file || uploading} className={primaryButtonClass}>{uploading ? "Processing…" : "Upload and process"}</button>{error && <p role="alert" className="text-sm text-red-700">{error}</p>}{result && <div role="status" className="rounded-md border border-slate-200 bg-slate-50 p-4 text-sm"><div className="flex flex-wrap gap-3"><span className="break-all font-medium">{result.filename} processed</span><StatusBadge status={result.security_status} /><span>Trust: {formatTrustScore(result.trust_score)}</span></div><p className="mt-2">{readableReason(result.flag_reason)}</p></div>}</form>
  </Panel>;
}
