import type { Metadata } from "next";
import { Suspense } from "react";
import { DocumentsWorkspace } from "@/components/documents-workspace";

export const metadata: Metadata = { title: "Documents" };

export default function Page() {
  return <Suspense fallback={<p role="status" className="text-sm text-slate-500">Loading documents…</p>}><DocumentsWorkspace /></Suspense>;
}
