import type { Metadata } from "next";
import { Suspense } from "react";
import { SecurityWorkspace } from "@/components/security-workspace";

export const metadata: Metadata = { title: "Security Review" };

export default function Page() {
  return <Suspense fallback={<p role="status" className="text-sm text-slate-500">Loading security review…</p>}><SecurityWorkspace /></Suspense>;
}
