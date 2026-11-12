import type { Metadata } from "next";
import { Suspense } from "react";
import { ActivityWorkspace } from "@/components/activity-workspace";

export const metadata: Metadata = { title: "Activity" };

export default function Page() {
  return <Suspense fallback={<p role="status" className="text-sm text-slate-500">Loading activity…</p>}><ActivityWorkspace /></Suspense>;
}
