import type { Metadata } from "next";
import { RagShieldDashboard } from "@/components/rag-shield-dashboard";

export const metadata: Metadata = { title: "Dashboard" };

export default function Page() {
  return <RagShieldDashboard />;
}
