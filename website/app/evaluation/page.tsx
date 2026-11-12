import type { Metadata } from "next";
import { EvaluationWorkspace } from "@/components/evaluation-workspace";

export const metadata: Metadata = { title: "Evaluation" };

export default function Page() {
  return <EvaluationWorkspace />;
}
