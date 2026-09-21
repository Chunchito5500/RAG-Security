import { demoScenarios } from "./mock-scenarios";
import { validTrustScore } from "./selectors";
import type { DemoDataset, RetrievalDataSource, RetrievalEvent, RetrievalMode } from "./types";

export const demoModes: { id: RetrievalMode; label: string; description: string }[] = [
  { id: "strict", label: "Strict", description: "Exclude flagged candidates." },
  { id: "balanced", label: "Balanced", description: "Allow flagged candidates scoring at least 0.60." },
  { id: "observe", label: "Observe", description: "Allow flagged candidates; retain warnings." },
];

/** Illustrative fixture decisions only. Real scoring/filtering belongs to the backend. */
export function createDemoEvent(scenarioId: string, mode: RetrievalMode, id: string, occurredAt: string): RetrievalEvent {
  const scenario = demoScenarios.find((candidate) => candidate.id === scenarioId);
  if (!scenario) throw new Error("This query is unavailable. Select another preset.");

  const documents = scenario.documents.map((document) => {
    if (document.status === "trusted") return { ...document };
    const allowed = mode === "observe" || (mode === "balanced" && validTrustScore(document.trustScore) && document.trustScore >= 0.6);
    return {
      ...document,
      status: allowed || document.status === "flagged" ? "flagged" as const : "blocked" as const,
      decision: allowed ? "allowed" as const : "excluded" as const,
    };
  });
  const allowedIds = new Set(documents.filter((document) => document.decision === "allowed").map((document) => document.id));
  const answer = { ...scenario.answer, citationIds: scenario.answer.citationIds.filter((citation) => allowedIds.has(citation)) };
  if (answer.status === "withheld" && allowedIds.size > 0) {
    answer.text = "No answer is available for this unverified evidence.";
    answer.status = "unavailable";
  }

  return { id, occurredAt, query: scenario.query, scenarioLabel: scenario.label, origin: "demo", mode, documents, answer };
}

const delay = () => new Promise<void>((resolve) => setTimeout(resolve, 450));

export function createDemoDataSource(dataset: DemoDataset): RetrievalDataSource {
  return {
    async listEvents() {
      await delay();
      if (dataset === "error") throw new Error("Retrieval history could not be loaded. Retry or choose another record set.");
      if (dataset === "empty") return [];
      const scenarios = dataset === "trusted-only" ? demoScenarios.slice(0, 1) : demoScenarios;
      return scenarios.map((scenario, index) => createDemoEvent(scenario.id, "strict", `event-${index + 1}`, `2026-04-16T${String(15 - index).padStart(2, "0")}:00:00.000Z`));
    },
  };
}

export async function runDemoScenario(scenarioId: string, mode: RetrievalMode): Promise<RetrievalEvent> {
  await delay();
  return createDemoEvent(scenarioId, mode, `event-${crypto.randomUUID()}`, new Date().toISOString());
}
