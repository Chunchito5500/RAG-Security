# Sprint 6 frontend audit and handoff

## Before implementation

Inspected every existing source/configuration file: `app/page.tsx`, `app/layout.tsx`, `app/globals.css`, `components/rag-shield-dashboard.tsx`, package manifests, TypeScript, Next.js, ESLint, Tailwind, PostCSS, Next type declarations, and ignore rules. No project or ancestor `AGENTS.md` instructions were present. The installed framework is Next.js 15.5.14 with React 19.2.4. Baseline lint passed.

| Area | Existing behavior verified from source |
| --- | --- |
| Pages/components | One route (`/`), one client dashboard, and a local `Panel` helper. Query, Answer, Trusted documents, Blocked documents, Ingestion queue, and Why this answer panels. |
| Navigation | Dashboard, Docs, Queue were all `href="#"` placeholders. |
| Upload/ingestion | No file input, upload handler, network call, or ingestion implementation. Four hardcoded queue examples only. |
| Query | Three scenario buttons updated local selection and question text. Run matched keywords to a scenario, falling back to the selected scenario, and updated a timestamp. No query submission or generated answer. Arbitrary questions could display unrelated canned answers. |
| Modes | Strict/Balanced/Observe updated state and the last-run message only. They did not change documents, filtering, or the answer. |
| Trusted documents | Hardcoded title, source/chunk text, excerpt, and numeric trust score on a 0–100 scale. The internal verified/reviewed status was not displayed. |
| Blocked documents | Hardcoded filename, risk severity, and reason. No trust score or source metadata on blocked entries. |
| IDs | Local numeric array IDs existed but were not displayed or used to link review records to source documents. |
| Events/review | No retrieval history, selected event, flagged-event filter, or review view. |
| States | Populated fixtures rendered immediately. No loading, empty, missing-metadata, or error handling. |
| Backend | No API routes, fetch calls, external data service, standalone data types, database, or backend integration. Answers and traces were entirely scripted; claims about signatures, vector retrieval, embeddings, and generation were illustrative text. |

## Implemented frontend

The existing single-page layout, neutral panels, query area, original policy/incident/vendor examples, and ingestion examples remain. Main navigation now targets real sections. No new dependencies were added.

- Event history with query, event ID/time, document counts, flagged counts, and selected-event details.
- Flagged-only review includes both flagged and blocked results. A review item shows its document/source IDs, 0–1 score, reason, and explicit allowed/excluded decision; its link targets the exact document card in the selected event.
- All retrieved documents remain inspectable, including excluded ones. Status, context eligibility, and answer citation membership are separate concepts.
- The workflow explicitly follows query → retrieved documents → trust evaluation → flag/filter decision → final answer. Answer citations link back to document cards.
- Six prepared scenarios: normal trusted retrieval, mixed retrieval, blocked vendor sources, all-low-trust retrieval, incomplete metadata, and no matching documents.
- Strict excludes flagged candidates; Balanced allows fixture candidates scoring at least 0.60; Observe allows flagged candidates while preserving warnings. These are **demo rules only**, not a production policy or threat detector. Scores and reasons are pre-authored, not calculated.
- Prepared answers remain scenario-specific. With no permitted evidence, the all-low-trust answer is withheld. Allowing unverified evidence does not invent an answer. Unsupported custom questions receive a validation message instead of an unrelated canned response.
- Empty history, no flags, no documents, missing titles/IDs/source/excerpts/scores/reasons, loading, load failure, retry, query validation, and demo-run failure states. Invalid numeric scores display “Not available”; zero remains `0.00`.
- A dataset selector demonstrates populated history, trusted-only history, empty history, and a repeatable simulated load error. Switch away from the error dataset to recover. Requests are guarded against stale completions after reload/dataset changes/unmount.
- Keyboard focus indicators, pressed-state controls, labels, status/alert announcements, responsive columns, and reduced-motion support.

## Data boundary and backend handoff

- `lib/retrieval/types.ts`: retrieval event, retrieved document/result identity, source metadata, normalized nullable trust score, nullable flag reason, security status, inclusion decision, mode, answer, and `RetrievalDataSource` read interface.
- `lib/retrieval/mock-scenarios.ts`: prepared evidence/answer fixtures and the preserved static queue.
- `lib/retrieval/demo-source.ts`: asynchronous demo data source and explicitly demo-only event creation/mode behavior.
- `lib/retrieval/selectors.ts`: display and review helpers; review includes flagged and blocked documents regardless of inclusion.
- `components/retrieval-event-detail.tsx`: renders event data without importing demo fixtures or calculating security decisions.
- `components/rag-shield-dashboard.tsx`: page orchestration, local history, controls, asynchronous states; the data-source construction is the replacement point.

Colin/Faizan's backend integration will need to supply event records, retrieved source/document metadata, normalized scores, status/reasons, actual allow/exclude decisions, and final answers/citation references. Implement `RetrievalDataSource.listEvents()` with an adapter to the agreed backend interface; no endpoint contract is presumed here. Validate backend payloads at that boundary (finite scores within 0–1, nullable missing metadata, stable unique IDs, valid timestamps, known status/decision values, and citations referencing allowed results). `RetrievedDocument.id` identifies a retrieved item/chunk; `documentId` and `source.id` identify the underlying document and source. Use `unknown`/null for unavailable decisions and metadata, never infer trust from missing data.

Replace the separate demo-run action when a real query submission interface is agreed. Actual trust scoring, detection, filtering, answer generation, persistence, ingestion, and reviewer disposition belong to backend work. The UI must render backend decisions rather than applying `createDemoEvent` rules to backend data. Configure event origin as `backend` and replace the local query/record-set controls when a real integration exists. No authentication, database, vector store, embeddings, LLM, upload pipeline, or reviewer approval action was implemented.

## Demo walkthrough

1. Run Normal retrieval in Strict mode and inspect the two trusted documents and linked answer citations.
2. Run Mixed retrieval, open Flagged events, and follow a flagged source link to its full metadata and exclusion decision.
3. Run Mixed retrieval in Balanced/Observe to show that flagging and inclusion are different decisions.
4. Select All documents low trust in Strict mode to show exclusions and a withheld answer.
5. Inspect Incomplete metadata and No matching documents for fallback states.
6. Choose Trusted only and open Flagged events for the no-flags state; choose No retrieval records for empty history; choose Loading failure for failure/retry.

History is in-memory only. Refreshing or changing datasets resets it. There is no upload control; the queue still uses the static fixture records.

## Validation

- `npm run lint`: includes app, components, and the new data layer.
- `npm run typecheck`: strict TypeScript validation.
- `npm test`: seven regression tests cover fixture semantics, flagged review membership, all-low-trust withholding, mode isolation, missing/invalid scores, citation integrity, empty/error/recovery datasets, and unique new events. Uses the existing TypeScript compiler and Node test runner; temporary compilation is cleaned up.
- `npm run build`: production build.
- Production HTTP smoke check validates the rendered page shell and initial loading state.
- Programmatic event-detail rendering passed for all 18 scenario/mode combinations, including source/citation anchor targets and missing-metadata/no-document/withheld-answer text.
- Browser interaction/visual verification could not be completed because computer access to Chrome was denied. This is a verification limit; automated data/markup checks do not establish browser layout or interactive behavior.

## UI copy cleanup

Removed the demonstration banner and implementation disclaimers from the interface. Query presets, retrieval mode, record sets, event details, answers, citations, and the ingestion queue use concise product labels. The retrieval trace is Query → Retrieved documents → Trust decision → Final answer. Security warnings, missing-data states, and unsupported-query validation remain. Mock data, source architecture, and behavior are unchanged; event IDs now use the neutral `event-` prefix. These documentation notes retain the backend integration details.
