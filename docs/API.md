# API contract

Base: configured HTTPS backend URL. Responses use `{ "data": ... }`; errors use `{ "error": "readable message" }`, with a `code` on authentication/rate-limit responses. No tracebacks or storage paths are returned.

Authenticate with `POST /api/session`, JSON `{ "access_code": "..." }`. Response: `{ "data": { "token": "...", "expires_at": 1234567890 } }` (Unix seconds). Send `Authorization: Bearer <token>` on protected requests. `DELETE /api/session` revokes that token. CORS allows Authorization and Content-Type; cookies are not used.

| Method | Route | Behavior |
| --- | --- | --- |
| GET | `/api/health` | Public minimal liveness |
| GET | `/api/ready` | Public readiness; 503 for inaccessible/corrupt/incompatible database or unavailable worker |
| GET | `/api/documents` | All document summaries, plus failed-processing `errors` |
| GET | `/api/documents/{uuid}` | Summary, provenance, hashes, chunk IDs/references |
| POST | `/api/documents/upload?filename=...` | Raw PDF/TXT bytes; returns 202 once file/job are durable |
| POST | `/api/query` | JSON `{ "query": "...", "top_k": 5 }` |
| GET | `/api/retrieval-events` | Persisted query snapshots, newest first |
| GET | `/api/flagged-events` | Persisted ingestion/retrieval warnings and quarantines |
| GET | `/api/evaluation-results` | Latest explicitly imported evaluation, or null |

## Upload lifecycle

Use `application/octet-stream`, `text/plain` for TXT, or `application/pdf` for PDF. Filename must be a basename, at most 200 UTF-8 bytes. Limit: 10 MiB. Validation errors: 400 for empty/unsafe filename, 413 for size/quota, 415 for unsupported/mismatched media type. Parsing happens asynchronously: a malformed but transport-valid upload returns 202 and subsequently becomes `failed`.

Poll document detail or library. `processing_status` is `queued`, `processing`, `processed`, or `failed`; `processing_error` explains failure. This is separate from `decision` (`allow`, `warn`, `quarantine`, or `unknown` before assessment). A quarantined document is successfully processed, not failed.

Document summaries retain existing UI fields: `document_id`, `filename`, `source`, `trust_score`, `security_status`, `decision`, `flag_reason`, `quarantine_status`, `status`. Additional fields include `uploaded_at`, `metadata`, `raw_sha256`, `text_sha256`, `trust_score_breakdown`, `policy_version`, `embedding_model`, and processing fields.

## Semantic query

Query is trimmed, 1–2,000 characters; top_k is a strict integer from 1–50; extra request fields are rejected. top_k counts eligible chunks, not excluded candidates. No qualifying result is a valid empty result.

Response retains `retrieval_event_id`, `query`, `timestamp`, `origin`, `retrieved_documents`, `trusted_documents`, `excluded_documents`, `trust_scores`, `security_decisions`, and `retrieval_trace`. It adds `pipeline: "semantic"`, model/policy version, and `answer: null`.

Each result has document UUID, distinct `result_id`/`chunk_id`, filename, source, `excerpt`, trust, raw/effective similarity, decision, inclusion (`allowed`/`excluded`), flag reason, and reference:

```json
{"page": 3, "chunk_index": 2, "start_char": 610, "end_char": 1420,
 "line_start": null, "line_end": null, "chunk_version": "tokens-192-overlap-32-v1"}
```

TXT references use null page and one-based line_start/line_end. `trusted_documents` means policy-eligible source passages, including visibly warned passages—not guaranteed-safe or generated text. Excluded entries have null excerpts and are deduplicated by document. Lists currently return complete collections; pagination is not implemented.

## Events and evaluation

Security events identify `event_type` (ingestion/retrieval), document UUID, timestamp, trust, reason, decision, high-severity flag, and optional retrieval request ID. Ingestion flags are available before the first query.

Evaluation is an explicitly imported, timestamped historical snapshot. It does not rerun tests or imply scores were produced by the currently active policy. The original ten-document fixture uses the legacy policy. See source/timestamp and the testing documentation when comparing results.

401 = authentication missing/expired; 422 = invalid query; 429 = shared-workspace rate limit; 500 = internal/storage failure; 503 = readiness failure. Authentication failures have no-store responses. Browser code handles expiry by returning to sign-in.
