# Testing and known limits

Run the commands in the README. Tests use temporary directories, not user uploads or repository JSON logs. `RAG_TEST_SEMANTIC=1` enables the real pinned model test; without it, that test is explicitly skipped, not substituted. Model download is a separate setup operation. Legacy test doubles are injected only in unit tests.

Coverage added:

- SQLite repeated initialization, restart, foreign keys, rollback, consistent backup, future-schema rejection.
- PDF pages, TXT lines, deterministic overlapping chunks and exact substring references.
- UUID duplicate uploads, original timestamps, hashes and vector persistence.
- Whole-document high-severity quarantine, no blocked excerpts, late-document content retrieval.
- Trust-adjusted ordering, no low-relevance filler results, model mismatch failure.
- Durable ingestion failure and interrupted-job recovery.
- Authentication, logout, expiry, CORS Authorization preflight, query validation and readiness failure.
- Evaluation import consistency/idempotency; original fixture scoring remains tested separately.
- Operator import idempotency and restoration into a new directory.
- Regression for duplicate validation fixtures corrupting the first stored assessment.
- Frontend authentication transport, malformed response errors, source references, and blocked-text suppression.

The real-model integration test uploads finance, astronomy and injection-bearing documents into temporary storage. It uses a paraphrased banking query, verifies relevant retrieval and quarantine exclusion, then restarts and checks unchanged metadata/results. It is deliberately a smoke test, not a semantic retrieval benchmark or broad attack evaluation.

Run package checks with `python -m pip_audit` and `npm audit`. The direct Next.js version stays on the existing major. Patched PostCSS and minimatch overrides remove vulnerable nested versions; any future lockfile refresh must rerun lint/build/tests and advisory checks.

Current limits: shared workspace rather than per-user isolation; English-oriented embedding model; no OCR, digital-signature validation, genuine document sanitization, generated answers, document deletion UI, event pagination/retention, or horizontal scaling. Rate limits are single-process and reset on restart. Similarity threshold needs broader evaluation. Parser subprocess bounds do not constitute a complete sandbox. Original legacy JSON files remain intact; standalone old flagged-event files are not imported. A browser-host access outage may prevent automated UI verification; distinguish HTTP/API integration checks from actual browser interaction in release reports.

For legacy evaluation runners, work in a temporary repository copy because they write artifacts. Do not overwrite saved evaluation/trace files merely to run regression checks.

## Local verification recorded during implementation

- Python discovery with the real semantic test enabled: **55 passed**.
- Additional legacy traceability suite: **7 passed**.
- Frontend tests: **21 passed**; TypeScript, lint and production build passed on Node 24 / Next.js 15.5.29.
- `pip check`: passed. Updated `pip-audit` and `npm audit`: **no known vulnerabilities reported** at the time checked.
- Real HTTP smoke: authenticated PDF/TXT upload, asynchronous worker processing, semantic query, quarantine exclusion, page references and persisted events passed.
- Real process restart: all uploaded files, original UUIDs/timestamps/provenance, chunk references, assessments, historical events and semantic results remained unchanged.
- Actual frontend API adapter tested against the live local backend: login, document loading, semantic query and source reference adaptation passed. All six frontend routes returned the access screen over HTTP.
- Browser interaction was not verified: browser automation was unavailable and computer access to Chrome was not approved. Hosted Linux builds, production domain/CORS configuration, hosted restarts and provider disk behavior still require deployment approval and a hosted smoke test.

Reproduce the HTTP smoke only against a disposable test workspace:

```sh
RAG_SMOKE_ACCESS_CODE='<temporary test code>' python scripts/http_smoke.py --url http://127.0.0.1:8107 --output /tmp/rag-smoke-results.json
```

This uploads three fixtures and records events in the selected service. It is not a read-only command; never point it at a sensitive corpus.
