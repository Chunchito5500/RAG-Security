# Architecture

## Production flow

Vercel browser client → authenticated FastAPI → SQLite repository and immutable uploaded files on one persistent disk. A single ingestion thread consumes a durable SQLite jobs table. PDF/TXT extraction runs in a separate bounded subprocess. Chunk vectors are computed locally with MiniLM, persisted as little-endian float32, and searched using NumPy cosine similarity. No vector server, Redis, external AI service, or generated-answer component is required.

The existing scanner and trust-score implementation in `rag_sample_demo.py` are reused. The legacy `Application` and lexical retrieval remain testable, but `api:app` defaults exclusively to `PersistentApplication`.

## State and transactions

`storage.py` initializes version 1 using SQLite `user_version` and `schema_migrations`. Future unsupported schemas fail startup. Foreign keys and FULL synchronous writes are enabled. The default rollback journal is intentional for this one-process service.

Tables: documents (metadata and assessment JSON), chunks, embeddings, jobs, retrieval_events, security_events, evaluations, sessions, imports, schema_migrations. Assessments are stored transactionally on their document row rather than in an independent table. Retrieval events contain immutable result/decision snapshots rather than joining historical views to mutable current assessments.

Each upload has an independent UUID, full raw SHA-256, full extracted-text SHA-256, original filename, server-recorded upload time, and workspace uploader label. Duplicate content does not share identity. A repeat HTTP submission is a new upload; there is no idempotency-key API yet. Operator imports use an atomic import receipt to avoid importing the same source path/content twice.

Raw files are flushed before their document/job transaction is committed. A crash in the intervening filesystem window can leave an orphan; `manage.py inspect` reports it without deleting it. Related chunk/vector/assessment/job/security writes share one transaction. Queries persist their retrieval snapshot and associated security events together before returning success.

Normal restart reads stored metadata and vectors; it does not reinterpret upload timestamps, rescore completed documents, or regenerate their embeddings. Interrupted processing jobs are retried (maximum three interrupted attempts), while ordinary parsing failures become durable failed records. An explicit offline reindex preserves UUIDs, raw files, upload times, and historical events.

## Parsing and chunks

UTF-8 TXT preserves line references. PDF extraction preserves one-based page references; encrypted PDFs and pages with no usable text are unsupported. No OCR is performed. Raw and extracted text are limited to 10 MiB; PDFs default to 200 pages. The parser subprocess has a wall timeout, and on Linux has a 768 MiB address-space bound and 25-second CPU bound. This is resource containment, not a full OS security sandbox.

Chunks use 192 tokenizer tokens with 32-token overlap. IDs are deterministic UUID5 values under the upload UUID and chunk-policy version. Character offsets are page-local for PDFs and file-local for TXT. Exact source substrings are retained. Chunks never span PDF pages.

`embeddings.py` pins all-MiniLM-L6-v2 to revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. CPU inference uses safetensors and disables remote code. Cache the model during setup/build; runtime defaults to offline. Missing model files fail startup, with no mock fallback.

## Security decisions

Formula remains `0.35P + 0.25I + 0.20V + 0.20(1-R)`; ALLOW ≥ 0.80, WARN ≥ 0.50, otherwise quarantine. High-severity regex findings and integrity failure still force quarantine. The whole document is scanned, including text crossing chunk boundaries. Chunk scans can additionally force quarantine but cannot relax a document decision.

**Versioned correction:** `v2-formula-unsanitized-v1` records `sanitization_applied=false` for PDFs as well as TXT. Parsing was previously incorrectly treated as sanitization. With otherwise benign unknown-source content, both receive 0.6925 rather than the old PDF score 0.7525. The formula/thresholds did not change. Legacy scoring fixtures retain their old behavior for reproducibility; new regression tests verify the corrected input and score.

Provenance is recorded, not cryptographically authenticated: source type unknown, signature absent, uploader the protected workspace. Login does not imply a trusted publisher. Hashes establish consistency with the ingested content, not publisher authenticity.

Quarantine excludes every chunk from returned context. Similarity can still be calculated for quarantined chunks to explain relevant exclusions, but their text is omitted from query responses. Eligibility is applied before final top-k. WARN scores are multiplied by `(0.5 + 0.5 * trust)` and the adjusted score determines ordering. A default raw cosine floor of 0.30 avoids filling results with low-relevance passages. This threshold is provisional and must be calibrated against a broader project dataset.

Documents are never executed, passed to a shell, or interpreted as application instructions. React renders text as escaped text, not trusted HTML. No original-file download endpoint exists. Regex coverage remains limited; quarantining known fixtures does not prove broad injection resistance.

## Authentication and deployment boundaries

One shared capstone corpus, one-hour random bearer sessions, PBKDF2-hashed access code, hashed tokens persisted in SQLite. Frontend holds issued tokens only in memory. All corpus/API/debug routes require a session except minimal health/readiness and the login endpoint. Production disables OpenAPI/Swagger. This is not per-user/private document isolation.

CORS accepts explicit origins; production requires HTTPS origins. Global single-workspace limits: ten login attempts, twenty uploads, sixty queries per minute. Limits reset on process restart and are not a distributed abuse-prevention system. Corpus bounds default to 1 GiB raw uploads, 1,000 documents, and 20,000 chunks. Failed uploads continue to count against storage. Event retention/pagination and administrative deletion are deferred; monitor disk usage and back up histories.

A process lock enforces one production backend per data directory. SQLite is the authority; no in-memory index is necessary to recover state. Larger corpora, multiple tenants, or horizontal scaling require revisiting database/object storage and vector indexing.
