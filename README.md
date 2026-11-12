# RAG Security — Retrieval Poisoning Defense

A local Next.js dashboard connected to the team's Python document security pipeline. It loads PDF/TXT files, scans and scores them, excludes quarantined candidates, and displays retrieved passages and audit records. No LLM, external AI service, vector database, or generated answers are used.

## Run locally

Prerequisites: Python 3.11+ and Node.js 22.13+ (22 LTS or 24 LTS recommended), with npm. Run these commands from the repository root.

Terminal 1:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```sh
cd website
npm ci
npm run dev -- --hostname 127.0.0.1
```

Open **http://127.0.0.1:3000** (redirects to `/dashboard`). The API loads `documents/` and previously uploaded files at startup. Upload documents through the dashboard; use any query and select how many candidates to retrieve. Restart the backend after changing repository files manually.

The frontend defaults to `http://127.0.0.1:8000`. To change it, copy `website/.env.example` to `website/.env.local` and set `NEXT_PUBLIC_API_BASE_URL`, then restart/rebuild Next.js. This public variable must not contain secrets. The backend accepts `RAG_CORS_ORIGINS` as a comma-separated origin list, defaulting to `http://localhost:3000,http://127.0.0.1:3000`.

Optional backend environment variables: `RAG_DOCUMENTS_DIR`, `RAG_UPLOADS_DIR`, `RAG_RETRIEVAL_LOG`, `RAG_FLAGGED_LOG`, and `RAG_EVALUATION_FILE`. Defaults are repository-relative `documents/`, `uploads/`, `retrieval_logs.json`, `flagged_events.json`, and `validation_results_updated.json`. Export variables in the backend terminal; a root `.env` is not loaded automatically. Use one backend worker. This is a local application without authentication; bind to loopback.

## Dashboard pages

The shared Retrieval Shield sidebar opens six workspaces:

- `/dashboard`: collection metrics, recent retrieval activity, and a compact security overview.
- `/search`: query submission, eligible passages, compact candidate tables, security decisions, and retrieval trace.
- `/documents`: filename search, status filters, PDF/TXT upload, and individual document details.
- `/security`: flagged documents, quarantined documents, and recorded security events with readable explanations and expandable detection rules.
- `/activity`: retrieval history, flagged-event filtering, and linked event details.
- `/evaluation`: metrics and individual results from the saved test dataset.

Document warnings, quarantine, and context eligibility are displayed separately. Security event records are created during retrieval; the event log can be empty while documents are already quarantined. Historical records use document identifiers when filenames are missing. Each page loads only its relevant APIs; refreshing reads the current data without rerunning evaluation.

## Pipeline and existing work

- `rag_sample_demo.py` supplies `read_document`, `DocumentIngestionPipeline`, `ContentRiskScanner`, `TrustScoringEngine`, and `SafeRetrievalLayer`. Extraction uses UTF-8 for TXT and strict pypdf parsing for PDF. The trust formula, risk patterns, thresholds, warning weighting, and quarantine behavior are unchanged.
- The existing `MockEmbeddingService` actually performs deterministic token-set/Jaccard retrieval over the first 512 characters. It ranks candidates before security filtering; excluded candidates are not backfilled, and zero-similarity candidates can be returned. The dashboard displays actual source excerpts (up to 2,000 characters), not query-specific summaries.
- `retrieval_monitor.py` remains the separate historical prototype with inline example documents. Its log format is supported for review; it is not used to retrieve uploaded files. Existing records are labeled “Historical log”; missing source text, quarantine metadata, and timestamps remain unavailable. Legacy timestamps without a timezone retain their original value.
- `flagged_events.py` builds source-linked records for new retrieval warnings and quarantines. New events append to the existing retrieval log and `flagged_events.json`. The JSON writers use atomic file replacement and reject corrupt files instead of resetting history. `sample_flagged_events.json` and the duplicate `Tracebility_FlaggedResults/` examples remain unchanged and are not presented as live events.
- `application.py` connects these implementations and normalizes records. `api.py` exposes them to the existing dashboard. The live UI no longer imports demo scenarios; those files and their regression tests are retained.
- All files use unknown provenance and no verified signature, including uploads. Filename/category labels do not grant trust. With the existing formula, benign unknown-source files may receive WARN and remain eligible. The UI labels WARN “Flagged”, QUARANTINE “Blocked”, and ALLOW “Trusted”. “Trusted passages” means eligible results, including explicitly warned results.
- Quarantine is logical exclusion, not moving or deleting files. Uploaded files are stored under random subdirectories, never executed or served directly. TXT/PDF uploads have a 10 MB byte and extracted-text limit. Unsupported, empty, invalid UTF-8, malformed, encrypted/unextractable PDFs, and binary TXT input receive errors. OCR, actual signature verification, reviewer approval, and production isolation/resource controls for PDF parsing are not implemented.
- The existing content-derived ID deduplicates identical extracted text in retrieval. All uploaded file entries remain visible; the latest ingested copy supplies retrieval source metadata. Repository files are reprocessed on backend restart.

## API contract

Successful responses use `{ "data": ... }`. Errors use `{ "error": "message" }` with a non-2xx status. Missing optional metadata is `null`; missing collections are empty. `GET /api/documents` also includes an `errors` array for files that could not be processed. OpenAPI is available at `/docs`.

| Method | Endpoint | Data |
| --- | --- | --- |
| GET | `/api/health` | Readiness, loaded document count, processing error count |
| GET | `/api/documents` | Document IDs, filenames, sources, trust scores/breakdowns, decisions, flag reasons, quarantine status |
| POST | `/api/documents/upload?filename=notes.txt` | Raw file body (`application/octet-stream`, or matching `text/plain` / `application/pdf`); returns a processing result with status 201 |
| POST | `/api/query` | JSON `{ "query": "document security", "top_k": 3 }`; query length 1–2000, top_k 1–50 |
| GET | `/api/retrieval-events` | Newest-first normalized retrieval history |
| GET | `/api/flagged-events` | Newest-first saved source-linked warning/quarantine events |
| GET | `/api/evaluation-results` | Saved evaluation summary and individual results, or null when absent |

Query results include `query`, `retrieval_event_id`, `timestamp`, `retrieved_documents`, `trust_scores`, `security_decisions`, `excluded_documents`, `trusted_documents`, and `retrieval_trace`. Each candidate includes the backend inclusion decision, source, score, similarity, status, reason, and quarantine state. Quarantined source text is omitted from retrieved passages. Retrieval snapshots remain traceable after a restart. Each JSON file is written atomically, but the two log files are not a cross-file transaction; a disk failure between writes can leave a retrieval record without its flagged-log entries.

Evaluation reads the saved file without running validation. Accuracy is calculated from records with known `correct` values, on a 0–100 scale; unknown metrics remain null. The committed evaluation contains five benign PDFs and five controlled malicious PDF/TXT documents. These labeled results do not establish correctness for arbitrary uploads.

## Tests

From the repository root:

```sh
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m unittest test_api test_flagged_events -v
python -m unittest discover -s Tracebility_FlaggedResults -p 'test_*.py' -v
cd website
npm test
npm run lint
npm run typecheck
npm run build
```

API tests use temporary uploads and logs and leave the saved evaluation intact. They cover real PDF/TXT scores, malicious-upload exclusion, malformed/oversized uploads, restart persistence, concurrent log writes, legacy normalization, missing data, evaluation, and CORS. Frontend tests cover the retained demo behavior, API adaptation and request sharing, network/error states, and rendered retrieval, document, activity, and evaluation views.

`python run_validation.py` runs the original ten-document evaluation and writes **`validation_results.json`** and appends to **`validation_retrieval_logs.json`**; it does not update the dashboard's saved `validation_results_updated.json`. To avoid writing evaluation artifacts into the repository, run the original validator in a temporary copy:

```sh
python - <<'PY'
import shutil, subprocess, sys, tempfile
from pathlib import Path
root = Path.cwd()
with tempfile.TemporaryDirectory() as directory:
    target = Path(directory)
    for name in ('run_validation.py', 'rag_sample_demo.py', 'json_storage.py'):
        shutil.copy2(root / name, target / name)
    shutil.copytree(root / 'documents', target / 'documents')
    subprocess.run([sys.executable, str(target / 'run_validation.py')], cwd=target, check=True)
PY
```
