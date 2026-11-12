# Deployment, data migration and rollback

No hosted service has been created or changed. `render.yaml` is a proposed configuration, not a deployed resource. Owner approval is required before provisioning, deploying, or changing hosted settings.

## Environment

Backend examples: `.env.example` and `.env.production.example`. The root `.env` is not auto-loaded: source it using `set -a; source .env; set +a`, or configure hosting environment values. Keep hashes single-quoted in shell files because they contain dollar signs. Hosting dashboards receive the raw hash without shell quotes.

| Variable | Meaning/default |
| --- | --- |
| `RAG_ENV` | `production` activates strict configuration and disables public schema/docs |
| `RAG_DATA_DIR` | Whole durable state root; local `data`, hosted `/var/data/rag` |
| `RAG_ACCESS_CODE_HASH` | Required PBKDF2 hash from `python manage.py hash-password` |
| `RAG_CORS_ORIGINS` | Explicit comma-separated frontend origins; HTTPS required in production |
| `RAG_MODEL_CACHE` | Model cache; `.model-cache`, populated at build/setup |
| `RAG_MODEL_OFFLINE` | Runtime `1`; `embeddings.py` explicitly enables download during model setup |
| `RAG_MIN_SIMILARITY` | Raw cosine floor, default `0.30`; calibrate using project queries |
| `RAG_MAX_STORAGE_BYTES` | Raw upload quota, default 1 GiB; monitor additional DB/event disk use |
| `RAG_MAX_DOCUMENTS` | Default 1,000 including failed uploads |
| `RAG_MAX_CHUNKS` | Default 20,000 persisted chunks |
| `RAG_MAX_PDF_PAGES` | Default 200 |
| `RAG_PARSE_TIMEOUT_SECONDS` | Default 30 |
| `PORT` | Hosting port; startup script defaults to 8000 locally |
| `NEXT_PUBLIC_API_BASE_URL` | Frontend only: public HTTPS backend base URL, never a secret |

Model identity, revision, token sizes, session lifetime, upload limit and security policy are versioned in code. Changing model/policy requires explicit reindexing, rather than silently mixing incompatible vectors. Legacy JSON-path variables do not control the production SQLite service.

## Existing data

Stop the backend before applying imports. Originals are read/copied, never overwritten. Normal startup never imports repository fixture directories automatically.

```sh
python manage.py import-documents ./uploads
python manage.py import-documents ./uploads --apply
python manage.py import-documents ./documents --apply  # Optional controlled fixture corpus
python manage.py import-history ./retrieval_logs.json
python manage.py import-history ./retrieval_logs.json --apply
python manage.py import-evaluation ./validation_results_updated.json --apply
python manage.py inspect
```

Imports are explicit and repeatable. Legacy files without an original upload registry receive a new registration timestamp and an uploader label stating that original uploader/time is unknown. This does not rewrite the original files or pretend the timestamp is historical. Existing short document IDs in historical events remain historical references; ambiguous identity matches are not guessed. Imported retrieval records retain raw legacy payloads. Standalone legacy flagged-event JSON remains preserved on disk; this release does not import that separate format.

Evaluation import copies the normalized saved results and original evaluation timestamp into SQLite. Reimporting identical content is idempotent. The old fixture result is a legacy-policy snapshot, not a live semantic evaluation. A future evaluation can be imported using the same `results`-array schema.

## Vercel frontend

After approval:

1. Ensure the complete reviewed source and lockfiles are available to the deployment, without committing data, model files, access codes, or environment secrets.
2. Create/select a Vercel Next.js project with root directory `website`.
3. Use Node 24, `npm ci`, and `npm run build`; keep the default Next.js output configuration.
4. Set `NEXT_PUBLIC_API_BASE_URL=https://<backend-host>` before building.
5. Add the resulting exact frontend origin to backend CORS. Prefer a stable production domain. Preview origins must be explicitly approved/configured.
6. Verify the sign-in screen and authenticated document workflow.

The browser uploads directly to the Python service. No Vercel Function proxies document bodies. Vercel filesystem storage is not used for uploads or application state. Changing the public API URL requires a frontend rebuild.

Reference: https://vercel.com/docs/frameworks/full-stack/nextjs

## Python backend on Render

The proposed Blueprint uses a paid service and persistent disk. Its `standard` plan is a conservative starting point for local CPU embeddings; validate memory/cost and obtain owner approval before creating it. Do not assume a 512 MiB instance can load the model and parse PDFs concurrently. No paid service or disk has been provisioned.

Build sequence:

```sh
pip install --upgrade pip==26.3
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python embeddings.py
```

The explicit CPU wheel avoids unnecessary CUDA packages on Linux. Model downloads occur during build; runtime is offline. Dependencies are pinned with transitive constraints. Linux-specific wheel resolution still needs confirmation in the hosted build; local checks ran on macOS.

Startup: `sh scripts/start-backend.sh`. It initializes compatible SQLite schema on the mounted disk, then starts one Uvicorn process. The application also enforces a single-process file lock. Do not enable multiple workers or independent instances against this disk. Existing state lives entirely under `/var/data/rag`; an app redeploy must retain the mount and directory.

Set the credential hash and exact CORS origin through hosting secrets/settings. Health path: `/api/ready`. Readiness checks database integrity/schema/write access and worker availability. `/api/health` is only liveness. Production will not start without explicit auth, storage and CORS configuration or compatible model cache/index.

Render disks are available on paid services; only mounted paths persist. Disk-backed deployments have scaling and zero-downtime limitations. Budget for deploy downtime. See https://render.com/docs/disks and https://render.com/docs/deploy-fastapi.

## Backups, reindex and rollback

```sh
python manage.py backup /path/to/new-backup-directory
python manage.py restore /path/to/backup --data-dir /path/to/NEW-restored-directory
```

Backup uses SQLite's backup API, then copies immutable original files referenced by the snapshot and validates hashes. Copy completed backups off the application disk. Restore refuses an existing target directory. Verify restored data before switching `RAG_DATA_DIR`. Raw files, SQLite and the backup manifest must be kept together. Backup artifacts contain sensitive corpus data and hashed sessions; restrict access.

For an intentional model/policy change, stop the backend and back up first:

```sh
python manage.py reindex
python manage.py reindex --apply
```

Reindex deletes derived chunks/vectors, queues all documents, and preserves originals, UUIDs, timestamps and historical snapshots. It is an offline operation with temporary retrieval downtime. Startup rejects model/policy mismatch until an explicit reindex or rollback. This version does not maintain parallel index versions.

Before release, record code commit, schema version, model revision, policy version and environment settings. Prefer code rollback compatible with the existing schema. Do not run destructive down-migrations automatically. If restoration is required, stop new writes, preserve the current database and post-backup uploads, restore into a new directory, and reconcile later submissions before switching. Never silently discard post-backup work. Roll frontend and backend to compatible contracts.

## Approval and release checklist

- [ ] Review implementation and explicit PDF sanitization-metadata correction.
- [ ] Review complete source changes; no automatic commit/push is performed.
- [ ] Confirm production budget and backend memory sizing.
- [ ] Run clean Linux dependency/model build and review runtime patch level.
- [ ] Configure strong access code, explicit HTTPS CORS and persistent mount.
- [ ] Run upload → processing → query → reference/security/event inspection.
- [ ] Restart backend and confirm identical metadata/results/history.
- [ ] Confirm unauthorized access and expired sessions are rejected.
- [ ] Verify readiness fails when storage is unusable.
- [ ] Verify backup restoration and copy backup off-disk.
- [ ] Confirm fixture evaluation is labeled as historical and limited.
- [ ] Obtain explicit approval before deploying either hosted service.
