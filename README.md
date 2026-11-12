# RAG Security — Retrieval Shield

A protected Next.js workspace backed by FastAPI, SQLite, real MiniLM embeddings, and the team's existing prompt-injection scanner/trust-scoring engine. Upload PDF/TXT documents, inspect their security assessments, and retrieve eligible source chunks with page/line references. **No generated answers or LLM calls.**

## Local development

Use Python 3.13 and Node 24. The backend uses one process; do not use multiple workers with this storage design.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==26.3
python -m pip install -r requirements-dev.txt
python embeddings.py
cp .env.example .env
python manage.py hash-password
```

Put the generated hash in `.env` as the **single-quoted** value of `RAG_ACCESS_CODE_HASH`. Choose a strong access code and share it only with approved demo participants. Do not paste it into a frontend environment variable. Existing `.env` files should be edited, not overwritten.

```sh
set -a
source .env
set +a
python manage.py init
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
```

In another terminal:

```sh
cd website
npm ci
npm run dev -- --hostname 127.0.0.1
```

Open http://127.0.0.1:3000 and sign in. The six existing pages remain: Dashboard, Search, Documents, Security Review, Activity, and Evaluation. Uploads return immediately after durable registration; the Documents page polls until processing finishes. Quarantine is a security decision, not a processing error. Search returns eligible passages, with WARN sources still visibly warned.

The frontend defaults to `http://127.0.0.1:8000`. Set `NEXT_PUBLIC_API_BASE_URL` in `website/.env.local` to change it, then restart/rebuild Next.js. Sessions are held in browser memory and expire after one hour; refreshing the entire page requires sign-in again.

The new backend starts with an empty database. It **does not overwrite or automatically ingest existing repository uploads, logs, or evaluation artifacts**. Follow [data migration instructions](docs/DEPLOYMENT.md#existing-data) to import selected records.

## Documentation

- [Architecture and security policy](docs/ARCHITECTURE.md)
- [API contract](docs/API.md)
- [Environment, migration, Vercel/Render setup, backup and rollback](docs/DEPLOYMENT.md)
- [Verification commands and limitations](docs/TESTING.md)

## Tests

```sh
source .venv/bin/activate
python -m unittest discover -p 'test_*.py' -v
python -m unittest discover -s Tracebility_FlaggedResults -p 'test_*.py' -v
RAG_TEST_SEMANTIC=1 python -m unittest test_semantic -v
cd website
npm test
npm run typecheck
npm run lint
npm run build
```

The real-model test requires `python embeddings.py` once. All application tests use temporary storage; no user corpus is used. Existing lexical/demo tests are retained and labeled as such. The ten-document evaluation is a controlled fixture result, **not evidence of general prompt-injection resistance**.

## Legacy tools

`application.py`, `rag_sample_demo.py`, `retrieval_monitor.py`, and the JSON log adapters remain available for historical tests/CLI examples. Production `api:app` uses `PersistentApplication`; there is no production switch to mock embeddings. Legacy `create_app(Application(...))` injection exists for backwards-compatible tests only and is rejected in production mode.

`run_validation.py` and `run_validation_trace.py` write evaluation artifacts. Run them in a temporary copy to protect existing results. The duplicate-content crash in these runners is corrected; duplicate fixtures are reported as errors without corrupting the first document. The historical frontend handoff in `website/docs/sprint-6-frontend.md` is not the current setup guide.
