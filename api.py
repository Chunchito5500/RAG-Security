"""Local HTTP API for the existing retrieval security pipeline."""

import os
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from application import Application
from persistent_application import PersistentApplication
from auth import WorkspaceAuth


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=50, strict=True)

    @field_validator("query")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("Enter a nonempty query")
        return value.strip()


def create_app(service=None, access_code_hash=None):
    service = service or PersistentApplication()
    persistent = getattr(service, "persistent", False)
    production = os.getenv("RAG_ENV") == "production"
    origins = [origin.strip() for origin in os.getenv("RAG_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if origin.strip()]
    auth = None
    rate_windows = {}

    @asynccontextmanager
    async def lifespan(app):
        nonlocal auth
        if production and (not persistent or not os.getenv("RAG_DATA_DIR") or not os.getenv("RAG_CORS_ORIGINS")):
            raise RuntimeError("Production requires persistent data and explicit CORS origins")
        if production and any(not origin.startswith("https://") or "*" in origin for origin in origins):
            raise RuntimeError("Production CORS must use explicit HTTPS origins")
        if persistent:
            auth = WorkspaceAuth(service.store, access_code_hash or os.getenv("RAG_ACCESS_CODE_HASH", ""))
        try:
            await run_in_threadpool(service.load)
            yield
        finally:
            if persistent:
                await run_in_threadpool(service.close)

    app = FastAPI(title="RAG Security API", lifespan=lifespan,
                  docs_url=None if production else "/docs", redoc_url=None,
                  openapi_url=None if production else "/openapi.json")
    app.add_middleware(CORSMiddleware,
                       allow_origins=origins,
                       allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type", "Authorization"])

    @app.middleware("http")
    async def protect(request, call_next):
        path = request.url.path
        if persistent and request.method != "OPTIONS":
            public = path in ("/api/health", "/api/ready", "/api/session")
            token = request.headers.get("authorization", "").removeprefix("Bearer ")
            if not public and (auth is None or not await run_in_threadpool(auth.valid, token)):
                response = JSONResponse({"error": "Workspace authentication required", "code": "unauthorized"}, status_code=401)
            else:
                # Single-instance shared workspace limits; never trust user-supplied forwarded IPs.
                group = "login" if path == "/api/session" else "upload" if path.endswith("/upload") else "query" if path == "/api/query" else None
                limit = {"login": 10, "upload": 20, "query": 60}.get(group, 0)
                window = int(time.monotonic() // 60)
                key = (group, window)
                for old in list(rate_windows):
                    if old[1] != window:
                        del rate_windows[old]
                if group and request.method == "POST" and rate_windows.get(key, 0) >= limit:
                    response = JSONResponse({"error": "Workspace rate limit reached; retry in a minute", "code": "rate_limit"}, status_code=429)
                else:
                    if group and request.method == "POST":
                        rate_windows[key] = rate_windows.get(key, 0) + 1
                    response = await call_next(request)
            # Auth responses must also be readable by the explicitly allowed frontend.
            origin = request.headers.get("origin")
            if origin in origins:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Vary"] = "Origin"
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Content-Type-Options"] = "nosniff"
            return response
        return await call_next(request)

    @app.post("/api/session")
    async def login(request: Request):
        if auth is None:
            raise HTTPException(503, "Authentication is not configured")
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 2048:
                raise HTTPException(413, "Login request too large")
        import json
        try:
            body = json.loads(raw)
            code = body["access_code"]
            if not isinstance(code, str) or not 1 <= len(code) <= 256:
                raise ValueError()
        except (ValueError, KeyError, TypeError):
            raise HTTPException(422, "Enter a valid access code")
        session = await run_in_threadpool(auth.login, code)
        if session is None:
            raise HTTPException(401, "Invalid access code")
        return {"data": session}

    @app.delete("/api/session")
    def logout(request: Request):
        if auth:
            auth.logout(request.headers.get("authorization", "").removeprefix("Bearer "))
        return {"data": {"signed_out": True}}

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        return JSONResponse({"error": str(error.detail)}, status_code=error.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        return JSONResponse({"error": "; ".join(item["msg"] for item in error.errors())}, status_code=422)

    @app.exception_handler(Exception)
    async def internal_error(request, error):
        logging.error("Request failed: %s", request.url.path, exc_info=error)
        return JSONResponse({"error": "Unable to process the request or access application storage. Check the backend log."}, status_code=500)

    @app.get("/api/health")
    def health():
        if persistent:
            return {"data": {"status": "ok"}}
        return {"data": {"status": "ok", "documents_loaded": len(service.documents), "processing_errors": len(service.errors)}}

    @app.get("/api/ready")
    def ready():
        try:
            if persistent:
                service.ready()
        except Exception:
            raise HTTPException(503, "Service is not ready")
        return {"data": {"status": "ready"}}

    @app.get("/api/documents/{identity}")
    def document(identity: str):
        result = service.document(identity) if persistent else next((d for d in service.documents if d["document_id"] == identity), None)
        if result is None:
            raise HTTPException(404, "Document not found")
        return {"data": result}

    @app.get("/api/documents")
    def documents():
        with service.lock:
            return {"data": list(service.documents), "errors": list(service.errors)}

    @app.post("/api/documents/upload", status_code=201)
    async def upload(request: Request, filename: str):
        if not filename or len(filename.encode("utf-8")) > 200 or filename in (".", "..") or any(
                char in filename for char in ("/", "\\")) or any(ord(char) < 32 for char in filename):
            raise HTTPException(400, "Use a filename without paths or control characters")
        suffix = Path(filename).suffix.lower()
        if suffix not in (".pdf", ".txt"):
            raise HTTPException(415, "Only PDF and UTF-8 TXT documents are supported")
        content_type = request.headers.get("content-type", "").split(";")[0]
        if content_type not in ("application/octet-stream", "application/pdf" if suffix == ".pdf" else "text/plain"):
            raise HTTPException(415, "Content-Type does not match the document type")
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw) + len(chunk) > (service.max_file_bytes if persistent else service.ingestion.MAX_FILE_SIZE_BYTES):
                raise HTTPException(413, "Document exceeds 10 MB")
            raw.extend(chunk)
        if not raw:
            raise HTTPException(400, "Document is empty")

        if persistent:
            try:
                result = await run_in_threadpool(service.upload, filename, raw)
            except ValueError as error:
                raise HTTPException(413, str(error))
            return JSONResponse({"data": result}, status_code=202)

        def process_upload():
            with service.lock:
                directory = service.uploads_dir / str(uuid.uuid4())
                directory.mkdir(parents=True)
                path = directory / filename
                try:
                    path.write_bytes(raw)
                    return service.process(path, f"uploads/{directory.name}/{filename}")
                except Exception as error:
                    path.unlink(missing_ok=True)
                    directory.rmdir()
                    raise HTTPException(422, f"Document could not be processed: {error}") from error
        return {"data": await run_in_threadpool(process_upload)}

    @app.post("/api/query")
    def query(body: QueryRequest):
        return {"data": service.query(body.query, body.top_k)}

    @app.get("/api/retrieval-events")
    def events():
        return {"data": service.events()}

    @app.get("/api/flagged-events")
    def flagged():
        return {"data": service.flagged_events()}

    @app.get("/api/evaluation-results")
    def evaluation():
        return {"data": service.evaluation()}

    return app


app = create_app()
