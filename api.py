"""Local HTTP API for the existing retrieval security pipeline."""

import os
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


def create_app(service=None):
    service = service or Application(**{key: os.environ[env] for key, env in (
        ("documents_dir", "RAG_DOCUMENTS_DIR"), ("uploads_dir", "RAG_UPLOADS_DIR"),
        ("log_file", "RAG_RETRIEVAL_LOG"), ("flagged_file", "RAG_FLAGGED_LOG"),
        ("evaluation_file", "RAG_EVALUATION_FILE")) if os.environ.get(env)})

    @asynccontextmanager
    async def lifespan(app):
        await run_in_threadpool(service.load)
        yield

    app = FastAPI(title="RAG Security API", lifespan=lifespan)
    app.add_middleware(CORSMiddleware,
                       allow_origins=os.getenv("RAG_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        return JSONResponse({"error": str(error.detail)}, status_code=error.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        return JSONResponse({"error": "; ".join(item["msg"] for item in error.errors())}, status_code=422)

    @app.exception_handler(Exception)
    async def internal_error(request, error):
        return JSONResponse({"error": "Unable to process the request or access application storage. Check the backend log."}, status_code=500)

    @app.get("/api/health")
    def health():
        return {"data": {"status": "ok", "documents_loaded": len(service.documents), "processing_errors": len(service.errors)}}

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
            if len(raw) + len(chunk) > service.ingestion.MAX_FILE_SIZE_BYTES:
                raise HTTPException(413, "Document exceeds 10 MB")
            raw.extend(chunk)
        if not raw:
            raise HTTPException(400, "Document is empty")

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
