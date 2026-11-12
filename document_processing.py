"""Bounded extraction and deterministic, source-aware token chunks."""
import json
import os
import subprocess
import sys
import uuid
from io import BytesIO
from pathlib import Path

CHUNK_VERSION = "tokens-192-overlap-32-v1"


def extract(path, max_bytes=10 * 1024 * 1024, max_pages=200):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > max_bytes:
        raise ValueError("Document is a symlink or exceeds the size limit")
    raw = path.read_bytes()
    if not raw:
        raise ValueError("Document is empty")
    if path.suffix.lower() == ".txt":
        text = raw.decode("utf-8")
        if any(ord(c) < 32 and c not in "\n\r\t\f" for c in text):
            raise ValueError("Unsupported binary/control characters")
        segments = [{"text": text, "page": None}]
    elif path.suffix.lower() == ".pdf":
        if not raw.startswith(b"%PDF-"):
            raise ValueError("Invalid PDF header")
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(raw), strict=True)
        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are unsupported")
        if len(reader.pages) > max_pages:
            raise ValueError("PDF exceeds the page limit")
        segments, size = [], 0
        for index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            size += len(text.encode("utf-8"))
            if size > max_bytes:
                raise ValueError("Extracted text exceeds the size limit")
            segments.append({"text": text, "page": index + 1})
    else:
        raise ValueError("Only PDF and UTF-8 TXT are supported")
    if not any(s["text"].strip() for s in segments):
        raise ValueError("No extractable text; scanned PDFs require OCR, which is not supported")
    return segments


def extract_isolated(path, timeout=30, max_pages=200):
    # No shell, no evaluation of document content. Subprocess has CPU/memory bounds on Linux.
    try:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(path), str(max_pages)],
                                capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Document parsing timed out") from exc
    if result.returncode:
        raise ValueError("Document parsing failed or exceeded resource limits")
    payload = json.loads(result.stdout)
    if "error" in payload:
        raise ValueError(payload["error"])
    return payload["segments"]


def chunk_segments(document_id, segments, embedder, size=192, overlap=32):
    if not 0 <= overlap < size:
        raise ValueError("Invalid chunk overlap")
    chunks = []
    for segment in segments:
        text = segment["text"]
        offsets = embedder.offsets(text)
        for start in range(0, len(offsets), size - overlap):
            end = min(start + size, len(offsets))
            left, right = offsets[start][0], offsets[end - 1][1]
            passage = text[left:right]
            if passage.strip():
                ordinal = len(chunks)
                reference = {"page": segment["page"], "chunk_index": ordinal + 1,
                             "start_char": left, "end_char": right,
                             "line_start": text.count("\n", 0, left) + 1 if segment["page"] is None else None,
                             "line_end": text.count("\n", 0, right) + 1 if segment["page"] is None else None,
                             "chunk_version": CHUNK_VERSION}
                chunks.append({"id": str(uuid.uuid5(uuid.UUID(document_id), f"{CHUNK_VERSION}:{ordinal}")),
                               "ordinal": ordinal, "text": passage, "reference": reference})
            if end == len(offsets):
                break
    if not chunks:
        raise ValueError("Document contains no indexable tokens")
    return chunks


if __name__ == "__main__":
    if sys.platform.startswith("linux"):
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (768 * 1024**2, 768 * 1024**2))
        resource.setrlimit(resource.RLIMIT_CPU, (25, 25))
    try:
        print(json.dumps({"segments": extract(sys.argv[1], max_pages=int(sys.argv[2]))}))
    except Exception as error:
        # Return curated parser errors; never expose a traceback or arbitrary exception text.
        message = str(error) if isinstance(error, ValueError) else "Unable to extract PDF/UTF-8 text"
        print(json.dumps({"error": message}))
