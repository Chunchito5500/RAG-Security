"""Production service: durable ingestion and semantic retrieval, reusing security policy."""
import hashlib
import json
import logging
import os
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from application import Application, STATUS
from document_processing import chunk_segments, extract_isolated
from embeddings import SemanticEmbedder
from rag_sample_demo import ContentRiskScanner, DocumentMetadata, TrustScoringEngine
from storage import Storage

POLICY = "v2-formula-unsanitized-v1"


def now():
    return datetime.now(timezone.utc).isoformat()


class PersistentApplication:
    persistent = True
    max_file_bytes = 10 * 1024 * 1024

    def __init__(self, data_dir=None, embedder=None, start_worker=True, min_similarity=None):
        self.root = Path(data_dir or os.getenv("RAG_DATA_DIR", "data")).resolve()
        self.uploads_dir = self.root / "uploads"
        self.store = Storage(self.root / "app.sqlite3")
        self.embedder = embedder or SemanticEmbedder()
        self.scanner, self.scorer = ContentRiskScanner(), TrustScoringEngine()
        self.lock = threading.RLock()
        self.model_lock = threading.RLock()
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.worker = None
        self.process_lock = None
        self.start_worker = start_worker
        self.min_similarity = float(os.getenv("RAG_MIN_SIMILARITY", ".30")) if min_similarity is None else min_similarity
        self.max_storage = int(os.getenv("RAG_MAX_STORAGE_BYTES", str(1024**3)))
        self.max_chunks = int(os.getenv("RAG_MAX_CHUNKS", "20000"))
        self.max_documents = int(os.getenv("RAG_MAX_DOCUMENTS", "1000"))
        if not 0 < self.min_similarity <= 1 or min(self.max_storage, self.max_chunks, self.max_documents) <= 0:
            raise ValueError("Invalid retrieval or storage limits")

    def load(self):
        if self.start_worker:
            import fcntl
            self.root.mkdir(parents=True, exist_ok=True)
            self.process_lock = (self.root / ".service.lock").open("a")
            try:
                fcntl.flock(self.process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.process_lock.close()
                raise RuntimeError("Only one backend process may use this data directory")
        self.store.initialize()
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.embedder.load()  # Failure is explicit; never fall back to lexical retrieval.
        with self.store.connect() as db:
            incompatible = db.execute("SELECT COUNT(*) FROM documents WHERE status='processed' AND (model!=? OR policy!=?)",
                                      (self.embedder.identity, POLICY)).fetchone()[0]
            if incompatible:
                raise RuntimeError("Stored index model/policy differs; explicit reindex required")
            db.execute("UPDATE jobs SET status='queued' WHERE status='processing'")
            db.execute("UPDATE documents SET status='queued' WHERE status='processing'")
        self.stop.clear()
        if self.start_worker:
            self.worker = threading.Thread(target=self._work, name="rag-ingestion", daemon=True)
            self.worker.start()

    def close(self):
        self.stop.set()
        self.wake.set()
        if self.worker:
            self.worker.join(timeout=60)
        if self.process_lock and (not self.worker or not self.worker.is_alive()):
            self.process_lock.close()

    def ready(self):
        self.store.ready()
        if self.start_worker and (not self.worker or not self.worker.is_alive()):
            raise RuntimeError("Ingestion worker unavailable")
        if not os.access(self.uploads_dir, os.W_OK):
            raise RuntimeError("Upload directory is not writable")

    @property
    def errors(self):
        return [{"filename": d["filename"], "source": d["source"], "error": d["processing_error"]}
                for d in self.documents if d["processing_status"] == "failed"]

    @property
    def documents(self):
        return [self.document_result(row) for row in self.store.rows("SELECT * FROM documents ORDER BY uploaded_at, id")]

    def document_result(self, row):
        assessment = json.loads(row["assessment"]) if row["assessment"] else {}
        decision = assessment.get("decision", "unknown")
        return {"document_id": row["id"], "filename": row["filename"], "source": f"uploads/{row['id']}/{row['filename']}",
                "uploaded_at": row["uploaded_at"], "processing_status": row["status"], "status": row["status"],
                "processing_error": row["error"], "trust_score": assessment.get("final_score"),
                "decision": decision, "security_status": STATUS.get(decision, "unknown"),
                "quarantine_status": "flagged" if decision == "quarantine" else "none" if assessment else None,
                "flag_reason": "; ".join(assessment.get("flags", [])) or ("Source authenticity is not verified" if assessment else None),
                "trust_score_breakdown": assessment or None,
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {
                    "source_type": "unknown", "uploader_id": row["uploader"], "uploaded_at": row["uploaded_at"]},
                "raw_sha256": row["raw_hash"], "text_sha256": row["text_hash"], "policy_version": row["policy"],
                "embedding_model": row["model"]}

    def document(self, identity):
        rows = self.store.rows("SELECT * FROM documents WHERE id=?", (identity,))
        if not rows:
            return None
        result = self.document_result(rows[0])
        result["chunks"] = [{"chunk_id": row["id"], "reference": json.loads(row["reference"])}
                            for row in self.store.rows("SELECT id, reference FROM chunks WHERE document_id=? ORDER BY ordinal", (identity,))]
        return result

    def upload(self, filename, raw, uploader="capstone-workspace", import_fingerprint=None):
        with self.lock:
            # Count orphan files too: an interrupted registration must not evade quotas.
            used = sum(p.stat().st_size for p in self.uploads_dir.rglob("*") if p.is_file())
            if used + len(raw) > self.max_storage or len(self.documents) >= self.max_documents:
                raise ValueError("Workspace storage quota exceeded")
            identity, timestamp = str(uuid.uuid4()), now()
            folder = self.uploads_dir / identity
            folder.mkdir()
            path = folder / ("original" + Path(filename).suffix.lower())
            try:
                with path.open("xb") as file:
                    file.write(raw)
                    file.flush()
                    os.fsync(file.fileno())
                with self.store.connect() as db:
                    db.execute("INSERT INTO documents(id,filename,path,raw_hash,size,uploaded_at,uploader,status) VALUES (?,?,?,?,?,?,?,?)",
                               (identity, filename, str(path.relative_to(self.root)), hashlib.sha256(raw).hexdigest(),
                                len(raw), timestamp, uploader, "queued"))
                    db.execute("INSERT INTO jobs(document_id,status) VALUES (?, 'queued')", (identity,))
                    if import_fingerprint:
                        db.execute("INSERT INTO imports VALUES (?,?)", (import_fingerprint, timestamp))
            except Exception:
                # Only remove the new file created by this unsuccessful registration.
                path.unlink(missing_ok=True)
                folder.rmdir()
                raise
            self.wake.set()
            return self.document(identity)

    def _work(self):
        while not self.stop.is_set():
            try:
                if self.process_next():
                    continue
            except Exception:
                logging.exception("Ingestion worker storage failure")
            self.wake.wait(1)
            self.wake.clear()

    def process_next(self):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT d.*,j.attempts FROM documents d JOIN jobs j ON d.id=j.document_id WHERE j.status='queued' ORDER BY d.uploaded_at LIMIT 1").fetchone()
            if row is None:
                return False
            row = dict(row)
            db.execute("UPDATE jobs SET status='processing',attempts=attempts+1 WHERE document_id=?", (row["id"],))
            db.execute("UPDATE documents SET status='processing' WHERE id=?", (row["id"],))
        try:
            if row["attempts"] >= 3:
                raise ValueError("Processing interrupted too many times; operator review required")
            path = self.root / row["path"]
            if path.is_symlink() or not path.resolve().is_relative_to(self.uploads_dir.resolve()):
                raise ValueError("Invalid stored document path")
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["raw_hash"]:
                raise ValueError("Original file integrity check failed")
            segments = extract_isolated(path, timeout=int(os.getenv("RAG_PARSE_TIMEOUT_SECONDS", "30")),
                                        max_pages=int(os.getenv("RAG_MAX_PDF_PAGES", "200")))
            text = "\n".join(s["text"] for s in segments)
            if len(text.encode()) > self.max_file_bytes:
                raise ValueError("Extracted text exceeds 10 MB")
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            meta = DocumentMetadata(row["id"], "unknown", row["uploader"], row["uploaded_at"],
                                    path.suffix[1:], row["size"], text_hash, True, "none", None,
                                    "success", False, False, trust_score_version=POLICY)
            # Honest validation metadata: parsing does not sanitize a PDF. Same formula/thresholds.
            risk = self.scanner.scan(text)
            assessment = self.scorer.score(text, meta, risk)
            with self.model_lock:
                chunks = chunk_segments(row["id"], segments, self.embedder)
                # Full-document checks cover boundaries; chunk-only checks may only tighten decisions.
                chunk_hard = any(self.scanner.scan(c["text"])["hard_quarantine"] for c in chunks)
                if chunk_hard and not assessment.hard_quarantine:
                    assessment.hard_quarantine, assessment.decision = True, "quarantine"
                    assessment.flags.append("HIGH:chunk instruction pattern")
                count = self.store.rows("SELECT COUNT(*) AS n FROM chunks")[0]["n"]
                if count + len(chunks) > self.max_chunks:
                    raise ValueError("Workspace chunk quota exceeded")
                vectors = self.embedder.embed_batch([c["text"] for c in chunks])
            meta.trust_score = assessment.final_score
            meta.trust_score_breakdown = asdict(assessment)
            meta.quarantine_status = "flagged" if assessment.decision == "quarantine" else "none"
            meta.quarantine_reason = "; ".join(assessment.flags) or None
            with self.store.connect() as db:
                for chunk, vector in zip(chunks, vectors, strict=True):
                    vector = np.asarray(vector, dtype="<f4")
                    if vector.shape != (self.embedder.dimensions,) or not np.isfinite(vector).all():
                        raise ValueError("Invalid embedding dimensions")
                    db.execute("INSERT INTO chunks VALUES (?,?,?,?,?)", (chunk["id"], row["id"], chunk["ordinal"], chunk["text"], json.dumps(chunk["reference"])))
                    db.execute("INSERT INTO embeddings VALUES (?,?,?,?)", (chunk["id"], self.embedder.identity, self.embedder.dimensions, vector.tobytes()))
                db.execute("UPDATE documents SET status='processed',error=NULL,metadata=?,assessment=?,text_hash=?,model=?,policy=? WHERE id=?",
                           (json.dumps(asdict(meta)), json.dumps(asdict(assessment)), text_hash, self.embedder.identity, POLICY, row["id"]))
                db.execute("UPDATE jobs SET status='completed',error=NULL WHERE document_id=?", (row["id"],))
                if assessment.decision != "allow":
                    self._flag(db, row, asdict(assessment), None, None)
        except Exception as error:
            logging.exception("Document processing failed: %s", row["id"])
            message = str(error) if isinstance(error, ValueError) else "Processing failed; contact the workspace operator"
            with self.store.connect() as db:
                db.execute("UPDATE documents SET status='failed',error=? WHERE id=?", (message, row["id"]))
                db.execute("UPDATE jobs SET status='failed',error=? WHERE document_id=?", (message, row["id"]))
        return True

    def _flag(self, db, doc, assessment, request_id, query):
        payload = {"event_id": str(uuid.uuid4()), "document_id": doc["id"], "request_id": request_id,
                   "timestamp": now(), "query": query, "event_type": "retrieval" if request_id else "ingestion",
                   "trust_score": assessment["final_score"], "status": assessment["decision"].upper(),
                   "flag_reason": "; ".join(assessment["flags"]) or "Source authenticity is not verified",
                   "source": f"uploads/{doc['id']}/{doc['filename']}",
                   "quarantine_status": "flagged" if assessment["decision"] == "quarantine" else "none",
                   "high_severity_flag": assessment["hard_quarantine"],
                   "source_document": {"document_id": doc["id"], "found": True}, "policy_version": POLICY}
        db.execute("INSERT INTO security_events VALUES (?,?,?,?,?)", (payload["event_id"], doc["id"], request_id, payload["timestamp"], json.dumps(payload)))

    def query(self, query, top_k):
        with self.model_lock:
            query_vector = self.embedder.embed(query)
        candidates, excluded, flagged = [], {}, {}
        rows = self.store.rows("SELECT c.*,e.vector,e.model,e.dimensions,d.filename,d.assessment,d.policy FROM chunks c JOIN embeddings e ON e.chunk_id=c.id JOIN documents d ON d.id=c.document_id WHERE d.status='processed'")
        for row in rows:
            if row["model"] != self.embedder.identity or row["policy"] != POLICY:
                raise RuntimeError("Incompatible persisted index")
            vector = np.frombuffer(row["vector"], dtype="<f4")
            if vector.shape != (self.embedder.dimensions,) or not np.isfinite(vector).all():
                raise RuntimeError("Corrupt persisted embedding")
            similarity = float(np.clip(np.dot(query_vector, vector), -1, 1))
            if similarity < self.min_similarity:
                continue
            assessment = json.loads(row["assessment"])
            decision, trust = assessment["decision"], assessment["final_score"]
            if decision not in STATUS or not 0 <= trust <= 1:
                raise RuntimeError("Invalid persisted security decision")
            blocked = decision == "quarantine" or assessment["hard_quarantine"]
            effective = similarity * (.5 + .5 * trust) if decision == "warn" else similarity
            result = {"document_id": row["document_id"], "result_id": row["id"], "chunk_id": row["id"],
                      "filename": row["filename"], "source": f"uploads/{row['document_id']}/{row['filename']}",
                      "reference": json.loads(row["reference"]), "trust_score": trust,
                      "security_status": "blocked" if blocked else STATUS[decision], "decision": "quarantine" if blocked else decision,
                      "inclusion": "excluded" if blocked else "allowed", "excerpt": None if blocked else row["text"],
                      "quarantine_status": "flagged" if blocked else "none",
                      "flag_reason": "; ".join(assessment["flags"]) or "Source authenticity is not verified",
                      "similarity": round(similarity, 6), "effective_similarity": round(effective, 6)}
            if blocked:
                excluded.setdefault(row["document_id"], result)
            else:
                candidates.append(result)
            flagged[row["document_id"]] = ({"id": row["document_id"], "filename": row["filename"]}, assessment)
        candidates.sort(key=lambda r: (-r["effective_similarity"], r["result_id"]))
        selected = candidates[:top_k]
        documents = selected + list(excluded.values())
        event = {"retrieval_event_id": str(uuid.uuid4()), "timestamp": now(), "query": query,
                 "origin": "file_pipeline", "pipeline": "semantic", "embedding_model": self.embedder.identity,
                 "policy_version": POLICY, "retrieved_documents": documents, "trusted_documents": selected,
                 "excluded_documents": list(excluded.values()), "retrieval_trace": documents,
                 "trust_scores": {r["document_id"]: r["trust_score"] for r in documents},
                 "security_decisions": [{k: r[k] for k in ("document_id", "decision", "inclusion")} for r in documents],
                 "answer": None}
        with self.store.connect() as db:
            db.execute("INSERT INTO retrieval_events VALUES (?,?,?)", (event["retrieval_event_id"], event["timestamp"], json.dumps(event)))
            for identity in {r["document_id"] for r in documents}:
                doc, assessment = flagged[identity]
                if assessment["decision"] != "allow":
                    self._flag(db, doc, assessment, event["retrieval_event_id"], query)
        return event

    def events(self):
        return self.store.records("retrieval_events")

    def flagged_events(self):
        return self.store.records("security_events")

    def evaluation(self):
        rows = self.store.records("evaluations")
        return rows[0] if rows else None

    def import_evaluation(self, path):
        path = Path(path)
        # Reuse existing schema normalization without loading or changing legacy documents/logs.
        result = Application(evaluation_file=path).evaluation()
        if result is None:
            raise ValueError("Evaluation file does not exist")
        identity = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.store.connect() as db:
            db.execute("INSERT OR IGNORE INTO evaluations VALUES (?,?,?)", (identity, now(), json.dumps(result)))
        return result
