"""Connect file ingestion, security decisions, and the existing audit formats."""

import json
import threading
from dataclasses import asdict
from pathlib import Path

from flagged_events import extract_flagged_events, save_events
from json_storage import read_records
from rag_sample_demo import (
    ContentRiskScanner, DocumentIngestionPipeline, MockEmbeddingService,
    SafeRetrievalLayer, SourceType, TrustScoringEngine, read_document,
)

ROOT = Path(__file__).resolve().parent
STATUS = {"allow": "trusted", "warn": "flagged", "quarantine": "blocked"}


def normalize_event(record, index=0):
    """Read both legacy monitor logs and file-pipeline logs without rewriting them."""
    documents = []
    selected = {row.get("id") for row in record.get("selected_results") or []}
    for position, result in enumerate(record.get("results") or []):
        decision = str(result.get("decision", result.get("status", "unknown"))).lower()
        inclusion = "unknown"
        if "selected_results" in record:
            inclusion = "allowed" if result.get("id") in selected else "excluded"
        elif decision in STATUS:
            inclusion = "excluded" if decision == "quarantine" else "allowed"
        documents.append({
            "document_id": result.get("id"),
            "result_id": f"{result.get('id') or 'unknown'}-{position}",
            "filename": result.get("filename"), "source": result.get("source"),
            "trust_score": result.get("trust", result.get("trust_score")),
            "security_status": STATUS.get(decision, "unknown"),
            "decision": decision, "inclusion": inclusion,
            "flag_reason": result.get("flag_reason"),
            "quarantine_status": result.get("quarantine_status"),
            "excerpt": result.get("excerpt"),
            "similarity": result.get("similarity"),
            "effective_similarity": result.get("effective_similarity", result.get("effective_retrieval_score")),
        })
    # Legacy quarantined selections remain inspectable but never become trusted passages.
    trusted = [d for d in documents if d["inclusion"] == "allowed" and d["decision"] in ("allow", "warn")]
    return {
        "retrieval_event_id": record.get("id", record.get("request_id", f"legacy-{index}")),
        "query": record.get("query"),
        "timestamp": record.get("timestamp", record.get("time")),
        "origin": "file_pipeline" if record.get("pipeline") == "file_pipeline" else "legacy_log",
        "retrieved_documents": documents,
        "trust_scores": {d["document_id"]: d["trust_score"] for d in documents if d["document_id"]},
        "security_decisions": [{"document_id": d["document_id"], "decision": d["decision"], "inclusion": d["inclusion"]} for d in documents],
        "excluded_documents": [d for d in documents if d["inclusion"] == "excluded"],
        "trusted_documents": trusted,
        "retrieval_trace": documents,
    }


class Application:
    def __init__(self, documents_dir=None, uploads_dir=None, log_file=None,
                 flagged_file=None, evaluation_file=None):
        self.documents_dir = Path(documents_dir or ROOT / "documents")
        self.uploads_dir = Path(uploads_dir or ROOT / "uploads")
        self.log_file = Path(log_file or ROOT / "retrieval_logs.json")
        self.flagged_file = Path(flagged_file or ROOT / "flagged_events.json")
        self.evaluation_file = Path(evaluation_file or ROOT / "validation_results_updated.json")
        self.lock = threading.RLock()
        self.ingestion = DocumentIngestionPipeline()
        self.scanner = ContentRiskScanner()
        self.scorer = TrustScoringEngine()
        self.documents = []
        self.errors = []

    def assess(self, doc):
        meta = doc["metadata"]
        trust = self.scorer.score(doc["text"], meta, self.scanner.scan(doc["text"]))
        doc["trust_breakdown"] = trust
        meta.trust_score = trust.final_score
        meta.trust_score_breakdown = asdict(trust)
        meta.quarantine_status = "flagged" if trust.decision == "quarantine" else "none"
        meta.quarantine_reason = "; ".join(trust.flags) or (
            "Trust score below quarantine threshold" if trust.decision == "quarantine" else None)
        return trust

    def process(self, path, source):
        text = read_document(path)
        if len(text.encode("utf-8")) > self.ingestion.MAX_FILE_SIZE_BYTES:
            raise ValueError("Extracted text exceeds 10 MB")
        if path.suffix.lower() == ".txt" and any(ord(char) < 32 and char not in "\n\r\t\f" for char in text):
            raise ValueError("Document contains unsupported binary/control characters")
        meta = self.ingestion.ingest(text, uploader_id="local-upload" if source.startswith("uploads/") else "repository",
                                     source_type=SourceType.UNKNOWN, file_type=path.suffix.lower()[1:])
        doc = self.ingestion.doc_store[meta.doc_id]
        self.assess(doc)
        # Identical extracted content shares the existing pipeline's document ID.
        doc["filename"] = path.name
        doc["source"] = source
        result = self.document_result(doc)
        self.documents.append(result)
        return result

    def document_result(self, doc):
        meta, trust = doc["metadata"], doc["trust_breakdown"]
        return {
            "document_id": meta.doc_id, "filename": doc["filename"], "source": doc["source"],
            "trust_score": trust.final_score, "security_status": STATUS[trust.decision],
            "decision": trust.decision, "flag_reason": "; ".join(trust.flags) or (
                "Moderate trust score" if trust.decision == "warn" else meta.quarantine_reason),
            "quarantine_status": meta.quarantine_status,
            "trust_score_breakdown": asdict(trust), "status": "processed",
        }

    def load(self):
        with self.lock:
            for folder, prefix in ((self.documents_dir, "documents"), (self.uploads_dir, "uploads")):
                if not folder.exists():
                    continue
                for path in sorted(folder.rglob("*")):
                    if not path.is_file() or path.suffix.lower() not in (".pdf", ".txt"):
                        continue
                    source = f"{prefix}/{path.relative_to(folder).as_posix()}"
                    try:
                        if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):
                            raise ValueError("Document symlinks outside the document directory are unsupported")
                        self.process(path, source)
                    except Exception as error:
                        self.errors.append({"filename": path.name, "source": source,
                                            "status": "error", "error": str(error)})

    def query(self, query, top_k):
        with self.lock:
            # Refuse to lose existing audit data if either store is corrupt.
            read_records(self.log_file)
            read_records(self.flagged_file)
            for doc in self.ingestion.doc_store.values():
                self.assess(doc)
            retrieval = SafeRetrievalLayer(MockEmbeddingService(), log_path=str(self.log_file))
            retrieval.retrieve(query, self.ingestion.doc_store, top_k=top_k)
            record = retrieval.logs[-1]
            record["pipeline"] = "file_pipeline"
            record["top_k"] = top_k
            record["timestamp"] = record["time"].replace(" ", "T") + "Z"
            sources = []
            for result in record["results"]:
                doc = self.ingestion.doc_store[result["id"]]
                details = self.document_result(doc)
                result.update({key: details[key] for key in ("filename", "source", "flag_reason", "quarantine_status")})
                result["excerpt"] = doc["text"][:2000] if result["decision"] != "quarantine" else None
                result["trust_score"] = result["trust"]
                result["status"] = result["decision"].upper()
                result["high_severity_flag"] = any(flag.startswith("HIGH:") for flag in doc["trust_breakdown"].flags)
                trust = doc["trust_breakdown"]
                sources.append({"id": result["id"], "content": doc["text"],
                                "provenance": trust.provenance, "integrity": trust.integrity,
                                "validation": trust.validation, "risk": trust.content_risk})
            flags = extract_flagged_events({**record, "request_id": record["id"]}, sources)
            for flag in flags:
                flag["timestamp"] = record["timestamp"]
                doc = self.ingestion.doc_store[flag["document_id"]]
                flag["source"] = doc["source"]
                flag["quarantine_status"] = doc["metadata"].quarantine_status
            retrieval.export_logs()
            save_events(flags, path=self.flagged_file)
            return normalize_event(record)

    def events(self):
        with self.lock:
            return [normalize_event(row, index) for index, row in reversed(list(enumerate(read_records(self.log_file))))]

    def flagged_events(self):
        with self.lock:
            return list(reversed(read_records(self.flagged_file)))

    def evaluation(self):
        if not self.evaluation_file.exists():
            return None
        result = json.loads(self.evaluation_file.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or not isinstance(result.get("results"), list):
            raise ValueError("Evaluation file must contain a results array")
        rows = [{**row, **{key: row.get(key) for key in (
            "document_id", "filename", "expected", "trust_score", "decision", "correct", "entered_context")},
                 "flags": row.get("flags") or []} for row in result["results"]]
        classified = [r for r in rows if r.get("status") == "processed" and isinstance(r.get("correct"), bool)]
        correct = sum(r["correct"] for r in classified)
        return {
            "source": self.evaluation_file.name, "timestamp": result.get("timestamp"),
            "total_documents_tested": result.get("total_documents", len(rows)),
            "processed_documents": result.get("processed_documents"),
            "correctly_classified": correct if classified else None,
            "false_positives": result.get("false_positives"),
            "missed_malicious_documents": result.get("false_negatives"),
            "accuracy": round(correct / len(classified) * 100, 2) if classified else None,
            "quarantine_exclusion_passed": result.get("quarantine_exclusion_passed"),
            "all_low_trust_test_passed": result.get("all_low_trust_test_passed"),
            "individual_test_results": rows,
        }
