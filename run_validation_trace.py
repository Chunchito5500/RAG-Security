"""
Run the benign / malicious validation set and write traceability output.

Reads documents/benign/* and documents/malicious/* (TXT or PDF), scores them
exactly the way run_validation.py does, then writes:

  validation_trace.json            -- one trace record per file (benign + malicious)
  validation_flagged_events.json   -- one flagged event per quarantined file,
                                      linked back to its source file and signals

Usage:
    python run_validation_trace.py
    python run_validation_trace.py --documents-dir path/to/documents
"""

import argparse
from pathlib import Path

from rag_sample_demo import (
    read_document,
    DocumentIngestionPipeline,
    ContentRiskScanner,
    TrustScoringEngine,
    MockEmbeddingService,
    SafeRetrievalLayer,
    SourceType,
)
from validation_trace import trace_validation_run, save_trace

QUERY = "What information is contained in these documents?"


def score_folder(documents_dir):
    ingestion = DocumentIngestionPipeline()
    scanner = ContentRiskScanner()
    scorer = TrustScoringEngine()
    results, seen = [], set()

    for category in ("benign", "malicious"):
        folder = Path(documents_dir) / category
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in (".pdf", ".txt"):
                continue
            try:
                text = read_document(path)
                meta = ingestion.ingest(
                    text,
                    uploader_id="sprint8-validation",
                    source_type=SourceType.UNKNOWN,
                    file_type=path.suffix.lower().lstrip("."),
                )
                if meta.doc_id in seen:
                    raise ValueError("Duplicate document content or document ID")
                seen.add(meta.doc_id)

                trust = scorer.score(text, meta, scanner.scan(text))
                ingestion.doc_store[meta.doc_id]["trust_breakdown"] = trust
                results.append({
                    "filename": path.name,
                    "document_id": meta.doc_id,
                    "expected": category,
                    "trust_score": trust.final_score,
                    "decision": trust.decision,
                    "flags": trust.flags,
                    "status": "processed",
                })
            except Exception as error:
                results.append({
                    "filename": path.name,
                    "document_id": None,
                    "expected": category,
                    "trust_score": None,
                    "decision": None,
                    "flags": [],
                    "status": "error",
                    "error": str(error),
                })
    return results, ingestion.doc_store


def run(documents_dir, trace_path="validation_trace.json",
        events_path="validation_flagged_events.json"):
    results, doc_store = score_folder(documents_dir)

    retrieval = SafeRetrievalLayer(MockEmbeddingService(), log_path="unused.json")
    allowed = retrieval.retrieve(QUERY, doc_store, top_k=max(len(doc_store), 1))
    allowed_ids = {d["metadata"].doc_id for d in allowed}

    traces, events = trace_validation_run(
        results, doc_store, allowed_ids=allowed_ids, query=QUERY)
    payload = save_trace(traces, events, trace_path, events_path)
    return payload, events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents-dir",
                        default=str(Path(__file__).resolve().parent / "documents"))
    args = parser.parse_args()

    payload, events = run(args.documents_dir)
    s = payload["summary"]
    print("\n===== VALIDATION TRACEABILITY =====")
    print(f"Benign:    {s['benign_allowed_or_warned']}/{s['benign_total']} allowed or warned")
    print(f"Malicious: {s['malicious_quarantined']}/{s['malicious_total']} quarantined")
    print(f"Flagged events written: {len(events)}")
    print(f"Quarantined docs that entered context: {s['quarantined_in_context'] or 'none'}")
    if s["errors"]:
        print(f"Files with errors: {s['errors']}")
    for e in events:
        print(f"  {e['filename']:<22} {e['document_id']}  score={e['trust_score']}  "
              f"{e['flag_reason']}")


if __name__ == "__main__":
    main()
