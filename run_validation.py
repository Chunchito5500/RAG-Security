
import json
from pathlib import Path
from datetime import datetime, timezone

from rag_sample_demo import (
    read_document,
    DocumentIngestionPipeline,
    ContentRiskScanner,
    TrustScoringEngine,
    MockEmbeddingService,
    SafeRetrievalLayer,
    SourceType,
)

ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = ROOT / "documents"
RESULTS_FILE = ROOT / "validation_results.json"
LOG_FILE = ROOT / "validation_retrieval_logs.json"

ingestion = DocumentIngestionPipeline()
scanner = ContentRiskScanner()
scorer = TrustScoringEngine()

retrieval = SafeRetrievalLayer(
    MockEmbeddingService(),
    log_path=str(LOG_FILE),
)

results = []
seen_ids = set()

#Read and evaluate real documents
for category in ("benign", "malicious"):
    folder = DOCUMENTS_DIR / category

    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")

    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in (".pdf", ".txt"):
            continue

        print(f"Processing: {path.name}")

        try:
            text = read_document(path)

            # Use the same provenance for both categories.
            # Expected labels must not determine trust scores.
            meta = ingestion.ingest(
                text,
                uploader_id="sprint8-validation",
                source_type=SourceType.UNKNOWN,
                file_type=path.suffix.lower().lstrip("."),
            )

            if meta.doc_id in seen_ids:
                raise ValueError(
                    "Duplicate document content or document ID"
                )

            seen_ids.add(meta.doc_id)

            risk = scanner.scan(text)
            trust = scorer.score(text, meta, risk)

            ingestion.doc_store[meta.doc_id][
                "trust_breakdown"
            ] = trust

            results.append({
                "filename": path.name,
                "document_id": meta.doc_id,
                "expected": category,
                "trust_score": trust.final_score,
                "decision": trust.decision,
                "flagged": trust.decision == "quarantine",
                "warning": trust.decision == "warn",
                "flags": trust.flags,
                "status": "processed",
            })

        except Exception as error:
            print(f"ERROR processing {path.name}: {error}")

            results.append({
                "filename": path.name,
                "document_id": None,
                "expected": category,
                "trust_score": None,
                "decision": None,
                "flagged": None,
                "flags": [],
                "status": "error",
                "error": str(error),
            })

#  Run retrieval against the ingested documents
query = "What information is contained in these documents?"

allowed_docs = retrieval.retrieve(
    query,
    ingestion.doc_store,
    top_k=len(ingestion.doc_store),
)

allowed_ids = {
    doc["metadata"].doc_id
    for doc in allowed_docs
}

#  Check classifications and document exclusion
tp = tn = fp = fn = 0
exclusion_failures = []

for result in results:
    if result["status"] != "processed":
        result["entered_context"] = None
        result["correct"] = None
        continue

    doc_id = result["document_id"]
    result["entered_context"] = doc_id in allowed_ids

    expected_malicious = result["expected"] == "malicious"
    predicted_malicious = result["flagged"]

    result["correct"] = (
        expected_malicious == predicted_malicious
    )

    if expected_malicious and predicted_malicious:
        tp += 1
    elif not expected_malicious and not predicted_malicious:
        tn += 1
    elif not expected_malicious and predicted_malicious:
        fp += 1
    else:
        fn += 1

    if (
        result["decision"] == "quarantine"
        and result["entered_context"]
    ):
        exclusion_failures.append(result["filename"])

# 4. Verify all-low-trust retrieval behavior
quarantined_store = {
    doc_id: doc
    for doc_id, doc in ingestion.doc_store.items()
    if doc["trust_breakdown"].decision == "quarantine"
}

all_low_trust_passed = None

if quarantined_store:
    low_trust_results = retrieval.retrieve(
        "Test retrieval with only quarantined documents",
        quarantined_store,
        top_k=len(quarantined_store),
    )

    all_low_trust_passed = len(low_trust_results) == 0

#  Save results
processed = sum(
    r["status"] == "processed" for r in results
)
total = len(results)
accuracy = (
    round((tp + tn) / processed * 100, 2)
    if processed else None
)

summary = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "total_documents": total,
    "processed_documents": processed,
    "expected_documents": 10,
    "true_positives": tp,
    "true_negatives": tn,
    "false_positives": fp,
    "false_negatives": fn,
    "accuracy_percent": accuracy,
    "quarantine_exclusion_passed": (
        len(exclusion_failures) == 0
    ),
    "exclusion_failures": exclusion_failures,
    "all_low_trust_test_passed": all_low_trust_passed,
    "results": results,
}

RESULTS_FILE.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

retrieval.export_logs()

#  Print summary
print("\n===== SPRINT 8 VALIDATION RESULTS =====")
print(f"Documents found: {total}")
print(f"Successfully processed: {processed}")
print(f"Correctly flagged malicious: {tp}")
print(f"Correctly allowed benign: {tn}")
print(f"False positives: {fp}")
print(f"Missed malicious documents: {fn}")
print(f"Accuracy: {accuracy}%")
print(f"Quarantine exclusion failures: {len(exclusion_failures)}")
print(f"All-low-trust test passed: {all_low_trust_passed}")
print(f"\nResults saved to: {RESULTS_FILE}")

# Enforce acceptance checks
if total != 10 or processed != 10:
    raise SystemExit(
        "Validation incomplete: 10 documents must process successfully."
    )

if exclusion_failures:
    raise SystemExit(
        "Validation failed: quarantined documents entered context."
    )

if all_low_trust_passed is not True:
    raise SystemExit(
        "Validation incomplete: all-low-trust test did not pass."
    )

print("\nSprint 8 validation checks passed.")
