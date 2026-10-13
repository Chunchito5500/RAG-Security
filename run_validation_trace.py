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


LINE = "=" * 78
THIN = "-" * 78


def _fmt(value, missing="n/a"):
    return missing if value is None else value


def format_document(t, index, total):
    """Detailed block for one document."""
    sb = t.get("score_breakdown") or {}
    sig = t.get("signals") or {}
    src = t.get("source") or {}
    verdict = {True: "CORRECT", False: "INCORRECT", None: "NOT SCORED"}[t.get("correct")]
    icon = {"allow": "[ALLOW]", "warn": "[WARN] ", "quarantine": "[BLOCK]"}.get(
        t.get("decision"), "[ERROR]")

    out = [
        THIN,
        f"Document {index}/{total}: {t['filename']}   {icon}   {verdict}",
        THIN,
        f"  Document ID      : {t['document_id']}",
        f"  Expected / Pred. : {_fmt(t['expected'])} / {_fmt(t['predicted'])}",
        f"  Decision         : {str(_fmt(t['decision'])).upper()}",
        f"  Trust score      : {_fmt(t['trust_score'])}   "
        f"(score band: {_fmt(sb.get('score_band'))})",
        f"  Decided by       : {_fmt(sb.get('decision_basis'))}",
        f"  Reason           : {_fmt(t['reason'])}",
        f"  Entered context  : {_fmt(t['entered_context'])}",
    ]
    if t.get("processing_status") != "processed":
        out.append(f"  Processing error : {_fmt(t.get('reason'))}")
        return out

    out += [
        "",
        "  Trust signals                 value   weight   contribution",
        f"    Provenance (P)              {_fmt(sig.get('provenance')):<7} 0.35     "
        f"{_fmt(sb.get('weighted_contributions', {}).get('provenance'))}",
        f"    Integrity (I)               {_fmt(sig.get('integrity')):<7} 0.25     "
        f"{_fmt(sb.get('weighted_contributions', {}).get('integrity'))}",
        f"    Validation (V)              {_fmt(sig.get('validation')):<7} 0.20     "
        f"{_fmt(sb.get('weighted_contributions', {}).get('validation'))}",
        f"    Content risk (R, uses 1-R)  {_fmt(sig.get('content_risk')):<7} 0.20     "
        f"{_fmt(sb.get('weighted_contributions', {}).get('content_risk'))}",
        f"  Calculation      : {_fmt(sb.get('calculation'))}",
        f"  Hard rule applied: {sb.get('hard_rule_applied', False)}",
        "",
    ]
    patterns = t.get("matched_patterns") or []
    if patterns:
        out.append(f"  Matched patterns ({len(patterns)}):")
        for m in patterns:
            out.append(f"    - [{m['severity']}] {m['pattern']}")
    else:
        out.append("  Matched patterns : none")

    stats = t.get("content_stats") or {}
    out += [
        "",
        f"  Source type      : {_fmt(src.get('source_type'))}   "
        f"file type: {_fmt(src.get('file_type'))}",
        f"  Uploaded by / at : {_fmt(src.get('uploader_id'))} / {_fmt(src.get('uploaded_at'))}",
        f"  SHA-256 hash     : {_fmt(src.get('content_hash_sha256'))}",
        f"  Content size     : {_fmt(stats.get('words'))} words, "
        f"{_fmt(stats.get('characters'))} characters",
        f"  Preview          : \"{_fmt(t.get('content_preview'))}\"",
        f"  Trace ID         : {t['trace_id']}",
    ]
    return out


def format_report(payload, events, documents_dir):
    s = payload["summary"]
    cm = s["confusion_matrix"]
    traces = payload["traces"]
    out = [
        LINE,
        "  RAG SECURITY - VALIDATION TRACEABILITY REPORT",
        LINE,
        f"  Generated     : {payload['generated_at']}",
        f"  Documents dir : {documents_dir}",
        f"  Run ID        : {traces[0]['run_id'] if traces else 'n/a'}",
        f"  Query         : {QUERY}",
        f"  Scoring       : TrustScore = 0.35P + 0.25I + 0.20V + 0.20(1-R)",
        f"  Thresholds    : allow >= 0.80 | warn 0.50-0.80 | quarantine < 0.50",
        f"  Hard rule     : any HIGH-severity pattern => quarantine regardless of score",
        "",
        LINE,
        "  PER-DOCUMENT DETAIL",
        LINE,
    ]
    for n, t in enumerate(traces, 1):
        out += format_document(t, n, len(traces))

    out += [
        "",
        LINE,
        "  FLAGGED EVENTS (quarantined documents)",
        LINE,
    ]
    if not events:
        out.append("  none")
    for e in events:
        out += [
            f"  {e['filename']}",
            f"    event_id  : {e['event_id']}",
            f"    trace_id  : {e['trace_id']}",
            f"    doc_id    : {e['document_id']}   score: {e['trust_score']}   "
            f"high severity: {e['high_severity_flag']}",
            f"    reason    : {e['flag_reason']}",
            f"    in context: {e['entered_context']}",
        ]

    out += [
        "",
        LINE,
        "  SUMMARY",
        LINE,
        f"  Benign    : {s['benign_allowed_or_warned']}/{s['benign_total']} allowed or warned",
        f"  Malicious : {s['malicious_quarantined']}/{s['malicious_total']} quarantined",
        "",
        "  Confusion matrix            predicted malicious   predicted benign",
        f"    actually malicious        {cm['true_positives']:<21} {cm['false_negatives']}",
        f"    actually benign           {cm['false_positives']:<21} {cm['true_negatives']}",
        "",
        f"  Accuracy            : {_fmt(s['accuracy_percent'])}%",
        f"  Precision           : {_fmt(s['precision_percent'])}%",
        f"  Recall (detection)  : {_fmt(s['recall_percent'])}%",
        f"  False positive rate : {_fmt(s['false_positive_rate_percent'])}%",
        "",
        f"  Decisions           : {s['decision_counts']}",
        f"  Avg trust score     : benign {_fmt(s['average_trust_score']['benign'])} | "
        f"malicious {_fmt(s['average_trust_score']['malicious'])}",
        f"  Hard-rule quarantines: {s['hard_rule_quarantines']}",
        f"  Caught ONLY by hard rule (score alone would not quarantine): "
        f"{s['quarantined_only_by_hard_rule'] or 'none'}",
        "",
        "  Pattern match counts:",
    ]
    for pat, count in s["pattern_match_counts"].items() or []:
        out.append(f"    {count}x  {pat}")
    if not s["pattern_match_counts"]:
        out.append("    none")

    in_ctx = s["quarantined_in_context"]
    out += [
        "",
        f"  Misclassified files : {s['misclassified'] or 'none'}",
        f"  Processing errors   : {s['errors'] or 'none'}",
        f"  Quarantined docs that entered context: {in_ctx or 'none'}",
        "",
        "  RESULT: " + (
            "PASS - every quarantined document was kept out of the model's context."
            if not in_ctx and not s["errors"] else
            "CHECK REQUIRED - see errors / context leaks above."),
        LINE,
    ]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents-dir",
                        default=str(Path(__file__).resolve().parent / "documents"))
    parser.add_argument("--report", default="validation_report.txt",
                        help="where to save the printed report")
    parser.add_argument("--quiet", action="store_true",
                        help="only print the summary line")
    args = parser.parse_args()

    payload, events = run(args.documents_dir)
    report = format_report(payload, events, args.documents_dir)
    Path(args.report).write_text(report + "\n", encoding="utf-8")

    if args.quiet:
        s = payload["summary"]
        print(f"Benign {s['benign_allowed_or_warned']}/{s['benign_total']} | "
              f"Malicious {s['malicious_quarantined']}/{s['malicious_total']} | "
              f"Accuracy {s['accuracy_percent']}%")
    else:
        print(report)
    print(f"\nFiles written: validation_trace.json, validation_flagged_events.json, "
          f"{args.report}")


if __name__ == "__main__":
    main()
