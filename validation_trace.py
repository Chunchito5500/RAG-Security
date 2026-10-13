"""
Traceability for the benign / malicious validation run.

Bridges the Sprint 8 validation pipeline (rag_sample_demo.py) to the
flagged-event format in flagged_events.py, so that:

  * every QUARANTINED document (expected malicious) becomes a flagged event
    that links back to its source file, trust signals and detector flags, and
  * every document (benign or malicious) gets a trace record, so an auditor
    can see why a benign file was allowed/warned as well as why a malicious
    file was blocked.

Nothing here changes scoring: it only reads the doc_store entries that
run_validation.py already builds ({"text", "metadata", "trust_breakdown"}).
"""

import json
import time
import uuid

from flagged_events import build_flagged_event, save_events

TRACE_FILE = "validation_trace.json"
VALIDATION_FLAGGED_EVENTS_FILE = "validation_flagged_events.json"


def _get(obj, name, default=None):
    """Read a field from a dataclass/object or a dict."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def describe_flags(flags):
    """Turn detector flags like 'HIGH:\\bbypass\\b' into a readable reason."""
    flags = list(flags or [])
    if not flags:
        return None
    high = [f for f in flags if str(f).startswith("HIGH:")]
    other = [f for f in flags if f not in high]
    parts = []
    if high:
        parts.append(f"{len(high)} high-severity injection pattern(s) matched")
    if "INTEGRITY_FAIL" in other:
        parts.append("content hash did not match (integrity failure)")
        other = [f for f in other if f != "INTEGRITY_FAIL"]
    if other:
        parts.append(f"{len(other)} medium-severity risk pattern(s) matched")
    return "; ".join(parts)


def build_flag_reason(trust_breakdown):
    decision = _get(trust_breakdown, "decision")
    score = _get(trust_breakdown, "final_score")
    flag_text = describe_flags(_get(trust_breakdown, "flags"))

    if decision == "quarantine":
        if _get(trust_breakdown, "hard_quarantine"):
            return f"Hard quarantine: {flag_text or 'high-severity rule triggered'}"
        return f"Trust score {score} below quarantine threshold (0.5)"
    if decision == "warn":
        return "Moderate trust score; downweighted at retrieval" + (
            f" ({flag_text})" if flag_text else ""
        )
    if decision == "allow":
        return "Document passed trust checks"
    return None


def to_result(doc_id, trust_breakdown):
    """Convert a TrustScoreBreakdown into the result dict flagged_events expects."""
    decision = _get(trust_breakdown, "decision")
    flags = list(_get(trust_breakdown, "flags", []) or [])
    return {
        "id": doc_id,
        "trust_score": _get(trust_breakdown, "final_score"),
        "status": decision.upper() if isinstance(decision, str) else "unknown",
        "flagged": decision == "quarantine",
        "flag_reason": build_flag_reason(trust_breakdown),
        "high_severity_flag": any(str(f).startswith("HIGH:") for f in flags),
    }


def to_source_doc(doc_id, entry):
    """Shape a doc_store entry like the dicts build_source_link() reads."""
    if not isinstance(entry, dict):
        return None
    tb = entry.get("trust_breakdown")
    return {
        "id": doc_id,
        "content": entry.get("text"),
        "provenance": _get(tb, "provenance"),
        "integrity": _get(tb, "integrity"),
        "validation": _get(tb, "validation"),
        "risk": _get(tb, "content_risk"),
    }


WEIGHTS = {"provenance": 0.35, "integrity": 0.25, "validation": 0.20, "content_risk": 0.20}
ALLOW_THRESHOLD = 0.80
QUARANTINE_THRESHOLD = 0.50


def score_band(score):
    if score is None:
        return None
    if score >= ALLOW_THRESHOLD:
        return "allow"
    if score >= QUARANTINE_THRESHOLD:
        return "warn"
    return "quarantine"


def build_score_breakdown(tb):
    """Show exactly how the trust score was computed and what decided the outcome."""
    if tb is None:
        return None
    p, i, v, r = (_get(tb, k) for k in ("provenance", "integrity", "validation", "content_risk"))
    if None in (p, i, v, r):
        return None
    contributions = {
        "provenance": round(WEIGHTS["provenance"] * p, 4),
        "integrity": round(WEIGHTS["integrity"] * i, 4),
        "validation": round(WEIGHTS["validation"] * v, 4),
        "content_risk": round(WEIGHTS["content_risk"] * (1 - r), 4),
    }
    score = _get(tb, "final_score")
    band = score_band(score)
    decision = _get(tb, "decision")
    hard = bool(_get(tb, "hard_quarantine"))
    if hard and band != "quarantine":
        basis = "hard rule (score alone would have been '%s')" % band
    elif decision == "quarantine":
        basis = "score below quarantine threshold" if band == "quarantine" else "hard rule"
    else:
        basis = "score threshold"
    return {
        "formula": "0.35*P + 0.25*I + 0.20*V + 0.20*(1-R)",
        "calculation": (
            f"0.35*{p} + 0.25*{i} + 0.20*{v} + 0.20*(1-{r}) = "
            f"{contributions['provenance']} + {contributions['integrity']} + "
            f"{contributions['validation']} + {contributions['content_risk']} = {score}"
        ),
        "weighted_contributions": contributions,
        "score_band": band,
        "hard_rule_applied": hard,
        "decision_basis": basis,
        "thresholds": {"allow": f">= {ALLOW_THRESHOLD}",
                       "warn": f"{QUARANTINE_THRESHOLD} - {ALLOW_THRESHOLD}",
                       "quarantine": f"< {QUARANTINE_THRESHOLD}"},
    }


def parse_flags(flags):
    """Split 'HIGH:pattern' strings into structured matches."""
    parsed = []
    for f in flags or []:
        f = str(f)
        if f == "INTEGRITY_FAIL":
            parsed.append({"severity": "HIGH", "rule": "integrity check",
                           "pattern": "content hash mismatch"})
        elif ":" in f:
            sev, pat = f.split(":", 1)
            parsed.append({"severity": sev, "rule": "content risk pattern", "pattern": pat})
        else:
            parsed.append({"severity": "UNKNOWN", "rule": "other", "pattern": f})
    return parsed


def _content_preview(entry, length=160):
    text = entry.get("text") if isinstance(entry, dict) else None
    if not isinstance(text, str):
        return None
    flat = " ".join(text.split())
    return flat[:length] + ("..." if len(flat) > length else "")


def _content_stats(entry):
    text = entry.get("text") if isinstance(entry, dict) else None
    if not isinstance(text, str):
        return None
    return {"characters": len(text), "words": len(text.split()),
            "lines": text.count("\n") + 1}


def build_trace_record(result_row, entry, run_id, entered_context=None):
    """One audit record per validated file (benign and malicious)."""
    result_row = result_row or {}
    doc_id = result_row.get("document_id")
    tb = (entry or {}).get("trust_breakdown") if isinstance(entry, dict) else None
    meta = (entry or {}).get("metadata") if isinstance(entry, dict) else None

    decision = result_row.get("decision") or _get(tb, "decision")
    expected = result_row.get("expected")
    predicted = (
        None if decision is None
        else ("malicious" if decision == "quarantine" else "benign")
    )

    return {
        "trace_id": str(uuid.uuid4()),
        "run_id": run_id,
        "filename": result_row.get("filename", "unknown"),
        "document_id": doc_id if doc_id is not None else "unknown",
        "expected": expected,
        "predicted": predicted,
        "correct": (expected == predicted) if predicted and expected else None,
        "decision": decision,
        "trust_score": result_row.get("trust_score", _get(tb, "final_score")),
        "entered_context": entered_context,
        "reason": build_flag_reason(tb) if tb is not None else result_row.get("error"),
        "flags": list(_get(tb, "flags", []) or result_row.get("flags", []) or []),
        "matched_patterns": parse_flags(
            _get(tb, "flags", []) or result_row.get("flags", []) or []),
        "score_breakdown": build_score_breakdown(tb),
        "content_preview": _content_preview(entry),
        "content_stats": _content_stats(entry),
        "signals": {
            "provenance": _get(tb, "provenance"),
            "integrity": _get(tb, "integrity"),
            "validation": _get(tb, "validation"),
            "content_risk": _get(tb, "content_risk"),
        },
        "source": {
            "found": entry is not None,
            "source_type": _get(meta, "source_type"),
            "file_type": _get(meta, "file_type"),
            "content_hash_sha256": _get(meta, "content_hash_sha256"),
            "uploader_id": _get(meta, "uploader_id"),
            "uploaded_at": _get(meta, "uploaded_at"),
        },
        "processing_status": result_row.get("status", "unknown"),
    }


def trace_validation_run(results, doc_store, allowed_ids=None, query=None, run_id=None):
    """
    Build trace records (all files) and flagged events (quarantined files).

    results     -- the `results` list run_validation.py builds
    doc_store   -- ingestion.doc_store
    allowed_ids -- ids returned by retrieval (to record entered_context)
    """
    run_id = run_id or f"validation-{uuid.uuid4()}"
    doc_store = doc_store or {}
    traces, events = [], []

    for row in results or []:
        doc_id = row.get("document_id")
        entry = doc_store.get(doc_id) if doc_id is not None else None
        entered = None
        if allowed_ids is not None and doc_id is not None:
            entered = doc_id in allowed_ids
        elif "entered_context" in row:
            entered = row["entered_context"]

        trace = build_trace_record(row, entry, run_id, entered)
        traces.append(trace)

        tb = entry.get("trust_breakdown") if isinstance(entry, dict) else None
        if tb is None:
            continue
        result = to_result(doc_id, tb)
        if result["flagged"]:
            event = build_flagged_event(
                result,
                doc=to_source_doc(doc_id, entry),
                request_id=run_id,
                query=query,
            )
            event["filename"] = row.get("filename")
            event["expected"] = row.get("expected")
            event["entered_context"] = entered
            event["trace_id"] = trace["trace_id"]
            event["matched_patterns"] = trace["matched_patterns"]
            event["score_breakdown"] = trace["score_breakdown"]
            event["content_hash_sha256"] = trace["source"]["content_hash_sha256"]
            event["detection_correct"] = trace["correct"]
            events.append(event)

    return traces, events


def summarize_traces(traces):
    by_expected = {"benign": [], "malicious": []}
    for t in traces:
        by_expected.setdefault(t.get("expected") or "unknown", []).append(t)
    return {
        "benign_total": len(by_expected["benign"]),
        "benign_allowed_or_warned": sum(
            t["decision"] in ("allow", "warn") for t in by_expected["benign"]),
        "malicious_total": len(by_expected["malicious"]),
        "malicious_quarantined": sum(
            t["decision"] == "quarantine" for t in by_expected["malicious"]),
        "quarantined_in_context": [
            t["filename"] for t in traces
            if t["decision"] == "quarantine" and t["entered_context"]
        ],
        "errors": [t["filename"] for t in traces if t["processing_status"] != "processed"],
        **_detection_metrics(traces),
    }


def _avg(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return round(sum(values) / len(values), 4) if values else None


def _detection_metrics(traces):
    scored = [t for t in traces if t["predicted"] and t["expected"] in ("benign", "malicious")]
    tp = sum(t["expected"] == "malicious" and t["predicted"] == "malicious" for t in scored)
    tn = sum(t["expected"] == "benign" and t["predicted"] == "benign" for t in scored)
    fp = sum(t["expected"] == "benign" and t["predicted"] == "malicious" for t in scored)
    fn = sum(t["expected"] == "malicious" and t["predicted"] == "benign" for t in scored)

    def pct(n, d):
        return round(100 * n / d, 2) if d else None

    decisions, patterns = {}, {}
    for t in traces:
        decisions[t["decision"] or "error"] = decisions.get(t["decision"] or "error", 0) + 1
        for m in t.get("matched_patterns") or []:
            patterns[m["pattern"]] = patterns.get(m["pattern"], 0) + 1

    benign = [t for t in traces if t["expected"] == "benign"]
    malicious = [t for t in traces if t["expected"] == "malicious"]
    return {
        "confusion_matrix": {"true_positives": tp, "true_negatives": tn,
                             "false_positives": fp, "false_negatives": fn},
        "accuracy_percent": pct(tp + tn, len(scored)),
        "precision_percent": pct(tp, tp + fp),
        "recall_percent": pct(tp, tp + fn),
        "false_positive_rate_percent": pct(fp, fp + tn),
        "decision_counts": decisions,
        "hard_rule_quarantines": sum(
            bool((t.get("score_breakdown") or {}).get("hard_rule_applied")) for t in traces),
        "quarantined_only_by_hard_rule": [
            t["filename"] for t in traces
            if t["decision"] == "quarantine"
            and (t.get("score_breakdown") or {}).get("score_band") not in (None, "quarantine")
        ],
        "average_trust_score": {
            "benign": _avg(t["trust_score"] for t in benign),
            "malicious": _avg(t["trust_score"] for t in malicious),
        },
        "pattern_match_counts": dict(sorted(patterns.items(), key=lambda kv: -kv[1])),
        "misclassified": [t["filename"] for t in scored if t["correct"] is False],
    }


def save_trace(traces, events, trace_path=TRACE_FILE,
               events_path=VALIDATION_FLAGGED_EVENTS_FILE):
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summarize_traces(traces),
        "traces": traces,
    }
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    save_events(events, path=events_path)
    return payload
