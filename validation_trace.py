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
