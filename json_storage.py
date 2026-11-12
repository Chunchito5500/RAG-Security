"""JSON audit storage for the single-process local application."""

import json
import os
import tempfile
from pathlib import Path


def read_records(path):
    path = Path(path)
    if not path.exists():
        return []
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError(f"{path.name} must contain an array of records")
    return records


def append_records(path, records):
    path = Path(path)
    existing = read_records(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace only after the complete file is written; corrupt logs are never reset.
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         delete=False) as output:
            name = output.name
            json.dump(existing + records, output, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)
