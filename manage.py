"""Operator-only initialization, imports, reindexing and consistent backups."""
import argparse
import getpass
import hashlib
import json
import os
import shutil
from pathlib import Path

from auth import hash_code
from persistent_application import PersistentApplication, now
from storage import Storage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "hash-password", "import-documents", "import-evaluation", "import-history", "backup", "restore", "reindex", "inspect"])
    parser.add_argument("path", nargs="?")
    parser.add_argument("--data-dir", default=os.getenv("RAG_DATA_DIR", "data"))
    parser.add_argument("--apply", action="store_true", help="Apply an import/reindex; default is dry-run")
    args = parser.parse_args()
    if args.command == "hash-password":
        password = getpass.getpass("Workspace access code (at least 16 characters): ")
        if len(password) < 16:
            parser.error("Use at least 16 characters")
        print(hash_code(password))
        return
    root = Path(args.data_dir).resolve()
    if args.command == "restore":
        if not args.path or root.exists():
            parser.error("Restore requires a backup path and a NEW, nonexistent --data-dir")
        backup = Path(args.path)
        Storage(backup / "app.sqlite3").ready()
        manifest = json.loads((backup / "manifest.json").read_text())
        for relative, digest in manifest["files"].items():
            source = backup / relative
            if not source.resolve().is_relative_to(backup.resolve()) or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise ValueError("Backup file validation failed")
        shutil.copytree(backup, root)
        print(f"Restored to new directory: {root}")
        return
    service = PersistentApplication(root, start_worker=False)
    service.store.initialize()
    service.uploads_dir.mkdir(parents=True, exist_ok=True)
    operator_lock = None
    if args.apply and args.command in ("import-documents", "reindex", "import-history"):
        import fcntl
        operator_lock = (root / ".service.lock").open("a")
        try:
            fcntl.flock(operator_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Stop the backend before applying this operation")
    if args.command == "init":
        service.store.ready()
        print(f"Initialized schema at {service.store.path}")
    elif args.command == "inspect":
        known = {str((root / r["path"]).resolve()) for r in service.store.rows("SELECT path FROM documents")}
        orphans = [str(p) for p in service.uploads_dir.rglob("*") if p.is_file() and str(p.resolve()) not in known]
        print(json.dumps({"documents": len(service.documents), "orphan_files_preserved": orphans,
                          "jobs": service.store.rows("SELECT status,COUNT(*) AS count FROM jobs GROUP BY status")}, indent=2))
    elif args.command == "import-documents":
        if not args.path:
            parser.error("Supply a document directory")
        paths = sorted(p for p in Path(args.path).rglob("*") if p.suffix.lower() in (".txt", ".pdf") and p.is_file())
        print(f"{'Import' if args.apply else 'Dry run'}: {len(paths)} documents; originals remain unchanged")
        for path in paths:
            if path.is_symlink():
                raise ValueError("Symlink imports are unsupported")
            raw = path.read_bytes()
            if not raw or len(raw) > service.max_file_bytes:
                raise ValueError(f"Invalid document size: {path.name}")
            fingerprint = hashlib.sha256(str(path.resolve()).encode() + raw).hexdigest()
            if service.store.rows("SELECT 1 FROM imports WHERE fingerprint=?", (fingerprint,)):
                continue
            if args.apply:
                # Offline import: registration is durable before receipt is recorded.
                service.upload(path.name, raw, uploader="legacy-import; original uploader/time unknown", import_fingerprint=fingerprint)
        print("Queued imports are processed on the next backend start.")
    elif args.command == "import-history":
        if not args.path:
            parser.error("Supply a legacy retrieval log JSON file")
        from application import normalize_event
        from json_storage import read_records
        rows = read_records(args.path)
        print(f"{'Import' if args.apply else 'Dry run'}: {len(rows)} legacy retrieval records; original payloads and timestamps retained")
        if args.apply:
            with service.store.connect() as db:
                for row in rows:
                    fingerprint = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
                    event = normalize_event(row)
                    event["retrieval_event_id"] = "legacy-" + fingerprint
                    event["origin"] = "legacy_log"
                    event["legacy_payload"] = row
                    db.execute("INSERT OR IGNORE INTO retrieval_events VALUES (?,?,?)", (event["retrieval_event_id"], event["timestamp"] or "", json.dumps(event)))
    elif args.command == "import-evaluation":
        if not args.path:
            parser.error("Supply an evaluation JSON file")
        if args.apply:
            print(json.dumps(service.import_evaluation(args.path), indent=2))
        else:
            print("Dry run: use --apply to import this evaluation snapshot")
    elif args.command == "reindex":
        print("Reindex preserves original files, UUIDs, upload times and historical event snapshots. Stop the backend first.")
        if args.apply:
            with service.store.connect() as db:
                db.execute("DELETE FROM chunks")
                db.execute("UPDATE documents SET status='queued',error=NULL,assessment=NULL,metadata=NULL,model=NULL,policy=NULL")
                db.execute("UPDATE jobs SET status='queued',attempts=0,error=NULL")
    elif args.command == "backup":
        if not args.path:
            parser.error("Supply a new backup directory")
        destination = Path(args.path).resolve()
        if destination.is_relative_to(service.uploads_dir):
            parser.error("Backup must not be inside uploads")
        destination.mkdir(parents=True, exist_ok=False)
        service.store.backup(destination / "app.sqlite3")
        files = {}
        for row in Storage(destination / "app.sqlite3").rows("SELECT path,raw_hash FROM documents"):
            source, target = root / row["path"], destination / row["path"]
            if not source.resolve().is_relative_to(service.uploads_dir):
                raise ValueError("Invalid document path")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest != row["raw_hash"]:
                raise ValueError("Original file changed; backup incomplete")
            files[row["path"]] = digest
        (destination / "manifest.json").write_text(json.dumps({"created_at": now(), "files": files}, indent=2))
        print(f"Backup complete: {destination}. Copy it off the service disk.")
    if operator_lock:
        operator_lock.close()


if __name__ == "__main__":
    main()
