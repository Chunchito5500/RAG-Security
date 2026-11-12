"""SQLite persistence. Connections are short-lived; related writes share a transaction."""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY);
CREATE TABLE IF NOT EXISTS documents(
 id TEXT PRIMARY KEY, filename TEXT NOT NULL, path TEXT NOT NULL UNIQUE,
 raw_hash TEXT NOT NULL, size INTEGER NOT NULL, uploaded_at TEXT NOT NULL,
 uploader TEXT NOT NULL, status TEXT NOT NULL, error TEXT,
 metadata TEXT, assessment TEXT, text_hash TEXT, model TEXT, policy TEXT);
CREATE TABLE IF NOT EXISTS chunks(
 id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id),
 ordinal INTEGER NOT NULL, text TEXT NOT NULL, reference TEXT NOT NULL,
 UNIQUE(document_id, ordinal));
CREATE TABLE IF NOT EXISTS embeddings(
 chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
 model TEXT NOT NULL, dimensions INTEGER NOT NULL, vector BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS jobs(
 document_id TEXT PRIMARY KEY REFERENCES documents(id), status TEXT NOT NULL,
 attempts INTEGER NOT NULL DEFAULT 0, error TEXT);
CREATE TABLE IF NOT EXISTS retrieval_events(
 id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS security_events(
 id TEXT PRIMARY KEY, document_id TEXT REFERENCES documents(id),
 request_id TEXT REFERENCES retrieval_events(id), timestamp TEXT NOT NULL, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evaluations(
 id TEXT PRIMARY KEY, imported_at TEXT NOT NULL, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions(token_hash TEXT PRIMARY KEY, expires REAL NOT NULL);
CREATE TABLE IF NOT EXISTS imports(fingerprint TEXT PRIMARY KEY, imported_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS security_document ON security_events(document_id);
"""


class Storage:
    def __init__(self, path):
        self.path = Path(path)

    def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError("Database requires a newer application; refusing downgrade")
            # Default rollback journal is adequate for one process and simplifies backup.
            db.executescript("BEGIN IMMEDIATE;" + SCHEMA +
                             f"INSERT OR IGNORE INTO schema_migrations VALUES ({SCHEMA_VERSION});"
                             f"PRAGMA user_version={SCHEMA_VERSION}; COMMIT;")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def rows(self, sql, args=()):
        with self.connect() as db:
            return [dict(row) for row in db.execute(sql, args)]

    def records(self, table):
        if table not in {"retrieval_events", "security_events", "evaluations"}:
            raise ValueError("Unsupported record type")
        order = "imported_at" if table == "evaluations" else "timestamp"
        return [json.loads(row["payload"]) for row in self.rows(
            f"SELECT payload FROM {table} ORDER BY {order} DESC, rowid DESC")]

    def ready(self):
        with self.connect() as db:
            if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise RuntimeError("Database integrity check failed")
            if db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
                raise RuntimeError("Database schema mismatch")
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE schema_migrations SET version=version")

    def backup(self, destination):
        destination = Path(destination)
        if destination.exists():
            raise ValueError("Backup destination already exists")
        target = sqlite3.connect(destination)
        try:
            with self.connect() as source:
                source.backup(target)
        finally:
            target.close()
