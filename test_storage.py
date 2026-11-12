import sqlite3
import tempfile
import unittest
from pathlib import Path

from storage import Storage


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "app.sqlite3"
        self.store = Storage(self.path)
        self.store.initialize()

    def test_initialization_restart_and_backup(self):
        with self.store.connect() as db:
            db.execute("INSERT INTO imports VALUES ('original', '2026-01-01')")
        restarted = Storage(self.path)
        restarted.initialize()
        restarted.ready()
        self.assertEqual(restarted.rows("SELECT * FROM imports")[0]["imported_at"], "2026-01-01")
        backup = self.path.with_name("backup.sqlite3")
        restarted.backup(backup)
        self.assertEqual(Storage(backup).rows("SELECT * FROM imports"), restarted.rows("SELECT * FROM imports"))

    def test_transaction_rollback_and_foreign_keys(self):
        with self.assertRaises(sqlite3.IntegrityError), self.store.connect() as db:
            db.execute("INSERT INTO imports VALUES ('must-rollback', 'today')")
            db.execute("INSERT INTO jobs VALUES ('missing-document', 'queued', 0, NULL)")
        self.assertEqual(self.store.rows("SELECT * FROM imports"), [])

    def test_future_schema_is_not_downgraded(self):
        with self.store.connect() as db:
            db.execute("PRAGMA user_version=999")
        with self.assertRaises(RuntimeError):
            self.store.initialize()
