import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from persistent_application import PersistentApplication
from test_persistent import TestEmbedder


class OperationsTests(unittest.TestCase):
    def test_import_backup_restore_preserves_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "originals"
            source.mkdir()
            (source / "notes.txt").write_text("Bank records retained seven years.")
            data = root / "data"
            def run(*args, directory=data):
                return subprocess.run([sys.executable, "manage.py", *args, "--data-dir", str(directory)], check=True, capture_output=True, text=True)
            run("import-documents", str(source), "--apply")
            run("import-documents", str(source), "--apply")
            service = PersistentApplication(data, TestEmbedder(), start_worker=False)
            service.load()
            self.assertEqual(len(service.documents), 1)
            service.process_next()
            original = service.documents
            run("backup", str(root / "backup"))
            run("restore", str(root / "backup"), directory=root / "restored")
            restored = PersistentApplication(root / "restored", TestEmbedder(), start_worker=False)
            restored.load()
            self.assertEqual(restored.documents, original)
            self.assertEqual(len(restored.query("bank", 1)["trusted_documents"]), 1)
            self.assertEqual((source / "notes.txt").read_text(), "Bank records retained seven years.")

    def test_duplicate_validation_fixture_does_not_corrupt_first_entry(self):
        from run_validation_trace import score_folder
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "benign").mkdir()
            (root / "malicious").mkdir()
            for name in ("first.txt", "duplicate.txt"):
                (root / "benign" / name).write_text("Bank records retained seven years.")
            results, store = score_folder(root)
            self.assertEqual(len(store), 1)
            self.assertIn("trust_breakdown", next(iter(store.values())))
            self.assertEqual(sum(r["status"] == "error" for r in results), 1)
