import re
import tempfile
import unittest
import uuid
from pathlib import Path

from document_processing import chunk_segments, extract, extract_isolated


class OffsetTokenizer:
    def offsets(self, text):
        return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


class ProcessingTests(unittest.TestCase):
    def test_txt_offsets_overlap_and_determinism(self):
        text = "\n".join(f"word{i}" for i in range(400))
        identity = str(uuid.uuid4())
        chunks = chunk_segments(identity, [{"text": text, "page": None}], OffsetTokenizer())
        self.assertEqual(chunks, chunk_segments(identity, [{"text": text, "page": None}], OffsetTokenizer()))
        self.assertEqual(chunks[1]["reference"]["line_start"], 161)
        self.assertIn("word399", chunks[-1]["text"])
        for chunk in chunks:
            ref = chunk["reference"]
            self.assertEqual(chunk["text"], text[ref["start_char"]:ref["end_char"]])

    def test_real_pdf_page_references(self):
        path = Path(__file__).parent / "documents/benign/benign_01.pdf"
        pages = extract(path)
        chunks = chunk_segments(str(uuid.uuid4()), pages, OffsetTokenizer())
        self.assertTrue(chunks)
        self.assertTrue(all(c["reference"]["page"] >= 1 for c in chunks))

    def test_isolated_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "notes.txt"
            path.write_text("First line\nSecond line")
            self.assertEqual(extract_isolated(path)[0]["text"], path.read_text())
            path.write_bytes(b"binary\x00text")
            with self.assertRaises(ValueError):
                extract_isolated(path)
