"""Pinned, local semantic embeddings. There is deliberately no mock fallback."""
import os
import threading

import numpy as np

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class SemanticEmbedder:
    dimensions = 384
    identity = f"{MODEL}@{REVISION}"

    def __init__(self):
        self.model = None
        self.lock = threading.RLock()

    def load(self):
        with self.lock:
            if self.model is None:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(
                    MODEL, revision=REVISION, device="cpu", trust_remote_code=False,
                    cache_folder=os.getenv("RAG_MODEL_CACHE", ".model-cache"),
                    local_files_only=os.getenv("RAG_MODEL_OFFLINE", "1") == "1",
                    model_kwargs={"use_safetensors": True},
                )
                self.model.max_seq_length = 256

    def offsets(self, text):
        self.load()
        return self.model.tokenizer(text, add_special_tokens=False, truncation=False,
                                    return_offsets_mapping=True, verbose=False)["offset_mapping"]

    def embed_batch(self, texts):
        self.load()
        with self.lock:
            vectors = self.model.encode(texts, normalize_embeddings=True,
                                        convert_to_numpy=True, show_progress_bar=False, batch_size=16)
        vectors = np.asarray(vectors, dtype="<f4")
        if vectors.shape != (len(texts), self.dimensions) or not np.isfinite(vectors).all():
            raise ValueError("Invalid embedding output")
        return vectors

    def embed(self, text):
        return self.embed_batch([text])[0]


if __name__ == "__main__":
    os.environ["RAG_MODEL_OFFLINE"] = "0"
    embedder = SemanticEmbedder()
    embedder.load()
    print(f"Model cached: {embedder.identity}")
