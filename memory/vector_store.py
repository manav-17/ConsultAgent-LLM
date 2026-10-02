"""
vector_store.py — persistent memory for the Second Brain.

Same stack as SecuRAG-LLM: sentence-transformers (MiniLM) + FAISS.
Uses cosine similarity (normalised vectors + inner-product index).
Saved to data/memory/ so knowledge survives restarts.
"""
import io
import json
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

from core.config import MEMORY_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

INDEX_FILE  = MEMORY_DIR / "index.faiss"
CHUNKS_FILE = MEMORY_DIR / "chunks.json"


# ── File loading ─────────────────────────────────────────────
def load_file_text(name: str, data: bytes) -> str:
    """Read .txt / .md / .pdf / .docx bytes into plain text."""
    suffix = Path(name).suffix.lower()
    if suffix in {".txt", ".md", ".csv", ".log", ".eml"}:
        return data.decode("utf-8", errors="ignore")
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix == ".docx":
        import docx
        document = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in document.paragraphs)
    raise ValueError(f"Unsupported file type: {suffix}. Use txt, md, pdf or docx.")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split on paragraphs first, then pack paragraphs into ~size-char chunks."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(para) > size:  # very long paragraph: hard-split it
            for start in range(0, len(para), size - overlap):
                piece = para[start:start + size]
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(piece)
            continue
        if len(current) + len(para) + 2 <= size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if current else ""
            current = f"{tail}\n\n{para}" if tail else para
    if current:
        chunks.append(current)
    return chunks


# ── Store ────────────────────────────────────────────────────
class MemoryStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._embedder = None
        self.index = None
        self.chunks: list[dict] = []
        self._load()

    # Embedding model loads lazily (first use) to keep startup fast
    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedder

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.embedder.encode(texts, convert_to_numpy=True,
                                       normalize_embeddings=True, batch_size=32)
        return vectors.astype("float32")

    def _load(self):
        import faiss
        if INDEX_FILE.exists() and CHUNKS_FILE.exists():
            self.index = faiss.read_index(str(INDEX_FILE))
            self.chunks = json.loads(CHUNKS_FILE.read_text())

    def _save(self):
        import faiss
        faiss.write_index(self.index, str(INDEX_FILE))
        CHUNKS_FILE.write_text(json.dumps(self.chunks, indent=1))

    # ── Public API ──
    def add_document(self, text: str, source: str) -> int:
        """Chunk, embed and store a document. Returns number of chunks added."""
        import faiss
        pieces = chunk_text(text)
        if not pieces:
            return 0
        vectors = self._embed(pieces)
        with self._lock:
            if self.index is None:
                self.index = faiss.IndexFlatIP(vectors.shape[1])
            start = len(self.chunks)
            self.index.add(vectors)
            stamp = datetime.now().isoformat(timespec="seconds")
            for i, piece in enumerate(pieces):
                self.chunks.append({"id": start + i, "source": source,
                                    "text": piece, "added_at": stamp})
            self._save()
        return len(pieces)

    def search(self, query: str, k: int = 5) -> list[dict]:
        if self.is_empty:
            return []
        vector = self._embed([query])
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(vector, k)
        hits = []
        for score, idx in zip(scores[0], ids[0]):
            if 0 <= idx < len(self.chunks):
                hit = dict(self.chunks[idx])
                hit["score"] = float(score)
                hits.append(hit)
        return hits

    @property
    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0

    def stats(self) -> dict:
        sources = sorted({c["source"] for c in self.chunks})
        return {"documents": len(sources), "chunks": len(self.chunks), "sources": sources}

    def clear(self):
        with self._lock:
            self.index, self.chunks = None, []
            for f in (INDEX_FILE, CHUNKS_FILE):
                if f.exists():
                    f.unlink()


_store = None


def get_store() -> MemoryStore:
    """Single shared store per process."""
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store
