"""On-disk index: chunks.jsonl + embeddings.npy + BM25 + manifest.json.

Layout: artifacts/index/<embedder>-c<chunk_tokens>[-<variant>]/
The manifest records the embedder identity, dimension, corpus hash and chunking so a stale or mismatched
index is detected instead of silently returning garbage (CHANGELOG B1-B3).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mhrag.types import Chunk

log = logging.getLogger(__name__)
MANIFEST = "manifest.json"


class IndexError_(RuntimeError):
    pass


class IndexMissingError(IndexError_):
    pass


class IndexMismatchError(IndexError_):
    pass


def index_name(embedder_key: str, chunk_tokens: int, variant: str = "") -> str:
    name = f"{embedder_key}-c{chunk_tokens}"
    return f"{name}-{variant}" if variant else name


# ---------------------------------------------------------------- BM25
_TOKEN = re.compile(r"[a-z0-9]+")


def _stemmer():
    try:
        import Stemmer

        return Stemmer.Stemmer("english")
    except ImportError:  # pragma: no cover
        return None


class BM25Index:
    def __init__(self, retriever, n_docs: int):
        self.retriever = retriever
        self.n_docs = n_docs

    @staticmethod
    def tokenize(texts: list[str]):
        import bm25s

        return bm25s.tokenize(texts, stopwords="en", stemmer=_stemmer(), show_progress=False)

    @classmethod
    def build(cls, texts: list[str]) -> BM25Index:
        import bm25s

        r = bm25s.BM25()
        r.index(cls.tokenize(texts), show_progress=False)
        return cls(r, len(texts))

    def search(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = self.tokenize([query])
        k = min(k, self.n_docs)
        if not q.vocab or all(len(ids) == 0 for ids in q.ids):
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        docs, scores = self.retriever.retrieve(q, k=k, show_progress=False)
        docs, scores = docs[0], scores[0]
        keep = scores > 0
        return docs[keep].astype(int), scores[keep].astype(np.float32)

    def save(self, path: Path):
        self.retriever.save(str(path))
        (Path(path) / "n_docs.txt").write_text(str(self.n_docs))

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        import bm25s

        return cls(bm25s.BM25.load(str(path)), int((Path(path) / "n_docs.txt").read_text()))


# ---------------------------------------------------------------- Dense
class DenseIndex:
    def __init__(self, embeddings: np.ndarray, use_faiss: bool = False):
        self.emb = np.ascontiguousarray(embeddings.astype(np.float32))
        self.faiss = None
        if use_faiss:
            try:
                import faiss

                self.faiss = faiss.IndexFlatIP(self.emb.shape[1])
                self.faiss.add(self.emb)
            except ImportError:
                log.warning("faiss not installed; using exact numpy inner-product search")

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Exact cosine search (vectors are L2-normalised so inner product == cosine)."""
        q = np.asarray(q, dtype=np.float32).reshape(1, -1)
        k = min(k, self.emb.shape[0])
        if self.faiss is not None:
            s, i = self.faiss.search(q, k)
            return i[0], s[0]
        scores = self.emb @ q[0]
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return idx, scores[idx]


# ---------------------------------------------------------------- Store
@dataclass
class LoadedIndex:
    path: Path
    manifest: dict
    chunks: list[Chunk]
    dense: DenseIndex
    bm25: BM25Index


def write_index(
    out_dir: Path, chunks: list[Chunk], embeddings: np.ndarray, manifest: dict, bm25: BM25Index
) -> None:
    """Atomic write: build in a temp dir next to the target, then rename."""
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=out_dir.name + ".tmp-", dir=out_dir.parent))
    os.chmod(tmp, 0o755)
    try:
        with open(tmp / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        np.save(tmp / "embeddings.npy", embeddings.astype(np.float32))
        bm25.save(tmp / "bm25")
        manifest = {**manifest, "n_chunks": len(chunks), "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2))
        if out_dir.exists():
            old = out_dir.with_name(out_dir.name + ".old")
            if old.exists():
                shutil.rmtree(old)
            os.replace(out_dir, old)
            os.replace(tmp, out_dir)
            shutil.rmtree(old)
        else:
            os.replace(tmp, out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def read_manifest(path: Path) -> dict | None:
    m = path / MANIFEST
    return json.loads(m.read_text()) if m.exists() else None


def load_index(path: Path, expected_embedder: dict | None = None, use_faiss: bool = False) -> LoadedIndex:
    path = Path(path)
    manifest = read_manifest(path)
    if manifest is None:
        raise IndexMissingError(
            f"No index found at {path}.\nBuild it first:\n    python -m scripts.build_index\n"
            "(or set MHRAG_INDEX__EMBEDDER / MHRAG_CHUNKING__CHUNK_TOKENS to match an existing index)."
        )
    if manifest.get("n_chunks", 0) <= 0:
        raise IndexMissingError(f"Index at {path} is empty. Rebuild with: python -m scripts.build_index --force")
    if expected_embedder:
        got = manifest["embedder"]
        for field in ("model", "query_prefix", "passage_prefix"):
            if got.get(field) != expected_embedder.get(field):
                raise IndexMismatchError(
                    f"Index at {path} was built with embedder {got.get('model')!r} "
                    f"({field}={got.get(field)!r}) but the server is configured for "
                    f"{expected_embedder.get('model')!r} ({field}={expected_embedder.get(field)!r}).\n"
                    "Rebuild: python -m scripts.build_index --force"
                )
        if got.get("backend") != expected_embedder.get("backend"):
            log.warning(
                "Index embedded with backend %s, queries use %s (same model; tiny numeric differences).",
                got.get("backend"),
                expected_embedder.get("backend"),
            )
    with open(path / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [Chunk.from_dict(json.loads(line)) for line in f]
    emb = np.load(path / "embeddings.npy")
    if emb.shape[0] != len(chunks):
        raise IndexMismatchError(f"Index at {path} is corrupt: {emb.shape[0]} vectors for {len(chunks)} chunks.")
    return LoadedIndex(path, manifest, chunks, DenseIndex(emb, use_faiss), BM25Index.load(path / "bm25"))
