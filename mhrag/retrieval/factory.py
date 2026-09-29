"""Build a Retriever for a configured (embedder, chunk size) index."""

from __future__ import annotations

from mhrag.config import Settings
from mhrag.index.builder import index_dir
from mhrag.index.embedders import make_embedder
from mhrag.index.store import load_index, read_manifest
from mhrag.retrieval.rerank import make_reranker
from mhrag.retrieval.retriever import Retriever


def build_retriever(settings: Settings, embedder_key: str | None = None, chunk_tokens: int | None = None,
                    variant: str = "", need_reranker: bool = True, reranker_key: str | None = None) -> Retriever:
    key = embedder_key or settings.index.embedder
    path = index_dir(settings, key, chunk_tokens, variant)
    manifest = read_manifest(path)
    backend = settings.embedder_backend
    if backend == "auto" and manifest:
        # Query with the backend the index was built with when it is installed.
        backend_built = manifest["embedder"].get("backend")
        backend = backend_built if backend_built in ("sentence_transformers", "fastembed", "hashing") else "auto"
        if backend == "sentence_transformers":
            try:
                import sentence_transformers  # noqa: F401
            except ImportError:
                backend = "fastembed"
    embedder = make_embedder(key, settings.embedder_cfg(key), backend) if manifest else None
    index = load_index(path, embedder.identity() if embedder else None, settings.index.use_faiss)
    reranker = None
    if need_reranker:
        rk = reranker_key or settings.retrieval.reranker
        rr_backend = "hashing" if backend == "hashing" else ("fastembed" if backend == "fastembed" else "auto")
        reranker = make_reranker(rk, settings.rerankers[rk], rr_backend)
    return Retriever(index, embedder, reranker, settings.retrieval.candidates, settings.retrieval.rrf_k)
