"""BM25, dense, hybrid (Reciprocal Rank Fusion) and reranked retrieval over a LoadedIndex."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from mhrag.cache import LRUCache, normalize_query
from mhrag.index.embedders import Embedder
from mhrag.index.store import LoadedIndex
from mhrag.retrieval.rerank import Reranker
from mhrag.types import Chunk

MODES = ("bm25", "dense", "hybrid", "dense_rerank", "hybrid_rerank")


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int
    signals: dict = field(default_factory=dict)


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion (Cormack et al., 2009): score(d) = sum_r 1 / (k + rank_r(d))."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for r, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + r + 1)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


class Retriever:
    def __init__(
        self,
        index: LoadedIndex,
        embedder: Embedder | None,
        reranker: Reranker | None = None,
        candidates: int = 30,
        rrf_k: int = 60,
        cache_size: int = 2048,
    ):
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self.candidates = candidates
        self.rrf_k = rrf_k
        self._qcache = LRUCache(cache_size)  # query embeddings
        self._rcache = LRUCache(cache_size)  # retrieval results
        self.last_timing: dict = {}

    # -- primitives ------------------------------------------------------
    def embed_query(self, q: str) -> np.ndarray:
        key = normalize_query(q)
        v = self._qcache.get(key)
        if v is None:
            v = self.embedder.embed_queries([q])[0]
            self._qcache.put(key, v)
        return v

    def dense(self, q: str, k: int) -> list[tuple[int, float]]:
        idx, sc = self.index.dense.search(self.embed_query(q), k)
        return list(zip(idx.tolist(), sc.tolist(), strict=True))

    def bm25(self, q: str, k: int) -> list[tuple[int, float]]:
        idx, sc = self.index.bm25.search(q, k)
        return list(zip(idx.tolist(), sc.tolist(), strict=True))

    # -- main entry --------------------------------------------------------
    def retrieve(self, query: str, k: int = 5, mode: str = "hybrid_rerank", exclude_doc_ids: set[str] | None = None) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"Unknown retrieval mode {mode!r}; choose from {MODES}")
        if mode.endswith("_rerank") and self.reranker is None:
            raise ValueError(f"Mode {mode} needs a reranker")
        cache_key = (mode, k, normalize_query(query), tuple(sorted(exclude_doc_ids or ())))
        cached = self._rcache.get(cache_key)
        if cached is not None:
            self.last_timing = {"cache_hit": True}
            return cached

        t0 = time.perf_counter()
        n = max(self.candidates, k)
        timing: dict = {}
        signals: dict[int, dict] = {}
        if mode == "bm25":
            fused = self.bm25(query, n)
            for i, s in fused:
                signals.setdefault(i, {})["bm25"] = s
        elif mode in ("dense", "dense_rerank"):
            fused = self.dense(query, n)
            for i, s in fused:
                signals.setdefault(i, {})["dense"] = s
        else:
            d = self.dense(query, n)
            b = self.bm25(query, n)
            for i, s in d:
                signals.setdefault(i, {})["dense"] = s
            for i, s in b:
                signals.setdefault(i, {})["bm25"] = s
            fused = rrf_fuse([[i for i, _ in d], [i for i, _ in b]], self.rrf_k)
        timing["first_stage_ms"] = (time.perf_counter() - t0) * 1000

        if exclude_doc_ids:
            fused = [(i, s) for i, s in fused if self.index.chunks[i].doc_id not in exclude_doc_ids]

        if mode.endswith("_rerank"):
            t1 = time.perf_counter()
            cand = fused[:n]
            rs = self.reranker.score(query, [self.index.chunks[i].embed_text for i, _ in cand])
            for (i, _), r in zip(cand, rs, strict=True):
                signals.setdefault(i, {})["rerank"] = float(r)
            fused = sorted(((i, float(r)) for (i, _), r in zip(cand, rs, strict=True)), key=lambda x: (-x[1], x[0]))
            timing["rerank_ms"] = (time.perf_counter() - t1) * 1000

        hits = [
            Hit(self.index.chunks[i], float(s), r + 1, signals.get(i, {})) for r, (i, s) in enumerate(fused[:k])
        ]
        timing["total_ms"] = (time.perf_counter() - t0) * 1000
        self.last_timing = timing
        self._rcache.put(cache_key, hits)
        return hits

    def cache_stats(self) -> dict:
        return {"query_embeddings": self._qcache.stats(), "results": self._rcache.stats()}
