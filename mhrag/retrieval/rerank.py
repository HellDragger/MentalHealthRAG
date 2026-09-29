"""Cross-encoder rerankers (sentence-transformers or fastembed/ONNX)."""

from __future__ import annotations

import numpy as np

from mhrag.config import RerankerCfg


class Reranker:
    backend = "base"

    def __init__(self, key: str, cfg: RerankerCfg):
        self.key = key
        self.cfg = cfg

    def score(self, query: str, passages: list[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class STCrossEncoder(Reranker):
    backend = "sentence_transformers"

    def __init__(self, key, cfg):
        super().__init__(key, cfg)
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(cfg.model, max_length=512)

    def score(self, query, passages):
        if not passages:
            return np.array([], dtype=np.float32)
        return np.asarray(self.model.predict([(query, p) for p in passages], show_progress_bar=False), dtype=np.float32)


class FastEmbedCrossEncoder(Reranker):
    backend = "fastembed"

    def __init__(self, key, cfg):
        super().__init__(key, cfg)
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(model_name=cfg.fastembed_model or cfg.model)

    def score(self, query, passages):
        if not passages:
            return np.array([], dtype=np.float32)
        return np.asarray(list(self.model.rerank(query, passages)), dtype=np.float32)


class OverlapReranker(Reranker):
    """Offline test stand-in: scores by word overlap."""

    backend = "overlap"

    def score(self, query, passages):
        q = set(query.lower().split())
        return np.array([len(q & set(p.lower().split())) / (len(q) or 1) for p in passages], dtype=np.float32)


def make_reranker(key: str, cfg: RerankerCfg, backend: str = "auto") -> Reranker:
    if backend == "hashing":
        return OverlapReranker(key, cfg)
    if backend == "fastembed":
        return FastEmbedCrossEncoder(key, cfg)
    if backend == "sentence_transformers":
        return STCrossEncoder(key, cfg)
    try:
        import sentence_transformers  # noqa: F401

        return STCrossEncoder(key, cfg)
    except ImportError:
        return FastEmbedCrossEncoder(key, cfg)
