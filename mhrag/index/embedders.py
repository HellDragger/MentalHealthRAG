"""Embedding backends. The same Embedder object embeds passages at build time and queries at run time, and
its identity (model, backend, prefixes, dim) is written to the index manifest (fixes CHANGELOG B3)."""

from __future__ import annotations

import gc
import hashlib
import logging
import re
import sys
from functools import lru_cache

import numpy as np

from mhrag.config import EmbedderCfg

log = logging.getLogger(__name__)


def _l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


class Embedder:
    backend = "base"

    def __init__(self, key: str, cfg: EmbedderCfg):
        self.key = key
        self.cfg = cfg
        self.model_name = cfg.model
        self.dim: int = 0

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def embed_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return _l2norm(self._encode([self.cfg.passage_prefix + t for t in texts], batch_size))

    def embed_queries(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return _l2norm(self._encode([self.cfg.query_prefix + t for t in texts], batch_size))

    def identity(self) -> dict:
        return {
            "key": self.key,
            "model": self.model_name,
            "backend": self.backend,
            "dim": self.dim,
            "query_prefix": self.cfg.query_prefix,
            "passage_prefix": self.cfg.passage_prefix,
            "max_tokens": self.cfg.max_tokens,
        }


class SentenceTransformerEmbedder(Embedder):
    backend = "sentence_transformers"

    def __init__(self, key: str, cfg: EmbedderCfg, device: str | None = None):
        super().__init__(key, cfg)
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(cfg.model, device=device)
        self.model.max_seq_length = min(self.model.max_seq_length or cfg.max_tokens, cfg.max_tokens)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def _encode(self, texts, batch_size):
        while True:
            try:
                return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
            except RuntimeError as e:  # torch.OutOfMemoryError subclasses RuntimeError
                if "out of memory" not in str(e).lower() or batch_size <= 1:
                    raise
                free_accelerator_memory()
                batch_size //= 2
                log.warning("%s: out of GPU memory; retrying with batch size %d", self.key, batch_size)


class FastEmbedEmbedder(Embedder):
    """ONNX runtime, no torch: used by the CPU deployment image."""

    backend = "fastembed"

    def __init__(self, key: str, cfg: EmbedderCfg):
        super().__init__(key, cfg)
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=cfg.model)
        self.dim = int(self._encode(["dimension probe"], 1).shape[1])

    def _encode(self, texts, batch_size):
        # .embed() does not add any prefix itself, so the configured prefixes are applied exactly once.
        return np.stack(list(self.model.embed(texts, batch_size=batch_size)))


class HashingEmbedder(Embedder):
    """Deterministic bag-of-words hashing embedder for offline tests. Not for real use."""

    backend = "hashing"

    def __init__(self, key: str, cfg: EmbedderCfg, dim: int = 256):
        super().__init__(key, cfg)
        self.dim = dim

    def _encode(self, texts, batch_size):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        return out


def fastembed_supports(model: str) -> bool:
    try:
        from fastembed import TextEmbedding
    except ImportError:
        return False
    return any(m["model"] == model for m in TextEmbedding.list_supported_models())


def make_embedder(key: str, cfg: EmbedderCfg, backend: str = "auto") -> Embedder:
    if backend == "hashing":
        return HashingEmbedder(key, cfg)
    if backend == "fastembed":
        return FastEmbedEmbedder(key, cfg)
    if backend == "sentence_transformers":
        return SentenceTransformerEmbedder(key, cfg)
    # auto: prefer sentence-transformers when torch is installed, else fastembed
    try:
        import sentence_transformers  # noqa: F401

        return SentenceTransformerEmbedder(key, cfg)
    except ImportError:
        if fastembed_supports(cfg.model):
            return FastEmbedEmbedder(key, cfg)
        raise RuntimeError(
            f"No embedding backend available for {cfg.model}: install sentence-transformers or fastembed."
        ) from None


def free_accelerator_memory() -> None:
    """Return cached CUDA memory to the device (only if torch is already imported)."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def release_models() -> None:
    """Drop cached embedders and free their GPU memory. Experiments that loop over many models call this between
    models; otherwise every embedder and cross-encoder stays resident and a 16 GB GPU runs out."""
    cached_embedder.cache_clear()
    free_accelerator_memory()


@lru_cache(maxsize=4)
def cached_embedder(key: str, model: str, qp: str, pp: str, max_tokens: int, backend: str) -> Embedder:
    return make_embedder(key, EmbedderCfg(model=model, query_prefix=qp, passage_prefix=pp, max_tokens=max_tokens), backend)
