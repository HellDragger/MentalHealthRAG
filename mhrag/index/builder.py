"""Idempotent index builder shared by scripts/build_index.py, the eval runner and tests."""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path

from mhrag.config import Settings
from mhrag.index.embedders import Embedder, make_embedder
from mhrag.index.store import BM25Index, index_name, read_manifest, write_index
from mhrag.ingest.chunk import get_token_counter
from mhrag.ingest.corpus import PIPELINE_VERSION, build_chunks, corpus_hash, load_documents
from mhrag.ingest.sources import RawData

log = logging.getLogger(__name__)


def index_dir(settings: Settings, embedder_key: str | None = None, chunk_tokens: int | None = None,
              variant: str = "") -> Path:
    return settings.artifacts_dir / "index" / index_name(
        embedder_key or settings.index.embedder, chunk_tokens or settings.chunking.chunk_tokens, variant
    )


@contextlib.contextmanager
def _build_lock(root: Path):
    """One index build at a time across processes (e.g. one generation process per GPU); the second process then
    finds the index up to date instead of racing the first."""
    root.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:  # not POSIX: no cross-process lock
        yield
        return
    with open(root / ".build.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def build_index(
    settings: Settings,
    embedder_key: str | None = None,
    chunk_tokens: int | None = None,
    variant: str = "",
    force: bool = False,
    backend: str | None = None,
    embedder: Embedder | None = None,
) -> tuple[Path, dict, bool]:
    """Build (or skip) one index. Returns (path, stats, built)."""
    with _build_lock(settings.artifacts_dir / "index"):
        return _build_index(settings, embedder_key, chunk_tokens, variant, force, backend, embedder)


def _build_index(settings, embedder_key, chunk_tokens, variant, force, backend, embedder) -> tuple[Path, dict, bool]:
    key = embedder_key or settings.index.embedder
    ecfg = settings.embedder_cfg(key)
    chunking = settings.chunking.model_copy(update={"chunk_tokens": chunk_tokens or settings.chunking.chunk_tokens})
    backend = backend or settings.embedder_backend
    out = index_dir(settings, key, chunking.chunk_tokens, variant)

    rd = RawData(settings.resolve(settings.paths.raw_data))
    tok_model = None if backend == "hashing" else ecfg.model
    count = get_token_counter(tok_model)
    chash = corpus_hash(rd, settings.ingest, chunking, tok_model if count.exact else "approx")

    old = read_manifest(out)
    if (
        not force
        and old
        and old.get("corpus_hash") == chash
        and old.get("embedder", {}).get("model") == ecfg.model
        and old.get("embedder", {}).get("query_prefix") == ecfg.query_prefix
        and old.get("embedder", {}).get("passage_prefix") == ecfg.passage_prefix
        # an explicitly requested backend (e.g. fastembed for the CPU Space) must match too; "auto" accepts any
        and (backend == "auto" or old.get("embedder", {}).get("backend") == backend)
    ):
        log.info("Index %s is up to date (corpus hash %s); skipping.", out, chash)
        return out, old.get("stats", {}), False

    t0 = time.perf_counter()
    corpus = load_documents(rd, settings.ingest)
    # Reserve room for the passage prefix and [CLS]/[SEP].
    max_tok = ecfg.max_tokens - count(ecfg.passage_prefix) - 2
    chunks, chunk_stats = build_chunks(corpus, chunking, count, max_tok, settings.ingest.near_dup_threshold)
    t_chunk = time.perf_counter() - t0

    emb_model = embedder or make_embedder(key, ecfg, backend)
    t1 = time.perf_counter()
    vectors = emb_model.embed_passages([c.embed_text for c in chunks], settings.index.embed_batch_size)
    t_embed = time.perf_counter() - t1
    bm25 = BM25Index.build([c.embed_text for c in chunks])

    stats = {
        "ingest": corpus.stats,
        "chunking": chunk_stats,
        "timing_seconds": {"load_and_chunk": round(t_chunk, 2), "embed": round(t_embed, 2)},
        "tokenizer_exact": count.exact,
    }
    manifest = {
        "index_name": out.name,
        "pipeline_version": PIPELINE_VERSION,
        "corpus_hash": chash,
        "embedder": emb_model.identity(),
        "chunking": chunking.model_dump(),
        "ingest": settings.ingest.model_dump(),
        "stats": stats,
    }
    write_index(out, chunks, vectors, manifest, bm25)
    log.info("Built %s: %d chunks in %.1fs (embed %.1fs)", out, len(chunks), time.perf_counter() - t0, t_embed)
    return out, stats, True
