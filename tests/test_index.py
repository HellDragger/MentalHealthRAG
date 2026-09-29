import json

import pytest

from mhrag.index.builder import build_index
from mhrag.index.embedders import make_embedder
from mhrag.index.store import IndexMismatchError, IndexMissingError, load_index


def test_index_roundtrip_persists(built_index):
    settings, path, stats = built_index
    emb = make_embedder("minilm", settings.embedder_cfg("minilm"), "hashing")
    idx = load_index(path, emb.identity())
    assert len(idx.chunks) == stats["chunking"]["chunks"] > 0
    assert idx.dense.emb.shape[0] == len(idx.chunks)
    m = json.loads((path / "manifest.json").read_text())
    assert m["embedder"]["model"] == settings.embedder_cfg("minilm").model
    assert m["n_chunks"] == len(idx.chunks)


def test_build_is_idempotent(built_index):
    settings, path, _ = built_index
    _, _, built = build_index(settings)
    assert not built
    _, _, built = build_index(settings, force=True)
    assert built


def test_rebuild_when_corpus_config_changes(built_index):
    settings, _, _ = built_index
    settings.ingest.include_faq = False
    _, _, built = build_index(settings)
    assert built


def test_missing_index_raises_actionable_error(tmp_path):
    with pytest.raises(IndexMissingError, match="build_index"):
        load_index(tmp_path / "nope")


def test_embedder_mismatch_detected(built_index):
    settings, path, _ = built_index
    other = make_embedder("e5-base", settings.embedder_cfg("e5-base"), "hashing")
    with pytest.raises(IndexMismatchError):
        load_index(path, other.identity())


def test_every_chunk_has_citation_metadata(built_index):
    settings, path, _ = built_index
    idx = load_index(path)
    for c in idx.chunks:
        assert c.source_file and c.title and c.source_type
        assert c.section is not None
    assert any(c.url and c.url.startswith("https://www.mind.org.uk") for c in idx.chunks)


def test_explicit_backend_change_triggers_rebuild(built_index):
    settings, path, _ = built_index
    _, _, built = build_index(settings, backend="hashing")
    assert not built  # same backend as the existing index -> skipped
    m = json.loads((path / "manifest.json").read_text())
    m["embedder"]["backend"] = "fastembed"  # pretend the index came from another runtime
    (path / "manifest.json").write_text(json.dumps(m))
    _, _, built = build_index(settings, backend="hashing")
    assert built  # explicit backend differs from the manifest -> rebuilt


def test_embedder_halves_batch_on_out_of_memory():
    import numpy as np

    from mhrag.index.embedders import SentenceTransformerEmbedder

    class FakeST:
        def __init__(self):
            self.batches = []

        def encode(self, texts, batch_size, **kw):
            self.batches.append(batch_size)
            if batch_size > 16:
                raise RuntimeError("CUDA out of memory. Tried to allocate 376.00 MiB")
            return np.ones((len(texts), 3), dtype=np.float32)

    emb = SentenceTransformerEmbedder.__new__(SentenceTransformerEmbedder)
    emb.key, emb.model = "fake", FakeST()
    out = emb._encode(["a", "b"], 64)
    assert out.shape == (2, 3) and emb.model.batches == [64, 32, 16]
