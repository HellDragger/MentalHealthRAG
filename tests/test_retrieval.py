import pytest

from mhrag.retrieval.factory import build_retriever
from mhrag.retrieval.retriever import rrf_fuse
from mhrag.retrieval.rewrite import heuristic_rewrite


@pytest.fixture()
def retriever(built_index):
    settings, _, _ = built_index
    return build_retriever(settings)


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "hybrid_rerank", "dense_rerank"])
def test_modes_find_relevant_chunk(retriever, mode):
    hits = retriever.retrieve("how can I improve my sleep routine", 3, mode)
    assert hits and "Sleep" in hits[0].chunk.title
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_query_embedding_and_results_cached(retriever):
    retriever.retrieve("anger management", 2, "hybrid")
    retriever.retrieve("Anger   management", 2, "hybrid")
    st = retriever.cache_stats()
    assert st["results"]["hits"] >= 1


def test_exclude_doc_ids(retriever):
    hits = retriever.retrieve("support group", 5, "bm25")
    faq_ids = {h.chunk.doc_id for h in hits if h.chunk.source_type == "faq"}
    assert faq_ids
    hits2 = retriever.retrieve("support group", 5, "bm25", exclude_doc_ids=faq_ids)
    assert not {h.chunk.doc_id for h in hits2} & faq_ids


def test_rrf_fuse():
    fused = rrf_fuse([[1, 2, 3], [3, 1]], k=60)
    assert fused[0][0] == 1  # ranked 1st and 2nd
    assert abs(fused[0][1] - (1 / 61 + 1 / 62)) < 1e-9


def test_unknown_mode(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("x", 1, "magic")


def test_heuristic_rewrite_adds_context_for_followups():
    hist = [{"role": "user", "content": "What is obsessive compulsive disorder?"}, {"role": "assistant", "content": "..."}]
    q = heuristic_rewrite("how is it treated?", hist)
    assert "obsessive" in q.lower()
    assert heuristic_rewrite("What are the symptoms of bipolar disorder in adults?", hist).startswith("What are")
