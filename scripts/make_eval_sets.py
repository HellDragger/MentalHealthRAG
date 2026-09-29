"""Build the evaluation sets in eval/data/.

    python -m scripts.make_eval_sets                                   # HeadingQA, ParaphraseQA, FAQ-Gen, Counsel-Gen
    python -m scripts.make_eval_sets --synth --synth-model qwen2.5-1.5b-gguf --synth-n 200
    python -m scripts.make_eval_sets --synth --synth-model llama-3.3-70b-groq   # stronger generator (API key)

Sets
- heading_qa.jsonl:     question-style section headings prefixed with the document topic; gold = that section.
                        No LLM involved -> fully reproducible. Caveat: chunks carry their section heading in the
                        embedded header, which favours lexical matching (reported in the paper).
- paraphrase_qa.jsonl:  KB.json fact / mentalhealth.json intent patterns; gold = the matching fact/FAQ documents.
- faq_gen.jsonl:        the 98 FAQ questions with reference answers, for generation with the FAQ *excluded* from
                        the index (index variant "nofaq").
- counsel_gen.jsonl:    100 held-out CounselChat contexts (hash split, never indexed) with all counsellor responses.
- synth_qa.jsonl:       LLM-written questions for sampled chunks, kept only if round-trip retrieval (hybrid, top-10)
                        finds the source span. synth_qa_verify.csv: a 100-question subset for human verification.
- oos_questions.jsonl:  50 hand-written out-of-scope / unanswerable / adversarial questions (checked in).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import Counter

from eval.datasets import EVAL_DATA, heading_qa, is_relevant, paraphrase_qa, write_jsonl
from mhrag.config import get_settings
from mhrag.ingest import qa
from mhrag.ingest.corpus import load_documents
from mhrag.ingest.sources import RawData

log = logging.getLogger("make_eval_sets")
SEED = 42

SYNTH_PROMPT = """Here is a passage from a mental-health information leaflet.

Passage:
\"\"\"{passage}\"\"\"

Write ONE question that a member of the public might type into a mental-health chatbot and that this passage answers.
- Ask it naturally, in your own words; do not copy long phrases from the passage.
- Do not mention "the passage", "the text" or the leaflet.
- Output only the question."""


def build_static(s) -> dict:
    rd = RawData(s.resolve(s.paths.raw_data))
    corpus = load_documents(rd, s.ingest)
    docs = corpus.documents
    stats = {}

    hq = heading_qa(docs)
    write_jsonl("heading_qa.jsonl", hq)
    stats["heading_qa"] = {"n": len(hq), "by_topic_top10": Counter(x["topic"] for x in hq).most_common(10)}

    faq_docs = [d for d in docs if d.source_type == "faq"]
    kb_docs = [d for d in docs if d.source_type == "kb_fact"]
    kb_int = qa.load_intents(rd, qa.KB_FILE)
    mh_int = qa.load_intents(rd, qa.MH_JSON_FILE)
    pq = paraphrase_qa(kb_int, mh_int, faq_docs, kb_docs)
    write_jsonl("paraphrase_qa.jsonl", pq)
    stats["paraphrase_qa"] = {"n": len(pq), "by_source_file": Counter(x["source"].split(":")[0] for x in pq)}

    fg = [{"id": f"fg{i:03d}", "query": d.meta["question"], "reference": d.meta["answer"], "faq_doc_id": d.doc_id}
          for i, d in enumerate(faq_docs)]
    write_jsonl("faq_gen.jsonl", fg)
    stats["faq_gen"] = {"n": len(fg)}

    cp = qa.load_counselling_pairs(rd)
    test = cp[cp.split == "test"]
    ctxs = sorted(test.Context.unique())
    random.Random(SEED).shuffle(ctxs)
    cg = [{"id": f"cg{i:03d}", "query": c, "references": test[test.Context == c].Response.tolist()}
          for i, c in enumerate(ctxs[:100])]
    write_jsonl("counsel_gen.jsonl", cg)
    stats["counsel_gen"] = {"n": len(cg), "test_split_contexts": len(ctxs),
                            "references_per_query_mean": sum(len(x["references"]) for x in cg) / max(1, len(cg))}
    return stats


def build_synth(s, model: str, n: int) -> dict:
    from mhrag.llm.base import GenerationParams
    from mhrag.llm.registry import ModelManager, load_catalog
    from mhrag.retrieval.factory import build_retriever

    retriever = build_retriever(s, need_reranker=False)
    chunks = [c for c in retriever.index.chunks if c.source_type in ("mind_web", "mind_booklet") and c.n_tokens >= 80]
    rng = random.Random(SEED)
    sample = rng.sample(chunks, min(n, len(chunks)))
    mm = ModelManager(load_catalog(), [model], model)
    backend = mm.get(model)
    params = GenerationParams(max_new_tokens=64, temperature=0.0)
    kept, dropped = [], 0
    for i, c in enumerate(sample):
        q = backend.generate([{"role": "user", "content": SYNTH_PROMPT.format(passage=c.text)}], params).strip()
        q = q.strip('"').splitlines()[0].strip() if q else ""
        if not q or len(q.split()) < 4:
            dropped += 1
            continue
        gold = {"span": {"doc_id": c.doc_id, "text": c.text}}
        hits = retriever.retrieve(q, 10, "hybrid")
        rank = next((h.rank for h in hits if is_relevant(h.chunk, gold)), None)
        rec = {"id": f"sq{i:04d}", "query": q, "gold": gold, "source_chunk": c.chunk_id, "source_title": c.title,
               "source_section": c.section, "generator": model, "roundtrip_rank": rank}
        if rank is None:
            dropped += 1
            continue
        kept.append(rec)
        if (i + 1) % 25 == 0:
            log.info("synth %d/%d kept=%d", i + 1, len(sample), len(kept))
    write_jsonl("synth_qa.jsonl", kept)
    verify = rng.sample(kept, min(100, len(kept)))
    with open(EVAL_DATA / "synth_qa_verify.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "passage_title", "passage_section", "passage", "answerable_from_passage (y/n)",
                    "natural_question (y/n)", "notes"])
        for r in verify:
            chunk = next(c for c in sample if c.chunk_id == r["source_chunk"])
            w.writerow([r["id"], r["query"], r["source_title"], r["source_section"], chunk.text, "", "", ""])
    return {"sampled": len(sample), "kept_after_roundtrip": len(kept), "dropped": dropped, "generator": model,
            "roundtrip": "hybrid (BM25+dense RRF) top-10 on the default index",
            "human_verified_subset": "TODO(human): fill eval/data/synth_qa_verify.csv"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--synth-model", default="qwen2.5-1.5b-gguf")
    ap.add_argument("--synth-n", type=int, default=200)
    ap.add_argument("--skip-static", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    s = get_settings()
    stats_path = s.results_dir / "eval_sets.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    if not args.skip_static:
        stats.update(build_static(s))
    if args.synth:
        stats["synth_qa"] = build_synth(s, args.synth_model, args.synth_n)
    stats["oos"] = {"n": len((EVAL_DATA / "oos_questions.jsonl").read_text(encoding="utf-8").splitlines())}
    s.results_dir.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
