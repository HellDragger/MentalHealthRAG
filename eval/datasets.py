"""Evaluation-set construction and chunk-size-independent relevance.

Gold labels are defined on *documents/sections/text spans*, not chunk ids, so the same query set scores every
index (any embedder, any chunk size):
- {"doc_ids": [...]}                          any chunk from one of these documents is relevant
- {"sections": [[source_file, heading], ...]} a chunk is relevant if it covers one of the sections
- {"span": {"doc_id": ..., "text": ...}}      a chunk is relevant if it contains >= 30 % of the span's 5-shingles
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from mhrag.config import PROJECT_ROOT
from mhrag.ingest.dedup import shingles
from mhrag.types import Chunk, Document

EVAL_DATA = PROJECT_ROOT / "eval" / "data"


def topic_of(doc: Document) -> str:
    """Mind URLs look like .../information-support/<section>/<topic>/<page>/ -> use <topic>."""
    if doc.source_type == "mind_web" and doc.url:
        parts = [p for p in doc.url.split("/") if p]
        for anchor in ("types-of-mental-health-problems", "tips-for-everyday-living", "drugs-and-treatments"):
            if anchor in parts and parts.index(anchor) + 1 < len(parts):
                return parts[parts.index(anchor) + 1].replace("-", " ").capitalize()
    return doc.title


def heading_qa(docs: list[Document]) -> list[dict]:
    """Question-style section headings, prefixed with the document topic, as queries."""
    items: dict[str, dict] = {}
    for d in docs:
        if d.source_type not in ("mind_web", "mind_booklet", "web_article"):
            continue
        topic = topic_of(d)
        for s in d.sections:
            h = s.heading.strip()
            if not h.endswith("?") or len(h.split()) < 3 or h == d.title or len(s.text.split()) < 15:
                continue
            q = f"{topic}: {h}"
            it = items.setdefault(q.lower(), {"query": q, "gold": {"sections": []}, "topic": topic})
            pair = [d.source_file, h]
            if pair not in it["gold"]["sections"]:
                it["gold"]["sections"].append(pair)
    out = []
    for i, it in enumerate(sorted(items.values(), key=lambda x: x["query"])):
        out.append({"id": f"hq{i:04d}", **it})
    return out


def paraphrase_qa(kb_intents, mh_intents, faq_docs: list[Document], kb_docs: list[Document]) -> list[dict]:
    """Intent patterns as queries; gold = the fact document and any FAQ document with the same answer."""
    faq_sh = {d.doc_id: shingles(d.sections[0].text) for d in faq_docs}
    faq_by_q = {re.sub(r"\W+", " ", d.sections[0].heading.lower()).strip(): d.doc_id for d in faq_docs}

    def faq_matches(text: str) -> list[str]:
        sh = shingles(text)
        return [k for k, v in faq_sh.items() if v and len(sh & v) / max(1, min(len(sh), len(v))) >= 0.5]

    kb_ids = {d.doc_id for d in kb_docs}
    out = []
    for it in kb_intents:
        if not it.tag.startswith("fact-"):
            continue
        gold = ([f"kb:{it.tag}"] if f"kb:{it.tag}" in kb_ids else []) + faq_matches(" ".join(it.responses))
        # an FAQ entry asking exactly the same question is also a correct answer source
        gold += [faq_by_q[k] for k in (re.sub(r"\W+", " ", p.lower()).strip() for p in it.patterns) if k in faq_by_q]
        if not gold:
            continue
        for p in it.patterns:
            out.append({"query": p, "gold": {"doc_ids": sorted(set(gold))}, "source": f"KB.json:{it.tag}"})
    for it in mh_intents:
        gold = set(faq_matches(" ".join(it.responses)))
        for p in it.patterns:
            key = re.sub(r"\W+", " ", p.lower()).strip()
            if key in faq_by_q:
                gold.add(faq_by_q[key])
        if not gold:
            continue
        for p in it.patterns:
            out.append({"query": p, "gold": {"doc_ids": sorted(gold)}, "source": f"mentalhealth.json:{it.tag}"})
    # de-duplicate identical queries (merge gold)
    merged: dict[str, dict] = {}
    for o in out:
        k = o["query"].strip().lower()
        if k in merged:
            merged[k]["gold"]["doc_ids"] = sorted(set(merged[k]["gold"]["doc_ids"]) | set(o["gold"]["doc_ids"]))
        else:
            merged[k] = o
    return [{"id": f"pq{i:03d}", **o} for i, o in enumerate(merged.values())]


def is_relevant(chunk: Chunk, gold: dict, _cache: dict | None = None) -> bool:
    if "doc_ids" in gold:
        return chunk.doc_id in gold["doc_ids"]
    if "sections" in gold:
        return any(chunk.source_file == sf and h in chunk.sections for sf, h in gold["sections"])
    if "span" in gold:
        sp = gold["span"]
        if chunk.doc_id != sp["doc_id"]:
            return False
        key = sp["text"]
        if _cache is not None and key in _cache:
            gs = _cache[key]
        else:
            gs = shingles(key)
            if _cache is not None:
                _cache[key] = gs
        cs = shingles(chunk.text)
        return bool(gs) and len(gs & cs) / len(gs) >= 0.3
    raise ValueError(f"bad gold spec {gold}")


def n_relevant(chunks: list[Chunk], gold: dict, cache: dict | None = None) -> int:
    return sum(is_relevant(c, gold, cache) for c in chunks)


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.is_absolute():
        path = EVAL_DATA / path
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = EVAL_DATA / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path
