"""Build the knowledge-base documents and chunks from the raw data according to the ingest config."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass, field

from mhrag.config import ChunkingCfg, IngestCfg
from mhrag.ingest import pdf, qa
from mhrag.ingest.chunk import chunk_document
from mhrag.ingest.dedup import drop_near_duplicates, jaccard, shingles
from mhrag.ingest.sources import RawData
from mhrag.types import Chunk, Document

PIPELINE_VERSION = "2.0.1"  # 2.0.1: mojibake repair in FAQ/intents  # bump when ingestion logic changes -> forces index rebuilds


@dataclass
class Corpus:
    documents: list[Document]
    stats: dict = field(default_factory=dict)


def source_files(rd: RawData, cfg: IngestCfg) -> list[str]:
    names: list[str] = []
    if cfg.include_pdfs:
        names += rd.list("PDF_Files", ".pdf")
    if cfg.include_faq and not cfg.exclude_faq_from_index:
        names.append(qa.FAQ_FILE)
    if cfg.include_kb_facts:
        names.append(qa.KB_FILE)
    if cfg.include_counselling:
        names.append(qa.COUNSEL_FILE)
    return names


def corpus_hash(rd: RawData, ingest: IngestCfg, chunking: ChunkingCfg, tokenizer: str | None) -> str:
    h = hashlib.sha256()
    h.update(rd.content_hash(source_files(rd, ingest)).encode())
    h.update(json.dumps(ingest.model_dump(), sort_keys=True).encode())
    h.update(json.dumps(chunking.model_dump(), sort_keys=True).encode())
    h.update(f"{tokenizer}|{PIPELINE_VERSION}".encode())
    return h.hexdigest()[:20]


def load_documents(rd: RawData, cfg: IngestCfg) -> Corpus:
    t0 = time.perf_counter()
    docs: list[Document] = []
    stats: dict = {}

    if cfg.include_pdfs:
        names = rd.list("PDF_Files", ".pdf")
        raws, skipped = [], []
        t_pdf = time.perf_counter()
        for n in names:
            r = pdf.parse_pdf(n, pdf.extract_pages(rd.read(n)))
            if r is None:
                skipped.append(n)
            elif r.source_type == "web_article" and not cfg.include_web_articles:
                continue
            else:
                raws.append(r)
        pdf_seconds = time.perf_counter() - t_pdf
        removed = pdf.remove_cross_doc_boilerplate(raws, min_docs=cfg.boilerplate_min_docs)
        pdf_docs = [pdf.to_document(r) for r in raws]
        docs += pdf_docs
        stats["pdf"] = {
            "files": len(names),
            "parsed": len(raws),
            "skipped_no_text_layer": skipped,
            "by_source_type": dict(Counter(r.source_type for r in raws)),
            "pages": sum(r.n_pages for r in raws),
            "cross_doc_boilerplate_lines_removed": sum(removed.values()),
            "sections": sum(len(d.sections) for d in pdf_docs),
            "extract_seconds_pypdfium2": round(pdf_seconds, 2),
            "web_articles": sorted(r.source_file for r in raws if r.source_type == "web_article"),
        }

    faq_docs: list[Document] = []
    if cfg.include_faq:
        faq_docs = qa.load_faq(rd)
        stats["faq"] = {"questions": len(faq_docs), **qa.faq_duplicate_stats(rd)}
        if not cfg.exclude_faq_from_index:
            docs += faq_docs
        else:
            stats["faq"]["excluded_from_index"] = True

    if cfg.include_kb_facts:
        kb_docs, kb_stats = qa.load_kb_facts(rd)
        if cfg.exclude_faq_from_index and faq_docs:
            # KB facts fact-8..24 copy FAQ answers; drop them too or the FAQ-Gen experiment would leak answers.
            faq_sh = [shingles(d.sections[0].text) for d in faq_docs]
            leaked = [
                d.doc_id
                for d in kb_docs
                if max(jaccard(shingles(d.sections[0].text), s) for s in faq_sh) >= 0.5
            ]
            kb_docs = [d for d in kb_docs if d.doc_id not in leaked]
            kb_stats["dropped_as_faq_leak"] = leaked
        docs += kb_docs
        stats["kb"] = {**kb_stats, "indexed_facts": len(kb_docs)}

    if cfg.include_counselling:
        c_docs = qa.load_counselling(rd)
        docs += c_docs
        stats["counselling"] = {"indexed_contexts": len(c_docs), **qa.counselling_duplicate_stats(rd)}

    stats["documents"] = len(docs)
    stats["documents_by_source_type"] = dict(Counter(d.source_type for d in docs))
    stats["load_seconds"] = round(time.perf_counter() - t0, 2)
    return Corpus(docs, stats)


def build_chunks(
    corpus: Corpus, chunking: ChunkingCfg, count, max_tokens: int, near_dup_threshold: float
) -> tuple[list[Chunk], dict]:
    size = min(chunking.chunk_tokens, max_tokens - 8)  # small margin: joined text can tokenize slightly longer
    chunks: list[Chunk] = []
    for d in corpus.documents:
        chunks += chunk_document(d, size, chunking.overlap_ratio, count, chunking.min_chunk_tokens)
    n_before = len(chunks)
    # exact duplicates first
    seen, uniq = set(), []
    for c in chunks:
        key = " ".join(c.text.lower().split())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    n_exact = n_before - len(uniq)
    chunks, dup_log = drop_near_duplicates(uniq, near_dup_threshold)
    toks = sorted(c.n_tokens for c in chunks)
    stats = {
        "chunk_tokens": size,
        "overlap_ratio": chunking.overlap_ratio,
        "chunks_before_dedup": n_before,
        "exact_duplicates_removed": n_exact,
        "near_duplicates_removed": len(dup_log),
        "chunks": len(chunks),
        "chunks_by_source_type": dict(Counter(c.source_type for c in chunks)),
        "tokens_per_chunk": {
            "min": toks[0] if toks else 0,
            "median": toks[len(toks) // 2] if toks else 0,
            "max": toks[-1] if toks else 0,
            "mean": round(sum(toks) / len(toks), 1) if toks else 0,
        },
        "near_duplicate_examples": dup_log[:20],
    }
    return chunks, stats
