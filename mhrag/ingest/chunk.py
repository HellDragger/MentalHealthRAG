"""Heading-aware, token-based chunking.

Sections (heading + body) are the unit of meaning. Each section is split on sentence boundaries into chunks
of at most `chunk_tokens` tokens with ~`overlap_ratio` overlap. Consecutive short sections of the same
document are merged so chunks are not tiny; a merged chunk records every heading it covers in `sections`.
Token counts use the embedder's own tokenizer (all five supported embedders share BERT-uncased WordPiece),
and the limit is clipped to the embedder's max sequence length so nothing is silently truncated.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from functools import lru_cache

from mhrag.types import Chunk, Document

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"“(A-Z0-9])|\n+")


class TokenCounter:
    """Counts tokens with a HF `tokenizers` tokenizer; falls back to a word-based estimate offline."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name
        self._tok = None
        if model_name:
            try:
                from tokenizers import Tokenizer

                self._tok = Tokenizer.from_pretrained(model_name)
                self._tok.no_truncation()
                self._tok.no_padding()
            except Exception:  # offline / unknown model
                self._tok = None

    @property
    def exact(self) -> bool:
        return self._tok is not None

    def __call__(self, text: str) -> int:
        if self._tok is not None:
            return len(self._tok.encode(text, add_special_tokens=False).ids)
        return int(len(re.findall(r"\w+|[^\w\s]", text)) * 1.1) + 1


@lru_cache(maxsize=8)
def get_token_counter(model_name: str | None) -> TokenCounter:
    return TokenCounter(model_name)


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    return parts


def _chunk_id(doc_id: str, text: str) -> str:
    return hashlib.sha1(f"{doc_id}\n{text}".encode()).hexdigest()[:16]


def _hard_split(sentence: str, limit: int, count: Callable[[str], int]) -> list[str]:
    """Split an over-long sentence on word boundaries."""
    words, out, cur = sentence.split(), [], []
    for w in words:
        if cur and count(" ".join(cur + [w])) > limit:
            out.append(" ".join(cur))
            cur = []
        cur.append(w)
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_document(
    doc: Document,
    chunk_tokens: int,
    overlap_ratio: float,
    count: Callable[[str], int],
    min_chunk_tokens: int = 24,
    header_tokens: int = 16,
) -> list[Chunk]:
    # Reserve room for the contextual header ("Title — Section") that is embedded with each chunk.
    if doc.sections:
        header_tokens = max(header_tokens, min(64, max(count(f"{doc.title} — {s.heading}") for s in doc.sections) + 2))
    limit = max(32, chunk_tokens - header_tokens)
    overlap = int(limit * overlap_ratio)

    # 1) sentence units tagged with their section heading
    units: list[tuple[str, str, int]] = []  # (heading, sentence, n_tokens)
    for sec in doc.sections:
        for s in split_sentences(sec.text):
            n = count(s)
            if n > limit:
                for piece in _hard_split(s, limit, count):
                    units.append((sec.heading, piece, count(piece)))
            else:
                units.append((sec.heading, s, n))

    # 2) greedy packing; a new section starts a new chunk unless the current chunk is still small
    chunks: list[Chunk] = []
    cur: list[tuple[str, str, int]] = []
    cur_tok = 0

    def flush():
        nonlocal cur, cur_tok
        if not cur:
            return
        headings = list(dict.fromkeys(h for h, _, _ in cur))
        body_lines: list[str] = []
        last_h = None
        for h, s, _ in cur:
            if h != last_h and len(headings) > 1:
                body_lines.append(f"\n{h}\n")
            body_lines.append(s)
            last_h = h
        body = re.sub(r"\n{3,}", "\n\n", " ".join(body_lines).replace(" \n", "\n").replace("\n ", "\n")).strip()
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(doc.doc_id, body),
                doc_id=doc.doc_id,
                text=body,
                title=doc.title,
                section=headings[0],
                sections=headings,
                source_file=doc.source_file,
                source_type=doc.source_type,
                url=doc.url,
                url_verified=doc.url_verified,
                year=doc.year,
                n_tokens=cur_tok,
            )
        )
        # overlap: carry trailing sentences of the same section
        carry, carry_tok = [], 0
        for u in reversed(cur):
            if u[0] != cur[-1][0]:
                break
            # always allow one trailing sentence if it is short relative to the chunk, so overlap exists
            fits = carry_tok + u[2] <= overlap or (not carry and u[2] <= limit // 3)
            if not fits:
                break
            carry.insert(0, u)
            carry_tok += u[2]
        cur, cur_tok = (carry, carry_tok) if carry_tok < cur_tok else ([], 0)

    prev_heading = None
    for h, s, n in units:
        new_section = prev_heading is not None and h != prev_heading
        if new_section:
            n += count(h) + 1  # merged chunks repeat the heading inline
        if cur and (cur_tok + n > limit or (new_section and cur_tok >= min(limit // 2, 128))):
            flush()
            if new_section:
                cur, cur_tok = [], 0  # never carry overlap across a section boundary
            while cur and cur_tok + n > limit:  # the overlap must not push the next chunk over the limit
                cur_tok -= cur.pop(0)[2]
        cur.append((h, s, n))
        cur_tok += n
        prev_heading = h
    flush()
    # drop fragments that are too small to be useful on their own (e.g. a stray caption)
    return [c for c in chunks if c.n_tokens >= min_chunk_tokens or len(chunks) == 1]
