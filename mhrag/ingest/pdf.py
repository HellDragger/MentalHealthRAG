"""PDF loading with pypdfium2.

The 134 PDFs fall into four groups (see results/ingest_stats.json):
- mind_web:     mind.org.uk pages printed from a browser; every page carries a "date time <Title> - Mind" header
                and the page URL with an "n/N" counter. The URL becomes citation metadata.
- mind_booklet: Mind's downloadable booklets ("© Mind 20XX" on every page, a table of contents, no URL).
- web_article:  other printed web pages (news/blog/hospital pages).
- image-only files with no text layer are skipped and reported.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import pypdfium2 as pdfium

from mhrag.ingest import clean
from mhrag.types import Document, Section

MIND_INFO_URL = "https://www.mind.org.uk/information-support/"
MIN_TEXT_CHARS = 200


@dataclass
class RawPdf:
    source_file: str
    title: str
    source_type: str
    url: str | None
    year: int | None
    lines: list[str]
    known_headings: set[str] = field(default_factory=set)
    n_pages: int = 0


def extract_pages(data: bytes) -> list[str]:
    doc = pdfium.PdfDocument(data)
    try:
        pages = []
        for i in range(len(doc)):
            page = doc[i]
            tp = page.get_textpage()
            pages.append(tp.get_text_range())
            tp.close()
            page.close()
        return pages
    finally:
        doc.close()


def _title_from_filename(name: str) -> str:
    stem = PurePosixPath(name).stem
    stem = re.sub(r"[-_]+", " ", stem)
    stem = re.sub(r"\b(pdf|version|download|downloadable|for|20\d\d)\b", " ", stem, flags=re.I)
    stem = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem)  # CamelCase -> words
    stem = re.sub(r"\d+$", "", stem.strip())
    return re.sub(r"\s+", " ", stem).strip().capitalize()


def parse_pdf(source_file: str, pages: list[str]) -> RawPdf | None:
    """Classify a PDF, pull citation metadata out of headers and strip headers/footers.

    Returns None for image-only files.
    """
    pages = [clean.normalize_unicode(p) for p in pages]
    full = "\n".join(pages)
    if len(full.strip()) < MIN_TEXT_CHARS:
        return None

    first_lines = [ln.strip() for ln in pages[0].split("\n") if ln.strip()][:4]
    header = clean.DATE_HEADER.match(first_lines[0]) if first_lines else None
    is_booklet = any(clean.BOOKLET_COPYRIGHT.match(ln) for ln in first_lines) or bool(
        re.search(r"^©\s*Mind\s+\d{4}$", full, re.M)
    )

    title: str | None = None
    url: str | None = None
    year: int | None = None
    if header:
        title = header.group("title").strip()
        for ln in full.split("\n"):
            m = clean.URL_LINE.match(ln.strip())
            if m:
                url = m.group("url")
                break
        is_mind = title.endswith(" - Mind") or (url is not None and "mind.org.uk" in url)
        title = re.sub(r"\s+-\s+Mind$", "", title)
        source_type = "mind_web" if is_mind else "web_article"
    elif is_booklet:
        source_type = "mind_booklet"
        m = re.search(r"^©\s*Mind\s+(\d{4})$", full, re.M)
        year = int(m.group(1)) if m else None
    else:
        source_type = "web_article"

    known: set[str] = set()
    lines: list[str] = []
    for page in pages:
        for raw in page.split("\n"):
            s = raw.strip()
            if not s:
                continue
            if clean.TOC_LINE.search(s):
                known.add(re.sub(r"\s*\.{4,}\s*\d+\s*$", "", s).strip().lower().rstrip("?"))
                continue
            if (
                clean.DATE_HEADER.match(s)
                or clean.URL_LINE.match(s)
                or clean.URL_CONT.match(s)
                or clean.PAGE_COUNTER.match(s)
                or clean.BOOKLET_COPYRIGHT.match(s)
                or (source_type == "mind_booklet" and clean.PAGE_NUMBER.match(s))
                or clean.is_boilerplate(s)
            ):
                continue
            lines.append(s)

    # Welsh-translation notice spans two lines; remove on the joined text then re-split.
    joined = clean.WELSH_BOILERPLATE.sub("", "\n".join(lines))
    lines = [ln for ln in joined.split("\n") if ln.strip()]

    if title is None:
        # Booklet covers wrap the title over 1-3 short lines ("Body dysmorphic" / "disorder").
        parts: list[str] = []
        for ln in lines[:3]:
            if len(ln) > 45 or ln.startswith(("Explains", "This ")) or ln.endswith("."):
                break
            parts.append(ln)
        title = " ".join(parts) if parts else _title_from_filename(source_file)
        lines = lines[len(parts):]
    if source_type == "mind_booklet":
        url = None  # booklets carry no URL; the UI links to Mind's information hub instead
    return RawPdf(
        source_file=source_file,
        title=title,
        source_type=source_type,
        url=url,
        year=year,
        lines=lines,
        known_headings=known,
        n_pages=len(pages),
    )


def _intro_zone(p: RawPdf) -> int:
    """Index of the first question-style heading; lines before it are the page intro."""
    for i, ln in enumerate(p.lines):
        if ln.endswith("?") and len(ln.split()) <= 16:
            return i
    return 0


def remove_cross_doc_boilerplate(pdfs: list[RawPdf], min_docs: int = 3, min_len: int = 30) -> dict[str, int]:
    """Drop lines that recur in >= min_docs documents (e.g. the topic intro repeated on every sub-page
    anger, anger2 ... anger7), keeping the first occurrence. Long lines are checked everywhere; short lines
    only inside a page's intro zone (before its first heading), so repeated headings elsewhere survive.
    Returns counts of removed lines per file."""

    def candidates(p: RawPdf) -> set[str]:
        zone = _intro_zone(p)
        return {ln for i, ln in enumerate(p.lines) if len(ln) >= min_len or i < zone}

    doc_freq: Counter[str] = Counter()
    for p in pdfs:
        doc_freq.update(candidates(p))
    seen: set[str] = set()
    removed: dict[str, int] = {}
    for p in pdfs:
        cand = candidates(p)
        kept = []
        for ln in p.lines:
            if ln in cand and doc_freq[ln] >= min_docs:
                if ln in seen:
                    removed[p.source_file] = removed.get(p.source_file, 0) + 1
                    continue
                seen.add(ln)
            kept.append(ln)
        p.lines = kept
    return removed


def to_document(p: RawPdf) -> Document:
    """Group lines into (heading, text) sections."""
    sections: list[Section] = []
    heading = p.title
    buf: list[str] = []
    lines = p.lines
    # Skip a leading line identical to the title.
    start = 1 if lines and lines[0].strip().lower() == p.title.lower() else 0
    for i in range(start, len(lines)):
        ln = lines[i]
        prev = lines[i - 1] if i > start else None
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        if clean.looks_like_heading(ln, prev, nxt, p.known_headings):
            head = ln.strip()
            # A wrapped heading ("What are the signs and symptoms of" / "BDD?"): pull the first half back.
            if len(head.split()) <= 3 and buf and buf[-1][-1:] not in ".?!:”\"" and len(buf[-1]) < 80:
                head = buf.pop().strip() + " " + head
            if buf:
                sections.append(Section(heading=heading, text=clean.join_lines(buf)))
            heading = head
            buf = []
        else:
            buf.append(ln)
    if buf:
        sections.append(Section(heading=heading, text=clean.join_lines(buf)))
    sections = [s for s in sections if s.text.strip()]
    doc_id = re.sub(r"[^a-z0-9]+", "-", PurePosixPath(p.source_file).stem.lower()).strip("-")
    return Document(
        doc_id=f"pdf:{doc_id}",
        title=p.title,
        source_file=p.source_file,
        source_type=p.source_type,
        sections=sections,
        url=p.url if p.url else (MIND_INFO_URL if p.source_type == "mind_booklet" else None),
        url_verified=bool(p.url),  # URLs printed in the PDF itself are exact
        year=p.year,
        meta={"n_pages": p.n_pages},
    )
