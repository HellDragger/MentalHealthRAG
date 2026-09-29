"""Text normalisation and boilerplate patterns for the PDF corpus."""

from __future__ import annotations

import re
import unicodedata

# Browser print header, e.g. "4/16/24, 9:21 AM When is anger a problem? - Mind"
DATE_HEADER = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}\s?[AP]M\s+(?P<title>.+)$")
# URL line with a page counter, e.g. "https://www.mind.org.uk/... 1/4"
URL_LINE = re.compile(r"^(?P<url>https?://\S+)(?:\s+\d+/\d+)?$")
# Continuation of a wrapped URL, e.g. "ital/5-best-mental-health-apps/ 4/12"
URL_CONT = re.compile(r"^\S*/\S*\s+\d+/\d+$")
PAGE_COUNTER = re.compile(r"^\d+/\d+$")
BOOKLET_COPYRIGHT = re.compile(r"^©\s*Mind\s+(?P<year>\d{4})$")
PAGE_NUMBER = re.compile(r"^\d{1,3}$")
TOC_LINE = re.compile(r"\.{4,}\s*\d+\s*$")

BOILERPLATE_LINES = [
    re.compile(p, re.I)
    for p in [
        r"^©\s*\d{4}\s+Mind\b.*registered charity",  # web footer
        r"^\(no\. \d+\) in England and Wales\.?$",
        r"^Call Mind Infoline$",
        r"^0300 123 3393$",
        r"^Contents$",
        r"^If you require this information in Word document format",
        r"^please email: publications@mind\.org\.uk",
        r"^©\s*Mind\s+\w+\s+\d{4}\. (To be revised|Next review)",
        r"^References are available on request",
        r"^(#\s*)?PREV NEXT",
        r"^\d+\s+SHARES",
        r"^Home About Us",
        r"^News & Events Contact",
        r"^Home \(/\)",
        r"^NEWS / READERS",
    ]
]

# "Mae'r dudalen hon hefyd ar gael yn Gymraeg. This link will take you to a Welsh translation of this page."
WELSH_BOILERPLATE = re.compile(
    r"Mae'r dudalen hon hefyd ar gael yn Gymraeg\.?\s*(This link will take you to a Welsh\s*translation of this page\.?)?",
    re.I,
)

_PRIVATE_USE = re.compile("[-￼�]")
_QUOTE_ONLY = re.compile(r"^[\s“”\"'‘’]+$")
_BULLET = re.compile(r"^\s*[•●▪◦·\-–]\s+")


def normalize_unicode(text: str) -> str:
    """NFKC (fixes ligatures such as ﬀ ﬃ ﬁ), soft-hyphen artefacts, private-use glyphs, CRLF."""
    text = text.replace("￾", "-").replace("­", "")
    text = unicodedata.normalize("NFKC", text)
    text = _PRIVATE_USE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


_MOJIBAKE = re.compile("â€|Ã[\x80-\xbf]|Â[\xa0-\xbf]")


def fix_mojibake(text: str) -> str:
    """Repair UTF-8 text that was decoded as cp1252 somewhere upstream ("personâ€™s" -> "person’s").
    Applied to the FAQ and intent files; the PDFs are not affected."""
    if not _MOJIBAKE.search(text):
        return text
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        # fix piecewise when the string mixes clean and broken spans
        return _MOJIBAKE_SPAN.sub(lambda m: _try_fix(m.group(0)), text)


_MOJIBAKE_SPAN = re.compile("(?:â€.|Ã.|Â.)+")


def _try_fix(span: str) -> str:
    try:
        return span.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return span


def is_boilerplate(line: str) -> bool:
    s = line.strip()
    if not s or _QUOTE_ONLY.match(s):
        return True
    if TOC_LINE.search(s):
        return True
    return any(p.search(s) for p in BOILERPLATE_LINES)


def is_bullet(line: str) -> bool:
    return bool(_BULLET.match(line))


def strip_bullet(line: str) -> str:
    return _BULLET.sub("", line).strip()


def looks_like_heading(line: str, prev: str | None, nxt: str | None, known: set[str] | None = None) -> bool:
    """Heuristic heading detector.

    Strong signals: a question line (Mind uses "What is anger?" style headings) or a heading listed in a
    booklet's table of contents. Weak signal: a short title-like line that follows a finished sentence and
    precedes a long line.
    """
    s = line.strip()
    if not s or len(s) > 110 or is_bullet(s):
        return False
    words = s.split()
    if known and s.lower().rstrip("?") in known:
        return True
    if s.endswith("?") and len(words) <= 16 and s[0].isupper():
        return True
    if len(words) > 8 or s[-1] in ".,;:!)”\"'" or not s[0].isupper():
        return False
    prev_ends = prev is None or prev.strip()[-1:] in ".?!:”\"" or prev.strip() == ""
    next_long = nxt is not None and len(nxt.strip()) > 60
    return prev_ends and next_long


def join_lines(lines: list[str]) -> str:
    """Re-flow PDF-wrapped lines into paragraphs, keeping list items on their own lines."""
    if not lines:
        return ""
    width = max(len(ln) for ln in lines)
    out: list[str] = []
    buf = ""
    for i, raw in enumerate(lines):
        ln = raw.strip()
        if not ln:
            continue
        if is_bullet(ln):
            if buf:
                out.append(buf)
            buf = "- " + strip_bullet(ln)
            continue
        if not buf:
            buf = ln
        else:
            buf = buf + " " + ln
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        ends_sentence = ln[-1:] in ".?!:”\""
        short = len(ln) < 0.7 * width
        if (ends_sentence and short) or (short and nxt[:1].isupper() and not ends_sentence) or not nxt:
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    text = "\n".join(out)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(\w)- (\w)", r"\1-\2", text)
    return text.strip()
