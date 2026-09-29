from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SOURCE_TYPES = ("mind_web", "mind_booklet", "web_article", "faq", "kb_fact", "counselling")


@dataclass
class Section:
    heading: str
    text: str


@dataclass
class Document:
    doc_id: str
    title: str
    source_file: str
    source_type: str
    sections: list[Section]
    url: str | None = None
    url_verified: bool = False
    year: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(f"{s.heading}\n{s.text}" if s.heading else s.text for s in self.sections)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str  # body shown to the LLM / UI
    title: str
    section: str
    sections: list[str]
    source_file: str
    source_type: str
    url: str | None
    url_verified: bool = False
    year: int | None = None
    n_tokens: int = 0

    @property
    def embed_text(self) -> str:
        """Text that gets embedded / BM25-indexed: a contextual header + body."""
        head = self.title if not self.section or self.section == self.title else f"{self.title} — {self.section}"
        return f"{head}\n{self.text}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        return cls(**d)

    def citation(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "section": self.section,
            "url": self.url,
            "url_verified": self.url_verified,
            "source_type": self.source_type,
            "source_file": self.source_file,
            "year": self.year,
        }
