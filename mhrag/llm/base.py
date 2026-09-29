"""The single LLM interface every backend implements."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass, field

from pydantic import BaseModel

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


class GenerationParams(BaseModel):
    max_new_tokens: int = 400
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    seed: int | None = 0

    @property
    def do_sample(self) -> bool:
        # Fixes CHANGELOG B7: temperature/top_p only matter when sampling is on.
        return self.temperature > 0


@dataclass
class ModelSpec:
    key: str
    backend: str
    label: str = ""
    family: str = "generic"
    context: int = 8192
    raw: dict = field(default_factory=dict)


class BackendError(RuntimeError):
    """Raised for backend failures; the message is safe to log, never shown verbatim to end users."""


class Backend:
    name = "base"

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.last_usage: dict = {}
        self.last_finish_reason: str | None = None  # "stop" | "length" (hit max_new_tokens) | None (unknown)
        # Local models (llama.cpp, transformers) are not safe to run concurrently on one instance; their stream()
        # holds this lock for the whole generation. HTTP backends do not need it.
        self.generation_lock = threading.Lock()

    def stream(self, messages: list[Message], params: GenerationParams) -> Iterator[str]:  # pragma: no cover
        raise NotImplementedError

    def generate(self, messages: list[Message], params: GenerationParams) -> str:
        return "".join(self.stream(messages, params))

    def count_tokens(self, text: str) -> int:
        """Rough count used for context budgeting when a backend has no tokenizer handy."""
        return max(1, len(text) // 4)

    def warmup(self) -> None:
        try:
            self.generate([{"role": "user", "content": "Hi"}], GenerationParams(max_new_tokens=2, temperature=0))
        except Exception:  # warm-up is best effort
            pass

    def info(self) -> dict:
        return {"key": self.spec.key, "backend": self.name, "label": self.spec.label, "family": self.spec.family}

    def close(self) -> None:
        pass
