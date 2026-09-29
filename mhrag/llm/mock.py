"""Deterministic offline backend for tests and UI development."""

from __future__ import annotations

import os
import re
import time

from mhrag.llm.base import Backend, BackendError, GenerationParams, Message

FAIL_TRIGGER = "__simulate_backend_error__"


class MockBackend(Backend):
    name = "mock"

    def __init__(self, spec, delay_s: float | None = None):
        super().__init__(spec)
        self.delay_s = float(os.environ.get("MHRAG_MOCK_DELAY", "0.0")) if delay_s is None else delay_s
        self.calls: list[list[Message]] = []

    def _answer(self, messages: list[Message]) -> str:
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if FAIL_TRIGGER in user or os.environ.get("MHRAG_MOCK_FAIL") == "1":
            raise BackendError("mock backend asked to fail")
        m = re.search(r"\[1\][^\n]*\n(.+?)(?:\n\n|\n\[2\]|$)", user, re.S)
        if m:
            first = re.split(r"(?<=[.!?])\s", m.group(1).strip())[0]
            return f"{first} [1]\n\nThis is a mock answer generated offline for testing."
        return "I don't have information about that in my sources. (mock answer)"

    def stream(self, messages: list[Message], params: GenerationParams):
        self.calls.append(messages)
        text = self._answer(messages)
        words = text.split(" ")
        n = 0
        self.last_finish_reason = "stop"
        for i, w in enumerate(words):
            if n >= params.max_new_tokens:
                self.last_finish_reason = "length"
                break
            if self.delay_s:
                time.sleep(self.delay_s)
            n += 1
            yield w if i == 0 else " " + w
        self.last_usage = {"prompt_tokens": sum(len(m["content"]) // 4 for m in messages), "completion_tokens": n}
