"""safety gate -> (query rewrite) -> retrieve -> rerank -> prompt with chat template -> stream -> post-check
-> citations.

`RAGPipeline.stream()` yields events consumed by the SSE endpoint; `RAGPipeline.answer()` collects them for
the evaluation scripts. Three profiles are used in the experiments:
- full:    gate + configured retrieval (hybrid + rerank by default) + grounded prompt + output checks
- naive:   the v1 set-up (dense top-2, chunks joined by a space, v1 markdown template, no system prompt/gate)
- no_rag:  the LLM alone with a safety-aware system prompt
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from mhrag import prompts
from mhrag.config import Settings
from mhrag.llm.base import BackendError, GenerationParams
from mhrag.llm.registry import ModelManager
from mhrag.retrieval.retriever import Retriever
from mhrag.retrieval.rewrite import heuristic_rewrite, llm_rewrite
from mhrag.safety.crisis import ELEVATED_PREFIX, crisis_message, helplines
from mhrag.safety.gate import GateResult, SafetyGate
from mhrag.safety.output_check import check_output

log = logging.getLogger(__name__)


@dataclass
class Event:
    type: str  # gate | sources | token | replace | append | done | error
    data: dict = field(default_factory=dict)


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class RAGPipeline:
    def __init__(self, settings: Settings, retriever: Retriever | None, models: ModelManager, gate: SafetyGate | None):
        self.s = settings
        self.retriever = retriever
        self.models = models
        self.gate = gate

    def params(self, **over) -> GenerationParams:
        p = GenerationParams(
            max_new_tokens=self.s.llm.max_new_tokens, temperature=self.s.llm.temperature, top_p=self.s.llm.top_p
        )
        return p.model_copy(update=over) if over else p

    def _history(self, history: list[dict] | None) -> list[dict]:
        h = [
            {"role": m["role"], "content": str(m["content"])[:1500]}
            for m in (history or [])
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        return h[-2 * self.s.llm.history_turns :]

    # ------------------------------------------------------------------
    def stream(
        self,
        message: str,
        history: list[dict] | None = None,
        model: str | None = None,
        region: str | None = None,
        profile: str = "full",
        params: GenerationParams | None = None,
        exclude_doc_ids: set[str] | None = None,
        retrieval_mode: str | None = None,
        top_k: int | None = None,
    ) -> Iterator[Event]:
        t0 = time.perf_counter()
        timing: dict = {}
        hist = self._history(history)
        params = params or self.params()
        region = region or self.s.safety.region

        # 1) safety gate
        gate: GateResult | None = None
        if profile == "full" and self.gate is not None:
            tg = time.perf_counter()
            gate = self.gate.assess(message, hist)
            timing["gate_ms"] = (time.perf_counter() - tg) * 1000
            yield Event("gate", {**gate.to_dict(), "helplines": helplines(region) if gate.label != "none" else []})
            if gate.blocks_generation or gate.label == "third_party":
                text = crisis_message(gate.label, region)
                yield Event("token", {"text": text})
                timing["total_ms"] = (time.perf_counter() - t0) * 1000
                yield Event("done", {"timing": timing, "answer": text, "sources": [], "model": None,
                                     "gate": gate.label, "generated": False})
                return

        # 2) retrieval
        used, hits, context = [], [], ""
        if profile in ("full", "naive") and self.retriever is not None:
            tr = time.perf_counter()
            if profile == "naive":
                hits = self.retriever.retrieve(message, top_k or 2, retrieval_mode or "dense", exclude_doc_ids)
                used = hits
            else:
                q = message
                if self.s.retrieval.query_rewrite == "heuristic":
                    q = heuristic_rewrite(message, hist)
                elif self.s.retrieval.query_rewrite == "llm":
                    q = llm_rewrite(message, hist, self.models.get(model), params)
                hits = self.retriever.retrieve(
                    q, top_k or self.s.retrieval.top_k, retrieval_mode or self.s.retrieval.mode, exclude_doc_ids
                )
                timing["query"] = q if q != message else None
            timing["retrieval_ms"] = (time.perf_counter() - tr) * 1000
            timing["retrieval_detail"] = self.retriever.last_timing

        # 3) prompt
        spec = self.models.catalog[model or self.models.default]
        # Small-context models (GPT-2: 1024, FLAN-T5: 512, MentaLLaMA: 2048) get a proportionally smaller budget.
        budget = min(self.s.retrieval.context_token_budget,
                     max(150, int(spec.context) - params.max_new_tokens - 400))
        if profile == "full" and self.retriever is not None:
            context, used = prompts.build_context(hits, budget, _approx_tokens)
            messages = prompts.grounded_messages(message, context, hist)
        elif profile == "naive":
            messages = prompts.v1_messages(message, used)
            context = " ".join(h.chunk.text for h in used)
        else:
            messages = prompts.no_context_messages(message, hist)
        sources = [
            {"n": i + 1, "chunk_id": h.chunk.chunk_id, "score": round(h.score, 4), **h.chunk.citation(),
             "snippet": h.chunk.text[:280]}
            for i, h in enumerate(used)
        ]
        yield Event("sources", {"sources": sources})

        # 4) generation
        prefix = ELEVATED_PREFIX if gate is not None and gate.label == "elevated" else ""
        if prefix:
            yield Event("token", {"text": prefix})
        if not self.models.is_loaded(model):
            key = model or self.models.default
            yield Event("status", {"text": f"Loading {self.models.catalog[key].label}… the first use can take a "
                                           "while (the model may need to download)."})
        backend = self.models.get(model)
        tgen = time.perf_counter()
        ttft = None
        parts: list[str] = []
        n_chunks = 0
        for piece in backend.stream(messages, params):
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            parts.append(piece)
            n_chunks += 1
            yield Event("token", {"text": piece})
            if time.perf_counter() - t0 > self.s.llm.timeout_s:
                raise BackendError("generation timed out")
        gen_s = time.perf_counter() - tgen
        answer = "".join(parts).strip()
        usage = dict(backend.last_usage or {})
        n_out = usage.get("completion_tokens") or n_chunks
        timing.update(
            ttft_ms=ttft, generation_ms=gen_s * 1000, output_tokens=n_out,
            tokens_per_s=(n_out / gen_s) if gen_s > 0 else None,
        )

        # 5) length-truncated answers: trim to a clean boundary and say so (instead of stopping mid-sentence)
        truncated = backend.last_finish_reason == "length"
        if truncated:
            trimmed = prompts.trim_to_boundary(answer) or answer
            answer = trimmed + prompts.TRUNCATION_NOTE
            yield Event("replace", {"text": prefix + answer, "truncated": True})

        # 6) output checks
        final = prefix + answer
        if profile == "full" and self.s.safety.output_checks:
            chk = check_output(answer)
            if chk.replaced:
                final = chk.text
                yield Event("replace", {"text": final, "issues": chk.issues})
            elif chk.issues:
                extra = chk.text[len(answer):]
                final = prefix + chk.text
                yield Event("append", {"text": extra, "issues": chk.issues})
        timing["total_ms"] = (time.perf_counter() - t0) * 1000
        yield Event("done", {
            "timing": timing, "answer": final, "raw_answer": answer, "sources": sources,
            "model": backend.spec.key, "gate": gate.label if gate else None, "generated": True,
            "usage": usage, "context": context if profile != "no_rag" else "", "truncated": truncated,
            "prompt_version": prompts.PROMPT_VERSION,
        })

    def answer(self, message: str, **kw) -> dict:
        out: dict = {"gate": None}
        for ev in self.stream(message, **kw):
            if ev.type == "gate":
                out["gate_detail"] = ev.data
            elif ev.type == "done":
                out.update(ev.data)
        return out
