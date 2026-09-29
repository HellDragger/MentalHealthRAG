"""llama.cpp backend for GGUF models (CPU, or Metal/CUDA offload when the wheel supports it)."""

from __future__ import annotations

import logging
import os
import time

from mhrag.llm.base import Backend, BackendError, GenerationParams, Message
from mhrag.llm.templates import prepare_messages, qwen3_no_think, strip_reasoning

log = logging.getLogger(__name__)


class LlamaCppBackend(Backend):
    name = "llamacpp"

    def __init__(self, spec, n_ctx: int | None = None, n_threads: int | None = None, n_gpu_layers: int | None = None):
        super().__init__(spec)
        from llama_cpp import Llama

        raw = spec.raw
        n_ctx = n_ctx or min(int(raw.get("context", 4096)), int(os.environ.get("MHRAG_LLAMACPP_CTX", "4096")))
        if n_gpu_layers is None:
            n_gpu_layers = int(os.environ.get("MHRAG_LLAMACPP_GPU_LAYERS", "-1"))
        n_threads = n_threads or int(os.environ.get("MHRAG_LLAMACPP_THREADS", "0")) or None
        t0 = time.perf_counter()
        common = dict(n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, n_threads=n_threads, verbose=False, seed=0)
        if raw.get("gguf_path"):
            self.llm = Llama(model_path=raw["gguf_path"], **common)
        else:
            self.llm = Llama.from_pretrained(repo_id=raw["gguf_repo"], filename=raw["gguf_file"], **common)
        self.load_seconds = time.perf_counter() - t0
        self.n_gpu_layers = n_gpu_layers
        log.info("Loaded GGUF %s in %.1fs (n_ctx=%d, gpu_layers=%d)", raw.get("gguf_file"), self.load_seconds,
                 n_ctx, n_gpu_layers)

    def count_tokens(self, text: str) -> int:
        return len(self.llm.tokenize(text.encode("utf-8"), add_bos=False))

    def stream(self, messages: list[Message], params: GenerationParams):
        # one generation at a time per loaded model (see Backend.generation_lock)
        with self.generation_lock:
            yield from self._stream(messages, params)

    def _stream(self, messages: list[Message], params: GenerationParams):
        msgs = prepare_messages(messages, self.spec.family)
        if self.spec.family == "qwen3":
            msgs = qwen3_no_think(msgs)
        try:
            it = self.llm.create_chat_completion(
                messages=msgs,
                stream=True,
                max_tokens=params.max_new_tokens,
                temperature=params.temperature if params.do_sample else 0.0,
                top_p=params.top_p,
                repeat_penalty=params.repetition_penalty,
                seed=params.seed if params.seed is not None else -1,
            )
            n = 0
            self.last_finish_reason = None
            for ev in it:
                fr = ev["choices"][0].get("finish_reason")
                if fr:
                    self.last_finish_reason = fr
                delta = ev["choices"][0].get("delta", {}).get("content")
                if delta:
                    n += 1
                    yield strip_reasoning(delta) if "<think>" in delta else delta
            self.last_usage = {"completion_tokens": n}
        except Exception as e:
            raise BackendError(f"llama.cpp generation failed: {type(e).__name__}: {str(e)[:200]}") from e

    def close(self):
        try:
            self.llm.close()
        except Exception:
            pass
