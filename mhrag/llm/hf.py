"""Local transformers backend (GPU / Apple MPS / CPU), with optional 4-bit NF4 quantisation.

Fixes CHANGELOG B4-B7: half precision + device_map, the model's own chat template, only new tokens are
decoded (no prompt echo), and sampling flags are consistent.
"""

from __future__ import annotations

import logging
import os
import threading
import time

from mhrag.llm.base import Backend, BackendError, GenerationParams, Message
from mhrag.llm.templates import (
    completion_prompt,
    llama2_prompt,
    merge_system,
    prepare_messages,
    strip_reasoning,
    template_kwargs,
)

log = logging.getLogger(__name__)


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFBackend(Backend):
    name = "hf"

    def __init__(self, spec, load_in_4bit: bool | None = None, dtype: str | None = None):
        super().__init__(spec)
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = os.environ.get("MHRAG_DEVICE") or pick_device()
        model_id = spec.raw["hf_id"]
        if load_in_4bit is None:
            load_in_4bit = os.environ.get("MHRAG_LOAD_IN_4BIT", "0") == "1"
        if dtype is None:  # MHRAG_DTYPE overrides everything; a model's catalogue `dtype` overrides "auto"
            dtype = os.environ.get("MHRAG_DTYPE") or spec.raw.get("dtype") or "auto"
        if dtype == "auto":
            if self.device == "cuda":
                # native bf16 only (Ampere+); on a T4 bf16 is emulated and much slower than fp16
                try:
                    bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
                except TypeError:  # torch < 2.4
                    bf16 = torch.cuda.get_device_capability()[0] >= 8
                torch_dtype = torch.bfloat16 if bf16 else torch.float16
            elif self.device == "mps":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32  # CPU: prefer the llamacpp backend for speed
        else:
            torch_dtype = getattr(torch, dtype)

        kwargs: dict = {}
        # transformers >= 4.56 renamed torch_dtype -> dtype
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        kwargs["dtype" if (major, minor) >= (4, 56) else "torch_dtype"] = torch_dtype
        if self.device == "cuda":
            # One visible GPU (the Kaggle runs give each process its own): put the whole model on it. "auto" can
            # misjudge free memory and try to offload 4-bit layers to the CPU, which fails ("Some modules are
            # dispatched on the CPU or the disk"); with several GPUs it splits models across them.
            kwargs["device_map"] = {"": 0} if torch.cuda.device_count() == 1 else "auto"
            try:
                import flash_attn  # noqa: F401

                kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                kwargs["attn_implementation"] = "sdpa"
            if load_in_4bit:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                )
        else:
            kwargs["attn_implementation"] = "sdpa"
            if load_in_4bit:
                log.warning("4-bit bitsandbytes needs CUDA; loading %s unquantised on %s", model_id, self.device)

        token = os.environ.get("HF_TOKEN")
        t0 = time.perf_counter()
        self.seq2seq = spec.raw.get("arch") == "seq2seq"
        auto = AutoModelForSeq2SeqLM if self.seq2seq else AutoModelForCausalLM
        if self.seq2seq or spec.family == "completion":
            kwargs.pop("attn_implementation", None)  # older architectures (GPT-2/J, BART, T5): library default
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        self.model = auto.from_pretrained(model_id, token=token, **kwargs)
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()
        self.load_seconds = time.perf_counter() - t0
        self.quantized = bool(load_in_4bit and self.device == "cuda")
        log.info("Loaded %s on %s (%s, 4bit=%s) in %.1fs", model_id, self.device, torch_dtype, self.quantized,
                 self.load_seconds)

    # ------------------------------------------------------------------
    def build_prompt(self, messages: list[Message]) -> str:
        fam = self.spec.family
        if fam in ("completion", "seq2seq"):
            return completion_prompt(messages)
        if fam == "llama2" or not getattr(self.tokenizer, "chat_template", None):
            return llama2_prompt(messages)
        msgs = prepare_messages(messages, fam)
        try:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, **template_kwargs(fam)
            )
        except Exception:  # some templates reject the system role
            return self.tokenizer.apply_chat_template(
                merge_system(msgs), tokenize=False, add_generation_prompt=True, **template_kwargs(fam)
            )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    def stream(self, messages: list[Message], params: GenerationParams):
        # one generation at a time per loaded model (see Backend.generation_lock)
        with self.generation_lock:
            yield from self._stream(messages, params)

    def _stream(self, messages: list[Message], params: GenerationParams):
        import torch
        from transformers import TextIteratorStreamer

        prompt = self.build_prompt(messages)
        ctx = int(self.spec.context)
        max_new = min(params.max_new_tokens, max(32, ctx // 2)) if not self.seq2seq else params.max_new_tokens
        # Chat templates already contain BOS; base LMs / seq2seq models use their normal special tokens.
        add_special = self.spec.family in ("completion", "seq2seq")
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special)
        limit = ctx if self.seq2seq else ctx - max_new
        ids = enc["input_ids"]
        if ids.shape[1] > limit:
            # keep the end of the prompt for decoder-only models (question + generation cue), the start for seq2seq
            sl = slice(0, limit) if self.seq2seq else slice(ids.shape[1] - limit, ids.shape[1])
            enc = {k: v[:, sl] for k, v in enc.items()}
            log.warning("%s: prompt truncated to %d tokens (context %d)", self.spec.key, limit, ctx)
        inputs = {k: v.to(self.model.device) for k, v in enc.items()}
        if (getattr(self.model.generation_config, "num_beams", 1) or 1) > 1:
            # Beam search (e.g. BART-large-CNN's default num_beams=4) cannot be streamed by transformers; keep the
            # model's own decoding settings and return the answer in one piece.
            yield from self._generate_whole(inputs, max_new, params)
            return
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=not self.seq2seq, skip_special_tokens=True,
                                        timeout=300)
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new,
            do_sample=params.do_sample,
            repetition_penalty=params.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if params.do_sample:
            gen_kwargs.update(temperature=params.temperature, top_p=params.top_p)
        if params.seed is not None:
            torch.manual_seed(params.seed)
        err: list[BaseException] = []

        def run():
            try:
                with torch.inference_mode():
                    self.model.generate(**gen_kwargs)
            except BaseException as e:  # surfaced below
                err.append(e)
                streamer.end()

        th = threading.Thread(target=run, daemon=True)
        th.start()
        n = 0
        in_think = False
        for piece in streamer:
            n += 1
            if self.spec.family == "qwen3":  # guard against stray reasoning blocks
                if "<think>" in piece:
                    in_think = True
                if in_think:
                    if "</think>" in piece:
                        in_think = False
                        piece = strip_reasoning("<think>" + piece.split("<think>")[-1])
                    else:
                        continue
            yield piece
        th.join()
        if err:
            raise BackendError(f"generation failed: {type(err[0]).__name__}: {str(err[0])[:200]}") from err[0]
        self.last_usage = {"prompt_tokens": int(inputs["input_ids"].shape[1]), "completion_tokens": n}
        self.last_finish_reason = "length" if n >= max_new - 1 else "stop"

    def _generate_whole(self, inputs: dict, max_new: int, params: GenerationParams):
        import torch

        try:
            kw = dict(max_new_tokens=max_new, repetition_penalty=params.repetition_penalty, do_sample=params.do_sample,
                      pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
            if params.do_sample:
                kw.update(temperature=params.temperature, top_p=params.top_p)
            if params.seed is not None:
                torch.manual_seed(params.seed)
            with torch.inference_mode():
                out = self.model.generate(**inputs, **kw)
        except Exception as e:
            raise BackendError(f"generation failed: {type(e).__name__}: {str(e)[:200]}") from e
        new = out[0] if self.seq2seq else out[0, inputs["input_ids"].shape[1]:]
        n = int((new != (self.tokenizer.pad_token_id or -1)).sum())
        self.last_usage = {"prompt_tokens": int(inputs["input_ids"].shape[1]), "completion_tokens": n}
        self.last_finish_reason = "length" if n >= max_new - 1 else "stop"
        yield self.tokenizer.decode(new, skip_special_tokens=True).strip()

    def close(self):
        import gc

        del self.model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
