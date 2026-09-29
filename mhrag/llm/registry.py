"""Model catalogue + a manager that loads each backend once and keeps local models bounded in memory."""

from __future__ import annotations

import importlib.util
import logging
import os
import threading
import time
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

from mhrag.config import load_yaml
from mhrag.llm.base import Backend, BackendError, ModelSpec

log = logging.getLogger(__name__)
LOCAL_BACKENDS = {"hf", "llamacpp"}
ALL = "*"  # served_models: ["*"] serves every catalogue model (unavailable ones are listed but disabled)

# UI groups, in display order
GROUPS = {
    "api": "API models (fast, need a key)",
    "gguf": "Local, llama.cpp (CPU / Metal)",
    "hf": "Local, transformers (GPU / Apple MPS)",
    "ollama": "Local servers (Ollama, vLLM)",
    "baseline": "Original v1 project models",
    "test": "Testing",
}


def load_catalog(path: str | Path = "models.yaml") -> dict[str, ModelSpec]:
    data = load_yaml(path)["models"]
    return {
        k: ModelSpec(key=k, backend=v["backend"], label=v.get("label", k), family=v.get("family", "generic"),
                     context=int(v.get("context", 8192)), raw=v)
        for k, v in data.items()
    }


def needs_api_key(spec: ModelSpec) -> bool:
    """A hosted API model that cannot run without the user's key (local servers such as vLLM/Ollama do not count)."""
    raw = spec.raw
    return spec.backend == "openai_compatible" and bool(raw.get("api_key_env")) and not raw.get("api_key_optional")


def group_of(spec: ModelSpec) -> str:
    if spec.backend == "mock":
        return "test"
    if spec.raw.get("v1_model"):
        return "baseline"
    if spec.backend == "openai_compatible" and not needs_api_key(spec):
        return "ollama"  # a server you run yourself (e.g. vLLM): no key, grouped with Ollama
    return {"openai_compatible": "api", "llamacpp": "gguf", "hf": "hf", "ollama": "ollama"}.get(spec.backend, "hf")


# ------------------------------------------------------------------------------ resource checks
@lru_cache(maxsize=1)
def device_memory_gb() -> tuple[float, str]:
    """Memory that a local model can use: CUDA VRAM if a GPU is present, otherwise system RAM (which is also what
    Apple-Silicon MPS and llama.cpp use)."""
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / 2**30, "GPU"
        except Exception:
            pass
    try:
        import psutil

        return psutil.virtual_memory().total / 2**30, "RAM"
    except ImportError:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, "RAM"


def estimated_memory_gb(spec: ModelSpec) -> float | None:
    """Rough resident size: GGUF Q4_K_M ~0.6 GB per billion params; transformers fp16/bf16 2 bytes per param (NF4
    4-bit on CUDA ~0.7 GB per billion); plus runtime overhead."""
    p = spec.raw.get("params_b")
    if not p:
        return None
    if spec.backend == "llamacpp":
        return 0.6 * p + 0.8
    if spec.backend == "hf":
        cuda = device_memory_gb()[1] == "GPU"
        per_b = 0.7 if (cuda and os.environ.get("MHRAG_LOAD_IN_4BIT") == "1") else 2.0  # NF4 vs fp16/bf16
        return per_b * p * 1.15 + 0.8
    return None


_ollama_cache: dict[str, tuple[float, set[str] | None]] = {}


def _ollama_models(base_url: str) -> set[str] | None:
    """Tags available on an Ollama server (cached 30 s); None if it is not reachable."""
    now = time.time()
    hit = _ollama_cache.get(base_url)
    if hit and now - hit[0] < 30:
        return hit[1]
    try:
        import httpx

        r = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=0.5)
        tags = {m["name"] for m in r.json().get("models", [])}
    except Exception:
        tags = None
    _ollama_cache[base_url] = (now, tags)
    return tags


def _cached_locally(spec: ModelSpec) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        if spec.backend == "llamacpp":
            return isinstance(try_to_load_from_cache(spec.raw["gguf_repo"], spec.raw["gguf_file"]), str)
        if spec.backend == "hf":
            return isinstance(try_to_load_from_cache(spec.raw["hf_id"], "config.json"), str)
    except Exception:
        pass
    return False


def availability(spec: ModelSpec) -> tuple[bool, str]:
    """Can this model be served on this machine? Cheap: no downloads, at most one 0.5 s Ollama ping."""
    raw = spec.raw
    if spec.backend == "mock":
        return True, "ok"
    if spec.backend == "openai_compatible":
        key_env = raw.get("api_key_env")
        if key_env and not os.environ.get(key_env) and not raw.get("api_key_optional"):
            return False, f"set {key_env}"
        for f in ("base_url", "api_model"):
            if not (raw.get(f) or os.environ.get(raw.get(f"{f}_env", "") or "_none_")):
                return False, f"set {raw.get(f + '_env')}"
        return True, "ok"
    if spec.backend == "ollama":
        base = os.environ.get(raw.get("base_url_env", "") or "_none_") or raw.get("base_url", "http://localhost:11434")
        tags = _ollama_models(base)
        if tags is None:
            return False, "start Ollama (ollama serve)"
        tag = raw["ollama_tag"]
        if tag not in tags and f"{tag}:latest" not in tags:
            return False, f"run: ollama pull {tag}"
        return True, "ok"
    if spec.backend == "llamacpp" and importlib.util.find_spec("llama_cpp") is None:
        return False, "pip install llama-cpp-python"
    if spec.backend == "hf":
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
            return False, "pip install '.[gpu]'"
        if raw.get("gated") and not os.environ.get("HF_TOKEN"):
            return False, "gated: accept the licence on Hugging Face and set HF_TOKEN"
    if spec.backend in LOCAL_BACKENDS:
        need = estimated_memory_gb(spec)
        have, kind = device_memory_gb()
        if need and need > 0.8 * have:
            return False, f"needs ~{need:.0f} GB, this machine has {have:.0f} GB {kind}"
        return True, "ok" if _cached_locally(spec) else "downloads on first use"
    if spec.backend not in LOCAL_BACKENDS:
        return False, f"unknown backend {spec.backend}"
    return True, "ok"


def make_backend(spec: ModelSpec, timeout_s: float = 120.0) -> Backend:
    if spec.backend == "mock":
        from mhrag.llm.mock import MockBackend

        return MockBackend(spec)
    if spec.backend == "openai_compatible":
        from mhrag.llm.remote import OpenAICompatibleBackend

        return OpenAICompatibleBackend(spec, timeout_s=timeout_s)
    if spec.backend == "ollama":
        from mhrag.llm.remote import OllamaBackend

        return OllamaBackend(spec, timeout_s=timeout_s)
    if spec.backend == "llamacpp":
        from mhrag.llm.llamacpp import LlamaCppBackend

        return LlamaCppBackend(spec)
    if spec.backend == "hf":
        from mhrag.llm.hf import HFBackend

        return HFBackend(spec)
    raise BackendError(f"unknown backend {spec.backend}")


class ModelManager:
    """Loads backends lazily and once. At most `max_local` heavy local models stay resident (LRU)."""

    def __init__(self, catalog: dict[str, ModelSpec], served: list[str], default: str, max_local: int = 1,
                 timeout_s: float = 120.0):
        if ALL in served:
            served = [k for k in catalog if k not in served] + [k for k in served if k != ALL]
        unknown = [k for k in served + [default] if k not in catalog]
        if unknown:
            raise KeyError(f"Models not in configs/models.yaml: {unknown}")
        self.catalog = catalog
        self.served = list(dict.fromkeys([default] + served))
        self.default = default
        self.max_local = max_local
        self.timeout_s = timeout_s
        self._loaded: OrderedDict[str, Backend] = OrderedDict()
        self._lock = threading.Lock()  # guards _loaded
        self._load_locks: dict[str, threading.Lock] = {}  # one per model: loading one does not block the others

    def is_loaded(self, key: str | None) -> bool:
        return (key or self.default) in self._loaded

    def list(self) -> list[dict]:
        out = []
        for k in self.served:
            spec = self.catalog[k]
            ok, why = availability(spec)
            alt = spec.raw.get("local_alternative")
            if not ok and alt in self.catalog and alt in self.served:
                why += f"; try {self.catalog[alt].label}"
            out.append({
                "key": k, "label": spec.label, "backend": spec.backend, "group": group_of(spec),
                "group_label": GROUPS[group_of(spec)], "available": ok, "status": why,
                "loaded": k in self._loaded, "default": k == self.default,
                "params_b": spec.raw.get("params_b"), "license": spec.raw.get("license"),
            })
        order = list(GROUPS)
        out.sort(key=lambda m: (not m["default"], order.index(m["group"]), not m["available"]))
        return out

    def get(self, key: str | None = None) -> Backend:
        key = key or self.default
        if key not in self.served:
            raise BackendError(f"model {key!r} is not served by this server")
        with self._lock:
            if key in self._loaded:
                self._loaded.move_to_end(key)
                return self._loaded[key]
            load_lock = self._load_locks.setdefault(key, threading.Lock())
        with load_lock:
            with self._lock:
                if key in self._loaded:  # loaded by a concurrent request while we waited
                    return self._loaded[key]
            spec = self.catalog[key]
            ok, why = availability(spec)
            if not ok:
                raise BackendError(f"model {key!r} unavailable: {why}")
            if spec.backend in LOCAL_BACKENDS:
                with self._lock:
                    local = [k for k in self._loaded if self.catalog[k].backend in LOCAL_BACKENDS]
                    evicted = [(k, self._loaded.pop(k)) for k in local[: max(0, len(local) - self.max_local + 1)]]
                for old, b in evicted:
                    log.info("Unloading %s to make room for %s", old, key)
                    with b.generation_lock:  # wait for an in-flight answer on that model to finish
                        b.close()
            backend = make_backend(spec, self.timeout_s)
            with self._lock:
                self._loaded[key] = backend
            return backend
