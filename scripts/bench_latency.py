"""Latency / memory benchmark: v1 settings ("before") vs the v2 pipeline ("after").

    python -m scripts.bench_latency --all                 # every config that can run on this machine
    python -m scripts.bench_latency --config v2_llamacpp_cpu --n 12
    python -m scripts.bench_latency --config v2_hf_cuda_4bit --model mistral-7b-instruct-v0.3   # on a GPU

Each config runs in its own subprocess (clean memory accounting) and is merged into results/latency.json.

Configs
- v1_hf_fp32_cpu: the v1 generation settings - transformers fp32 on CPU, raw markdown prompt (no chat template),
  no streaming (the whole answer is returned at once, so TTFT == end-to-end), max_new_tokens=500, full-text
  decoding. v1 used Mistral-7B, which needs ~28 GB in fp32 and cannot load on a 16 GB machine; the default here
  is Qwen2.5-1.5B-Instruct so before/after use the same weights. Use --model to change.
- v2_llamacpp_cpu:   GGUF Q4_K_M on CPU only (what the free HF Space runs), streaming, full pipeline.
- v2_llamacpp_metal: GGUF Q4_K_M with Metal offload (Apple Silicon), streaming, full pipeline.
- v2_hf_mps:         transformers fp16 on Apple MPS, streaming, full pipeline.
- v2_hf_cuda / v2_hf_cuda_4bit: transformers bf16 / NF4 on CUDA (run in the Colab notebook).
- v2_api:            openai_compatible backend (e.g. Groq); needs the API key.
- retrieval:         retrieval-only latency (cold vs cached) for each retrieval mode.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

QUERIES = [
    "What is a panic attack and what does it feel like?",
    "How can I manage my anger in the moment?",
    "What are the side effects of antidepressants?",
    "How can I improve my sleep?",
    "What is the difference between sadness and depression?",
    "How can I support a friend with bipolar disorder?",
    "What treatments are available for OCD?",
    "What is dissociation?",
    "How can I cope with loneliness?",
    "What should I know before starting a new medication?",
    "What are the symptoms of PTSD?",
    "How do I find a therapist?",
]

CONFIGS = {
    "v1_hf_fp32_cpu": {"kind": "v1", "model": "qwen2.5-1.5b-instruct", "n": 5},
    "v2_llamacpp_cpu": {"kind": "v2", "model": "qwen2.5-1.5b-gguf", "env": {"MHRAG_LLAMACPP_GPU_LAYERS": "0"}, "n": 12},
    "v2_llamacpp_metal": {"kind": "v2", "model": "qwen2.5-1.5b-gguf", "env": {"MHRAG_LLAMACPP_GPU_LAYERS": "-1"}, "n": 12,
                          "requires": "darwin"},
    "v2_hf_mps": {"kind": "v2", "model": "qwen2.5-1.5b-instruct", "env": {"MHRAG_DEVICE": "mps"}, "n": 12,
                  "requires": "mps"},
    # 16-bit Mistral-7B (~15 GB) does not fit a 16 GB T4, so the 16-bit CUDA row uses the same 1.5B model as the
    # laptop rows; the NF4 row shows a 7B model on the same GPU.
    "v2_hf_cuda": {"kind": "v2", "model": "qwen2.5-1.5b-instruct", "n": 12, "requires": "cuda"},
    "v2_hf_cuda_4bit": {"kind": "v2", "model": "mistral-7b-instruct-v0.3", "env": {"MHRAG_LOAD_IN_4BIT": "1"},
                        "n": 12, "requires": "cuda"},
    # pause between questions to stay under the free plan's tokens-per-minute limit, so TTFT is not throttling
    "v2_api": {"kind": "v2", "model": "gpt-oss-120b-groq", "n": 12, "requires": "GROQ_API_KEY", "pause_s": 30},
    "retrieval": {"kind": "retrieval", "n": 12},
}


class PeakRSS:
    def __init__(self, interval=0.05):
        import psutil

        self.p = psutil.Process()
        self.peak = self.p.memory_info().rss
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.p.memory_info().rss)
            time.sleep(0.05)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(rows: list[dict]) -> dict:
    def col(k):
        return [r[k] for r in rows if r.get(k) is not None]

    out = {"n": len(rows)}
    for k in ("ttft_ms", "e2e_ms", "tokens_per_s", "output_tokens", "retrieval_ms"):
        v = col(k)
        if v:
            out[k] = {"mean": statistics.mean(v), "p50": pct(v, 0.5), "p95": pct(v, 0.95)}
    waits = col("rate_limit_wait_s")
    if waits:
        out["throttled_queries"] = sum(w > 0 for w in waits)  # >0: API rate limiting is in these timings
    return out


def env_info() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(), "machine": platform.machine()}
    try:
        if sys.platform == "darwin":
            info["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            info["ram_gb"] = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)) / 2**30
        else:
            import psutil

            info["cpu"] = platform.processor()
            info["ram_gb"] = psutil.virtual_memory().total / 2**30
    except Exception:
        pass
    for mod in ("torch", "transformers", "llama_cpp"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------------------------- runners
def _ensure_index(s) -> None:
    """Build the default index if it is missing (a no-op when it is up to date), so the benchmark also runs on a
    fresh checkout such as a Kaggle session."""
    from mhrag.index.builder import build_index

    build_index(s)


def run_v1(cfg: dict, n: int) -> dict:
    """v1 generation settings with transformers, reproduced faithfully (see module docstring)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mhrag.config import get_settings
    from mhrag.llm.registry import load_catalog
    from mhrag.prompts import V1_TEMPLATE
    from mhrag.retrieval.factory import build_retriever

    s = get_settings()
    _ensure_index(s)
    retriever = build_retriever(s, need_reranker=False)
    model_id = load_catalog()[cfg["model"]].raw["hf_id"]
    with PeakRSS() as mem:
        t0 = time.perf_counter()
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to("cpu").eval()
        load_s = time.perf_counter() - t0
        rows = []
        for q in QUERIES[:n]:
            t1 = time.perf_counter()
            hits = retriever.retrieve(q, 2, "dense")
            r_ms = (time.perf_counter() - t1) * 1000
            prompt = V1_TEMPLATE.format(context=" ".join(h.chunk.text for h in hits), question=q)
            ids = tok(prompt, return_tensors="pt")
            with torch.inference_mode():
                out = model.generate(**ids, max_new_tokens=500, do_sample=False, repetition_penalty=1.1,
                                     pad_token_id=tok.eos_token_id)
            _ = tok.decode(out[0], skip_special_tokens=True)  # v1 returned prompt + answer (return_full_text)
            e2e = (time.perf_counter() - t1) * 1000
            n_new = int(out.shape[1] - ids["input_ids"].shape[1])
            gen_s = (e2e - r_ms) / 1000
            rows.append({"query": q, "ttft_ms": e2e, "e2e_ms": e2e, "output_tokens": n_new,
                         "tokens_per_s": n_new / gen_s if gen_s else None, "retrieval_ms": r_ms})
    return {"load_s": load_s, "startup_s_v1_debug_reloader": 2 * load_s,
            "startup_note": "v1 ran app.run(debug=True): the reloader imports rag.py in two processes, so the "
                            "model is loaded twice; startup_s_v1_debug_reloader = 2 x measured load_s.",
            "peak_rss_gb": mem.peak / 2**30, "rows": rows, "streaming": False, "dtype": "float32", "device": "cpu",
            "hf_id": model_id}


def run_v2(cfg: dict, n: int) -> dict:
    from mhrag.config import get_settings
    from mhrag.runtime import build_runtime

    s = get_settings()
    s.llm.model = cfg["model"]
    s.llm.served_models = [cfg["model"]]
    s.llm.allow_api_models = True  # the v2_api row measures the API model on purpose
    with PeakRSS() as mem:
        t0 = time.perf_counter()
        _ensure_index(s)
        rt = build_runtime(s, warmup=True)
        startup_s = time.perf_counter() - t0
        rows = []
        params = rt.pipeline.params(temperature=0.0)
        for i, q in enumerate(QUERIES[:n]):
            if i and cfg.get("pause_s"):
                time.sleep(cfg["pause_s"])
            rt.retriever._rcache.clear()
            rt.retriever._qcache.clear()
            res = rt.pipeline.answer(q, model=cfg["model"], params=params)
            t = res["timing"]
            rows.append({"query": q, "ttft_ms": t.get("ttft_ms"), "e2e_ms": t.get("total_ms"),
                         "output_tokens": t.get("output_tokens"), "tokens_per_s": t.get("tokens_per_s"),
                         "retrieval_ms": t.get("retrieval_ms"), "gate": res.get("gate"),
                         "usage": res.get("usage"),
                         # API rate-limit waiting inside this answer (0 = the timing is the model's own)
                         "rate_limit_wait_s": getattr(rt.models.get(cfg["model"]), "last_rate_wait_s", 0.0)})
    extra = {}
    try:
        import torch

        if torch.backends.mps.is_available():
            extra["mps_driver_allocated_gb"] = torch.mps.driver_allocated_memory() / 2**30
        if torch.cuda.is_available():
            extra["cuda_peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
    except Exception:
        pass
    return {"startup_s_including_warmup": startup_s, "peak_rss_gb": mem.peak / 2**30, "rows": rows,
            "streaming": True, "retrieval_mode": s.retrieval.mode, "index": rt.index.manifest["index_name"],
            "max_new_tokens": s.llm.max_new_tokens, **extra}


def run_retrieval(cfg: dict, n: int) -> dict:
    from mhrag.config import get_settings
    from mhrag.retrieval.factory import build_retriever

    s = get_settings()
    _ensure_index(s)
    r = build_retriever(s)
    r.retrieve("warm up", 1, "hybrid_rerank")
    out = {}
    for mode in ("bm25", "dense", "hybrid", "hybrid_rerank"):
        cold, warm = [], []
        for q in QUERIES[:n]:
            r._rcache.clear()
            r._qcache.clear()
            t = time.perf_counter()
            r.retrieve(q, 5, mode)
            cold.append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            r.retrieve(q, 5, mode)
            warm.append((time.perf_counter() - t) * 1000)
        out[mode] = {"cold_ms": {"p50": pct(cold, 0.5), "p95": pct(cold, 0.95)},
                     "cached_ms": {"p50": pct(warm, 0.5), "p95": pct(warm, 0.95)}}
    return {"modes": out, "index": r.index.manifest["index_name"], "n_chunks": r.index.manifest["n_chunks"]}


def run_one(name: str, n: int | None, model: str | None) -> dict:
    cfg = dict(CONFIGS[name])
    if model:
        cfg["model"] = model
    n = n or cfg["n"]
    for k, v in cfg.get("env", {}).items():
        os.environ[k] = v
    kind = cfg["kind"]
    t = time.time()
    res = run_v1(cfg, n) if kind == "v1" else run_v2(cfg, n) if kind == "v2" else run_retrieval(cfg, n)
    if "rows" in res:
        res["summary"] = summarize(res["rows"])
    res.update(config=name, model=cfg.get("model"), env_overrides=cfg.get("env", {}), queries_n=n,
               ran_at=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t)), environment=env_info())
    return res


def can_run(cfg: dict) -> tuple[bool, str]:
    req = cfg.get("requires")
    if not req:
        return True, ""
    if req == "darwin":
        return sys.platform == "darwin", "needs macOS/Metal"
    if req in ("mps", "cuda"):
        try:
            import torch

            ok = torch.cuda.is_available() if req == "cuda" else torch.backends.mps.is_available()
            return ok, f"needs {req}"
        except ImportError:
            return False, "needs torch"
    return bool(os.environ.get(req)), f"needs {req}"


def merge(out_path: Path, name: str, res: dict):
    data = json.loads(out_path.read_text()) if out_path.exists() else {}
    data[name] = res
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=list(CONFIGS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int)
    ap.add_argument("--model")
    ap.add_argument("--out", help="default: <results dir>/latency.json (follows MHRAG_PATHS__RESULTS)")
    args = ap.parse_args(argv)
    from mhrag.config import get_settings

    out = Path(args.out) if args.out else get_settings().results_dir / "latency.json"

    if args.all:
        for name, cfg in CONFIGS.items():
            ok, why = can_run(cfg)
            if not ok:
                print(f"skip {name}: {why}")
                merge(out, name, {"config": name, "status": f"TODO(run): {why}",
                                  "command": f"python -m scripts.bench_latency --config {name}"})
                continue
            print(f"== {name}")
            cmd = [sys.executable, "-m", "scripts.bench_latency", "--config", name, "--out", str(out)]
            if args.n:
                cmd += ["--n", str(args.n)]
            subprocess.run(cmd, check=False)
        return 0
    if not args.config:
        ap.error("--config or --all")
    ok, why = can_run(CONFIGS[args.config])
    if not ok:
        print(f"cannot run {args.config}: {why}")
        return 1
    res = run_one(args.config, args.n, args.model)
    merge(out, args.config, res)
    print(json.dumps(res.get("summary", res.get("modes")), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
