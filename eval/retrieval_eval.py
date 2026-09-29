"""Retrieval experiments: every (embedder x chunk size x system) on every query set."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np

from eval.datasets import is_relevant, load_jsonl, n_relevant
from eval.metrics import bootstrap_ci, holm, mrr_at_k, ndcg_at_k, paired_bootstrap_p
from mhrag.config import Settings
from mhrag.index.builder import build_index
from mhrag.index.embedders import release_models
from mhrag.retrieval.factory import build_retriever

log = logging.getLogger(__name__)
KS = (1, 5, 10)


def score_run(retriever, queries: list[dict], mode: str, k: int = 10) -> dict:
    per = {f"R@{x}": [] for x in KS} | {"MRR@10": [], "nDCG@10": [], "latency_ms": []}
    cache: dict = {}
    chunks = retriever.index.chunks
    nrel_cache: dict = {}
    for q in queries:
        retriever._rcache.clear()
        t = time.perf_counter()
        hits = retriever.retrieve(q["query"], k, mode)
        per["latency_ms"].append((time.perf_counter() - t) * 1000)
        rel = [is_relevant(h.chunk, q["gold"], cache) for h in hits]
        key = q["id"]
        if key not in nrel_cache:
            nrel_cache[key] = max(1, n_relevant(chunks, q["gold"], cache))
        for x in KS:
            per[f"R@{x}"].append(float(any(rel[:x])))  # success@k: >=1 relevant chunk in the top k
        per["MRR@10"].append(mrr_at_k(rel, 10))
        per["nDCG@10"].append(ndcg_at_k(rel, 10, nrel_cache[key]))
    summary = {m: dict(zip(("mean", "lo", "hi"), bootstrap_ci(v, n_boot=1000), strict=True))
               for m, v in per.items() if m != "latency_ms"}
    summary["latency_ms"] = {"mean": float(np.mean(per["latency_ms"])), "p50": float(np.median(per["latency_ms"]))}
    return {"summary": summary, "per_query": {m: v for m, v in per.items() if m != "latency_ms"}}


def _run_key(system: str, embedder: str | None, chunk: int, dataset: str, queries: list[dict]) -> str:
    """Identifies a finished run in the checkpoint; the query-id hash invalidates it if the eval set changes."""
    qh = hashlib.sha1("|".join(str(q["id"]) for q in queries).encode()).hexdigest()[:12]
    return f"{system}|{embedder or '-'}|c{chunk}|{dataset}|{qh}"


def _load_runs(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    runs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:  # truncated last line of an interrupted run
            continue
        runs[r["_key"]] = r
    return runs


def run_retrieval_experiment(cfg: dict, settings: Settings, checkpoint: Path | None = None) -> dict:
    """`checkpoint`: a JSONL file with one line per finished (system, embedder, chunk size, dataset) run. Runs
    already in it are reused, so an interrupted experiment resumes where it stopped."""
    done = _load_runs(checkpoint)
    if done:
        log.info("resuming: %d finished runs in %s", len(done), checkpoint)
    datasets = {}
    for name in cfg["datasets"]:
        try:
            datasets[name] = load_jsonl(f"{name}.jsonl")[: cfg.get("limit")]
        except FileNotFoundError:
            log.warning("dataset %s missing - run scripts.make_eval_sets", name)
    runs = []
    for chunk in cfg["chunk_sizes"]:
        for emb in cfg["embedders"]:
            built = False
            for sysdef in cfg["systems"]:
                if sysdef.get("embedders") and emb not in sysdef["embedders"]:
                    continue
                if sysdef["mode"] == "bm25" and emb != cfg.get("bm25_on", cfg["embedders"][0]):
                    continue  # BM25 does not depend on the embedder; score it once per chunk size
                if sysdef.get("chunk_sizes") and chunk not in sysdef["chunk_sizes"]:
                    continue
                run_emb = None if sysdef["mode"] == "bm25" else emb
                keys = {d: _run_key(sysdef["name"], run_emb, chunk, d, q) for d, q in datasets.items()}
                if all(k in done for k in keys.values()):
                    runs += [{k: v for k, v in done[keys[d]].items() if k != "_key"} for d in datasets]
                    continue
                if not built:
                    build_index(settings, emb, chunk)
                    built = True
                need_rr = sysdef["mode"].endswith("_rerank")
                r = build_retriever(settings, emb, chunk, need_reranker=need_rr, reranker_key=sysdef.get("reranker"))
                for dname, queries in datasets.items():
                    if keys[dname] in done:
                        runs.append({k: v for k, v in done[keys[dname]].items() if k != "_key"})
                        continue
                    t0 = time.time()
                    res = score_run(r, queries, sysdef["mode"])
                    run = {"system": sysdef["name"], "mode": sysdef["mode"], "reranker": sysdef.get("reranker"),
                           "embedder": None if sysdef["mode"] == "bm25" else emb, "chunk_tokens": chunk,
                           "dataset": dname, "n_queries": len(queries), "n_chunks": len(r.index.chunks),
                           "seconds": round(time.time() - t0, 1), **res}
                    runs.append(run)
                    if checkpoint is not None:
                        checkpoint.parent.mkdir(parents=True, exist_ok=True)
                        with open(checkpoint, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"_key": keys[dname], **run}) + "\n")
                    s = res["summary"]
                    log.info("%-14s %-10s c%-4d %-14s R@1 %.3f R@10 %.3f MRR %.3f nDCG %.3f", sysdef["name"],
                             run["embedder"] or "-", chunk, dname, s["R@1"]["mean"], s["R@10"]["mean"],
                             s["MRR@10"]["mean"], s["nDCG@10"]["mean"])
                r = None  # drop this retriever's cross-encoder before the next system loads its own
            if built:
                release_models()  # free this embedder's GPU memory before the next one
    return {"name": cfg["name"], "config": cfg, "runs": runs, "significance": significance(runs, cfg)}


def run_id(r: dict) -> str:
    return f"{r['system']}|{r['embedder'] or '-'}|c{r['chunk_tokens']}"


def significance(runs: list[dict], cfg: dict) -> dict:
    """Paired bootstrap (and Wilcoxon) of every system vs the baseline on nDCG@10 and MRR@10, Holm-corrected
    within each (dataset, chunk size, metric) family."""
    from eval.metrics import wilcoxon_p

    base = cfg.get("baseline", {"system": "dense", "embedder": "minilm"})
    out: dict = {}
    for d in {r["dataset"] for r in runs}:
        for c in {r["chunk_tokens"] for r in runs}:
            fam = [r for r in runs if r["dataset"] == d and r["chunk_tokens"] == c]
            b = next((r for r in fam if r["system"] == base["system"] and r["embedder"] == base["embedder"]), None)
            if b is None:
                continue
            for metric in ("nDCG@10", "MRR@10"):
                raw, wil, delta = {}, {}, {}
                for r in fam:
                    if r is b:
                        continue
                    a, bb = r["per_query"][metric], b["per_query"][metric]
                    raw[run_id(r)] = paired_bootstrap_p(a, bb)
                    wil[run_id(r)] = wilcoxon_p(a, bb)
                    delta[run_id(r)] = float(np.mean(a) - np.mean(bb))
                out[f"{d}|c{c}|{metric}"] = {
                    "baseline": run_id(b), "delta": delta, "p_bootstrap": raw, "p_bootstrap_holm": holm(raw),
                    "p_wilcoxon": wil, "p_wilcoxon_holm": holm(wil),
                }
    return out
