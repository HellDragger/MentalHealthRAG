"""Generation experiments: every model x profile (no_rag / naive / full) x dataset, with checkpoint/resume."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np

from eval.datasets import load_jsonl
from eval.judges import (
    JUDGE_VERSION,
    NLIJudge,
    abstained,
    answer_relevance,
    citation_stats,
    strip_citations,
)
from eval.metrics import bootstrap_ci, holm, paired_bootstrap_p, wilcoxon_p
from eval.tables import esc, fmt, write_table
from mhrag.config import Settings
from mhrag.index.builder import build_index
from mhrag.index.embedders import free_accelerator_memory
from mhrag.llm.registry import ModelManager, availability, load_catalog
from mhrag.pipeline import RAGPipeline
from mhrag.retrieval.factory import build_retriever
from mhrag.runtime import build_gate

log = logging.getLogger(__name__)
PROFILES = ("no_rag", "naive", "full")
_NO_ACCESS = re.compile(r"HTTP (401|403|404)\b")  # the API key cannot use this model: retrying will not help


class RateLimited(RuntimeError):
    """An API quota ran out. Everything so far is checkpointed; the run exits non-zero and a later run continues."""


# ------------------------------------------------------------------------------ generation
def _index_for(settings: Settings, cfg: dict, dataset: str, profile: str):
    """FAQ-Gen must not see the FAQ (or KB facts that copy it) -> 'nofaq' index variants."""
    nofaq = dataset == "faq_gen"
    ing = settings.ingest.model_copy(update={"exclude_faq_from_index": nofaq})
    s2 = settings.model_copy(update={"ingest": ing})
    if profile == "naive":
        emb, chunk = cfg["naive"]["embedder"], cfg["naive"]["chunk_tokens"]
    else:
        emb, chunk = settings.index.embedder, settings.chunking.chunk_tokens
    variant = "nofaq" if nofaq else ""
    build_index(s2, emb, chunk, variant)
    return s2, emb, chunk, variant


def generate(cfg: dict, settings: Settings, out_dir: Path, limit: int | None) -> list[dict]:
    catalog = load_catalog()
    retrievers: dict = {}
    gate = build_gate(settings)
    records = []
    for model in cfg["models"]:
        if cfg.get("run_models") and model not in cfg["run_models"]:
            records += _load_checkpoint(cfg, model, out_dir)  # reuse answers generated in earlier sessions
            continue
        status_path = out_dir / f"{model}__status.json"
        if _complete(cfg, model, out_dir, limit):  # finished in an earlier session: no need to load the model
            log.info("%s: all answers already in the checkpoint; skipping", model)
            records += _load_checkpoint(cfg, model, out_dir)
            records.append(json.loads(status_path.read_text()) if status_path.exists()
                           else {"model": model, "status": "ok"})
            continue
        ok, why = availability(catalog[model])
        if not ok:
            log.warning("skip %s: %s", model, why)
            records.append({"model": model, "status": f"TODO(run): {why}"})
            continue
        mm = ModelManager(catalog, [model], model, timeout_s=cfg.get("timeout_s", 300))
        load_t = time.perf_counter()
        try:
            try:
                backend = mm.get(model)
            except Exception as e:
                if not any(t in str(e).lower() for t in ("out of memory", "dispatched on the cpu")):
                    raise
                # memory still held from the previous model or the failed load: release it and try once more
                log.warning("%s: out of GPU memory while loading; freeing memory and retrying once", model)
                mm = ModelManager(catalog, [model], model, timeout_s=cfg.get("timeout_s", 300))
                free_accelerator_memory()
                backend = mm.get(model)
        except Exception as e:  # e.g. out of memory or disk; record it and continue with the next model
            log.error("skip %s: failed to load: %s: %s", model, type(e).__name__, e)
            records.append({"model": model, "status": f"TODO(run): failed to load: {type(e).__name__}: {str(e)[:200]}"})
            records += _load_checkpoint(cfg, model, out_dir)
            continue
        load_s = time.perf_counter() - load_t
        peak = _Peak()
        no_access = None
        for dataset in cfg["datasets"]:
            rows = _rows(cfg, dataset, limit)
            for row in rows:
                row.setdefault("query", row.get("text"))
            for profile in cfg["profiles"]:
                path = out_dir / f"{model}__{profile}__{dataset}.jsonl"
                done = {i: r for i, r in _answers(path).items() if not r.get("error")}  # failed ones are retried
                if profile == "no_rag":  # no retrieval, so no index to build
                    s2, retriever = settings, None
                else:
                    s2, emb, chunk, variant = _index_for(settings, cfg, dataset, profile)
                    key = (emb, chunk, variant)
                    if key not in retrievers:
                        retrievers[key] = build_retriever(s2, emb, chunk, variant,
                                                          need_reranker=s2.retrieval.mode.endswith("_rerank"))
                    retriever = retrievers[key]
                pipe = RAGPipeline(s2, retriever, mm, gate if profile == "full" else None)
                params = pipe.params(temperature=cfg.get("temperature", 0.0))
                with open(path, "a", encoding="utf-8") as f:
                    for row in rows:
                        if row["id"] in done:
                            continue
                        with peak:
                            try:
                                res = pipe.answer(row["query"], model=model, profile=profile, params=params)
                                err = None
                            except Exception as e:  # keep going; record the failure
                                res, err = {}, f"{type(e).__name__}: {e}"
                        rec = {"id": row["id"], "dataset": dataset, "model": model, "profile": profile,
                               "query": row["query"], "answer": res.get("answer", ""),
                               "context": res.get("context", ""), "sources": res.get("sources", []),
                               "gate": res.get("gate"), "generated": res.get("generated"),
                               "timing": res.get("timing", {}), "usage": res.get("usage", {}), "error": err}
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                        if err and "HTTP 429" in err:
                            raise RateLimited(f"{model}: {err[:300]}")
                        if err and _NO_ACCESS.search(err):
                            no_access = err
                            break
                        done[row["id"]] = rec
                if no_access:
                    break
                log.info("%s %s %s: %d answers", model, profile, dataset, len(done))
                records += [{**r, "reference": _ref(row_by_id(rows, r["id"]))} for r in done.values()
                            if row_by_id(rows, r["id"]) is not None]
            if no_access:
                break
        if no_access:
            log.error("skip %s: %s", model, no_access[:300])
            records = [r for r in records if r.get("model") != model]
            records.append({"model": model, "status": f"TODO(run): no access with this API key: {no_access[:200]}"})
        else:
            records.append({"model": model, "status": "ok", "load_seconds": load_s, "peak_rss_gb": peak.gb,
                            **_gpu_mem()})
            if _complete(cfg, model, out_dir, limit):
                status_path.write_text(json.dumps(records[-1]))  # marks the model finished (read by the notebook)
            else:  # e.g. out-of-memory on some answers: no status file, so the next run retries just those
                log.warning("%s: some answers failed; they are retried on the next run", model)
        mm._loaded.clear()
        backend.close()
        del backend, mm
        free_accelerator_memory()  # return the model's GPU memory before the next one loads
        if os.environ.get("MHRAG_FREE_MODEL_CACHE") == "1":
            free_model_cache(catalog[model])
    return records


def _rows(cfg: dict, dataset: str, limit: int | None) -> list[dict]:
    return load_jsonl(f"{dataset}.jsonl")[: cfg.get("limits", {}).get(dataset) or limit]


def _complete(cfg: dict, model: str, out_dir: Path, limit: int | None) -> bool:
    for dataset in cfg["datasets"]:
        want = {r["id"] for r in _rows(cfg, dataset, limit)}
        for profile in cfg["profiles"]:
            answers = _answers(out_dir / f"{model}__{profile}__{dataset}.jsonl")
            have = {i for i, r in answers.items() if not r.get("error")}
            if not want <= have:
                return False
    return True


def _answers(path: Path) -> dict:
    """Latest answer per question id; a successful retry replaces an earlier failed attempt, never the reverse."""
    out: dict = {}
    for r in read_checkpoint(path):
        if r["id"] not in out or out[r["id"]].get("error"):
            out[r["id"]] = r
    return out


def read_checkpoint(path: Path) -> list[dict]:
    """Answers in a checkpoint file. A run killed mid-write (e.g. at a session time limit) can leave a truncated
    last line; it is dropped and the file rewritten, so the next append starts on a clean line."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    rows, bad = [], 0
    for line in text.splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad or (text and not text.endswith("\n")):
        log.warning("%s: dropped %d incomplete line(s) from an interrupted run", path.name, bad)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return rows


def free_model_cache(spec) -> None:
    """Delete a finished model's weights from the Hugging Face cache (MHRAG_FREE_MODEL_CACHE=1). A 21-model run
    downloads several hundred GB, which does not fit on a Kaggle/Colab disk."""
    repo = spec.raw.get("hf_id") or spec.raw.get("gguf_repo")
    if not repo:
        return
    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
        revs = [rev.commit_hash for r in cache.repos if r.repo_id == repo for rev in r.revisions]
        if revs:
            strategy = cache.delete_revisions(*revs)
            strategy.execute()
            log.info("freed %s from the model cache (%s)", strategy.expected_freed_size_str, repo)
    except Exception as e:  # never fail the experiment over cache housekeeping
        log.warning("could not free the cache for %s: %s", repo, e)


def _load_checkpoint(cfg: dict, model: str, out_dir: Path) -> list[dict]:
    """Answers already on disk for a model that is not being (re)generated in this session."""
    recs = []
    for dataset in cfg["datasets"]:
        rows = load_jsonl(f"{dataset}.jsonl")
        for profile in cfg["profiles"]:
            path = out_dir / f"{model}__{profile}__{dataset}.jsonl"
            for r in _answers(path).values():
                row = row_by_id(rows, r["id"])
                if row is not None:
                    recs.append({**r, "reference": _ref(row)})
    return recs


def row_by_id(rows, rid):
    for r in rows:
        if r["id"] == rid:
            return r
    return None


def _ref(row):
    if row is None:
        return None
    return row.get("reference") or row.get("references")


class _Peak:
    """Peak resident memory across generations; None when psutil (an [eval] extra) is not installed."""

    def __init__(self):
        try:
            import psutil
        except ImportError:
            self.p, self.gb = None, None
            return
        self.p = psutil.Process()
        self.gb = self.p.memory_info().rss / 2**30

    def __enter__(self):
        return self

    def __exit__(self, *a):
        if self.p is not None:
            self.gb = max(self.gb, self.p.memory_info().rss / 2**30)


def _gpu_mem() -> dict:
    try:
        import torch

        if torch.cuda.is_available():
            return {"cuda_peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30}
    except Exception:
        pass
    return {}


# ------------------------------------------------------------------------------ scoring
def score(records: list[dict], cfg: dict, cache_dir: Path | None = None) -> list[dict]:
    answers = [r for r in records if "answer" in r]
    nli = NLIJudge(cfg.get("nli_model", "cross-encoder/nli-deberta-v3-base"))
    try:
        from rouge_score import rouge_scorer

        rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    except ImportError:
        rouge = None
    import textstat

    for r in answers:
        a = r["answer"] or ""
        refs = r.get("reference")
        refs = [refs] if isinstance(refs, str) else (refs or [])
        passages = [s["snippet"] for s in r.get("sources", [])]
        # use full chunk text when available (context) for NLI: split context blocks
        if r.get("context"):
            passages = [b for b in r["context"].split("\n\n[") if b.strip()] if r["profile"] == "full" else [r["context"]]
        m: dict = {}
        if r.get("generated") and r["profile"] != "no_rag":
            m.update(nli.score(a, passages[:8]))
        m["answer_relevance"] = answer_relevance(r["query"], a) if a else None
        if refs and rouge:
            m["rougeL"] = max(rouge.score(ref, strip_citations(a))["rougeL"].fmeasure for ref in refs)
        text = strip_citations(a).strip()
        m["flesch"] = float(textstat.flesch_reading_ease(text)) if text else None  # none for citation-only answers
        m["words"] = len(a.split())
        m["abstained"] = abstained(a)
        m.update(citation_stats(a, len(r.get("sources", []))))
        r["metrics"] = m
    _bertscore(answers, cfg)
    cfg["_judge_status"] = _llm_judge(answers, cfg, cache_dir)
    return answers


def _bertscore(answers, cfg):
    model = cfg.get("bertscore_model", "roberta-large")
    todo = [r for r in answers if r.get("reference") and r.get("answer")]
    if not todo or cfg.get("bertscore") is False:
        return
    from bert_score import score as bscore

    cands, refs, scored = [], [], []
    for r in todo:
        cand = strip_citations(r["answer"]).strip()
        if not cand:
            # an answer that is only citation markers (e.g. FLAN-T5's "[1]") has no content to compare: 0, as
            # bert_score itself scores an empty candidate (newer transformers break its empty-string path)
            r["metrics"]["bertscore_f1"] = 0.0
            r["metrics"]["bertscore_model"] = model
            continue
        ref = r["reference"]
        refs.append([ref] if isinstance(ref, str) else ref)
        cands.append(cand)
        scored.append(r)
    todo = scored
    if not todo:
        return
    # multi-reference: bert_score takes list of lists
    _, _, f1 = bscore(cands, refs, model_type=model, lang="en", verbose=False, batch_size=16)
    for r, v in zip(todo, f1.tolist(), strict=True):
        r["metrics"]["bertscore_f1"] = float(v)
        r["metrics"]["bertscore_model"] = model


def judge_sample_ids(cfg: dict, dataset: str, ids) -> set | None:
    """The questions the LLM judge scores: all, or `judge_sample` per dataset chosen by a fixed hash, so every model
    and setting is judged on the same questions (API quotas rarely allow judging every answer)."""
    n = cfg.get("judge_sample")
    if not n:
        return None
    return set(sorted(ids, key=lambda i: hashlib.sha1(f"{dataset}|{i}".encode()).hexdigest())[:n])


def _strip_judge(answers) -> None:
    for x in answers:
        for k in [k for k in x.get("metrics", {}) if k.startswith("judge_")]:
            del x["metrics"][k]


def _llm_judge(answers, cfg, cache_dir: Path | None = None) -> str:
    """Returns "complete", "incomplete" (quota ran out: verdicts so far are cached, and the scores are withheld so
    that no model is compared on a partial sample) or "unavailable"."""
    jm = cfg.get("judge_model")
    if not jm:
        return "unavailable"
    catalog = load_catalog()
    ok, why = availability(catalog[jm])
    if not ok:
        log.warning("LLM judge %s unavailable (%s): rubric scores left as TODO(run)", jm, why)
        return "unavailable"
    from eval.judges import LLMJudge
    from mhrag.llm.base import BackendError

    try:
        judge = LLMJudge(ModelManager(catalog, [jm], jm).get(jm))
    except BackendError as e:
        log.error("LLM judge %s could not be loaded (%s): rubric scores left as TODO(run)", jm, e)
        return "unavailable"
    samples = {d: judge_sample_ids(cfg, d, {r.get("id") for r in answers if r.get("dataset") == d})
               for d in {r.get("dataset") for r in answers}}
    todo = [r for r in answers if r.get("answer")
            and (samples[r.get("dataset")] is None or r.get("id") in samples[r.get("dataset")])]
    # Verdicts are cached per (judge, prompt version, answer), so an interrupted pass (session limit, daily quota)
    # continues where it stopped and repeated scoring does not call the API again.
    use_faith = cfg.get("judge_faithfulness", True)  # NLI already measures faithfulness; this saves ~half the tokens
    cache_path = cache_dir / "judge_cache.jsonl" if cache_dir else None
    cache = {c["key"]: c["metrics"] for c in (read_checkpoint(cache_path) if cache_path else [])}
    before = len(cache)
    for r in todo:
        ref = r.get("reference")
        ref = ref if isinstance(ref, str) else (ref[0] if ref else None)
        ctx = r.get("context") if r["profile"] != "no_rag" and use_faith else None
        key = hashlib.sha1(json.dumps([jm, JUDGE_VERSION, r["query"], r["answer"], ref, ctx]).encode()).hexdigest()
        if key in cache:
            continue
        jmetrics = {}
        try:
            rub = judge.rubric(r["query"], r["answer"], ref)
            if rub:
                jmetrics.update({f"judge_{k}": v for k, v in rub.items()})
            if ctx:
                jmetrics["judge_faithfulness"] = judge.faithfulness(r["answer"], ctx)
        except BackendError as e:
            if _NO_ACCESS.search(str(e)):
                log.error("LLM judge disabled: %s. Rubric scores are left as TODO(run).", e)
                _strip_judge(answers)
                return "unavailable"
            log.warning("LLM judge stopped after %d new verdicts (%s); the next run continues from the cache",
                        len(cache) - before, e)
            _strip_judge(answers)
            return "incomplete"
        # A verdict that could not be parsed is kept as missing (the judge answered; asking again gives the same)
        cache[key] = jmetrics
        if cache_path:
            with open(cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "metrics": jmetrics}) + "\n")
    for r in todo:
        ref = r.get("reference")
        ref = ref if isinstance(ref, str) else (ref[0] if ref else None)
        ctx = r.get("context") if r["profile"] != "no_rag" and use_faith else None
        key = hashlib.sha1(json.dumps([jm, JUDGE_VERSION, r["query"], r["answer"], ref, ctx]).encode()).hexdigest()
        r["metrics"].update(cache[key])
        r["metrics"]["judge_model"] = jm
    log.info("LLM judge %s: %d answers judged (%d new calls this run)", jm, len(todo), len(cache) - before)
    return "complete"


# ------------------------------------------------------------------------------ aggregation
METRICS = ["faithfulness", "contradiction_rate", "answer_relevance", "rougeL", "bertscore_f1", "flesch", "words",
           "has_citation", "abstained", "judge_empathy", "judge_safety", "judge_helpfulness", "judge_correctness",
           "judge_faithfulness"]


def aggregate(answers: list[dict], model_info: list[dict], cfg: dict) -> dict:
    groups: dict = {}
    for r in answers:
        groups.setdefault((r["model"], r["profile"], r["dataset"]), []).append(r)
    summary = []
    for (model, profile, dataset), rs in sorted(groups.items()):
        row = {"model": model, "profile": profile, "dataset": dataset, "n": len(rs),
               "errors": sum(bool(r.get("error")) for r in rs)}
        for m in METRICS:
            vals = [float(r["metrics"][m]) for r in rs if r.get("metrics", {}).get(m) is not None]
            if vals:
                mean, lo, hi = bootstrap_ci(vals, n_boot=1000)
                row[m] = {"mean": mean, "lo": lo, "hi": hi, "n": len(vals)}
        t = [r["timing"] for r in rs if r.get("generated")]
        for k in ("ttft_ms", "total_ms", "tokens_per_s"):
            v = [x[k] for x in t if x.get(k) is not None]
            if v:
                row[k] = {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "mean": float(np.mean(v))}
        row["gate_labels"] = {g: sum(r.get("gate") == g for r in rs) for g in {r.get("gate") for r in rs}}
        usage = [r.get("usage") or {} for r in rs]
        pin = sum(u.get("prompt_tokens") or 0 for u in usage)
        pout = sum(u.get("completion_tokens") or 0 for u in usage)
        spec = load_catalog().get(model)
        if spec and spec.backend == "openai_compatible":
            pi, po = spec.raw.get("price_per_m_input"), spec.raw.get("price_per_m_output")
            row["tokens_per_query"] = {"input": pin / len(rs), "output": pout / len(rs)}
            row["cost_per_1k_queries_usd"] = (
                1000 * (pin / len(rs) * pi + pout / len(rs) * po) / 1e6 if pi is not None and po is not None
                else "TODO(run): set price_per_m_input/output in configs/models.yaml"
            )
        summary.append(row)
    return {"summary": summary, "models": model_info, "significance": _significance(groups),
            "verbosity_bias": _verbosity(answers), "judge_version": JUDGE_VERSION}


def _significance(groups) -> dict:
    out = {}
    for model, dataset in {(m, d) for m, _, d in groups}:
        for metric in ("faithfulness", "rougeL", "bertscore_f1", "answer_relevance", "judge_helpfulness"):
            raw, wil, delta = {}, {}, {}
            for a, b in (("full", "naive"), ("full", "no_rag"), ("naive", "no_rag")):
                ga, gb = groups.get((model, a, dataset)), groups.get((model, b, dataset))
                if not ga or not gb:
                    continue
                va = {r["id"]: r["metrics"].get(metric) for r in ga if r.get("metrics")}
                vb = {r["id"]: r["metrics"].get(metric) for r in gb if r.get("metrics")}
                ids = [i for i in va if va[i] is not None and vb.get(i) is not None]
                if len(ids) < 5:
                    continue
                x, y = [va[i] for i in ids], [vb[i] for i in ids]
                k = f"{a}>{b}"
                raw[k], wil[k], delta[k] = paired_bootstrap_p(x, y), wilcoxon_p(x, y), float(np.mean(x) - np.mean(y))
            if raw:
                out[f"{model}|{dataset}|{metric}"] = {"delta": delta, "p_bootstrap": raw, "p_bootstrap_holm": holm(raw),
                                                      "p_wilcoxon": wil, "p_wilcoxon_holm": holm(wil)}
    return out


def _verbosity(answers) -> dict:
    from scipy.stats import spearmanr

    out = {}
    for k in ("judge_helpfulness", "judge_empathy", "judge_correctness"):
        pairs = [(r["metrics"]["words"], r["metrics"][k]) for r in answers if r.get("metrics", {}).get(k) is not None]
        if len(pairs) >= 10:
            rho, p = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
            out[k] = {"spearman_rho_vs_length": float(rho), "p": float(p), "n": len(pairs)}
    return out


def run_generation_experiment(cfg: dict, settings: Settings, limit: int | None = None) -> dict:
    out_dir = settings.results_dir / "generation" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    records = generate(cfg, settings, out_dir, limit)
    model_info = [r for r in records if "status" in r]
    answers = score([r for r in records if "answer" in r], cfg, cache_dir=out_dir)
    with open(out_dir / "scored.jsonl", "w", encoding="utf-8") as f:
        for r in answers:
            f.write(json.dumps({k: v for k, v in r.items() if k != "context"}, ensure_ascii=False) + "\n")
    judge_status = cfg.pop("_judge_status", "unavailable")
    res = aggregate(answers, model_info, cfg)
    res.update(name=cfg["name"], config=cfg, judge_status=judge_status)
    return res


# ------------------------------------------------------------------------------ tables
PROFILE_NAMES = {"no_rag": "No RAG", "naive": "Naive RAG (v1)", "full": "Full pipeline"}


def _cell(row, m, nd=3):
    v = row.get(m)
    if not v:
        return "--"
    return f"{v['mean']:.{nd}f}"


SAFETY_TEMPLATES = ("crisis", "harmful_request", "third_party")  # gate labels answered with a template, not generated


def _template_share(row: dict) -> str:
    labels = row.get("gate_labels") or {}
    if row["profile"] != "full" or not labels:
        return "--"
    return f"{sum(v for k, v in labels.items() if k in SAFETY_TEMPLATES) / row['n']:.2f}"


# Benchmarks with many models are split into these groups (one table each), plus a one-row-per-model summary.
MODEL_GROUPS = (("large", "Instruction-tuned models, 6--12B"), ("small", "Instruction-tuned models, up to 4B"),
                ("base", "Non-chat baselines (base language models, a summariser, seq2seq)"))
SPLIT_ABOVE = 6  # models


def model_group(key: str, catalog: dict) -> str:
    spec = catalog.get(key)
    if spec is None:
        return "small"
    if spec.family in ("completion", "seq2seq"):
        return "base"
    return "large" if (spec.raw.get("params_b") or 0) >= 6 else "small"


def _quality_caption(dataset: str, res: dict) -> str:
    return (f"Generation quality on {dataset.replace('_', '-')}. Faithfulness = share of answer sentences "
            "entailed by a retrieved passage (NLI, DeBERTa-v3); Relev. = question-answer embedding "
            "cosine; Cites = share of answers with an inline citation; Judge = LLM-as-judge (1--5), "
            "`--' where not run; Tmpl. = share of questions the risk gate answered with a fixed safety "
            "template instead of the model (full pipeline only; these are included in the other columns)."
            + (f" The judge scored a fixed sample of {res['config']['judge_sample']} questions per "
               "dataset, the same for every model and setting."
               if (res.get("config") or {}).get("judge_sample") else ""))


def _write_quality(name: str, suffix: str, dataset: str, rs: list, res: dict, prefix: str = "") -> None:
    table = []
    for r in rs:
        table.append([esc(r["model"]), PROFILE_NAMES.get(r["profile"], r["profile"]), _cell(r, "faithfulness"),
                      _cell(r, "answer_relevance"), _cell(r, "rougeL"), _cell(r, "bertscore_f1"),
                      _cell(r, "flesch", 1), _cell(r, "has_citation", 2), _cell(r, "judge_helpfulness", 2),
                      _template_share(r), fmt(r["ttft_ms"]["p50"] / 1000, 2) if r.get("ttft_ms") else "--"])
    write_table(f"generation_{name}_{dataset}{suffix}",
                ["Model", "Setting", "Faithful.", "Relev.", "ROUGE-L", "BERTScore", "Flesch", "Cites", "Judge help.",
                 "Tmpl.", "TTFT s"], table,
                caption=prefix + _quality_caption(dataset, res),
                label=f"tab:gen-{name}-{dataset.replace('_', '-')}{suffix.replace('_', '-')}", align="llrrrrrrrrr",
                source=f"results/generation_{name}.json")


def _write_oos(name: str, suffix: str, rs: list, prefix: str = "") -> None:
    table = [[esc(r["model"]), PROFILE_NAMES.get(r["profile"], r["profile"]), _cell(r, "abstained"),
              f"{1 - r['abstained']['mean']:.3f}" if r.get("abstained") else "--",
              _cell(r, "judge_safety", 2), str(r["n"])] for r in rs]
    write_table(f"generation_{name}_oos{suffix}", ["Model", "Setting", "Abstain/refuse", "Answered (halluc.)",
                                                  "Judge safety", "n"], table,
                caption=prefix + "Out-of-scope questions (50): rate at which the system declines or says it lacks "
                        "information (higher is better; answering is counted as a potential hallucination). "
                        "Abstention is detected with a fixed phrase list (eval/judges.py).",
                label=f"tab:gen-{name}-oos{suffix.replace('_', '-')}", align="llrrrr",
                source=f"results/generation_{name}.json")


def _run_notes(res: dict, groups: dict) -> str:
    """Caption note on runs left out (with the reason) and listed models without results yet."""
    shown = {m for ms in groups.values() for m in ms}
    pending = [m for m in (res.get("report_models") or []) if m not in shown]
    notes = ""
    if res.get("excluded_runs"):
        notes += " Excluded runs: " + "; ".join(f"{esc(m)} ({esc(why)})" for m, why in res["excluded_runs"].items()) + "."
    if pending:
        notes += " Pending: " + ", ".join(esc(m) for m in pending) + " \\todorun{notebooks/kaggle\\_experiments.ipynb}."
    return notes


def _write_summary(name: str, res: dict, groups: dict, catalog: dict) -> None:
    """One row per model: the comparisons that matter most, grouped like the per-setting tables."""
    rows = {(r["model"], r["profile"], r["dataset"]): r for r in res["summary"]}
    sig = res.get("significance", {})
    table = []
    for g, label in MODEL_GROUPS:
        if not groups.get(g):
            continue
        table.append([f"\\addlinespace\\multicolumn{{8}}{{l}}{{\\textit{{{label}}}}}"])
        for m in groups[g]:
            faq_naive, faq_full = rows.get((m, "naive", "faq_gen")) or {}, rows.get((m, "full", "faq_gen")) or {}
            oos_naive, oos_full = (rows.get((m, "naive", "oos_questions")) or {},
                                   rows.get((m, "full", "oos_questions")) or {})
            p = sig.get(f"{m}|faq_gen|faithfulness", {}).get("p_bootstrap_holm", {}).get("full>naive")
            faith_full = _cell(faq_full, "faithfulness")
            if faith_full != "--" and p is not None and p < 0.05:
                faith_full += "$^{*}$"
            size = catalog[m].raw.get("params_b") if m in catalog else None
            table.append([esc(m), f"{size:g}" if size else "--", _cell(faq_naive, "faithfulness"), faith_full,
                          _cell(faq_full, "answer_relevance"), _cell(faq_full, "has_citation", 2),
                          _cell(oos_naive, "abstained", 2), _cell(oos_full, "abstained", 2)])
    write_table(f"generation_{name}_summary",
                ["Model", "Size (B)", "Faithful. naive", "Faithful. full", "Relev. full", "Cites full",
                 "OOS declined naive", "OOS declined full"], table,
                caption="All models, naive RAG (the v1 configuration) versus the full pipeline. Faithfulness and "
                        "relevance on FAQ-Gen; Cites = share of FAQ-Gen answers with an inline citation; OOS declined "
                        "= share of the 50 out-of-scope questions declined. $^{*}$ faithfulness differs significantly "
                        "between the two settings (paired bootstrap, Holm-corrected $p<0.05$). Per-setting results, "
                        "including no retrieval and Counsel-Gen, are in the appendix." + _run_notes(res, groups),
                label=f"tab:gen-{name}-summary", align="lrrrrrrr", source=f"results/generation_{name}.json")


def render_tables(res: dict) -> None:
    name = res["name"]
    rows = res["summary"]
    if res.get("report_models") is not None:  # the current config's list (excluded runs are not in it)
        rows = [r for r in rows if r["model"] in res["report_models"]]
    listed = res.get("report_models") or (res.get("config") or {}).get("models", [])
    order = [m for m in listed if any(r["model"] == m for r in rows)]
    order += sorted({r["model"] for r in rows} - set(order))
    grouped = len(order) > SPLIT_ABOVE
    groups: dict = {}
    if grouped:  # one table per model group (each fits a page) plus a summary table
        catalog = load_catalog()
        for m in order:
            groups.setdefault(model_group(m, catalog), []).append(m)
        _write_summary(name, res, groups, catalog)
    for dataset in sorted({r["dataset"] for r in rows}):
        rs = [r for r in rows if r["dataset"] == dataset]
        parts = ([("", rs, "")] if not grouped else
                 [(f"_{g}", [r for m in groups.get(g, []) for r in rs if r["model"] == m], f"{label}. ")
                  for g, label in MODEL_GROUPS])
        for suffix, sub, prefix in parts:
            if not sub:
                continue
            if dataset == "oos_questions":
                _write_oos(name, suffix, sub, prefix)
            else:
                _write_quality(name, suffix, dataset, sub, res, prefix)
