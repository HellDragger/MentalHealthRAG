"""Generate paper/numbers.tex (LaTeX macros) and paper/tables/latency.tex from results/*.json.

    python -m scripts.paper_numbers

Every number quoted in the paper text is a macro defined here from a results file. A result that does not exist yet
becomes \\todorun{<command that produces it>}, so the manuscript can never show an invented value.
"""

from __future__ import annotations

import json

from eval.tables import esc as tex_esc
from eval.tables import fmt, fmt_ci, write_table
from mhrag.config import PROJECT_ROOT, get_settings

_WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
R = get_settings().results_dir
OUT = PROJECT_ROOT / "paper" / "numbers.tex"


def load(name: str) -> dict | None:
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def f3(x) -> str:
    return f"{x:.3f}"


class Macros:
    def __init__(self):
        self.lines: list[str] = []

    def set(self, name: str, value, cmd: str):
        if value is None:
            v = f"\\todorun{{{cmd}}}"
        else:
            v = str(value).replace("%", "\\%").replace("_", "\\_") if not str(value).startswith("\\") else str(value)
        self.lines.append(f"\\newcommand{{\\{name}}}{{{v}}}")

    def text(self, name: str, value: str | None, cmd: str):
        self.lines.append(f"\\newcommand{{\\{name}}}{{{value if value else chr(92) + 'todorun{' + cmd + '}'}}}")


SYS = {"bm25": "BM25", "dense": "dense retrieval", "hybrid": "hybrid (RRF) retrieval",
       "hybrid+ce": "hybrid retrieval with MiniLM cross-encoder reranking",
       "hybrid+bge-rr": "hybrid retrieval with BGE reranking"}
EMB = {"minilm": "MiniLM-L6", "bge-small": "BGE-small", "bge-base": "BGE-base", "e5-base": "E5-base", "gte-base": "GTE-base"}
WORDS = {2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}
DS = {"heading_qa": "HeadingQA", "synth_qa": "SynthQA", "paraphrase_qa": "ParaphraseQA"}
# Short labels used in running text and tables (defined in the paper's experimental setup)
SYS_SHORT = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid", "hybrid+ce": "H+CE", "hybrid+bge-rr": "H+BGE"}
EMB_SHORT = {"minilm": "MiniLM", "bge-small": "BGE-s", "bge-base": "BGE-b", "e5-base": "E5", "gte-base": "GTE"}


def short(r) -> str:
    return SYS_SHORT.get(r["system"], r["system"]) + (f" ({EMB_SHORT.get(r['embedder'], r['embedder'])})"
                                                      if r["embedder"] else "")


def get_catalog_label(key: str) -> str:
    from mhrag.llm.registry import load_catalog

    spec = load_catalog().get(key)
    return spec.raw.get("api_model", key) if spec else key


def sysname(r) -> str:
    return SYS.get(r["system"], r["system"]) + (f" ({EMB.get(r['embedder'], r['embedder'])} embeddings)" if r["embedder"] else "")


def get(d, *path):
    for p in path:
        if d is None:
            return None
        d = d.get(p) if isinstance(d, dict) else None
    return d


def main():
    m = Macros()

    # ---------------------------------------------------------------- ingestion
    ing = load("ingest_stats.json")
    c_ing = "python -m scripts.build_index"
    pdf = get(ing, "ingest", "pdf")
    m.set("NPDFs", get(pdf, "files"), c_ing)
    m.set("NPDFsParsed", get(pdf, "parsed"), c_ing)
    bst = get(pdf, "by_source_type") or {}
    m.set("NMindWeb", bst.get("mind_web"), c_ing)
    m.set("NMindBooklet", bst.get("mind_booklet"), c_ing)
    m.set("NMindPDFs", (bst.get("mind_web", 0) + bst.get("mind_booklet", 0)) or None, c_ing)
    m.set("NWebArticles", bst.get("web_article"), c_ing)
    m.set("NBoilerplateLines", get(pdf, "cross_doc_boilerplate_lines_removed"), c_ing)
    m.set("PdfExtractSeconds", get(pdf, "extract_seconds_pypdfium2"), c_ing)
    m.set("NChunks", get(ing, "chunking", "chunks"), c_ing)
    m.set("ChunkTokens", get(ing, "chunking", "chunk_tokens"), c_ing)
    m.set("ChunkMedianTokens", get(ing, "chunking", "tokens_per_chunk", "median"), c_ing)
    m.set("NExactDups", get(ing, "chunking", "exact_duplicates_removed"), c_ing)
    m.set("NNearDups", get(ing, "chunking", "near_duplicates_removed"), c_ing)
    m.set("NFAQ", get(ing, "ingest", "faq", "questions"), c_ing)
    m.set("NKBFactIntents", get(ing, "ingest", "kb", "kb_fact_intents"), c_ing)
    m.set("NKBFactsIndexed", get(ing, "ingest", "kb", "indexed_facts"), c_ing)
    import re

    changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text()
    m.set("NBugsAudited", len(set(re.findall(r"^\| (B\d+) \|", changelog, re.M))), "")

    # ---------------------------------------------------------------- eval sets
    es = load("eval_sets.json")
    c_es = "python -m scripts.make_eval_sets"
    for key, name in (("heading_qa", "NHeadingQA"), ("paraphrase_qa", "NParaphraseQA"), ("faq_gen", "NFAQGen"),
                      ("counsel_gen", "NCounselGen"), ("oos", "NOOS")):
        m.set(name, get(es, key, "n"), c_es)
    m.set("NSynthQA", get(es, "synth_qa", "kept_after_roundtrip"), c_es + " --synth")
    m.set("NCounselRows", get(es, "counsel_source", "rows"), "python -m scripts.make\\_eval\\_sets --skip-static")

    # ---------------------------------------------------------------- classifier
    rc = load("risk_classifier.json")
    c_rc = "python -m scripts.train_risk_classifier"
    m.set("NClfRows", get(rc, "split", "dedup", "after_dedup"), c_rc)
    for key, pre in (("tfidf_lr", "Tfidf"), ("transformer", "Trans")):
        mod = get(rc, "models", key)
        t = get(mod, "test_at_tau")
        m.set(f"{pre}AUROC", f3(t["auroc"]) if t else None, c_rc)
        m.set(f"{pre}Recall", f3(t["recall"]) if t else None, c_rc)
        m.set(f"{pre}Precision", f3(t["precision"]) if t else None, c_rc)
        ks = get(mod, "external", "kb_suicide_intent_patterns")
        m.set(f"{pre}KBSuicideRecall", f3(ks["recall"]) if ks else None, c_rc)
        kd = get(mod, "external", "kb_distress_intent_patterns")
        m.set(f"{pre}DistressFlag", f3(kd["flag_rate_at_tau"]) if kd else None, c_rc)
    m.set("NKBSuicide", get(rc, "external_sets", "kb_suicide_intent_patterns", "n"), c_rc)

    # ---------------------------------------------------------------- safety gate
    dep = "lexicon+tfidf_lr"
    for tag, pre in (("v1_devset", "GateVOneDev"), ("v1_heldout", "GateVOneHeld"), ("v2_devset", "GateVTwoDev"),
                     ("v2_heldout", "GateVTwoHeld")):
        g = load(f"safety_gate_{tag}.json")
        cmd = f"python -m scripts.eval_safety --tag {tag}"
        v = get(g, "variants", dep)
        m.set(f"{pre}Recall", f3(v["crisis_recall_any_escalation"]) if v else None, cmd)
        m.set(f"{pre}FPR", f3(v["escalation"]["false_positive_rate"]) if v else None, cmd)
        m.set(f"{pre}Acc", f3(v["overall_accuracy"]) if v else None, cmd)
        m.set(f"{pre}AnySafety", f3(v["crisis_recall_any_safety_response"]) if v else None, cmd)
        if v:  # counts behind the rates (the sets are small)
            pc = v["per_category"]
            nc = pc["explicit_crisis"]["n"] + pc["implicit_crisis"]["n"]
            m.set(f"{pre}RecallK", round(v["crisis_recall_any_escalation"] * nc), cmd)
            m.set(f"{pre}CrisisN", nc, cmd)
            m.set(f"{pre}FP", f"{v['escalation']['fp']}/{v['escalation']['fp'] + v['escalation']['tn']}", cmd)
        else:
            for k in ("RecallK", "CrisisN", "FP"):
                m.set(f"{pre}{k}", None, cmd)
        if tag.startswith("v2"):
            lex = get(g, "variants", "lexicon_only")
            m.set("LexHeldRecallVTwo" if "held" in tag else "LexDevRecallVTwo",
                  f3(lex["crisis_recall_any_escalation"]) if lex else None, cmd)
    m.set("NSafetyDev", get(load("safety_gate_v2_devset.json"), "n_prompts"), "python -m scripts.eval_safety")
    m.set("NSafetyHeld", get(load("safety_gate_v2_heldout.json"), "n_prompts"), "python -m scripts.eval_safety")

    # ---------------------------------------------------------------- latency
    lat = load("latency.json") or {}
    c_lat = "python -m scripts.bench_latency --all"
    env = next((v.get("environment") for v in lat.values() if isinstance(v, dict) and v.get("environment")), None)
    m.set("Hardware", f"{env.get('cpu', env.get('machine'))} ({env.get('ram_gb', 0):.0f} GB RAM)" if env else None, c_lat)
    rows = []
    # Row labels as the paper presents them: one system, with an unoptimised baseline (the result keys keep their
    # historical v1_/v2_ prefixes).
    names = {"v1_hf_fp32_cpu": "Unoptimised baseline (fp32, CPU, no streaming)",
             "v2_llamacpp_cpu": "llama.cpp Q4\\_K\\_M, CPU", "v2_llamacpp_metal": "llama.cpp Q4\\_K\\_M, Metal",
             "v2_hf_mps": "transformers fp16, MPS", "v2_hf_cuda": "transformers 16-bit, CUDA",
             "v2_hf_cuda_4bit": "transformers NF4, CUDA"}
    # (no API row: the app serves local models only, and bench_latency's v2_api config is kept for manual runs)
    for k, label in names.items():
        v = lat.get(k)
        s = get(v, "summary")
        if not s:
            rows.append([label, "--", "\\multicolumn{5}{l}{\\todorun{python -m scripts.bench\\_latency --config " +
                         k.replace("_", "\\_") + "}}"])
            continue
        if s.get("throttled_queries"):
            label += f" ({s['throttled_queries']} of {s['n']} rate-limited)"
        mem = v.get("peak_rss_gb")
        rows.append([label, str(v.get("model", "")).replace("_", "\\_"), f"{s['ttft_ms']['p50'] / 1000:.2f}",
                     f"{s['e2e_ms']['p50'] / 1000:.1f}", f"{s['e2e_ms']['p95'] / 1000:.1f}",
                     f"{s['tokens_per_s']['p50']:.1f}", f"{mem:.1f}" if mem else "--"])
    if lat:
        write_table("latency", ["Configuration", "Model", "TTFT p50 (s)", "E2E p50 (s)", "E2E p95 (s)", "tok/s", "Peak RSS (GB)"],
                    rows, caption="Latency on the laptop for 12 fixed questions (the unoptimised baseline answered 5). "
                                  "TTFT: time to first streamed token; E2E: end-to-end time (the same for the "
                                  "non-streaming baseline). All rows but the baseline include the gate, hybrid retrieval "
                                  "and reranking; peak RSS excludes GPU memory held by Metal.",
                    label="tab:latency", align="llrrrrr", source="results/latency.json")
    v1, cpu, metal = get(lat, "v1_hf_fp32_cpu", "summary"), get(lat, "v2_llamacpp_cpu", "summary"), get(lat, "v2_llamacpp_metal", "summary")
    same = None
    if v1 and cpu:
        import statistics

        # the baseline answered 5 of the 12 benchmark questions; compare on exactly those
        cpu_rows = {r["query"]: r for r in lat["v2_llamacpp_cpu"]["rows"]}
        qs = [r["query"] for r in lat["v1_hf_fp32_cpu"]["rows"] if r["query"] in cpu_rows]
        same = {"n": len(qs), "base_e2e": statistics.median(r["e2e_ms"] for r in lat["v1_hf_fp32_cpu"]["rows"]
                                                              if r["query"] in qs) / 1000,
                "e2e": statistics.median(cpu_rows[q]["e2e_ms"] for q in qs) / 1000,
                "ttft": statistics.median(cpu_rows[q]["ttft_ms"] for q in qs) / 1000}
        txt = (f"The two optimisations act on different quantities. On the {same['n']} questions both configurations "
               f"answered, quantised weights in llama.cpp cut the median end-to-end time on the laptop CPU from "
               f"{same['base_e2e']:.1f}~s (full-precision transformers) to {same['e2e']:.1f}~s "
               f"({v1['tokens_per_s']['p50']:.1f} versus {cpu['tokens_per_s']['p50']:.1f} tokens/s over all questions). "
               f"Streaming does not shorten the answer, but it shows the first words after a median {same['ttft']:.1f}~s "
               "instead of only when the answer is complete")
        if metal:
            txt += (f"; with Metal offload the first token appeared after {metal['ttft_ms']['p50'] / 1000:.2f}~s at "
                    f"{metal['tokens_per_s']['p50']:.1f} tokens/s")
        txt += (" (Table~\\ref{tab:latency}). These laptop figures are not comparable with the TTFT column of the "
                "generation tables, which was measured on the Kaggle GPU (llama.cpp offloaded to the GPU for the GGUF "
                "model)")
        lc = load("latency_cpu.json") or {}
        keys_c = ("v1_hf_fp32_cpu", "v2_hf_fp32_cpu", "v2_llamacpp_f16_cpu", "v2_llamacpp_cpu")
        k_ = {k: get(lc, k, "summary") for k in keys_c}
        if all(k_.values()):
            import datetime as _dt

            b_, h_, f_, q_ = (k_[k] for k in keys_c)
            # configurations measured more than an hour apart ran in different Kaggle sessions (possibly other CPUs)
            when = sorted((_dt.datetime.fromisoformat(lc[k]["ran_at"]), k) for k in keys_c if lc[k].get("ran_at"))
            sessions, last = [], None
            for t_, k in when:
                if last is None or (t_ - last).total_seconds() > 3600:
                    sessions.append([])
                sessions[-1].append(k)
                last = t_
            envs = {json.dumps({x: (lc[k].get("environment") or {}).get(x) for x in ("platform", "ram_gb", "torch",
                                                                                   "transformers", "llama_cpp")})
                    for k in keys_c}
            cpus = {(lc[k].get("environment") or {}).get("cpu") for k in keys_c}
            cpu_known = len(cpus) == 1 and next(iter(cpus)) not in (None, "", "x86_64")
            nw = {2: "two", 3: "three", 4: "four"}.get(len(sessions), str(len(sessions)))
            where = "in one session" if len(sessions) <= 1 else (
                f"in {nw} sessions" + (" with the same software versions and memory" if len(envs) == 1 else ""))
            caveat = "" if len(sessions) <= 1 or cpu_known else (
                f"; the CPU model was not recorded, so the {nw} sessions may have run on different processors")
            txt += (f". To separate the effects, all four configurations were also run on Kaggle CPUs {where} "
                    f"(Table~\\ref{{tab:latency-cpu}}){caveat}. Median end-to-end times were "
                    f"{b_['e2e_ms']['p50'] / 1000:.1f}~s (transformers fp32, no streaming), "
                    f"{h_['e2e_ms']['p50'] / 1000:.1f}~s (the same with streaming and the full pipeline), "
                    f"{f_['e2e_ms']['p50'] / 1000:.1f}~s (llama.cpp, 16-bit) and {q_['e2e_ms']['p50'] / 1000:.1f}~s "
                    f"(llama.cpp, 4-bit), and the first words appeared after {b_['ttft_ms']['p50'] / 1000:.1f}, "
                    f"{h_['ttft_ms']['p50'] / 1000:.1f}, {f_['ttft_ms']['p50'] / 1000:.1f} and "
                    f"{q_['ttft_ms']['p50'] / 1000:.1f}~s")
            names_c = {"v1_hf_fp32_cpu": "transformers fp32, no streaming (baseline)",
                       "v2_hf_fp32_cpu": "transformers fp32, streaming", "v2_llamacpp_f16_cpu": "llama.cpp F16, streaming",
                       "v2_llamacpp_cpu": "llama.cpp Q4\\_K\\_M, streaming"}
            sess_of = {k: i + 1 for i, ks in enumerate(sessions) for k in ks}
            from eval.tables import esc as _esc
            cap = ("Qwen2.5-1.5B-Instruct on Kaggle CPUs" + (f" ({_esc(next(iter(cpus)))})" if cpu_known else "")
                   + ", varying precision, runtime and streaming one at a time. The baseline uses naive retrieval and "
                   "its own n questions; the other rows run the full pipeline. TTFT: time to first token; E2E: "
                   "end-to-end time.")
            if len(sessions) > 1:
                notes = (["same software"] if len(envs) == 1 else []) + ([] if cpu_known else ["CPU model not recorded"])
                cap += " Session: the Kaggle session each row was measured in" + (f" ({'; '.join(notes)})" if notes else "") + "."
            write_table("latency_cpu", ["Configuration", "n", "TTFT p50 (s)", "E2E p50 (s)", "tok/s"]
                        + (["Session"] if len(sessions) > 1 else []),
                        [[names_c[k], str(v["n"]), f"{v['ttft_ms']['p50'] / 1000:.2f}", f"{v['e2e_ms']['p50'] / 1000:.1f}",
                          f"{v['tokens_per_s']['p50']:.1f}"] + ([str(sess_of.get(k, "--"))] if len(sessions) > 1 else [])
                         for k, v in k_.items()],
                        cap, "tab:latency-cpu", align="lrrrr" + ("r" if len(sessions) > 1 else ""),
                        source="results/latency_cpu.json")
        txt += "."
        m.text("LatencySummary", txt, c_lat)
    else:
        m.text("LatencySummary", None, c_lat)
    rl = get(lat, "retrieval", "modes")
    m.set("RetrievalHybridRerankMs", f"{rl['hybrid_rerank']['cold_ms']['p50']:.0f}" if rl else None, c_lat)

    # ---------------------------------------------------------------- retrieval
    rm = load("retrieval_main.json")
    c_rm = "python -m scripts.run_eval --config configs/experiments/retrieval_main.yaml"
    if rm:
        parts = []
        for ds, label in (("heading_qa", "HeadingQA"), ("synth_qa", "SynthQA"), ("paraphrase_qa", "ParaphraseQA")):
            runs = [r for r in rm["runs"] if r["dataset"] == ds]
            if not runs:
                continue
            best = max(runs, key=lambda r: r["summary"]["nDCG@10"]["mean"])
            base = next((r for r in runs if r["system"] == "dense" and r["embedder"] == "minilm"), None)
            bm = next((r for r in runs if r["system"] == "bm25"), None)
            b = best["summary"]["nDCG@10"]
            s = f"{label}: {short(best)}, {f3(b['mean'])} [{f3(b['lo'])}, {f3(b['hi'])}]"
            if base:
                s += f" (baseline {f3(base['summary']['nDCG@10']['mean'])}"
                s += f", BM25 {f3(bm['summary']['nDCG@10']['mean'])})" if bm else ")"
            parts.append(s)
        m.text("RetrievalSummary", "Table~\\ref{tab:retrieval-summary} summarises retrieval; the full tables for every "
               "configuration are in \\ref{app:retrieval}. The best configuration and its nDCG@10 on each set "
               "were " + "; ".join(parts) + ".", c_rm)
    else:
        m.text("RetrievalSummary", None, c_rm)

    # ---------------------------------------------------------------- generation
    gl = load("generation_local.json")
    c_gl = "python -m scripts.run_eval --config configs/experiments/generation_local.yaml"
    if gl:
        parts = []
        order = (gl.get("config") or {}).get("models") or []
        models = sorted(dict.fromkeys(r["model"] for r in gl["summary"]),
                        key=lambda k: order.index(k) if k in order else len(order))
        for model in models[:1]:  # the deployed model; the weak baselines are reported in the appendix tables
            row = {(r["profile"], r["dataset"]): r for r in gl["summary"] if r["model"] == model}
            full, naive, norag = row.get(("full", "faq_gen")), row.get(("naive", "faq_gen")), row.get(("no_rag", "faq_gen"))
            if full and naive and full.get("faithfulness") and naive.get("faithfulness"):
                s = (f"With Qwen2.5-1.5B (GGUF), NLI faithfulness on FAQ-Gen was {f3(full['faithfulness']['mean'])} with "
                     f"the full pipeline and {f3(naive['faithfulness']['mean'])} with naive RAG, while ROUGE-L barely moved "
                     f"({f3(full['rougeL']['mean'])}, {f3(naive['rougeL']['mean'])} and {f3(norag['rougeL']['mean'])} "
                     "without retrieval)" if full.get("rougeL") and norag and norag.get("rougeL") else
                     f"With Qwen2.5-1.5B (GGUF), NLI faithfulness on FAQ-Gen was {f3(full['faithfulness']['mean'])} with "
                     f"the full pipeline and {f3(naive['faithfulness']['mean'])} with naive RAG")
                oos, oosn = row.get(("full", "oos_questions")), row.get(("naive", "oos_questions"))
                if oos and oos.get("abstained") and oosn and oosn.get("abstained"):
                    s += (f". It declined {oos['abstained']['mean']:.2f} of out-of-scope questions, against "
                          f"{oosn['abstained']['mean']:.2f} with naive RAG")
                parts.append(s + ". Per-setting results for GPT-2 and BART-large-CNN, kept as weak baselines, are reported in the code repository.")
        if gl.get("judge_status") == "complete":
            tail = ""
        else:
            tail = (" The LLM-judge column remains "
                    "\\todorun{" + c_gl.replace("_", "\\_") + " (continues until the judge sample is complete)}.")
        m.text("GenerationSummary", " ".join(parts) + tail, c_gl)
    else:
        m.text("GenerationSummary", None, c_gl)

    # ---------------------------------------------------------------- judge design (from the experiment config)
    import yaml

    jcfg = yaml.safe_load((PROJECT_ROOT / "configs" / "experiments" / "generation_local.yaml").read_text())
    jm = jcfg.get("judge_model")
    m.set("JudgeModel", (get_catalog_label(jm) if jm else None), "set judge_model in generation_local.yaml")
    m.set("JudgeSample", jcfg.get("judge_sample") or "all", "set judge_sample in generation_local.yaml")

    # ---------------------------------------------------------------- gate on ordinary questions
    ge = load("gate_escalation.json")
    c_ge = "python -m scripts.eval_gate_escalation"
    if ge and "faq_gen" in ge["datasets"] and "oos_questions" in ge["datasets"]:
        def esc(ds):
            d = ge["datasets"][ds]
            return d["labels"].get("crisis", 0), d["labels"].get("elevated", 0), d["n"], d["escalations_by_source"]
        fc, fe, fn, fsrc = esc("faq_gen")
        oc, oe, on, osrc = esc("oos_questions")
        srcs = {k.split("|")[1] for k in list(fsrc) + list(osrc)}
        who = "the classifier alone" if srcs == {"classifier"} else "+".join(sorted(srcs))
        m.text("GateOnRealQuestions",
               f"On the {fn} FAQ-Gen questions, all of them informational, the deployed gate sent {fc} to the crisis "
               f"protocol and {fe} to the elevated tier; on the {on} out-of-scope questions it sent {oc} and {oe}. "
               f"All of these escalations were triggered by {who}.", c_ge)
    else:
        m.text("GateOnRealQuestions", None, c_ge)

    # ---------------------------------------------------------------- held-out harmful requests the gate missed
    gh_ = load("safety_gate_v2_heldout.json")
    gd_ = load("safety_gate_v2_devset.json")
    c_hm = "python -m scripts.eval\\_safety --data eval/data/safety\\_prompts\\_heldout.jsonl --tag v2\\_heldout"
    vh, vd = get(gh_, "variants", dep), get(gd_, "variants", dep)
    if vh and vd and get(vh, "per_category", "harmful_request"):
        ph, pd_ = vh["per_category"]["harmful_request"], vd["per_category"]["harmful_request"]
        lab = ph["labels"]
        caught = lab.get("harmful_request", 0) + lab.get("crisis", 0) + lab.get("third_party", 0)
        miss = [e for e in vh["errors"] if e["expected"] == "harmful_request"]
        none_ = [e for e in miss if e["got"] == "none"]
        thr = vh.get("crisis_threshold")
        guarded = [e for e in none_ if thr is not None and (e.get("score") or 0) >= thr]
        texts = {}
        hp = PROJECT_ROOT / "eval" / "data" / "safety_prompts_heldout.jsonl"
        for line in hp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r_ = json.loads(line)
                texts[r_["id"]] = r_["text"]
        ex = next((texts.get(e["id"]) for e in guarded if texts.get(e["id"])), None)
        dev_caught = round(pd_["escalated"] * pd_["n"])
        m.text("HarmfulMissSummary",
               f"(iv) Requests for suicide or self-harm methods: the gate escalated {caught} of {ph['n']} held-out "
               f"harmful requests, against {dev_caught} of {pd_['n']} on the development set. Of the {len(miss)} it "
               f"missed, {lab.get('elevated', 0)} went to the non-blocking elevated tier and {len(none_)} were not "
               f"flagged at all. {_WORDS.get(len(guarded), str(len(guarded))).capitalize()} of the unflagged ones had a classifier score above the crisis threshold "
               "but were phrased as general questions without a first-person pronoun"
               + (f" (``{tex_esc(ex.rstrip('?'))}?'')" if ex else "")
               + ", so the informational-question guard, which keeps factual questions from being escalated, overrode "
               "the classifier, and the lexicon's method patterns did not match their wording. I left the gate "
               "unchanged after seeing these results so that the held-out set stays held out; exempting method "
               "questions from the guard is the obvious fix, and it needs a new held-out set to evaluate.", c_hm)
    else:
        m.text("HarmfulMissSummary", None, c_hm)

    # ---------------------------------------------------------------- SynthQA human verification
    sv = load("synth_qa_verification.json")
    c_sv = "python -m scripts.synth_qa_agreement"
    if sv:
        cr, nr, k = sv["criteria"], len(sv["raters"]), sv["items"]
        need = nr // 2 + 1
        ans, nat = cr["answerable"], cr["natural"]
        m.text("SynthVerifySummary",
               f"{WORDS.get(nr, str(nr))} raters, including me, independently judged a random {k}-question subset: whether the "
               "source passage answers the question, and whether the question reads like one a person would ask. "
               f"With every rater weighted equally and an item counted only when at least {need} of {nr} said yes "
               f"(ties fail), {ans['pass_strict_majority']} were answerable, {nat['pass_strict_majority']} natural and "
               f"{sv['verified']['n']} both (\\ref{{app:synthqa}}). Raw agreement between rater pairs was "
               f"{ans['mean_pairwise_agreement']:.2f} and {nat['mean_pairwise_agreement']:.2f} for the two questions, but Fleiss' $\\kappa$ "
               f"was low ({f3(ans['fleiss_kappa'][0])} and {f3(nat['fleiss_kappa'][0])}) because almost every answer "
               "was yes, a known limitation of $\\kappa$ \\citep{feinstein1990high}; Gwet's AC1, which is robust to "
               f"this \\citep{{gwet2008computing}}, was {f3(ans['gwet_ac1'][0])} and {f3(nat['gwet_ac1'][0])}.", c_sv)
        rows = []
        for key, name in (("answerable", "Answerable from passage"), ("natural", "Natural question")):
            a, wa = cr[key], sv["without_author"].get(key)
            p = a["pass_rate"]
            rows.append([name, f"{a['pass_strict_majority']}/{a['n_items']} {{\\scriptsize [{p[1]:.2f}, {p[2]:.2f}]}}",
                         str(a["ties"]), f"{a['mean_pairwise_agreement']:.2f}", fmt_ci(*a["fleiss_kappa"]),
                         fmt_ci(*a["gwet_ac1"]), fmt_ci(*wa["gwet_ac1"]) if wa else "--"])
        v = sv["verified"]
        rows.append(["Both (verified)", f"{v['n']}/{k} {{\\scriptsize [{v['rate'][1]:.2f}, {v['rate'][2]:.2f}]}}",
                     "--", "--", "--", "--", "--"])
        write_table("synth_qa_verification",
                    ["Criterion", "Pass (95\\% CI)", "Ties", "Raw agr.", "Fleiss $\\kappa$", "Gwet AC1",
                     "AC1 w/o author"], rows,
                    f"Human verification of {k} SynthQA questions by {nr} raters. Pass: at least {need} of {nr} raters "
                    "said yes (Wilson 95\\% CI); " + (f"Ties: questions on which {nr // 2} of {nr} raters said yes; " if nr % 2 == 0 else "") + "Raw agr.: mean agreement over rater pairs; $\\kappa$ "
                    "and AC1 with 95\\% bootstrap CIs; AC1 w/o author: AC1 without my own ratings.",
                    "tab:synthqa-verification", source="results/synth_qa_verification.json")
        rr = sv.get("retrieval_robustness")
        if rr:
            ra, vb = rr["rank_agreement"]["nDCG@10"], rr["vs_baseline_verified"]["nDCG@10"]
            sig = [r for r, p in vb["p_bootstrap_holm"].items() if p < 0.05]
            hyb = [r for r in vb["delta"] if r.startswith("hybrid")]
            m.text("SynthVerifiedRobustness",
                   f"On the {rr['n']['verified']} verified SynthQA questions, the ordering of the "
                   f"{len(rr['runs'])} configurations by nDCG@10 was largely unchanged (Kendall $\\tau$ = "
                   f"{ra['kendall_tau']:.2f}), and {sum(r in sig for r in hyb)} of {len(hyb)} hybrid configurations "
                   "beat the dense MiniLM baseline on this subset after Holm correction (Table~\\ref{tab:synthqa-robustness}).",
                   c_sv)
            label = lambda r: sysname({"system": r.split("|")[0], "embedder": None if r.split("|")[1] == "-"  # noqa: E731
                                       else r.split("|")[1]})
            show = [rr["baseline"], "bm25|-|c256", "hybrid|minilm|c256", "hybrid+ce|minilm|c256",
                    "hybrid+bge-rr|minilm|c256", ra["best_all"], ra["best_verified"]]
            rows = []
            for r in dict.fromkeys(s for s in show if s in rr["runs"]):
                nd = rr["runs"][r]["nDCG@10"]
                base = r == rr["baseline"]
                rows.append([label(r) + ("$^{\\dagger}$" if base else ""), fmt(nd["all"]["mean"]),
                             fmt_ci(nd["verified"]["mean"], nd["verified"]["lo"], nd["verified"]["hi"]),
                             "--" if base else f"{vb['delta'][r]:+.3f}",
                             "--" if base else f"{vb['p_bootstrap_holm'][r]:.3f}"])
            write_table("synth_qa_robustness",
                        ["System", f"All ({rr['n']['all']})", f"Verified ({rr['n']['verified']})",
                         "$\\Delta$ vs $^{\\dagger}$", "$p_{\\mathrm{Holm}}$"], rows,
                        "SynthQA nDCG@10 (95\\% CI) on all and on the human-verified questions. $\\Delta$, $p$: paired "
                        "bootstrap against the dense MiniLM baseline ($^{\\dagger}$) on the verified subset, Holm-corrected "
                        f"over {len(vb['delta'])} configurations.",
                        "tab:synthqa-robustness", align="lrrrr", source="results/synth_qa_verification.json")
        else:
            m.text("SynthVerifiedRobustness", None, c_sv)
    else:
        m.text("SynthVerifySummary", None, c_sv)
        m.text("SynthVerifiedRobustness", None, c_sv)

    # ---------------------------------------------------------------- chunk-size ablation
    rc = load("retrieval_chunks.json")
    c_rc = "python -m scripts.run_eval --config configs/experiments/retrieval_chunks.yaml"
    if rc:
        sizes = sorted({r["chunk_tokens"] for r in rc["runs"]})
        per_ds = []
        for ds in ("heading_qa", "paraphrase_qa", "synth_qa"):
            best = [max((r["summary"]["nDCG@10"]["mean"] for r in rc["runs"]
                         if r["dataset"] == ds and r["chunk_tokens"] == c), default=None) for c in sizes]
            if all(b is not None for b in best):
                per_ds.append(f"{'/'.join(f3(b) for b in best)} on {DS[ds]}")
        m.text("ChunkSummary", "The best nDCG@10 over all systems and embedders with "
               f"{'/'.join(str(c) for c in sizes)}-token chunks was " + ", ".join(per_ds) +
               ". MiniLM is capped at its 256-token input, so its 512-token results repeat those at 256 tokens.", c_rc)
    else:
        m.text("ChunkSummary", None, c_rc)

    # ---------------------------------------------------------------- baseline vs system overview
    st = get_settings()
    ge = load("gate_escalation.json")
    gl_first = None
    if gl:
        order = (gl.get("config") or {}).get("models") or []
        gl_first = order[0] if order else None
    ov = []

    def cell(x, nd=3):
        return "--" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    if rm:
        for ds in ("heading_qa", "synth_qa", "paraphrase_qa"):
            runs = [r for r in rm["runs"] if r["dataset"] == ds and r["chunk_tokens"] == st.chunking.chunk_tokens]
            b = next((r for r in runs if r["system"] == "dense" and r["embedder"] == "minilm"), None)
            d = next((r for r in runs if r["mode"] == st.retrieval.mode and r.get("reranker") == st.retrieval.reranker
                      and r["embedder"] == st.index.embedder), None)
            if not (b and d):
                continue
            fam = rm["significance"].get(f"{ds}|c{d['chunk_tokens']}|nDCG@10", {})
            p = fam.get("p_bootstrap_holm", {}).get(f"{d['system']}|{d['embedder']}|c{d['chunk_tokens']}")
            mark = "$^{*}$" if p is not None and p < 0.05 else ""
            ov.append(["Retrieval", f"nDCG@10, {DS[ds]} (n={d['n_queries']})",
                       cell(b["summary"]["nDCG@10"]["mean"]), cell(d["summary"]["nDCG@10"]["mean"]) + mark])
    if v1 and cpu:
        ov.append(["Latency", f"End-to-end time, CPU, same {same['n']} questions (s)", f"{same['base_e2e']:.1f}",
                   f"{same['e2e']:.1f}"])
        ov.append(["", "Time to first text, CPU (s)", f"{same['base_e2e']:.1f}", f"{same['ttft']:.1f}"])
        ov.append(["", "Generation speed, CPU (tokens/s)", f"{v1['tokens_per_s']['p50']:.1f}",
                   f"{cpu['tokens_per_s']['p50']:.1f}"])
    if gl and gl_first:
        row = {(r["profile"], r["dataset"]): r for r in gl["summary"] if r["model"] == gl_first}

        def g(profile, ds, k):
            return get(row.get((profile, ds)), k, "mean")
        ov.append(["Generation", "NLI faithfulness, FAQ-Gen", cell(g("naive", "faq_gen", "faithfulness")),
                   cell(g("full", "faq_gen", "faithfulness"))])
        ov.append(["", "Answers with a citation, FAQ-Gen", cell(g("naive", "faq_gen", "has_citation"), 2),
                   cell(g("full", "faq_gen", "has_citation"), 2)])
        ov.append(["", "Out-of-scope questions declined", cell(g("naive", "oos_questions", "abstained"), 2),
                   cell(g("full", "oos_questions", "abstained"), 2)])
        if gl.get("judge_status") == "complete" and g("full", "faq_gen", "judge_helpfulness") is not None:
            ov.append(["", f"LLM-judge helpfulness (1--5), FAQ-Gen sample of {gl['config'].get('judge_sample')}",
                       cell(g("naive", "faq_gen", "judge_helpfulness"), 2),
                       cell(g("full", "faq_gen", "judge_helpfulness"), 2)])
    hv = get(load("safety_gate_v2_heldout.json"), "variants", dep)
    if hv:
        ov.append(["Safety", "Crisis recall, held-out red-team set", "no gate", cell(hv["crisis_recall_any_escalation"])])
        ov.append(["", "Escalation false-positive rate, held-out", "no gate", cell(hv["escalation"]["false_positive_rate"])])
    if ge and "faq_gen" in ge["datasets"]:
        fq = ge["datasets"]["faq_gen"]
        ov.append(["", "FAQ-Gen questions sent to the crisis protocol", "no gate",
                   f"{fq['labels'].get('crisis', 0)}/{fq['n']}"])
    for i in range(len(ov) - 1, 0, -1):  # name each aspect once
        if ov[i][0] and ov[i][0] == next((r[0] for r in reversed(ov[:i]) if r[0]), None):
            ov[i][0] = ""
    if ov:
        model = gl_first.replace("_", "\\_") if gl_first else "--"
        write_table("baseline_vs_system", ["Aspect", "Measure", "Naive baseline", "This system"], ov,
                    caption="The system against a naive baseline on the same knowledge base and questions. Retrieval: "
                            "dense MiniLM versus the deployed "
                            f"{SYS.get('hybrid+ce' if st.retrieval.reranker == 'minilm-ce' else 'hybrid+bge-rr')} "
                            f"({EMB.get(st.index.embedder, st.index.embedder)}), {st.chunking.chunk_tokens}-token chunks "
                            "($^{*}$ Holm-corrected $p<0.05$). Latency: laptop CPU, full precision without streaming "
                            f"versus quantised with streaming. Generation: {model}. The baseline has no safety gate.",
                    label="tab:baseline-vs-system", align="llrr", source="results/*.json")

    # ---------------------------------------------------------------- multi-model benchmark (GPU)
    gg = load("generation_gpu.json")
    c_gg = "notebooks/kaggle\\_experiments.ipynb step generation\\_gpu"
    gpu_txt = None
    if gg:
        import statistics

        from eval.generation_eval import model_group
        from mhrag.llm.registry import load_catalog

        cat = load_catalog()
        import yaml

        gcfg = yaml.safe_load((PROJECT_ROOT / "configs" / "experiments" / "generation_gpu.yaml").read_text())
        rows = {(r["model"], r["profile"], r["dataset"]): r for r in gg["summary"]}
        present = set(dict.fromkeys(r["model"] for r in gg["summary"]))
        chat = [m for m in gcfg["models"] if m in present and model_group(m, cat) != "base"]  # excluded runs left out
        up = sig_up = 0
        faith_full = {}
        for mdl in chat:
            fn, fu = (get(rows.get((mdl, p, "faq_gen")), "faithfulness", "mean") for p in ("naive", "full"))
            if fn is None or fu is None:
                continue
            faith_full[mdl] = fu
            if fu > fn:
                up += 1
                pv = get(gg, "significance", f"{mdl}|faq_gen|faithfulness", "p_bootstrap_holm", "full>naive")
                sig_up += pv is not None and pv < 0.05
        oos = {p: [v for mdl in chat if (v := get(rows.get((mdl, p, "oos_questions")), "abstained", "mean")) is not None]
               for p in ("naive", "full")}
        if faith_full and oos["naive"] and oos["full"]:
            best = max(faith_full, key=faith_full.get)
            n = len(faith_full)
            if sig_up == up == n:
                won = f"for all {n}, each significantly (paired bootstrap, Holm-corrected)"
            elif up == n:
                won = f"for all {n} ({sig_up} significantly)"
            else:
                won = f"for {up} ({sig_up} significantly)"
            gpu_txt = (f"Across the {n} instruction-tuned models of the multi-model benchmark "
                       f"(Table~\\ref{{tab:benchmark-summary}}), the full pipeline raised the median share of "
                       f"out-of-scope questions declined from {statistics.median(oos['naive']):.2f} (naive RAG) to "
                       f"{statistics.median(oos['full']):.2f} and added inline citations. At full context length it also "
                       f"gave higher NLI faithfulness on FAQ-Gen than naive RAG {won}, but that comparison is confounded "
                       "by the amount of retrieved text, as the ablation showed.")
            excl = gcfg.get("excluded_runs") or {}
            if excl:
                count = {1: "One further run", 2: "Two further runs", 3: "Three further runs"}.get(
                    len(excl), f"{len(excl)} further runs")
                reruns = [(m, r) for m in excl for r in present if r.startswith(m + "-")]  # e.g. gemma-3-4b-it-fp32
                gpu_txt += (f" {count} ({', '.join(m.replace('_', chr(92) + '_') for m in excl)}) "
                            "produced degenerate output and are excluded"
                            + "".join(f"; {m} was rerun with different settings and is reported as {r}" for m, r in reruns)
                            + "; the table caption gives the reasons.")
    m.text("GPUBenchmarkSummary", gpu_txt, c_gg)

    # ---------------------------------------------------------------- LLM judge (sample, local experiment)
    c_j = "python -m scripts.run_eval --config configs/experiments/generation_local.yaml"
    judge_txt = None
    if gl and gl_first and gl.get("judge_status") == "complete":
        jrow = {(r["profile"], r["dataset"]): r for r in gl["summary"] if r["model"] == gl_first}

        def jm(p, k):
            return get(jrow.get((p, "faq_gen")), k, "mean")
        if None not in (jm("no_rag", "judge_helpfulness"), jm("full", "judge_helpfulness")):
            sig = gl.get("significance", {}).get(f"{gl_first}|faq_gen|judge_helpfulness", {})
            pv, dv = sig.get("p_bootstrap_holm", {}).get("full>no_rag"), sig.get("delta", {}).get("full>no_rag")
            vb = get(gl, "verbosity_bias", "judge_helpfulness", "spearman_rho_vs_length")
            name = gl_first.replace("_", "\\_")
            judge_txt = (
                f"On the judged sample ({gl['config'].get('judge_sample')} questions per dataset), the LLM judge rated "
                f"{name}'s FAQ-Gen answers {jm('no_rag', 'judge_helpfulness'):.2f} (no retrieval), "
                f"{jm('naive', 'judge_helpfulness'):.2f} (naive RAG) and {jm('full', 'judge_helpfulness'):.2f} (full "
                f"pipeline) for helpfulness, and {jm('no_rag', 'judge_correctness'):.2f}, "
                f"{jm('naive', 'judge_correctness'):.2f} and {jm('full', 'judge_correctness'):.2f} for correctness "
                "(1--5)")
            if pv is not None and dv is not None:
                judge_txt += (f"; the full pipeline scored {abs(dv):.2f} {'lower' if dv < 0 else 'higher'} than no "
                              f"retrieval on helpfulness (paired bootstrap, Holm-corrected $p={pv:.3f}$)")
            if vb is not None and jm("no_rag", "words") and jm("full", "words"):
                judge_txt += (f". Judge scores rose with answer length (Spearman $\\rho={vb:.2f}$), and answers "
                              f"without retrieval were longer ({jm('no_rag', 'words'):.0f} vs "
                              f"{jm('full', 'words'):.0f} words on average over all FAQ-Gen answers)")
            judge_txt += "."
    m.text("JudgeSummary", judge_txt, c_j)

    # ---------------------------------------------------------------- re-analyses of saved answers
    ra = load("generation_reanalysis.json")
    c_ra = "python -m scripts.generation_reanalysis"
    tf = get(ra, "experiments", "local", gl_first, "template_free", "faq_gen") if gl_first else None
    if tf:
        mm = tf["metrics"]

        def sv(metric, prof):
            return get(mm, metric, "settings", prof, "mean")

        def pv(metric, prof):
            return get(mm, metric, "vs_full", "p_bootstrap_holm", prof)
        txt = (f"The risk gate answered {tf['templated']} of {tf['n']} FAQ-Gen questions with a safety template. On the "
               f"{tf['template_free_n']} questions the model answered itself in every setting "
               f"(Table~\\ref{{tab:gen-template-free}}), relevance was {f3(sv('answer_relevance', 'full'))} with the full "
               f"pipeline against {f3(sv('answer_relevance', 'naive'))} with naive RAG and "
               f"{f3(sv('answer_relevance', 'no_rag'))} without retrieval")
        if sv("judge_helpfulness", "full") is not None:
            p = pv("judge_helpfulness", "no_rag")
            txt += (f", and judge helpfulness was {sv('judge_helpfulness', 'full'):.2f} against "
                    f"{sv('judge_helpfulness', 'no_rag'):.2f} without retrieval"
                    + (f" (Holm-corrected $p={p:.3f}$)" if p is not None else "")
                    + ", so the lower helpfulness is not caused by the templates")
        m.text("TemplateFreeSummary", txt + ".", c_ra)
        aw = get(ra, "experiments", "local", gl_first, "abstention_wording", "full")
        gpu_aw = get(ra, "experiments", "gpu") or {}
        chat = [k for k in gpu_aw if k in ((load("generation_gpu.json") or {}).get("config") or {}).get("models", [])]
        from eval.generation_eval import model_group as _mg
        from mhrag.llm.registry import load_catalog as _lc

        _cat = _lc()
        chat = [k for k in chat if _mg(k, _cat) != "base"]
        tot = sum(gpu_aw[k]["abstention_wording"]["full"]["abstained"] for k in chat)
        ph = sum(gpu_aw[k]["abstention_wording"]["full"]["with_prompt_phrase"] for k in chat)
        if aw and tot:
            m.text("AbstentionWordingSummary",
                   f"The full pipeline's system prompt suggests a sentence for missing information, and the phrase list "
                   f"detects it. Of the {aw['abstained']} out-of-scope answers the deployed model declined, "
                   f"{aw['with_prompt_phrase']} used that sentence; across the {len(chat)} instruction-tuned benchmark "
                   f"models, {ph} of {tot} detected refusals did ({100 * ph / tot:.0f}\\%), and models such as Phi and "
                   "Llama declined about as often in their own words. The phrase list also missed typographic "
                   "apostrophes in an earlier version of this analysis; all results here use the corrected version.", c_ra)
        else:
            m.text("AbstentionWordingSummary", None, c_ra)
        q = get(ra, "experiments", "gpu", "qwen2.5-1.5b-instruct", "diagnostics")
        qg = get(ra, "experiments", "local", gl_first, "diagnostics")
        qa = get(ra, "experiments", "gpu", "qwen2.5-1.5b-instruct", "abstention_wording")
        qga = get(ra, "experiments", "local", gl_first, "abstention_wording")
        ge4 = get(ra, "experiments", "gpu", "gemma-3-4b-it-fp32", "abstention_wording")
        if q and qg and qa and qga:
            faiths = sorted((get(gpu_aw, k, "diagnostics", "full", "faq_faithfulness") or 0 for k in chat), reverse=True)
            rank = faiths.index(q["full"]["faq_faithfulness"]) + 1 if q["full"]["faq_faithfulness"] in faiths else None
            where = ("the highest faithfulness" if rank == 1 else
                     f"the {rank}{'nd' if rank == 2 else 'rd' if rank == 3 else 'th'}-highest faithfulness of {len(chat)}"
                     if rank else "a faithfulness")
            txt = (f"Two results stand out. Qwen2.5-1.5B served through transformers with 4-bit NF4 weights, as in the whole "
                   f"benchmark, had {where} ({f3(q['full']['faq_faithfulness'])}) but cited a source in only "
                   f"{q['full']['faq_citation_rate']:.2f} of its FAQ-Gen answers and declined {qa['full']['abstained']} of "
                   f"{qa['full']['generated']} out-of-scope questions with the full pipeline, no more than without "
                   f"retrieval ({qa['no_rag']['abstained']} of {qa['no_rag']['generated']}); it often answered off-topic "
                   "questions from its own knowledge, sometimes attaching a citation to an unrelated passage. The same "
                   f"weights in llama.cpp with a different 4-bit scheme (Q4\\_K\\_M) declined {qga['full']['abstained']} of "
                   f"{qga['full']['generated']} under the same prompt and decoding parameters; the runtime comparison "
                   "below shows where the difference comes from.")
            if ge4:
                txt += (f" Gemma-3-4B declined {ge4['full']['abstained']} of {ge4['full']['generated']} out-of-scope "
                        f"questions with the full pipeline against {ge4['naive']['abstained']} with naive RAG; an apparent "
                        "drop in an earlier analysis came from the apostrophe error above.")
            m.text("OutlierSummary", txt, c_ra)
        else:
            m.text("OutlierSummary", None, c_ra)
    else:
        for k in ("TemplateFreeSummary", "AbstentionWordingSummary", "OutlierSummary"):
            m.text(k, None, c_ra)

    # ---------------------------------------------------------------- gate threshold trade-off
    gt = load("gate_tradeoff.json")
    c_gt = "python -m scripts.gate_tradeoff"
    if gt:
        dep_pt = next(p for p in gt["points"] if p["deployed"])
        cap = next(p for p in gt["points"] if p["cap_at_elevated"])
        m.text("GateTradeoffSummary",
               f"Raising the classifier's crisis threshold trades these escalations against crisis recall "
               f"(Table~\\ref{{tab:gate-tradeoff}}). At the deployed threshold the gate sent "
               f"{dep_pt['faq_gen']['crisis']} of {dep_pt['faq_gen']['n']} FAQ-Gen questions to the crisis protocol and "
               f"escalated {dep_pt['heldout']['crisis_k']} of {dep_pt['heldout']['crisis_n']} held-out crisis prompts; "
               f"capping the classifier at the elevated tier sent {cap['faq_gen']['crisis']} FAQ-Gen questions but "
               f"escalated only {cap['heldout']['crisis_k']} crisis prompts. No threshold removes the over-escalation "
               "without a large loss of recall, so a better fix needs a different signal (for example a rule for "
               "help-seeking questions), evaluated on a new held-out set.", c_gt)
    else:
        m.text("GateTradeoffSummary", None, c_gt)

    # ---------------------------------------------------------------- negation, safety baselines
    ng = load("negation_test.json")
    c_ng = "python -m scripts.eval_negation"
    nv, nt = get(ng, "variants", dep), get(ng, "variants", "tfidf_lr_only")
    if nv and nt:
        m.text("NegationSummary",
               f"On {ng['n_pairs']} minimal pairs that differ only in polarity (Table~\\ref{{tab:negation-test}}), the "
               f"TF-IDF classifier scored risk and no-risk statements almost identically (mean "
               f"{nt['mean_score_risk']:.2f} vs {nt['mean_score_no_risk']:.2f}): the stop-word normalisation of its "
               f"training data removes the negation. The deployed gate escalated {nv['risk_to_crisis']} of "
               f"{nv['risk_n']} risk statements but also {nv['no_risk_to_crisis']} of {nv['no_risk_n']} negated ones "
               f"(``I don't want to die''), separating only {nv['pairs_separated']} pairs; the lexicon's negation "
               "guard moves some of them to the non-blocking tier, but not most. The error is on the safe side, but it "
               "adds to the over-escalation of ordinary messages.", c_ng)
    else:
        m.text("NegationSummary", None, c_ng)
    sb = load("safety_baselines.json")
    c_sb = "notebooks/kaggle\\_experiments.ipynb step safety\\_baselines"

    def paired_sb(sb, names) -> str:
        """Significant per-prompt differences from the gate (exact McNemar, Holm-corrected; eval_safety_baselines)."""
        pv = sb.get("paired_vs_gate") or {}
        if not pv:
            return ""
        what = {"heldout_crisis": "held-out crisis prompts", "heldout_harmful": "held-out harmful requests",
                "heldout_fp": "held-out venting or informational prompts", "faq_gen": "FAQ-Gen questions"}

        def fp_(x):
            return "$p<0.001$" if x < 0.001 else f"$p={x:.3f}$"
        sig, n_tests = [], sum(len(r) for r in pv.values())
        for m_, res in pv.items():
            for k_, c in res.items():
                if c["p_mcnemar_holm"] >= 0.05:
                    continue
                more = "more" if c["method"] > c["gate"] else "fewer"
                verb = f"separated {more} negation pairs" if k_ == "negation_separated" else f"escalated {more} {what[k_]}"
                sig.append(f"{names.get(m_, m_)} {verb} ({c['method']} vs {c['gate']} of {c['n']}, {fp_(c['p_mcnemar_holm'])})")
        txt = (f" Compared with the gate prompt by prompt (exact McNemar, Holm-corrected over {n_tests} contrasts), "
               + ("; ".join(sig) if sig else "no difference was significant") + ".")
        hm = [(names.get(m_, m_), res["heldout_harmful"]) for m_, res in pv.items() if "heldout_harmful" in res]
        if hm:
            c0 = hm[0][1]
            txt += (f" On held-out requests for methods, the gate escalated {c0['gate']} of {c0['n']}; "
                    + "; ".join(f"{nm} escalated {c['method']} ({c['only_method']} that the gate did not, "
                                f"{c['only_gate']} the other way; Holm-corrected {fp_(c['p_mcnemar_holm'])})"
                                for nm, c in hm)
                    + f". With {c0['n']} requests these tests have little power (Section~\\ref{{sec:error-analysis}}).")
        z = pv.get("zeroshot", {})
        ns = [f"{what[k_]} ({z[k_]['method']} vs {z[k_]['gate']})" for k_ in ("heldout_crisis", "faq_gen")
              if k_ in z and z[k_]["p_mcnemar_holm"] >= 0.05]
        if ns:
            txt += (" The zero-shot classifier's differences on " + " and ".join(ns) + " were not significant; it "
                    "sends every message to an external API, which the deployed system avoids.")
        return txt
    sbm = {k: v for k, v in (sb or {}).get("methods", {}).items() if "error" not in v and k != "gate"}
    if sb and sbm:
        g_ = sb["methods"]["gate"]
        names_sb = {"llama_guard": "Llama Guard 3 (1B)", "shieldgemma": "ShieldGemma (2B)",
                    "zeroshot": "a zero-shot LLM classifier (gpt-oss-120b)"}
        parts_sb = [f"{names_sb.get(k, k)} escalated {v['heldout']['crisis_k']} of {v['heldout']['crisis_n']} held-out "
                    f"crisis prompts, {v['heldout']['fp']} of {v['heldout']['neg_n']} venting or informational ones and "
                    f"{v['faq_gen']['escalated']} of {v['faq_gen']['n']} FAQ-Gen questions" for k, v in sbm.items()]
        miss = [names_sb.get(k, k) for k, v in sb["methods"].items() if "error" in v]
        m.text("SafetyBaselinesSummary",
               f"Off-the-shelf classifiers, scored the same way (Table~\\ref{{tab:safety-baselines}}): the gate escalated "
               f"{g_['heldout']['crisis_k']} of {g_['heldout']['crisis_n']}, {g_['heldout']['fp']} of "
               f"{g_['heldout']['neg_n']} and {g_['faq_gen']['escalated']} of {g_['faq_gen']['n']}; "
               + "; ".join(parts_sb) + "."
               + (f" {', '.join(miss)} could not be run (no access)." if miss else "") + paired_sb(sb, names_sb), c_sb)
    else:
        m.text("SafetyBaselinesSummary", None, c_sb)

    # ---------------------------------------------------------------- citation precision, matched length, runtime
    cp = load("citation_precision.json")
    c_cp = "python -m scripts.citation\\_precision"
    if cp:
        gpu_cp = {k.split(":")[1]: v for k, v in cp["sets"].items() if k.startswith("gpu:") and v["citation_precision"] is not None}
        loc_cp = next((v for k, v in cp["sets"].items() if k.startswith("local:")), None)
        abl_cp = {k.split(":")[2]: v for k, v in cp["sets"].items() if k.startswith("ablation:")}
        if gpu_cp:
            vals = sorted(v["citation_precision"] for v in gpu_cp.values())
            import statistics as _st
            txt = (f"Citation precision, which checks each cited passage against the sentence that cites it and so does "
                   f"not grow with the amount of retrieved text (Table~\\ref{{tab:citation-precision}}), ranged from "
                   f"{vals[0]:.2f} to {vals[-1]:.2f} across the {len(vals)} models that cited (median "
                   f"{_st.median(vals):.2f})")
            if loc_cp and loc_cp["citation_precision"] is not None:
                txt += f"; it was {loc_cp['citation_precision']:.2f} for the deployed model"
            if "abl_full_full" in abl_cp and "abl_full_full_matched" in abl_cp and \
                    None not in (abl_cp["abl_full_full"]["citation_precision"], abl_cp["abl_full_full_matched"]["citation_precision"]):
                af, am = abl_cp["abl_full_full"], abl_cp["abl_full_full_matched"]
                txt += (f", and {af['citation_precision']:.2f} at full context length against "
                        f"{am['citation_precision']:.2f} at naive length in the ablation")
                if am["invalid"] > af["invalid"]:  # the precision denominator leaves these out
                    txt += (f", although at naive length {am['invalid']} citations (against {af['invalid']}) named a "
                            "source that was not in the prompt and are not counted")
            m.text("CitationPrecisionSummary", txt + ".", c_cp)
            ck = load("citation_check.json")
            if ck and ck.get("raters"):
                rs_ = list(ck["raters"].values())
                r0 = rs_[0]
                ns_, ss_ = r0["strata"]["nli_unsupported"], r0["strata"]["nli_supported"]
                lo_, hi_ = r0["corrected_ci"]
                txt_ck = (f"To check whether the NLI model is too strict, I labelled {ck['pairs']} (sentence, cited "
                          "passage) pairs by hand, sampled from the deployed and benchmark models and blind to the NLI "
                          f"verdict ({ss_['n']} that NLI counted as entailed, {ns_['n']} that it did not). Of the "
                          f"{ns_['n']} that NLI rejected, I judged {ns_['supported']} supported and {ns_['partly']} partly "
                          f"supported; of the {ss_['n']} it accepted, {ss_['supported']} supported (agreement with NLI "
                          f"{r0['agreement']:.2f}, $\\kappa={r0['cohen_kappa']:.2f}$). Weighting each group by its share of "
                          f"all {ck['population']['pairs']} citation pairs, the estimated citation precision is "
                          f"{r0['corrected_citation_precision']:.2f} (95\\% CI {lo_:.2f}--{hi_:.2f}) rather than "
                          f"{r0['nli_citation_precision']:.2f}, and {r0['corrected_partly_as_supported']:.2f} if partial "
                          "support counts. The NLI model therefore understates how often citations are supported, so its "
                          "absolute values should not be read as citation rates. The same model scores NLI faithfulness, "
                          "whose absolute values are probably understated in the same way; the comparisons between "
                          "settings assume that its errors are similar across them, which this check does not test."
                          + (f" Agreement between raters: $\\kappa={ck['inter_rater_kappa']:.2f}$."
                             if ck.get("inter_rater_kappa") is not None else ""))
                m.text("CitationCheckSummary", txt_ck, "python -m scripts.citation\\_check")
            else:
                m.lines.append("\\newcommand{\\CitationCheckSummary}{\\todohuman{label the 30 pairs in "
                               "eval/human\\_eval/out/citation\\_check/sheet\\_rater1.csv, then python -m "
                               "scripts.citation\\_check}}")
        else:
            m.text("CitationPrecisionSummary", None, c_cp)
    else:
        m.text("CitationPrecisionSummary", None, c_cp)
    mt = load("generation_matched.json")
    c_mt = "notebooks/kaggle\\_experiments.ipynb step generation\\_matched"
    if mt:
        sig_m = mt.get("significance", {})
        ms = [x for x in mt["config"]["models"] if f"{x}|faq_gen|faithfulness" in sig_m]

        def dp(x, a, b):
            g = sig_m[f"{x}|faq_gen|faithfulness"]
            return g["delta"].get(f"{a}>{b}"), g["p_bootstrap_holm"].get(f"{a}>{b}")
        eq = [dp(x, "abl_full_full_matched", "abl_naive_naive") for x in ms]
        ln = [dp(x, "abl_full_full", "abl_full_full_matched") for x in ms]
        up = sum(d > 0 and pv < 0.05 for d, pv in eq if d is not None)
        down = sum(d < 0 and pv < 0.05 for d, pv in eq if d is not None)
        lup = sum(d > 0 and pv < 0.05 for d, pv in ln if d is not None)
        bigger = sum(l_[0] > e_[0] for e_, l_ in zip(eq, ln, strict=True) if None not in (e_[0], l_[0]))
        import statistics as _st
        m.text("MatchedSummary",
               f"Across the {len(ms)} instruction-tuned models (Table~\\ref{{tab:gen-matched}}; one run, no gate), the "
               f"full pipeline cut to the naive context length was significantly more faithful than naive RAG for {up} "
               f"models and less faithful for {down}, while restoring the full context raised faithfulness "
               f"significantly for {lup}. The length step was the larger of the two for {bigger} of {len(ms)} models "
               f"(median {_st.median(d for d, _ in ln):+.3f} against {_st.median(d for d, _ in eq):+.3f}). The "
               "equal-length contrast changes the prompt as well as retrieval, and in the ablation the prompt alone "
               "raised faithfulness, so it bounds the contribution of better retrieval from above.", c_mt)
    else:
        m.text("MatchedSummary", None, c_mt)
    rt = load("generation_runtime.json")
    c_rt = "notebooks/kaggle\\_experiments.ipynb step generation\\_runtime"

    def runtime_tests() -> str:
        """Paired runtime contrasts from generation_reanalysis.json (exact McNemar, Holm-corrected)."""
        t = get(load("generation_reanalysis.json"), "runtime_oos_abstention") or {}
        q4, rtm, nf4 = (t.get(k) for k in ("qwen2.5-1.5b-gguf-f16>qwen2.5-1.5b-gguf",
                                            "qwen2.5-1.5b-gguf-f16>qwen2.5-1.5b-instruct-f16",
                                            "qwen2.5-1.5b-instruct-f16>qwen2.5-1.5b-instruct"))
        if not (q4 and rtm and nf4):
            return ""

        def pv(c):
            p_ = c["p_mcnemar_holm"]
            return "$p<0.001$" if p_ < 0.001 else f"$p={p_:.3f}$"

        def sig(c):
            return c["p_mcnemar_holm"] < 0.05
        if sig(q4) or not (sig(rtm) and sig(nf4)) or rtm["only_a"] <= rtm["only_b"] or nf4["only_a"] <= nf4["only_b"]:
            return (" Paired by question (exact McNemar, Holm-corrected): Q4\\_K\\_M vs 16 bits in llama.cpp "
                    f"{pv(q4)}, llama.cpp vs transformers at 16 bits {pv(rtm)}, 16 bits vs NF4 in transformers {pv(nf4)}.")
        return (" Paired by question (exact McNemar, Holm-corrected), 4-bit Q4\\_K\\_M quantisation did not "
                f"significantly change abstention in llama.cpp ({q4['only_b']} questions declined only at 4 bits, "
                f"{q4['only_a']} only at 16 bits; {pv(q4)}), whereas both the runtime and NF4 quantisation lowered it: at "
                f"16 bits, {rtm['only_a']} questions were declined only in llama.cpp and {rtm['only_b']} only in "
                f"transformers ({pv(rtm)}), and within transformers {nf4['only_a']} only at 16 bits and "
                f"{nf4['only_b']} only with NF4 ({pv(nf4)}). The benchmark's abstention rates, all measured with "
                "transformers and NF4, may therefore understate what the same models do in other runtimes; this was "
                "measured for one model only.")
    if rt:
        rr = {r["model"]: r for r in rt["summary"] if r["dataset"] == "oos_questions"}

        def kdec(x):
            a_ = get(rr.get(x), "abstained")
            return f"{round(a_['mean'] * a_['n'])} of {a_['n']}" if a_ else "--"
        m.text("RuntimeSummary",
               "Run four ways in one session on one GPU (Table~\\ref{tab:gen-runtime}), the same Qwen2.5-1.5B weights "
               f"declined {kdec('qwen2.5-1.5b-gguf')} out-of-scope questions with llama.cpp at 4 bits (Q4\\_K\\_M), "
               f"{kdec('qwen2.5-1.5b-gguf-f16')} with llama.cpp at 16 bits, {kdec('qwen2.5-1.5b-instruct-f16')} with "
               f"transformers at 16 bits and {kdec('qwen2.5-1.5b-instruct')} with transformers at 4 bits (NF4), the "
               "setting used for every model in the multi-model benchmark." + runtime_tests(), c_rt)
    else:
        m.text("RuntimeSummary", None, c_rt)

    # ---------------------------------------------------------------- component ablation (Kaggle)
    ab = load("generation_ablation.json")
    c_ab = "notebooks/kaggle\\_experiments.ipynb step generation\\_ablation"
    ab_txt = None
    if ab:
        rows_ab = {(r["profile"], r["dataset"]): r for r in ab["summary"]}
        mdl = ab["config"]["models"][0]
        sig_ab = ab.get("significance", {})

        def fa(arm, ds="faq_gen", k="faithfulness"):
            return get(rows_ab.get((arm, ds)), k, "mean")

        def ct(a, b, ds="faq_gen", k="faithfulness"):
            g = sig_ab.get(f"{mdl}|{ds}|{k}", {})
            return get(g, "delta", f"{a}>{b}"), get(g, "p_bootstrap_holm", f"{a}>{b}")

        def fp(pv):
            return "$p<0.001$" if pv < 0.001 else f"$p={pv:.3f}$"
        arms = ("abl_naive_naive", "abl_full_naive", "abl_naive_full", "abl_full_full", "abl_full_full_matched")
        need = [ct("abl_full_full_matched", "abl_naive_full"), ct("abl_full_full", "abl_full_full_matched"),
                ct("abl_full_naive", "abl_naive_naive"), ct("abl_naive_full", "abl_naive_naive"),
                ct("abl_naive_full", "abl_naive_naive", "oos_questions", "abstained"),
                ct("abl_full_naive", "abl_naive_naive", "oos_questions", "abstained"),
                ct("abl_full_full", "abl_naive_full", "oos_questions", "abstained")]
        if all(fa(a) is not None for a in arms) and all(None not in x for x in need):
            (eq_d, eq_p), (len_d, len_p), (ret_d, ret_p), (pr_d, pr_p), (ab_pr, ab_pr_p), (ab_r1, ab_r1_p), \
                (ab_r2, ab_r2_p) = need
            no_diff = eq_p >= 0.05
            ab_txt = (
                "The component ablation (Tables~\\ref{tab:gen-ablation} and~\\ref{tab:gen-ablation-contrasts}; the "
                "same model, no gate) separates these effects. With the minimal prompt, full retrieval raised "
                f"faithfulness from {f3(fa('abl_naive_naive'))} to {f3(fa('abl_full_naive'))} ({fp(ret_p)}), and the full "
                f"prompt alone raised it to {f3(fa('abl_naive_full'))} ({fp(pr_p)}). Most of the retrieval gain came "
                "from the amount of retrieved text: cut to the naive context length, full retrieval scored "
                f"{f3(fa('abl_full_full_matched'))}, "
                + (f"no different from naive retrieval with the same prompt ({f3(fa('abl_naive_full'))}, {fp(eq_p)})"
                   if no_diff else f"against {f3(fa('abl_naive_full'))} for naive retrieval ({fp(eq_p)})")
                + f", while the full-length context added ${len_d:+.3f}$ ({fp(len_p)}). Because NLI faithfulness counts a "
                "sentence as supported when any passage entails it, this part of the gain reflects how much text the "
                "metric can match against, not necessarily better answers; retrieval quality itself is measured in "
                "Section~\\ref{sec:results-retrieval}. The prompt, not retrieval, produced the abstention: it raised "
                f"the share of out-of-scope questions declined by {ab_pr:.2f} with naive retrieval ({fp(ab_pr_p)}), "
                f"whereas retrieval changed it by ${ab_r1:+.2f}$ ({fp(ab_r1_p)}) with the minimal prompt and "
                f"${ab_r2:+.2f}$ ({fp(ab_r2_p)}) with the full one.")
    m.text("AblationSummary", ab_txt, c_ab)

    # ---------------------------------------------------------------- benchmark summary (main text)
    if gg:
        from eval.generation_eval import model_group as _mg2
        from mhrag.llm.registry import load_catalog as _lc2

        cat2 = _lc2()
        g_rows = {(r["model"], r["profile"], r["dataset"]): r for r in gg["summary"]}
        cp_sets = {k.split(":")[1]: v for k, v in (cp or {}).get("sets", {}).items() if k.startswith("gpu:")}
        gcfg2 = gg.get("config", {}).get("models", [])
        brows = []
        for x in gcfg2:  # every column from the multi-model benchmark run (the length-matched run is its own table)
            if _mg2(x, cat2) == "base" or (x, "full", "faq_gen") not in g_rows:
                continue

            def mean_(r, k, nd=3):
                v = get(r, k, "mean")
                return "--" if v is None else f"{v:.{nd}f}"
            cpv = get(cp_sets.get(x), "citation_precision")
            size = cat2[x].raw.get("params_b") if x in cat2 else None
            brows.append([x.replace("_", "\\_"), f"{size:g}" if size else "--",
                          mean_(g_rows.get((x, "naive", "faq_gen")), "faithfulness"),
                          mean_(g_rows.get((x, "full", "faq_gen")), "faithfulness"),
                          mean_(g_rows.get((x, "full", "faq_gen")), "has_citation", 2),
                          "--" if cpv is None else f"{cpv:.2f}",
                          mean_(g_rows.get((x, "naive", "oos_questions")), "abstained", 2),
                          mean_(g_rows.get((x, "full", "oos_questions")), "abstained", 2)])
        if brows:
            write_table("benchmark_summary", ["Model", "B", "Faith. naive", "Faith. full", "Cites", "Cit. prec.",
                                              "Decl. naive", "Decl. full"], brows,
                        "Multi-model benchmark run: naive RAG versus the full pipeline (with the risk gate; model answers "
                        "only). Faithfulness is not length-matched (see Table~\\ref{tab:gen-matched}). B: parameters "
                        "(billions); Faith.: NLI faithfulness on FAQ-Gen; Cites: share of FAQ-Gen answers with a citation; "
                        "Cit. prec.: citation precision (Table~\\ref{tab:citation-precision}); Decl.: share of "
                        "out-of-scope questions declined. Per-setting results for every model are in the code repository.",
                        "tab:benchmark-summary", align="lrrrrrrr", source="results/generation_gpu.json, "
                        "results/citation_precision.json")

    # ---------------------------------------------------------------- run-to-run divergence (main run vs ablation)
    def _scored(exp):
        import gzip as _gz
        d_ = R / "generation" / exp
        for nm, op in (("scored.jsonl", open), ("scored.jsonl.gz", _gz.open)):
            if (d_ / nm).exists():
                with op(d_ / nm, "rt", encoding="utf-8") as f_:
                    return [json.loads(x) for x in f_]
        return []
    div_txt = None
    if ab and gl_first:
        loc_r = {r["id"]: r for r in _scored("local") if r["model"] == gl_first and r["profile"] == "full"
                 and r["dataset"] == "faq_gen" and r.get("generated")}
        abl_r = {r["id"]: r for r in _scored("ablation") if r["profile"] == "abl_full_full" and r["dataset"] == "faq_gen"}
        ids = [i for i in loc_r if i in abl_r and loc_r[i].get("gate") in (None, "none")
               and [x["chunk_id"] for x in loc_r[i]["sources"]] == [x["chunk_id"] for x in abl_r[i]["sources"]]]
        same_ans = sum(loc_r[i]["answer"].strip() == abl_r[i]["answer"].strip() for i in ids)
        fl = [loc_r[i]["metrics"].get("faithfulness") for i in ids]
        fa_ = [abl_r[i]["metrics"].get("faithfulness") for i in ids]
        pairs_ = [(x, y) for x, y in zip(fl, fa_, strict=True) if x is not None and y is not None]
        if ids and pairs_:
            n_pass = len(loc_r)
            main_f = get(next((r for r in (load("generation_local.json") or {}).get("summary", [])
                               if r["model"] == gl_first and r["profile"] == "full" and r["dataset"] == "faq_gen"), None),
                         "faithfulness")
            abl_f = get(next((r for r in ab["summary"] if r["profile"] == "abl_full_full" and r["dataset"] == "faq_gen"),
                             None), "faithfulness")
            div_txt = ("The full pipeline with the deployed model ran twice: in the main experiment, with the gate "
                       f"(faithfulness {main_f['mean']:.3f} on the {main_f['n']} FAQ-Gen questions the gate passed), and "
                       f"in the ablation, without it ({abl_f['mean']:.3f} on all {abl_f['n']}, "
                       "Table~\\ref{tab:gen-ablation}). "
                       f"On the {len(ids)} of the {n_pass} gate-passed questions for which both runs retrieved the same "
                       f"passages, only {same_ans} answers were identical. They diverged after a shared opening, which "
                       "points to numerical differences between the two Kaggle sessions rather than to a configuration "
                       "difference; the llama.cpp version was not recorded for these runs and is now pinned. Mean "
                       f"faithfulness on these {len(pairs_)} questions was {sum(x for x, _ in pairs_) / len(pairs_):.3f} "
                       f"in the main run and {sum(y for _, y in pairs_) / len(pairs_):.3f} in the ablation, a measure of "
                       "run-to-run variation. Every table comes from a single run, which its caption names.")
    m.text("AblationRunDivergence", div_txt, "python -m scripts.rescore\\_faithfulness")

    # ---------------------------------------------------------------- abstention cross-check (LLM judge)
    ac = load("abstention_check.json")
    c_ac = "notebooks/kaggle\\_experiments.ipynb step abstention\\_check"
    loc = next((v for k, v in (ac or {}).get("sets", {}).items() if k.startswith("local:")), None)
    if ac and ac.get("complete") and loc and "full" in loc and "naive" in loc:
        f_, n_ = loc["full"], loc["naive"]
        m.text("AbstentionCheckSummary",
               f"An LLM judge that does not see the system prompt labelled the deployed model as declining "
               f"{f_['judge_rate']:.2f} of out-of-scope questions with the full pipeline and {n_['judge_rate']:.2f} with "
               f"naive RAG, against {f_['regex_rate']:.2f} and {n_['regex_rate']:.2f} by the phrase list (Cohen's "
               f"$\\kappa$ {f_['cohen_kappa']:.2f} and {n_['cohen_kappa']:.2f}; Table~\\ref{{tab:abstention-check}}). "
               + (f"The phrase list therefore overstates the full pipeline's gain over naive RAG, but the gain remains: "
                  f"{loc['full>naive']['judge_delta']:+.2f} by the judge "
                  f"(paired bootstrap $p{'<0.001' if loc['full>naive']['p_bootstrap'] < 0.001 else '=' + format(loc['full>naive']['p_bootstrap'], '.3f')}$) "
                  f"against {f_['regex_rate'] - n_['regex_rate']:+.2f} by the phrase list."
                  if loc.get("full>naive") else ""), c_ac)
    else:
        m.text("AbstentionCheckSummary", None, c_ac)

    # ---------------------------------------------------------------- abstract: four findings, few figures
    parts = []
    mt = load("generation_matched.json")
    if ab:
        rows_ab = {(r["profile"], r["dataset"]): r for r in ab["summary"]}

        def fab(arm):
            return get(rows_ab.get((arm, "faq_gen")), "faithfulness", "mean")
        txt = "(1) Higher NLI faithfulness came mainly from more retrieved text: better retrieval added nothing at equal length for the deployed model"
        txt += "."  # the 13-model length-matched result is in the main text (MatchedSummary)
        cp_all = (load("citation_precision.json") or {}).get("sets", {})
        loc_p = next((v["citation_precision"] for k, v in cp_all.items() if k.startswith("local:")), None)
        gpu_p = sorted(v["citation_precision"] for k, v in cp_all.items()
                       if k.startswith("gpu:") and v["citation_precision"] is not None)
        if loc_p is not None and gpu_p:
            import statistics as _st
            ck_ = load("citation_check.json")
            cor_ = [r_["corrected_citation_precision"] for r_ in (ck_ or {}).get("raters", {}).values()
                    if r_.get("corrected_citation_precision") is not None]
            if cor_:
                ck0 = next(iter(ck_["raters"].values()))
                lo_c, hi_c = ck0["corrected_ci"]
                txt += (f" NLI judged {100 * loc_p:.0f}\\% of the deployed model's citations supported; a blinded "
                        f"hand check of {ck_['pairs']} citation pairs by the author suggested NLI was too strict "
                        f"(estimated {100 * cor_[0]:.0f}\\% of citations across models supported, 95\\% CI "
                        f"{100 * lo_c:.0f}--{100 * hi_c:.0f}\\%).")
            else:
                txt += (f" Citations were weak evidence: the cited passage entailed the citing sentence in "
                        f"{100 * loc_p:.0f}\\% of the deployed model's citations (median "
                        f"{100 * _st.median(gpu_p):.0f}\\% across models).")
        parts.append(txt)
    ac_loc = next((v for k, v in (load("abstention_check.json") or {}).get("sets", {}).items() if k.startswith("local:")), None)
    ac_abl = next((v for k, v in (load("abstention_check.json") or {}).get("sets", {}).items() if k.startswith("ablation:")), None)
    if ac_loc and ac_abl and "naive" in ac_loc and "abl_naive_full" in ac_abl:
        parts.append("(2) Declining out-of-scope questions came from the system prompt, not retrieval.")
    gd, gh = (get(load(f"safety_gate_v2_{t}.json"), "variants", dep) for t in ("devset", "heldout"))
    neg = get(load("negation_test.json"), "variants", dep)
    if gd and gh:
        def kn(v):
            pc = v["per_category"]
            n = pc["explicit_crisis"]["n"] + pc["implicit_crisis"]["n"]
            return f"{round(v['crisis_recall_any_escalation'] * n)} of {n}"
        hr = gh["per_category"].get("harmful_request")
        txt = (f"(3) The risk gate escalated {kn(gd)} crisis prompts on the development set but {kn(gh)} on a frozen "
               "held-out set" + (f", caught {round(hr['escalated'] * hr['n'])} of {hr['n']} held-out requests for "
                                 "suicide methods" if hr else ""))
        extra = []
        if neg:
            extra.append(f"{neg['no_risk_to_crisis']} of {neg['no_risk_n']} negated statements")
        if ge and "faq_gen" in ge["datasets"]:
            fq = ge["datasets"]["faq_gen"]
            extra.append(f"{fq['labels'].get('crisis', 0)} of {fq['n']} ordinary questions")
        if extra:
            txt += ", and escalated " + " and ".join(extra)
        parts.append(txt + ".")
    if tf:
        hf, hn = (get(tf, "metrics", "judge_helpfulness", "settings", p_, "mean") for p_ in ("full", "no_rag"))
        if None not in (hf, hn) and hf < hn:
            parts.append("(4) An LLM judge rated grounded answers less helpful than ungrounded ones.")
    m.text("AbstractResults", " ".join(p for p in parts if p), "run all experiments (scripts/run_all_local.sh)")

    OUT.parent.mkdir(parents=True, exist_ok=True)  # paper/ is not in the repository (kept private)
    OUT.write_text("% Auto-generated by scripts/paper_numbers.py from results/ -- do not edit by hand.\n"
                   + "\n".join(m.lines) + "\n")
    todo = sum("todorun" in ln for ln in m.lines)
    print(f"wrote {OUT} ({len(m.lines)} macros, {todo} pending)")


if __name__ == "__main__":
    main()
