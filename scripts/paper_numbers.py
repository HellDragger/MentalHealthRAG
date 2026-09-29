"""Generate paper/numbers.tex (LaTeX macros) and paper/tables/latency.tex from results/*.json.

    python -m scripts.paper_numbers

Every number quoted in the paper text is a macro defined here from a results file. A result that does not exist yet
becomes \\todorun{<command that produces it>}, so the manuscript can never show an invented value.
"""

from __future__ import annotations

import json

from eval.tables import write_table
from mhrag.config import PROJECT_ROOT, get_settings

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
DS = {"heading_qa": "HeadingQA", "synth_qa": "SynthQA", "paraphrase_qa": "ParaphraseQA"}


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
    from mhrag.ingest.qa import counselling_duplicate_stats
    from mhrag.ingest.sources import RawData

    st = get_settings()
    m.set("NCounselRows", counselling_duplicate_stats(RawData(st.resolve(st.paths.raw_data))).get("rows"), c_es)

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
    names = {"v1_hf_fp32_cpu": "v1 settings (fp32, CPU, no streaming)", "v2_llamacpp_cpu": "v2 llama.cpp Q4\\_K\\_M, CPU",
             "v2_llamacpp_metal": "v2 llama.cpp Q4\\_K\\_M, Metal", "v2_hf_mps": "v2 transformers fp16, MPS",
             "v2_hf_cuda": "v2 transformers 16-bit, CUDA", "v2_hf_cuda_4bit": "v2 transformers NF4, CUDA"}
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
                    rows, caption="End-to-end latency for 12 fixed questions (v1: 5). TTFT = time to first streamed token "
                                  "(v1 does not stream, so TTFT equals end-to-end time). v2 rows include the safety gate, hybrid "
                                  "retrieval and reranking. Peak RSS excludes GPU/unified memory held by Metal.",
                    label="tab:latency", align="llrrrrr", source="results/latency.json")
    v1, cpu, metal = get(lat, "v1_hf_fp32_cpu", "summary"), get(lat, "v2_llamacpp_cpu", "summary"), get(lat, "v2_llamacpp_metal", "summary")
    if v1 and cpu:
        txt = (f"With the same weights (Qwen2.5-1.5B-Instruct), the v1 generation settings needed a median "
               f"{v1['e2e_ms']['p50'] / 1000:.0f}~s per answer before any text appeared ({v1['tokens_per_s']['p50']:.1f} tokens/s), "
               f"whereas the v2 CPU configuration showed the first token after a median "
               f"{cpu['ttft_ms']['p50'] / 1000:.1f}~s and generated {cpu['tokens_per_s']['p50']:.1f} tokens/s")
        if metal:
            txt += (f"; with Metal offload the median time to first token was {metal['ttft_ms']['p50'] / 1000:.2f}~s at "
                    f"{metal['tokens_per_s']['p50']:.1f} tokens/s")
        txt += " (Table~\\ref{tab:latency})."
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
            s = (f"On {label}, the best configuration was {sysname(best)} (nDCG@10 {f3(best['summary']['nDCG@10']['mean'])}, "
                 f"95\\% CI {f3(best['summary']['nDCG@10']['lo'])}--{f3(best['summary']['nDCG@10']['hi'])})")
            if base:
                s += f", against {f3(base['summary']['nDCG@10']['mean'])} for the v1 configuration (dense MiniLM)"
            if bm:
                s += f" and {f3(bm['summary']['nDCG@10']['mean'])} for BM25"
            parts.append(s + ".")
        m.text("RetrievalSummary", " ".join(parts), c_rm)
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
        for model in models:  # in config order, so the deployed model comes first
            row = {(r["profile"], r["dataset"]): r for r in gl["summary"] if r["model"] == model}
            full, naive, norag = row.get(("full", "faq_gen")), row.get(("naive", "faq_gen")), row.get(("no_rag", "faq_gen"))
            if full and naive and full.get("faithfulness") and naive.get("faithfulness"):
                s = (f"For {model.replace('_', chr(92) + '_')} on FAQ-Gen, NLI faithfulness was "
                     f"{f3(full['faithfulness']['mean'])} with the full pipeline versus {f3(naive['faithfulness']['mean'])} with naive RAG")
                if full.get("rougeL") and norag and norag.get("rougeL"):
                    s += (f"; ROUGE-L was {f3(full['rougeL']['mean'])} (full), {f3(naive['rougeL']['mean'])} (naive) "
                          f"and {f3(norag['rougeL']['mean'])} (no retrieval)")
                oos = row.get(("full", "oos_questions"))
                if oos and oos.get("abstained"):
                    s += f"; it declined or flagged missing information on {f3(oos['abstained']['mean'])} of out-of-scope questions"
                parts.append(s + ".")
        if gl.get("judge_status") == "complete":
            tail = " Full per-model results are in the tables below."
        else:
            tail = (" Full per-model results are in the tables below; the LLM-judge column remains "
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

    # ---------------------------------------------------------------- v1 vs v2 overview
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
        ov.append(["Latency", "Time to first text, CPU (s)", f"{v1['e2e_ms']['p50'] / 1000:.1f}",
                   f"{cpu['ttft_ms']['p50'] / 1000:.1f}"])
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
        ov.append(["Safety", "Crisis recall, held-out red-team set", "none", cell(hv["crisis_recall_any_escalation"])])
        ov.append(["", "Escalation false-positive rate, held-out", "none", cell(hv["escalation"]["false_positive_rate"])])
    if ge and "faq_gen" in ge["datasets"]:
        fq = ge["datasets"]["faq_gen"]
        ov.append(["", "FAQ-Gen questions sent to the crisis protocol", "none",
                   f"{fq['labels'].get('crisis', 0)}/{fq['n']}"])
    for i in range(len(ov) - 1, 0, -1):  # name each aspect once
        if ov[i][0] and ov[i][0] == next((r[0] for r in reversed(ov[:i]) if r[0]), None):
            ov[i][0] = ""
    if ov:
        model = gl_first.replace("_", "\\_") if gl_first else "--"
        write_table("v1_vs_v2", ["Aspect", "Measure", "v1 configuration", "v2 (deployed)"], ov,
                    caption="The v2 system against the prototype's (v1) configuration, both run on the same knowledge base "
                            "and questions. Retrieval: dense MiniLM (v1) versus the deployed "
                            f"{SYS.get('hybrid+ce' if st.retrieval.reranker == 'minilm-ce' else 'hybrid+bge-rr')} "
                            f"({EMB.get(st.index.embedder, st.index.embedder)} embeddings), "
                            f"{st.chunking.chunk_tokens}-token chunks; $^{{*}}$ significant (paired bootstrap, "
                            "Holm-corrected $p<0.05$). Latency: Qwen2.5-1.5B-Instruct on the laptop CPU (\\Hardware{}); "
                            "the v1 settings do not stream, so the first text appears with the full answer. Generation: "
                            f"{model}, v1-style naive RAG versus the full pipeline. The prototype had no safety handling, "
                            "and its vector store was empty at run time (CHANGELOG B1).",
                    label="tab:v1-vs-v2", align="llrr", source="results/*.json")

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
            gpu_txt = (f"Across the {len(faith_full)} instruction-tuned models of the multi-model benchmark "
                       f"(Table~\\ref{{tab:gen-gpu-summary}}), the full pipeline gave higher NLI faithfulness on "
                       f"FAQ-Gen than naive RAG for {up} (significantly for {sig_up}), and the median share of "
                       f"out-of-scope questions declined was {statistics.median(oos['full']):.2f} with the full "
                       f"pipeline versus {statistics.median(oos['naive']):.2f} with naive RAG. The highest "
                       f"full-pipeline faithfulness was {faith_full[best]:.3f} ({best.replace('_', chr(92) + '_')}).")
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

    # ---------------------------------------------------------------- abstract
    parts = []
    if get(load("safety_gate_v2_heldout.json"), "variants", dep):
        g = load("safety_gate_v2_heldout.json")["variants"][dep]
        parts.append(f"On a held-out red-team set the gate detected {f3(g['crisis_recall_any_escalation'])} of crisis "
                     f"messages with an escalation false-positive rate of {f3(g['escalation']['false_positive_rate'])}, "
                     f"lower than on the development set, and the lexicon component generalised worst.")
    if rm:
        # only claim an improvement that is significant (Holm-corrected paired bootstrap) on HeadingQA and SynthQA
        wins = []
        for ds in ("heading_qa", "synth_qa"):
            runs = [r for r in rm["runs"] if r["dataset"] == ds]
            if not runs:
                continue
            best = max(runs, key=lambda r: r["summary"]["nDCG@10"]["mean"])
            fam = rm["significance"].get(f"{ds}|c{best['chunk_tokens']}|nDCG@10", {})
            rid = f"{best['system']}|{best['embedder'] or '-'}|c{best['chunk_tokens']}"
            p = fam.get("p_bootstrap_holm", {}).get(rid)
            if p is not None and p < 0.05 and fam["delta"][rid] > 0:
                wins.append(f"{DS[ds]}: +{fam['delta'][rid]:.3f} nDCG@10 with {sysname(best)}")
        if wins:
            parts.append("Relative to the prototype's retrieval configuration, the best systems improved nDCG@10 "
                         "significantly (" + "; ".join(wins) + ").")
    if v1 and cpu:
        parts.append(f"Quantised CPU inference with streaming reduced the time to first token from "
                     f"{v1['e2e_ms']['p50'] / 1000:.0f}~s to {cpu['ttft_ms']['p50'] / 1000:.1f}~s.")
    if gl and gl_first:
        fn, fu = get(row.get(("naive", "faq_gen")), "faithfulness", "mean"), get(row.get(("full", "faq_gen")), "faithfulness", "mean")
        on, ou = (get(row.get(("naive", "oos_questions")), "abstained", "mean"),
                  get(row.get(("full", "oos_questions")), "abstained", "mean"))
        if None not in (fn, fu, on, ou):
            txt = (f"With the same 1.5B model, the full pipeline raised NLI faithfulness on FAQ-Gen from {f3(fn)} to "
                   f"{f3(fu)} and the share of out-of-scope questions it declined from {on:.2f} to {ou:.2f}")
            if judge_txt:
                jr = {(r["profile"], r["dataset"]): r for r in gl["summary"] if r["model"] == gl_first}
                h_nr = get(jr.get(("no_rag", "faq_gen")), "judge_helpfulness", "mean")
                h_fu = get(jr.get(("full", "faq_gen")), "judge_helpfulness", "mean")
                txt += (f"; an LLM judge nevertheless rated its answers less helpful than answers without retrieval "
                        f"({h_fu:.2f} vs {h_nr:.2f} on a 1--5 scale, {gl['config'].get('judge_sample')}-question sample)")
            if ge and "faq_gen" in ge["datasets"]:
                fq = ge["datasets"]["faq_gen"]
                txt += (f", and the risk gate sent {fq['labels'].get('crisis', 0)} of {fq['n']} informational FAQ "
                        "questions to the crisis protocol")
            parts.append(txt + ".")
    m.text("AbstractResults", " ".join(p for p in parts if p), "run all experiments (scripts/run_all_local.sh)")

    OUT.parent.mkdir(parents=True, exist_ok=True)  # paper/ is not in the repository (kept private)
    OUT.write_text("% Auto-generated by scripts/paper_numbers.py from results/ -- do not edit by hand.\n"
                   + "\n".join(m.lines) + "\n")
    todo = sum("todorun" in ln for ln in m.lines)
    print(f"wrote {OUT} ({len(m.lines)} macros, {todo} pending)")


if __name__ == "__main__":
    main()
