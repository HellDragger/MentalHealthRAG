"""Render experiment JSON into LaTeX tables (paper/tables/)."""

from __future__ import annotations

from eval.tables import esc, fmt, fmt_ci, write_table

EMB_NAMES = {"minilm": "MiniLM-L6", "bge-small": "BGE-small", "bge-base": "BGE-base", "e5-base": "E5-base",
             "gte-base": "GTE-base", None: "--"}
SYS_NAMES = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid (RRF)", "hybrid+ce": "Hybrid + MiniLM-CE",
             "hybrid+bge-rr": "Hybrid + BGE-reranker"}
DS_NAMES = {"heading_qa": "HeadingQA", "paraphrase_qa": "ParaphraseQA", "synth_qa": "SynthQA"}


def _sig_mark(res: dict, run: dict) -> str:
    key = f"{run['dataset']}|c{run['chunk_tokens']}|nDCG@10"
    fam = res.get("significance", {}).get(key)
    if not fam:
        return ""
    rid = f"{run['system']}|{run['embedder'] or '-'}|c{run['chunk_tokens']}"
    if rid == fam["baseline"]:
        return "$^{\\dagger}$"
    p = fam["p_bootstrap_holm"].get(rid)
    if p is None:
        return ""
    return "$^{*}$" if p < 0.05 else ""


def retrieval_tables(res: dict) -> None:
    runs = res["runs"]
    datasets = list(dict.fromkeys(r["dataset"] for r in runs))
    chunks = sorted({r["chunk_tokens"] for r in runs})
    name = res["name"]
    if len(chunks) == 1:
        c = chunks[0]
        retrieval_summary(res, datasets, c)
        for d in datasets:
            rows = []
            best = max(r["summary"]["nDCG@10"]["mean"] for r in runs if r["dataset"] == d)
            for r in [r for r in runs if r["dataset"] == d and r["chunk_tokens"] == c]:
                s = r["summary"]
                nd = s["nDCG@10"]
                cell = fmt_ci(nd["mean"], nd["lo"], nd["hi"])
                if abs(nd["mean"] - best) < 1e-12:
                    cell = "\\textbf{" + f"{nd['mean']:.3f}" + "}" + cell[len(f"{nd['mean']:.3f}"):]
                rows.append([esc(SYS_NAMES.get(r["system"], r["system"])), EMB_NAMES.get(r["embedder"], r["embedder"]),
                             fmt(s["R@1"]["mean"]), fmt(s["R@5"]["mean"]), fmt(s["R@10"]["mean"]),
                             fmt_ci(s["MRR@10"]["mean"], s["MRR@10"]["lo"], s["MRR@10"]["hi"]),
                             cell + _sig_mark(res, r), f"{r['summary']['latency_ms']['p50']:.0f}"])
            n = next(r["n_queries"] for r in runs if r["dataset"] == d)
            write_table(f"retrieval_{name}_{d}", ["System", "Embedder", "R@1", "R@5", "R@10", "MRR@10", "nDCG@10",
                                                  "ms"], rows,
                        caption=f"Retrieval on {DS_NAMES.get(d, d)} (n={n}, {c}-token chunks; 95\\% bootstrap CIs). R@k: "
                                "a relevant chunk in the top $k$; ms: median latency. $^{\\dagger}$ baseline (dense "
                                "MiniLM); $^{*}$ differs from it on nDCG@10 (Holm-corrected $p<0.05$).",
                        label=f"tab:retrieval-{name}-{d.replace('_', '-')}", align="llrrrrrr",
                        source=f"results/retrieval_{name}.json")
    else:
        for d in datasets:
            rows = []
            keys = list(dict.fromkeys((r["system"], r["embedder"]) for r in runs if r["dataset"] == d))
            for sysname, emb in keys:
                row = [esc(SYS_NAMES.get(sysname, sysname)), EMB_NAMES.get(emb, emb)]
                for c in chunks:
                    r = next((r for r in runs if r["dataset"] == d and r["system"] == sysname and r["embedder"] == emb
                              and r["chunk_tokens"] == c), None)
                    row.append(fmt(r["summary"]["nDCG@10"]["mean"]) if r else "--")
                rows.append(row)
            n = next(r["n_queries"] for r in runs if r["dataset"] == d)
            write_table(f"retrieval_{name}_{d}", ["System", "Embedder"] + [f"nDCG@10 c{c}" for c in chunks], rows,
                        caption=f"Chunk-size ablation on {DS_NAMES.get(d, d)} (n={n}): nDCG@10 for 128/256/512-token "
                                "chunks (MiniLM is capped at its 256-token input limit).",
                        label=f"tab:retrieval-{name}-{d.replace('_', '-')}", source=f"results/retrieval_{name}.json")


SYS_SHORT = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid", "hybrid+ce": "H+CE", "hybrid+bge-rr": "H+BGE"}
EMB_SHORT = {"minilm": "MiniLM", "bge-small": "BGE-s", "bge-base": "BGE-b", "e5-base": "E5", "gte-base": "GTE"}


def short_name(system: str, embedder: str | None) -> str:
    return SYS_SHORT.get(system, system) + (f" ({EMB_SHORT.get(embedder, embedder)})" if embedder else "")


def retrieval_summary(res: dict, datasets: list[str], c: int) -> None:
    """Main-text table: the baseline, BM25, the MiniLM pipelines, the deployed configuration and the best one per
    query set, with nDCG@10 on every set (the full tables go to the appendix)."""
    from mhrag.config import get_settings

    st = get_settings()
    dep_sys = "hybrid+ce" if st.retrieval.reranker == "minilm-ce" else "hybrid+bge-rr"
    runs = [r for r in res["runs"] if r["chunk_tokens"] == c]
    keys = [("dense", "minilm"), ("bm25", None), ("hybrid", "minilm"), ("hybrid+ce", "minilm"),
            ("hybrid+bge-rr", "minilm"), (dep_sys, st.index.embedder)]
    for d in datasets:
        best = max((r for r in runs if r["dataset"] == d), key=lambda r: r["summary"]["nDCG@10"]["mean"])
        keys.append((best["system"], best["embedder"]))
    keys = list(dict.fromkeys(keys))
    by = {(r["system"], r["embedder"], r["dataset"]): r for r in runs}
    best_val = {d: max(r["summary"]["nDCG@10"]["mean"] for r in runs if r["dataset"] == d) for d in datasets}
    rows = []
    for sysk, emb in keys:
        label = short_name(sysk, emb) + (", deployed" if (sysk, emb) == (dep_sys, st.index.embedder) else "")
        row = [esc(label)]
        for d in datasets:
            r = by.get((sysk, emb, d))
            if not r:
                row.append("--")
                continue
            nd = r["summary"]["nDCG@10"]
            cell = fmt_ci(nd["mean"], nd["lo"], nd["hi"])
            if abs(nd["mean"] - best_val[d]) < 1e-12:
                cell = "\\textbf{" + f"{nd['mean']:.3f}" + "}" + cell[len(f"{nd['mean']:.3f}"):]
            row.append(cell + _sig_mark(res, r))
        rows.append(row)
    n = {d: next(r["n_queries"] for r in runs if r["dataset"] == d) for d in datasets}
    write_table(f"retrieval_{res['name']}_summary", ["Configuration"] + [f"{DS_NAMES.get(d, d)} ({n[d]})" for d in datasets],
                rows, caption=f"Retrieval summary: nDCG@10 with 95\\% bootstrap CIs ({c}-token chunks); best per set in "
                "bold. H = hybrid (BM25 + dense, RRF); CE = MiniLM cross-encoder; BGE = BGE reranker; embedder in brackets "
                "(BGE-s/-b = small/base). $^{\\dagger}$ baseline; $^{*}$ differs from it (Holm-corrected $p<0.05$). All "
                "configurations: \\ref{app:retrieval}.",
                label="tab:retrieval-summary", align="l" + "r" * len(datasets), source=f"results/retrieval_{res['name']}.json")


def generation_tables(res: dict) -> None:
    from eval.generation_eval import render_tables

    render_tables(res)
