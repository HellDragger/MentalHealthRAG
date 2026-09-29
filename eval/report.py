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
                        caption=f"Retrieval on {DS_NAMES.get(d, d)} (n={n}, {c}-token chunks). R@k = at least one "
                                "relevant chunk in the top $k$. 95\\% bootstrap CIs in brackets. $^{\\dagger}$ baseline "
                                "(v1 embedder, dense); $^{*}$ significantly different from the baseline on nDCG@10 "
                                "(paired bootstrap, Holm-corrected $p<0.05$). ms = median retrieval latency.",
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


def generation_tables(res: dict) -> None:
    from eval.generation_eval import render_tables

    render_tables(res)
