"""Re-analyses of the saved generation answers (no model is run again).

    python -m scripts.generation_reanalysis      # -> results/generation_reanalysis.json, paper/tables/gen_*.tex

1. Template-free comparison. In the full pipeline the risk gate answers some questions with a fixed safety template
   (11 of 98 FAQ-Gen questions). Every metric is recomputed on the questions the model answered itself, and the
   other settings are restricted to the same questions, so the comparison is paired and template-free.
2. Abstention wording. Out-of-scope abstention is detected with a phrase list, and the full pipeline's system
   prompt suggests one of those phrases. For each setting: the share of detected abstentions that use the prompt's
   suggested sentence, and the share detected only by the other patterns.
3. Per-model diagnostics used in the discussion (citation rate, answer length, abstention by setting).
"""

from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict

import numpy as np

from eval.generation_eval import SAFETY_TEMPLATES
from eval.metrics import bootstrap_ci, holm, paired_bootstrap_p
from eval.tables import write_table
from mhrag.config import get_settings

R = get_settings().results_dir
SOURCES = {"local": R / "generation" / "local" / "scored.jsonl", "gpu": R / "generation" / "gpu" / "scored.jsonl.gz"}
METRICS = ("faithfulness", "answer_relevance", "rougeL", "bertscore_f1", "has_citation", "judge_helpfulness")
# The full pipeline's system prompt suggests this sentence when the sources lack the answer (mhrag/prompts.py)
# ("... about that ...", or with the topic filled in: "... about changing the oil in my sources")
PROMPT_PHRASE = re.compile(r"(don[’']?t|do not) have (any )?information (about|on) [^.]{0,80}?\bin my sources", re.I)


def load(path):
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _vals(d: dict, ids, metric):
    return {i: float(d[i]["metrics"][metric]) for i in ids
            if i in d and (d[i].get("metrics") or {}).get(metric) is not None}


def template_free(by: dict, model: str, dataset: str) -> dict | None:
    full = by.get((model, "full", dataset))
    if not full:
        return None
    templated = sorted(i for i, r in full.items() if r.get("gate") in SAFETY_TEMPLATES)
    keep = [i for i in full if i not in templated]
    out: dict = {"n": len(full), "templated": len(templated), "template_free_n": len(keep), "metrics": {}}
    for metric in METRICS:
        m: dict = {"full_all": None, "settings": {}, "vs_full": {}}
        allv = _vals(full, list(full), metric)
        if allv:
            m["full_all"] = dict(zip(("mean", "lo", "hi"), bootstrap_ci(list(allv.values()), n_boot=1000), strict=True))
            m["full_all"]["n"] = len(allv)
        fv = _vals(full, keep, metric)
        raw, delta = {}, {}
        for prof in ("full", "naive", "no_rag"):
            v = _vals(by.get((model, prof, dataset), {}), keep, metric)
            if v:
                m["settings"][prof] = dict(zip(("mean", "lo", "hi"), bootstrap_ci(list(v.values()), n_boot=1000),
                                               strict=True)) | {"n": len(v)}
            if prof != "full" and v and fv:
                ids = [i for i in fv if i in v]
                if len(ids) >= 5:
                    a, b = [fv[i] for i in ids], [v[i] for i in ids]
                    raw[prof], delta[prof] = paired_bootstrap_p(a, b), float(np.mean(a) - np.mean(b))
        m["vs_full"] = {"delta": delta, "p_bootstrap": raw, "p_bootstrap_holm": holm(raw)}
        if m["settings"]:
            out["metrics"][metric] = m
    return out


def abstention_wording(by: dict, model: str) -> dict:
    out = {}
    for prof in ("no_rag", "naive", "full"):
        rs = [r for r in by.get((model, prof, "oos_questions"), {}).values() if r.get("generated")]
        flagged = [r for r in rs if (r.get("metrics") or {}).get("abstained")]
        phrase = [r for r in flagged if PROMPT_PHRASE.search(r["answer"] or "")]
        out[prof] = {"generated": len(rs), "abstained": len(flagged), "with_prompt_phrase": len(phrase),
                     "other_patterns_only": len(flagged) - len(phrase)}
    return out


def diagnostics(by: dict, model: str) -> dict:
    out = {}
    for prof in ("no_rag", "naive", "full"):
        faq = [r for r in by.get((model, prof, "faq_gen"), {}).values() if r.get("generated")]
        oos = [r for r in by.get((model, prof, "oos_questions"), {}).values() if r.get("generated")]
        def mean(rs, k):  # None when no answer has the metric (e.g. faithfulness without retrieval)
            v = [r["metrics"][k] for r in rs if r["metrics"].get(k) is not None]
            return float(np.mean(v)) if v else None
        out[prof] = {"faq_citation_rate": mean(faq, "has_citation"), "faq_words": mean(faq, "words"),
                     "faq_faithfulness": mean(faq, "faithfulness"), "oos_abstained": mean(oos, "abstained"),
                     "oos_citation_rate": mean(oos, "has_citation")}
    return out


def main():
    res: dict = {"note": __doc__.split("\n\n", 2)[2].strip(), "experiments": {}}
    for exp, path in SOURCES.items():
        if not path.exists():
            continue
        by: dict = defaultdict(dict)
        for r in load(path):
            by[(r["model"], r["profile"], r["dataset"])][r["id"]] = r
        models = sorted({m for m, _, _ in by})
        res["experiments"][exp] = {
            m: {"template_free": {d: template_free(by, m, d) for d in ("faq_gen", "counsel_gen")},
                "abstention_wording": abstention_wording(by, m), "diagnostics": diagnostics(by, m)}
            for m in models}
    (R / "generation_reanalysis.json").write_text(json.dumps(res, indent=2))

    loc = res["experiments"].get("local", {}).get("qwen2.5-1.5b-gguf", {}).get("template_free", {}).get("faq_gen")
    if loc:
        names = {"faithfulness": "NLI faithfulness", "answer_relevance": "Relevance", "rougeL": "ROUGE-L",
                 "bertscore_f1": "BERTScore", "has_citation": "Citation rate", "judge_helpfulness": "Judge helpfulness (1--5)"}
        rows = []
        for metric, name in names.items():
            m = loc["metrics"].get(metric)
            if not m:
                continue
            s, vs = m["settings"], m["vs_full"]
            def cell(p, s=s):
                return f"{s[p]['mean']:.3f}" if p in s else "--"

            def pv(p, vs=vs):
                return f"{vs['p_bootstrap_holm'][p]:.3f}" if p in vs["p_bootstrap_holm"] else "--"
            n = s.get("full", {}).get("n", "--")
            rows.append([name, f"{m['full_all']['mean']:.3f}" if m["full_all"] else "--", cell("full"), cell("naive"),
                         cell("no_rag"), pv("naive"), pv("no_rag"), str(n)])
        write_table("gen_template_free", ["Metric", "Full (all)", "Full", "Naive RAG", "No RAG", "$p$ vs naive",
                                          "$p$ vs no RAG", "$n$"], rows,
                    f"FAQ-Gen with Qwen2.5-1.5B (GGUF): the {loc['templated']} questions the risk gate answered with "
                    "a safety template are removed from every setting. ``Full (all)'' keeps them, as in the main "
                    "tables. $p$: paired bootstrap of the full pipeline against each setting on the same questions, "
                    "Holm-corrected per metric. Judge helpfulness is on the judge's fixed question sample.",
                    "tab:gen-template-free", align="lrrrrrrr", source="results/generation_reanalysis.json")
    print(f"wrote {R / 'generation_reanalysis.json'}")


if __name__ == "__main__":
    main()
