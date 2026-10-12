"""Citation precision: is each cited passage the one that supports the sentence citing it?

    python -m scripts.citation_precision        # -> results/citation_precision.json, paper/tables/citation_precision.tex
    python -m scripts.citation_precision --tables-only   # only the table, from the saved results

NLI faithfulness asks whether *any* retrieved passage entails a sentence, so it rises with the amount of retrieved
text. Citation precision does not: for every answer sentence that carries a citation ([2], [1][3] or [1, 3]), the
sentence (citations removed) is checked against exactly the passages it cites. Reported per model for the full
pipeline on FAQ-Gen, and for the ablation settings that use the citing prompt:
- citation precision: share of (sentence, cited passage) pairs where the passage entails the sentence;
- supported citing sentences: share of citing sentences entailed by at least one of their cited passages;
- citing sentences: share of answer sentences that carry a citation; invalid: citations to a source that does not exist.
Premises come from the saved generation checkpoints. Template answers (risk gate) are not model answers and are skipped.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict

from eval.judges import NLIJudge
from eval.metrics import bootstrap_ci
from eval.tables import esc, write_table
from mhrag.config import get_settings
from scripts.rescore_faithfulness import full_passages, latest

R = get_settings().results_dir
CITE = re.compile(r"\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]")
SENT = re.compile(r"(?<=[.!?])\s+|\n+")
LEAD = re.compile(r"^((?:\s*\[\d{1,2}(?:\s*,\s*\d{1,2})*\])+)\s*")
SETS = [("local", "qwen2.5-1.5b-gguf", "full"), ("ablation", "qwen2.5-1.5b-gguf", "abl_naive_full"),
        ("ablation", "qwen2.5-1.5b-gguf", "abl_full_full"), ("ablation", "qwen2.5-1.5b-gguf", "abl_full_full_matched")]


def cited_sentences(answer: str) -> tuple[list[tuple[str, list[int]]], int]:
    """(sentence without citations, cited source numbers) for each sentence, and the number of sentences.
    A marker that starts a sentence ("... support. [1] Next ...") belongs to the previous sentence."""
    parts = [p for p in SENT.split(answer) if p and p.strip()]
    sents: list[list] = []
    for p in parts:
        m = LEAD.match(p)
        if m and sents:
            sents[-1][1] += [int(x) for g in CITE.findall(m.group(1)) for x in g.split(",")]
            p = p[m.end():]
        cites = [int(x) for g in CITE.findall(p) for x in g.split(",")]
        text = re.sub(r"^[\s*\-\d.]+\s+", "", CITE.sub("", p)).strip()
        sents.append([text, cites])
    sents = [(t, c) for t, c in sents if len(t.split()) >= 4]
    return [(t, sorted(set(c))) for t, c in sents if c], len(sents)


def score_answer(nli: NLIJudge, answer: str, passages: list[str]) -> dict:
    cs, n_sent = cited_sentences(answer)
    pairs, supported, invalid = [], 0, 0
    for text, cites in cs:
        ok = False
        for c in cites:
            if not 1 <= c <= len(passages):
                invalid += 1
                continue
            ent = (nli.score(text, [passages[c - 1]], threshold=0.5)["faithfulness"] or 0.0) >= 1.0
            pairs.append(ent)
            ok = ok or ent
        supported += ok
    return {"sentences": n_sent, "citing": len(cs), "pairs": len(pairs), "pairs_entailed": int(sum(pairs)),
            "supported_citing": supported, "invalid": invalid}


def main(argv=None) -> int:
    if "--tables-only" in (sys.argv[1:] if argv is None else argv):
        write_tables(json.loads((R / "citation_precision.json").read_text()))
        return 0
    nli = NLIJudge("cross-encoder/nli-deberta-v3-base")
    sets = list(SETS)
    gpu = json.loads((R / "generation_gpu.json").read_text())
    from eval.generation_eval import model_group
    from mhrag.llm.registry import load_catalog

    cat = load_catalog()
    sets += [("gpu", m, "full") for m in gpu["config"]["models"] if model_group(m, cat) != "base"
             and (R / "generation" / "gpu" / f"{m}__full__faq_gen.jsonl").exists()]
    out: dict = {"definition": __doc__.split("\n\n", 2)[1].strip(), "sets": {}}
    for exp, model, profile in sets:
        path = R / "generation" / exp / f"{model}__{profile}__faq_gen.jsonl"
        if not path.exists():
            continue
        per = []
        for r in latest(path).values():
            if r.get("generated") is False or not r.get("context"):
                continue
            passages = r["passages"] if r.get("passages") else full_passages(r["context"])
            per.append(score_answer(nli, r.get("answer") or "", passages))
        tot = defaultdict(int)
        for p in per:
            for k, v in p.items():
                tot[k] += v
        prec = [p["pairs_entailed"] / p["pairs"] for p in per if p["pairs"]]
        out["sets"][f"{exp}:{model}:{profile}"] = {
            "answers": len(per), "answers_with_citations": sum(p["citing"] > 0 for p in per), **tot,
            "citation_precision": tot["pairs_entailed"] / tot["pairs"] if tot["pairs"] else None,
            "citation_precision_answer_ci": list(bootstrap_ci(prec)) if prec else None,
            "supported_citing_share": tot["supported_citing"] / tot["citing"] if tot["citing"] else None,
            "citing_share": tot["citing"] / tot["sentences"] if tot["sentences"] else None,
        }
        print(exp, model, profile, out["sets"][f"{exp}:{model}:{profile}"]["citation_precision"], flush=True)
    (R / "citation_precision.json").write_text(json.dumps(out, indent=2))
    write_tables(out)
    return 0


def write_tables(out: dict) -> None:
    names = {"full": "Full pipeline", "abl_naive_full": "Ablation: naive retrieval, full prompt",
             "abl_full_full": "Ablation: full retrieval, full prompt",
             "abl_full_full_matched": "Ablation: full retrieval cut to naive length"}
    rows = []
    for key, v in out["sets"].items():
        exp, model, profile = key.split(":")
        label = esc(model) + ("" if profile == "full" and exp == "gpu" else f" ({names.get(profile, profile)})")
        f = lambda x: "--" if x is None else f"{x:.2f}"  # noqa: E731
        rows.append([label, str(v["answers"]), f(v["citing_share"]), f(v["citation_precision"]),
                     f(v["supported_citing_share"]), str(v["invalid"])])
    write_table("citation_precision", ["Model / setting", "Answers", "Citing sent.", "Citation prec.",
                                       "Supported citing sent.", "Invalid"], rows,
                "Citation precision on FAQ-Gen (model answers only): does the cited passage entail the sentence (NLI)? "
                "Citing sent.: answer sentences with a citation; Citation prec.: entailed (sentence, cited passage) "
                "pairs; Supported citing sent.: citing sentences entailed by a cited passage; Invalid: citations to a "
                "source that was not given.", "tab:citation-precision", align="lrrrrr",
                source="results/citation_precision.json")


if __name__ == "__main__":
    raise SystemExit(main())
