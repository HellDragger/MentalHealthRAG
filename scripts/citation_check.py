"""Hand check of citation precision: is the NLI model too strict when it says a cited passage does not support a sentence?

    python -m scripts.citation_check --make     # -> eval/human_eval/out/citation_check/sheet_rater1.csv (+ key.csv)
    python -m scripts.citation_check            # after rating -> results/citation_check.json

--make samples (sentence, cited passage) pairs from the full-pipeline FAQ-Gen answers of the deployed model and the 13
instruction-tuned benchmark models (the pairs behind Table citation-precision), stratified by the NLI verdict: N_SUPPORTED
pairs the NLI model counts as entailed and N_UNSUPPORTED it does not, in random order, without the verdict. A rater
labels each pair: "supported" (the passage states the sentence or plainly implies it), "partly" (some of the sentence's
claims but not all) or "not" (the passage does not support it). Several raters can each fill a copy of the sheet.

Scoring reports, per stratum, the share a human judged supported, the agreement with NLI (Cohen's kappa), and citation
precision corrected for NLI error: the human-supported share in each stratum, weighted by the stratum's share of all
pairs (from results/citation_precision.json). "partly" counts as not supported; the share is reported separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter

from mhrag.config import PROJECT_ROOT, get_settings

R = get_settings().results_dir
OUT = PROJECT_ROOT / "eval" / "human_eval" / "out" / "citation_check"
N_SUPPORTED, N_UNSUPPORTED, SEED = 10, 20, 7
LABELS = ("supported", "partly", "not")


def _population() -> list[tuple[str, str, list[str]]]:
    """(set key, answer, passages) for every model answer in the citation-precision sets used here."""
    from eval.generation_eval import model_group
    from mhrag.llm.registry import load_catalog
    from scripts.rescore_faithfulness import full_passages, latest

    cat = load_catalog()
    gpu = json.loads((R / "generation_gpu.json").read_text())["config"]["models"]
    sets = [("local", "qwen2.5-1.5b-gguf")] + [("gpu", m) for m in gpu if model_group(m, cat) != "base"]
    out = []
    for exp, model in sets:
        path = R / "generation" / exp / f"{model}__full__faq_gen.jsonl"
        if not path.exists():
            continue
        for r in latest(path).values():
            if r.get("generated") is False or not r.get("context"):
                continue
            out.append((f"{exp}:{model}", r.get("answer") or "", r["passages"] if r.get("passages") else
                        full_passages(r["context"])))
    return out


def make() -> None:
    from eval.judges import NLIJudge
    from scripts.citation_precision import cited_sentences

    rng = random.Random(SEED)
    pairs = []
    for key, answer, passages in _population():
        for text, cites in cited_sentences(answer)[0]:
            pairs += [(key, text, passages[c - 1]) for c in cites if 1 <= c <= len(passages)]
    rng.shuffle(pairs)
    nli = NLIJudge("cross-encoder/nli-deberta-v3-base")
    sup, uns = [], []
    for key, text, passage in pairs:  # same rule as citation_precision.py: one passage, threshold 0.5
        if len(sup) >= N_SUPPORTED and len(uns) >= N_UNSUPPORTED:
            break
        ent = (nli.score(text, [passage], threshold=0.5)["faithfulness"] or 0.0) >= 1.0
        (sup if ent else uns).append((key, text, passage, ent))
    chosen = sup[:N_SUPPORTED] + uns[:N_UNSUPPORTED]
    rng.shuffle(chosen)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "sheet_rater1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "sentence", "cited_passage", "label (supported / partly / not)", "note"])
        for i, (_, text, passage, _) in enumerate(chosen, 1):
            w.writerow([f"c{i:02d}", text, passage, "", ""])
    with open(OUT / "key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "set", "nli_supported"])
        for i, (key, _, _, ent) in enumerate(chosen, 1):
            w.writerow([f"c{i:02d}", key, int(ent)])
    print(f"wrote {OUT / 'sheet_rater1.csv'} ({len(chosen)} pairs; key.csv holds the NLI verdicts, do not open it "
          "before rating)")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 1.0)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * (ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5 / d
    return (max(0.0, c - h), min(1.0, c + h))


def cohen(a: list[int], b: list[int]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def score() -> int:
    with open(OUT / "key.csv", encoding="utf-8") as f:
        key = {r["item_id"]: r for r in csv.DictReader(f)}
    sheets = sorted(OUT.glob("sheet_rater*.csv"))
    ratings: dict[str, dict[str, str]] = {}
    for sh in sheets:
        col = None
        with open(sh, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            col = col or next(k for k in r if k.startswith("label"))
            lab = (r[col] or "").strip().lower()
            if lab:
                if lab not in LABELS:
                    raise SystemExit(f"{sh.name} {r['item_id']}: label {lab!r} is not one of {LABELS}")
                ratings.setdefault(sh.stem, {})[r["item_id"]] = lab
    if not ratings or any(len(v) < len(key) for v in ratings.values()):
        print("not every sheet is fully rated:", {k: f"{len(v)}/{len(key)}" for k, v in ratings.items()})
        return 1
    cp = json.loads((R / "citation_precision.json").read_text())["sets"]
    keys = {k["set"] for k in key.values()}
    sel = [v for k, v in cp.items() if k.endswith(":full") and k.rsplit(":", 1)[0] in keys]  # exp:model:profile
    tot = sum(v["pairs"] for v in sel)
    ent = sum(v["pairs_entailed"] for v in sel)
    res = {"definition": __doc__.split("\n\n", 2)[1].strip(), "pairs": len(key),
           "population": {"pairs": tot, "nli_supported": ent}, "raters": {}}
    for rater, lab in ratings.items():
        strata = {}
        for s_, name in ((1, "nli_supported"), (0, "nli_unsupported")):
            ids = [i for i, k in key.items() if int(k["nli_supported"]) == s_]
            c = Counter(lab[i] for i in ids)
            strata[name] = {"n": len(ids), **{x: c.get(x, 0) for x in LABELS},
                            "human_supported_share": c.get("supported", 0) / len(ids) if ids else None}
        def corrected_for(share1, share0):
            return (ent * share1 + (tot - ent) * share0) / tot if tot else None
        w1, w0 = (wilson(strata[n_]["supported"], strata[n_]["n"]) for n_ in ("nli_supported", "nli_unsupported"))
        for n_, w_ in (("nli_supported", w1), ("nli_unsupported", w0)):
            strata[n_]["human_supported_ci"] = list(w_)
        # "partly" counted as supported: an upper reading of the same labels
        p1, p0 = ((strata[n_]["supported"] + strata[n_]["partly"]) / strata[n_]["n"]
                  for n_ in ("nli_supported", "nli_unsupported"))
        nli = [int(key[i]["nli_supported"]) for i in key]
        hum = [int(lab[i] == "supported") for i in key]
        s1, s0 = strata["nli_supported"]["human_supported_share"], strata["nli_unsupported"]["human_supported_share"]
        corrected = (ent * s1 + (tot - ent) * s0) / tot if tot and None not in (s1, s0) else None
        res["raters"][rater] = {"strata": strata, "agreement": sum(a == b for a, b in zip(nli, hum, strict=True)) / len(nli),
                                "cohen_kappa": cohen(nli, hum), "nli_citation_precision": ent / tot if tot else None,
                                "corrected_citation_precision": corrected,
                                # bounds from the two strata's Wilson intervals (conservative: both at their limits)
                                "corrected_ci": [corrected_for(w1[0], w0[0]), corrected_for(w1[1], w0[1])],
                                "corrected_partly_as_supported": corrected_for(p1, p0)}
    if len(ratings) > 1:
        names = sorted(ratings)
        a, b = ([int(ratings[n][i] == "supported") for i in key] for n in names[:2])
        res["inter_rater_kappa"] = cohen(a, b)
    (R / "citation_check.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["raters"], indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make", action="store_true")
    args = ap.parse_args(argv)
    if args.make:
        make()
        return 0
    return score()


if __name__ == "__main__":
    raise SystemExit(main())
