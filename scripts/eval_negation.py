"""Negation test of the risk gate on minimal pairs (eval/data/negation_pairs.jsonl).

    python -m scripts.eval_negation        # -> results/negation_test.json, paper/tables/negation_test.tex

The classifiers are trained on text that is lowercased and stripped of stop words, which removes most negations
("not", "no"; "dont" survives), so "I don't want to die" may look like "I want to die" to them. Each pair differs only
in polarity. Half the pairs carry risk without a negation and are safe with one ("I want to die" / "I don't want to
die"); the other half carry risk *through* a negation ("I don't want to be alive anymore" / "I want to be alive").
The pairs were written for this diagnostic only and were not used to design or tune the gate.
Reported per classifier and for the deployed gate: risk statements sent to the crisis protocol, no-risk statements
sent to it, and the mean classifier score of each polarity.
"""

from __future__ import annotations

import json

import numpy as np

from eval.tables import write_table
from mhrag.config import PROJECT_ROOT, get_settings
from mhrag.safety.classifier import load_classifier
from mhrag.safety.gate import SafetyGate

DATA = PROJECT_ROOT / "eval" / "data" / "negation_pairs.jsonl"


def main() -> int:
    s = get_settings()
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    out: dict = {"n_pairs": len({r["pair"] for r in rows}), "variants": {}}
    variants = {"lexicon_only": SafetyGate(None)}
    for kind in ("tfidf_lr", "transformer"):
        try:
            clf = load_classifier(kind, s.artifacts_dir)
        except OSError:  # weights not on this machine (the transformer is trained in the Kaggle classifier step)
            clf = None
        if clf is not None:
            variants[f"{kind}_only"] = SafetyGate(clf, use_lexicon=False)
            variants[f"lexicon+{kind}"] = SafetyGate(clf)
    for name, gate in variants.items():
        res = [(r, gate.assess(r["text"])) for r in rows]
        risk = [g for r, g in res if r["polarity"] == "risk"]
        safe = [g for r, g in res if r["polarity"] == "no_risk"]
        by_pair: dict = {}
        for r, g in res:
            by_pair.setdefault(r["pair"], {})[r["polarity"]] = g.label == "crisis"
        sc = lambda gs: float(np.mean([g.classifier_score for g in gs])) if gs[0].classifier_score is not None else None  # noqa: E731
        out["variants"][name] = {
            "risk_to_crisis": sum(g.label == "crisis" for g in risk), "risk_n": len(risk),
            "risk_any_safety": sum(g.label != "none" for g in risk),
            "no_risk_to_crisis": sum(g.label == "crisis" for g in safe), "no_risk_n": len(safe),
            "pairs_separated": sum(p.get("risk") and not p.get("no_risk") for p in by_pair.values()),
            "mean_score_risk": sc(risk), "mean_score_no_risk": sc(safe),
            "items": [{"pair": r["pair"], "polarity": r["polarity"], "text": r["text"], "label": g.label,
                       "score": g.classifier_score} for r, g in res],
        }
    (s.results_dir / "negation_test.json").write_text(json.dumps(out, indent=2))
    nice = {"lexicon_only": "Lexicon only", "tfidf_lr_only": "TF-IDF+LR only", "transformer_only": "DistilRoBERTa only",
            "lexicon+tfidf_lr": "Lexicon + TF-IDF+LR (deployed)", "lexicon+transformer": "Lexicon + DistilRoBERTa"}
    trows = []
    for name, v in out["variants"].items():
        f = lambda x: "--" if x is None else f"{x:.2f}"  # noqa: E731
        trows.append([nice.get(name, name), f"{v['risk_to_crisis']}/{v['risk_n']}",
                      f"{v['no_risk_to_crisis']}/{v['no_risk_n']}", f"{v['pairs_separated']}/{out['n_pairs']}",
                      f(v["mean_score_risk"]), f(v["mean_score_no_risk"])])
    write_table("negation_test", ["Gate", "Risk to crisis", "No-risk to crisis", "Pairs separated",
                                  "Score (risk)", "Score (no risk)"], trows,
                f"Negation test: {out['n_pairs']} minimal pairs that differ only in polarity (half carry risk through a "
                "negation, e.g. ``I don't want to be alive anymore''). Risk / no-risk to crisis: statements sent to the "
                "crisis protocol; Pairs separated: the risk statement is escalated and its no-risk twin is not; Score: "
                "mean classifier probability. The pairs were written for this test and not used to design the gate.",
                "tab:negation-test", align="lrrrrr", source="results/negation_test.json")
    for name, v in out["variants"].items():
        print(f"{name:22s} risk->crisis {v['risk_to_crisis']}/{v['risk_n']}  no_risk->crisis "
              f"{v['no_risk_to_crisis']}/{v['no_risk_n']}  separated {v['pairs_separated']}/{out['n_pairs']}  "
              f"score {v['mean_score_risk']} vs {v['mean_score_no_risk']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
