"""Trade-off of the gate's classifier crisis threshold: crisis recall against over-escalation of ordinary questions.

    python -m scripts.gate_tradeoff        # -> results/gate_tradeoff.json, paper/tables/gate_tradeoff.tex

The deployed gate sends a first-person message to the crisis protocol when the classifier score reaches a
high-precision threshold, and to the non-blocking elevated tier above a lower one. Raising the crisis threshold (up to
"cap at elevated", where the classifier can never block and only the lexicon can) trades crisis recall for fewer
help-seeking questions sent to the crisis protocol. This script traces that curve with everything else fixed. It
describes the trade-off on the existing sets; choosing a new operating point from it would need a fresh held-out set.
"""

from __future__ import annotations

import json

import numpy as np

from eval.datasets import load_jsonl
from eval.tables import write_table
from mhrag.config import PROJECT_ROOT, get_settings
from mhrag.safety.classifier import load_classifier
from mhrag.safety.gate import SafetyGate

ESCALATE = {"crisis", "harmful_request", "third_party"}
SETS = {"dev": PROJECT_ROOT / "eval" / "data" / "safety_prompts.jsonl",
        "heldout": PROJECT_ROOT / "eval" / "data" / "safety_prompts_heldout.jsonl"}


def rates(gate: SafetyGate, redteam: dict, ordinary: dict) -> dict:
    out = {}
    for name, rows in redteam.items():
        crisis = [r for r in rows if r["expected"] == "crisis"]
        neg = [r for r in rows if r["expected"] in ("support", "answer")]
        k = sum(gate.assess(r["text"]).label in ESCALATE for r in crisis)
        fp = sum(gate.assess(r["text"]).label in ESCALATE for r in neg)
        out[name] = {"crisis_recall": k / len(crisis), "crisis_k": k, "crisis_n": len(crisis),
                     "fp": fp, "neg_n": len(neg)}
    for name, rows in ordinary.items():
        labels = [gate.assess(r["query"]).label for r in rows]
        out[name] = {"crisis": sum(lab == "crisis" for lab in labels), "elevated": sum(lab == "elevated" for lab in labels),
                     "n": len(rows)}
    return out


def main() -> int:
    s = get_settings()
    clf = load_classifier(s.safety.classifier, s.artifacts_dir)
    if clf is None:
        raise SystemExit(f"classifier {s.safety.classifier} not found in {s.artifacts_dir}")
    redteam = {k: [json.loads(line) for line in v.read_text(encoding="utf-8").splitlines() if line.strip()]
               for k, v in SETS.items()}
    ordinary = {"faq_gen": load_jsonl("faq_gen.jsonl"), "counsel_gen": load_jsonl("counsel_gen.jsonl")}
    for rows in ordinary.values():
        for r in rows:
            r["query"] = r.get("query") or r.get("question") or r.get("text")
    deployed = SafetyGate(clf).crisis_threshold
    elevated = SafetyGate(clf).elevated_threshold
    grid = sorted({round(float(t), 3) for t in np.linspace(elevated, 0.95, 12)} | {round(deployed, 3)})
    points = []
    for t in grid + [1.01]:
        r = rates(SafetyGate(clf, crisis_threshold=t), redteam, ordinary)
        points.append({"crisis_threshold": t, "cap_at_elevated": t > 1, "deployed": abs(t - round(deployed, 3)) < 1e-9,
                       **r})
    out = {"classifier": s.safety.classifier, "deployed_threshold": deployed, "elevated_threshold": elevated,
           "points": points}
    (s.results_dir / "gate_tradeoff.json").write_text(json.dumps(out, indent=2))

    rows = []
    for p in points:
        label = "cap at elevated" if p["cap_at_elevated"] else f"{p['crisis_threshold']:.3f}"
        if p["deployed"]:
            label += " (deployed)"
        h, d = p["heldout"], p["dev"]
        rows.append([label, f"{h['crisis_k']}/{h['crisis_n']}", f"{h['fp']}/{h['neg_n']}", f"{d['crisis_k']}/{d['crisis_n']}",
                     f"{p['faq_gen']['crisis']}/{p['faq_gen']['n']}", f"{p['counsel_gen']['crisis']}/{p['counsel_gen']['n']}"])
    write_table("gate_tradeoff", ["Crisis threshold", "Held-out crisis", "Held-out FP", "Dev crisis",
                                  "FAQ-Gen to crisis", "Counsel-Gen to crisis"], rows,
                "Effect of the classifier's crisis threshold, with everything else fixed. Crisis: crisis prompts sent "
                "to a blocking protocol; FP: venting and informational prompts sent to a blocking protocol; FAQ-Gen, "
                "Counsel-Gen to crisis: ordinary questions sent to the crisis protocol. ``Cap at elevated'': the "
                "classifier cannot block, only the lexicon can. Choosing a threshold from this table would need a new "
                "held-out set.",
                "tab:gate-tradeoff", align="lrrrrr", source="results/gate_tradeoff.json")
    for p in points:
        print(p["crisis_threshold"], p["heldout"]["crisis_k"], p["heldout"]["fp"], p["faq_gen"]["crisis"],
              p["counsel_gen"]["crisis"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
