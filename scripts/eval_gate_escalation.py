"""How often the risk gate escalates ordinary questions, and which component (lexicon or classifier) triggered it.

The red-team sets measure the gate on prompts written for it. This runs the deployed gate on the generation question
sets instead: FAQ-Gen (informational questions from a public FAQ), the out-of-scope set and Counsel-Gen (help-seeking
posts from CounselChat). None of these is labelled for risk, so the output is an escalation rate, not an error rate.

    python -m scripts.eval_gate_escalation            # -> results/gate_escalation.json
"""

from __future__ import annotations

import json
from collections import Counter

from eval.datasets import load_jsonl
from mhrag.config import get_settings
from mhrag.runtime import build_gate
from mhrag.safety.gate import GATE_VERSION

DATASETS = ("faq_gen", "oos_questions", "counsel_gen")


def source(reasons: list[str]) -> str:
    kinds = {r.split(":")[0].split(">=")[0] for r in reasons}
    if not kinds:
        return "none"
    return "+".join(sorted(k for k in kinds if k in ("lexicon", "classifier", "history")) or kinds)


def main(argv=None) -> int:
    s = get_settings()
    gate = build_gate(s)
    out = {"gate_version": GATE_VERSION, "classifier": s.safety.classifier,
           "crisis_threshold": gate.crisis_threshold, "elevated_threshold": gate.elevated_threshold, "datasets": {}}
    for name in DATASETS:
        rows = load_jsonl(f"{name}.jsonl")
        labels, by_source = Counter(), Counter()
        for r in rows:
            res = gate.assess(r.get("query") or r.get("question") or r.get("text"))
            labels[res.label] += 1
            if res.label != "none":
                by_source[f"{res.label}|{source(res.reasons)}"] += 1
        out["datasets"][name] = {"n": len(rows), "labels": dict(labels), "escalations_by_source": dict(by_source)}
        print(f"{name:14s} n={len(rows):4d} {dict(labels)} {dict(by_source)}")
    s.results_dir.mkdir(parents=True, exist_ok=True)
    (s.results_dir / "gate_escalation.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
