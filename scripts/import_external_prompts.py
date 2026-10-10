"""Convert an externally written prompt sheet (eval/external/prompts_template.csv, filled) to the red-team format.

    python -m scripts.import_external_prompts eval/external/prompts_filled.csv   # -> eval/data/safety_prompts_external.jsonl

Writer ids are kept as given (use initials or numbers, not names). Empty rows are skipped; unknown categories stop it.
"""

from __future__ import annotations

import csv
import json
import sys

from mhrag.config import PROJECT_ROOT

EXPECTED = {"explicit_crisis": "crisis", "implicit_crisis": "crisis", "harmful_request": "harmful_request",
            "third_party": "third_party", "benign_venting": "support", "help_seeking": "answer",
            "informational": "answer", "negation": "support"}
# eval_safety reports per category; help-seeking and negation messages are scored like informational/venting ones
CATEGORY_AS = {"help_seeking": "informational", "negation": "benign_venting"}


def main(argv=None) -> int:
    src = (argv or sys.argv[1:])[0]
    out = PROJECT_ROOT / "eval" / "data" / "safety_prompts_external.jsonl"
    rows = []
    with open(src, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            text, cat = (r.get("text") or "").strip(), (r.get("category") or "").strip()
            if not text:
                continue
            if cat not in EXPECTED:
                raise SystemExit(f"unknown category {cat!r} for: {text[:60]}")
            rows.append({"id": f"ext{len(rows) + 1:03d}", "category": CATEGORY_AS.get(cat, cat), "subcategory": cat,
                         "expected": EXPECTED[cat], "text": text, "writer": (r.get("writer_id") or "").strip()})
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} prompts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
