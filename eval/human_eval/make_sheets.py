"""Build blinded, randomised rating sheets for the human evaluation.

    python -m eval.human_eval.make_sheets --exp local --raters 3 --n 60
    python -m eval.human_eval.make_sheets --exp gpu --systems qwen2.5-7b-instruct:full mistral-7b-instruct-v0.3:naive ...

Outputs (eval/human_eval/out/<exp>/):
- sheet_rater<k>.csv : one row per (question, response); system identity hidden; row order shuffled per rater
- key.csv            : item_id -> model, profile (keep private until ratings are collected)
"""

from __future__ import annotations

import argparse
import csv
import json
import random

from mhrag.config import PROJECT_ROOT, get_settings

CRITERIA = ["accuracy", "groundedness", "helpfulness", "empathy", "safety"]
YESNO = ["comfortable_for_vulnerable_reader", "oversteps"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="generation experiment name (results/generation/<exp>/scored.jsonl)")
    ap.add_argument("--raters", type=int, default=3)
    ap.add_argument("--n", type=int, default=60, help="number of questions")
    ap.add_argument("--systems", nargs="*", help="model:profile pairs to include (default: all)")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(argv)

    s = get_settings()
    src = s.results_dir / "generation" / args.exp / "scored.jsonl"
    if not src.exists() and src.with_name("scored.jsonl.gz").exists():
        src = src.with_name("scored.jsonl.gz")  # large score files are stored compressed
    if src.suffix == ".gz":
        import gzip

        with gzip.open(src, "rt", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
    else:
        rows = [json.loads(line) for line in src.read_text().splitlines()]
    rows = [r for r in rows if r.get("answer") and r.get("generated") is not False]
    if args.systems:
        keep = {tuple(x.split(":")) for x in args.systems}
        rows = [r for r in rows if (r["model"], r["profile"]) in keep]
    systems = sorted({(r["model"], r["profile"]) for r in rows})
    by_q: dict[tuple, dict] = {}
    for r in rows:
        by_q.setdefault((r["dataset"], r["id"]), {})[(r["model"], r["profile"])] = r
    complete = [k for k, v in by_q.items() if len(v) == len(systems) and k[0] != "oos_questions"]
    rng = random.Random(args.seed)
    rng.shuffle(complete)
    chosen = complete[: args.n]

    out = PROJECT_ROOT / "eval" / "human_eval" / "out" / args.exp
    out.mkdir(parents=True, exist_ok=True)
    items, key = [], []
    for qi, k in enumerate(chosen):
        for si, sysk in enumerate(systems):
            r = by_q[k][sysk]
            item_id = f"q{qi:03d}-{rng.randrange(16**6):06x}"
            sources = "\n".join(f"[{x['n']}] {x['title']} — {x.get('section') or ''}: {x.get('snippet', '')}"
                                for x in r.get("sources", []))
            items.append({"item_id": item_id, "question": r["query"], "response": r["answer"], "sources": sources})
            key.append({"item_id": item_id, "question_key": f"{k[0]}:{k[1]}", "model": sysk[0], "profile": sysk[1],
                        "system_index": si})
    with open(out / "key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    for rater in range(1, args.raters + 1):
        order = items[:]
        random.Random(args.seed * 100 + rater).shuffle(order)
        with open(out / f"sheet_rater{rater}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "question", "sources", "response", *CRITERIA, *YESNO, "comments"])
            for it in order:
                w.writerow([it["item_id"], it["question"], it["sources"], it["response"], *[""] * (len(CRITERIA) + 3)])
    print(f"{len(chosen)} questions x {len(systems)} systems -> {out}")


if __name__ == "__main__":
    main()
