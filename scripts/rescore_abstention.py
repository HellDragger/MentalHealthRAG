"""Recompute the abstention flag of saved answers after a fix to its phrase matching, and re-aggregate.

    python -m scripts.rescore_abstention            # local and gpu experiments
    python -m scripts.rescore_abstention --check    # only verify that re-aggregation reproduces the saved summary

Fix (D43): the phrase list matched only ASCII apostrophes, so answers with typographic ones ("I don’t have
information …", common in Gemma's output) were counted as not declining. Only `abstained` is recomputed from the saved
answer text; every other metric is kept, and the summary and significance tests are rebuilt with the same code and
seeds as the original run.
"""

from __future__ import annotations

import argparse
import gzip
import json

from eval.generation_eval import aggregate
from eval.judges import abstained
from mhrag.config import get_settings

R = get_settings().results_dir


def _scored(exp: str):
    d = R / "generation" / exp
    for name, op in (("scored.jsonl", open), ("scored.jsonl.gz", gzip.open)):
        if (d / name).exists():
            return d / name, op
    return None, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--experiments", nargs="+", default=["local", "gpu"])
    args = ap.parse_args(argv)
    for exp in args.experiments:
        path, op = _scored(exp)
        res_path = R / f"generation_{exp}.json"
        if path is None or not res_path.exists():
            print(f"{exp}: nothing to rescore")
            continue
        with op(path, "rt", encoding="utf-8") as f:
            answers = [json.loads(line) for line in f]
        res = json.loads(res_path.read_text())
        key = lambda r: (r["model"], r["profile"], r["dataset"])  # noqa: E731
        if args.check:
            new = {key(r): r for r in aggregate(answers, res["models"], res["config"])["summary"]}
            old = {key(r): r for r in res["summary"]}
            diff = [k for k in old if json.dumps(old[k], sort_keys=True) != json.dumps(new.get(k), sort_keys=True)]
            print(f"{exp}: {len(old)} summary rows, {len(diff)} differ" + (f", e.g. {diff[:3]}" if diff else ""))
            continue
        changed = []
        for r in answers:
            m = r.get("metrics") or {}
            if "abstained" in m:
                v = abstained(r.get("answer") or "")
                if v != m["abstained"]:
                    changed.append(key(r))
                m["abstained"] = v
        agg = aggregate(answers, res["models"], res["config"])
        before = {key(r): (r.get("abstained") or {}).get("mean") for r in res["summary"]}
        res.update(summary=agg["summary"], significance=agg["significance"], verbosity_bias=agg["verbosity_bias"])
        res["abstention_rescored"] = {"fix": "typographic apostrophes", "answers_changed": len(changed)}
        res_path.write_text(json.dumps(res, indent=1))
        with op(path, "wt", encoding="utf-8") as f:
            for r in answers:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{exp}: {len(changed)} answers changed")
        for r in agg["summary"]:
            b, a = before.get(key(r)), (r.get("abstained") or {}).get("mean")
            if b is not None and a is not None and abs(a - b) > 1e-9:
                print(f"   {key(r)}: {b:.3f} -> {a:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
