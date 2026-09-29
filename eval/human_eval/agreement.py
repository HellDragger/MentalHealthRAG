"""Inter-rater agreement and system comparison for completed human-evaluation sheets.

    python -m eval.human_eval.agreement --exp local

Reads eval/human_eval/out/<exp>/sheet_rater*.csv (filled in) + key.csv; writes results/human_eval_<exp>.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

from eval.human_eval.make_sheets import CRITERIA
from eval.metrics import bootstrap_ci, holm, wilcoxon_p
from mhrag.config import PROJECT_ROOT, get_settings


def krippendorff_alpha(matrix: np.ndarray, level: str = "ordinal") -> float:
    """matrix: raters x items, np.nan for missing."""
    import krippendorff

    return float(krippendorff.alpha(reliability_data=matrix, level_of_measurement=level))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    args = ap.parse_args(argv)
    d = PROJECT_ROOT / "eval" / "human_eval" / "out" / args.exp
    with open(d / "key.csv", encoding="utf-8") as f:
        key = {r["item_id"]: r for r in csv.DictReader(f)}
    sheets = sorted(d.glob("sheet_rater*.csv"))
    ratings: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)  # rater -> item -> crit -> score
    for sh in sheets:
        with open(sh, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                vals = {}
                for c in CRITERIA:
                    try:
                        vals[c] = float(r[c])
                    except (TypeError, ValueError):
                        pass
                if vals:
                    ratings[sh.stem][r["item_id"]] = vals
    if not ratings:
        raise SystemExit("No completed ratings found. TODO(human): collect ratings first.")
    raters = sorted(ratings)
    items = sorted(key)
    out: dict = {"raters": len(raters), "items": len(items), "alpha": {}, "systems": {}, "tests": {}}
    for c in CRITERIA:
        m = np.array([[ratings[r].get(i, {}).get(c, np.nan) for i in items] for r in raters], dtype=float)
        out["alpha"][c] = krippendorff_alpha(m)
    # per-item mean over raters
    item_mean = {c: {i: np.nanmean([ratings[r].get(i, {}).get(c, np.nan) for r in raters]) for i in items} for c in CRITERIA}
    systems = sorted({(v["model"], v["profile"]) for v in key.values()})
    for sysk in systems:
        ids = [i for i in items if (key[i]["model"], key[i]["profile"]) == sysk]
        out["systems"][":".join(sysk)] = {c: bootstrap_ci([item_mean[c][i] for i in ids if not np.isnan(item_mean[c][i])])
                                          for c in CRITERIA}
    # paired tests between systems on the same questions
    for c in CRITERIA:
        raw = {}
        for a in systems:
            for b in systems:
                if a >= b:
                    continue
                qa = {key[i]["question_key"]: item_mean[c][i] for i in items if (key[i]["model"], key[i]["profile"]) == a}
                qb = {key[i]["question_key"]: item_mean[c][i] for i in items if (key[i]["model"], key[i]["profile"]) == b}
                qs = [q for q in qa if q in qb and not np.isnan(qa[q]) and not np.isnan(qb[q])]
                if len(qs) >= 5:
                    raw[f"{':'.join(a)} vs {':'.join(b)}"] = wilcoxon_p([qa[q] for q in qs], [qb[q] for q in qs])
        out["tests"][c] = {"p_wilcoxon": raw, "p_holm": holm(raw)}
    s = get_settings()
    (s.results_dir / f"human_eval_{args.exp}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["alpha"], indent=1))


if __name__ == "__main__":
    main()
