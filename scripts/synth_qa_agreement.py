"""Human verification of SynthQA: rater agreement, the verified subset, and retrieval on that subset.

    python -m scripts.synth_qa_agreement --import a.csv b.csv c.csv d.csv   # copy filled sheets in (author first)
    python -m scripts.synth_qa_agreement                                   # -> results/synth_qa_verification.json

Raters fill eval/data/synth_qa_verify.csv (two yes/no questions per item). --import keeps only the id, the two answers
and the notes, re-encoded as UTF-8, under eval/data/synth_qa_verify/rater<N>.csv. Rater names are not stored; the
order given is the rater number, and --author marks the rater who also wrote the system (default rater1).

Aggregation, fixed before any per-item analysis: an item passes a criterion when a strict majority of raters say yes
(3 of 4); a tie counts as a fail. An item is verified when it passes both criteria. Equal weight for every rater.

Agreement: raw pairwise agreement, Cohen's kappa per pair, Fleiss' kappa and Gwet's AC1. Kappa is reported with AC1
because kappa collapses when almost every answer is "yes" (Feinstein & Cicchetti 1990, J Clin Epidemiol 43(6):543;
Gwet 2008, Br J Math Stat Psychol 61(1):29).

Robustness: per-query retrieval scores in results/retrieval_main.json (same order as eval/data/synth_qa.jsonl) are
re-averaged over the verified subset, so no retrieval run is repeated.
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from eval.datasets import EVAL_DATA, load_jsonl
from eval.metrics import bootstrap_ci, holm, paired_bootstrap_p
from mhrag.config import get_settings

SHEET = EVAL_DATA / "synth_qa_verify.csv"
RATER_DIR = EVAL_DATA / "synth_qa_verify"
CRITERIA = {"answerable": "answerable_from_passage (y/n)", "natural": "natural_question (y/n)"}
METRICS = ("nDCG@10", "MRR@10", "R@10")


def _decode(raw: bytes) -> str:
    """UTF-8, with stray Windows-1252 bytes (a spreadsheet re-save) repaired line by line."""
    out = []
    for line in raw.splitlines(keepends=True):
        try:
            out.append(line.decode("utf-8"))
        except UnicodeDecodeError:
            out.append(_mixed(line))
    return "".join(out).lstrip("﻿")


def _mixed(line: bytes) -> str:
    """Decode valid UTF-8 sequences as UTF-8 and every other byte as Windows-1252."""
    s, i = [], 0
    while i < len(line):
        for n in (4, 3, 2, 1):
            try:
                s.append(line[i:i + n].decode("utf-8"))
                i += n
                break
            except UnicodeDecodeError:
                continue
        else:
            s.append(line[i:i + 1].decode("cp1252", errors="replace"))
            i += 1
    return "".join(s)


def import_sheets(paths: list[str]) -> None:
    with open(SHEET, encoding="utf-8") as f:
        ids = [r["id"] for r in csv.DictReader(f)]
    RATER_DIR.mkdir(parents=True, exist_ok=True)
    for n, p in enumerate(paths, 1):
        rows = list(csv.DictReader(io.StringIO(_decode(Path(p).read_bytes()))))
        got = [r["id"] for r in rows]
        if sorted(got) != sorted(ids):
            raise SystemExit(f"{p}: ids differ from {SHEET.name} (missing {set(ids) - set(got)}, extra {set(got) - set(ids)})")
        out = RATER_DIR / f"rater{n}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", *CRITERIA.values(), "notes"])
            for r in rows:
                w.writerow([r["id"], *(r[c].strip() for c in CRITERIA.values()), r.get("notes", "").strip()])
        print(f"{p} -> {out.relative_to(EVAL_DATA.parent.parent)}")


def label(v: str) -> int | None:
    """y/yes -> 1, n/no -> 0, blank -> None. Typos made only of the letters of one answer ("yesy") count as it."""
    v = v.strip().lower()
    if not v:
        return None
    if v in ("y", "yes") or set(v) <= set("yes") and v[0] == "y":
        return 1
    if v in ("n", "no") or set(v) <= set("no") and v[0] == "n":
        return 0
    raise ValueError(f"unreadable answer {v!r}")


def cohen(a: list[int], b: list[int]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def _pa_p1(m: np.ndarray) -> tuple[float, float]:
    """Observed pairwise agreement and overall yes-rate for an items x raters 0/1 matrix."""
    k = m.shape[1]
    s = m.sum(axis=1)
    pa = float(((s * (s - 1) + (k - s) * (k - s - 1)) / (k * (k - 1))).mean())
    return pa, float(m.mean())


def fleiss(m: np.ndarray) -> float:
    pa, p = _pa_p1(m)
    pe = p * p + (1 - p) * (1 - p)
    return (pa - pe) / (1 - pe) if pe < 1 else float("nan")


def gwet_ac1(m: np.ndarray) -> float:
    pa, p = _pa_p1(m)
    pe = 2 * p * (1 - p)
    return (pa - pe) / (1 - pe)


def item_ci(m: np.ndarray, fn, n_boot: int = 2000, seed: int = 0) -> list[float]:
    """Point estimate and 95% bootstrap CI over items."""
    rng = np.random.default_rng(seed)
    boots = [fn(m[rng.integers(0, len(m), len(m))]) for _ in range(n_boot)]
    boots = [b for b in boots if not math.isnan(b)]
    return [fn(m), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    p = k / n
    c = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    return [p, mid - c, mid + c]


def agreement(ratings: dict[str, dict[str, int | None]], ids: list[str], raters: list[str]) -> dict:
    full = [i for i in ids if all(ratings[r][i] is not None for r in raters)]
    m = np.array([[ratings[r][i] for r in raters] for i in full], dtype=int)
    k = len(raters)
    yes = m.sum(axis=1)
    pairs = {}
    for (x, a), (y, b) in itertools.combinations(enumerate(raters), 2):
        pairs[f"{a}-{b}"] = {"agreement": float((m[:, x] == m[:, y]).mean()), "cohen_kappa": cohen(list(m[:, x]), list(m[:, y]))}
    vs_rest = {}
    for x, r in enumerate(raters):
        rest = np.delete(m, x, axis=1).sum(axis=1)
        maj = rest * 2 > k - 1
        tie = rest * 2 == k - 1
        vs_rest[r] = float((m[:, x] == maj)[~tie].mean())
    passed = int((yes * 2 > k).sum())
    return {
        "n_items": len(full), "n_raters": k,
        "yes_rate": {r: float(m[:, x].mean()) for x, r in enumerate(raters)},
        "pairwise": pairs,
        "mean_pairwise_agreement": _pa_p1(m)[0],
        "fleiss_kappa": item_ci(m, fleiss),
        "gwet_ac1": item_ci(m, gwet_ac1),
        "yes_votes": {str(v): int(c) for v, c in sorted(Counter(yes.tolist()).items())},
        "ties": int((yes * 2 == k).sum()),
        "pass_strict_majority": passed,
        "pass_rate": wilson(passed, len(full)),
        "agreement_with_others_majority": vs_rest,
        "_pass_ids": [i for i, v in zip(full, yes, strict=True) if v * 2 > k],
    }


def retrieval_on_subset(subset: list[str], rated: list[str]) -> dict | None:
    s = get_settings()
    path = s.results_dir / "retrieval_main.json"
    if not path.exists():
        return None
    res = json.loads(path.read_text())
    qids = [q["id"] for q in load_jsonl("synth_qa.jsonl")]
    runs = [r for r in res["runs"] if r["dataset"] == "synth_qa"]
    if not runs or any(r["n_queries"] != len(qids) for r in runs):
        raise SystemExit("results/retrieval_main.json does not match eval/data/synth_qa.jsonl; rerun the retrieval eval")
    pos = {q: n for n, q in enumerate(qids)}
    idx = {"all": list(range(len(qids))), "rated": [pos[i] for i in rated], "verified": [pos[i] for i in subset]}
    base_cfg = res["config"].get("baseline", {"system": "dense", "embedder": "minilm"})
    base = next(r for r in runs if r["system"] == base_cfg["system"] and r["embedder"] == base_cfg["embedder"])
    rid = lambda r: f"{r['system']}|{r['embedder'] or '-'}|c{r['chunk_tokens']}"  # noqa: E731
    out: dict = {"baseline": rid(base), "n": {k: len(v) for k, v in idx.items()}, "runs": {}, "rank_agreement": {},
                 "vs_baseline_verified": {}}
    from scipy.stats import kendalltau

    for metric in METRICS:
        means = {}
        for r in runs:
            v = np.asarray(r["per_query"][metric], float)
            out["runs"].setdefault(rid(r), {})[metric] = {k: dict(zip(("mean", "lo", "hi"), bootstrap_ci(v[i]), strict=True))
                                                         for k, i in idx.items()}
            means[rid(r)] = (float(v[idx["all"]].mean()), float(v[idx["verified"]].mean()))
        tau = kendalltau([a for a, _ in means.values()], [b for _, b in means.values()])
        best_all = max(means, key=lambda k: means[k][0])
        best_ver = max(means, key=lambda k: means[k][1])
        out["rank_agreement"][metric] = {"kendall_tau": float(tau.statistic), "p": float(tau.pvalue),
                                         "best_all": best_all, "best_verified": best_ver}
        b = np.asarray(base["per_query"][metric], float)[idx["verified"]]
        raw, delta = {}, {}
        for r in runs:
            if r is base:
                continue
            a = np.asarray(r["per_query"][metric], float)[idx["verified"]]
            raw[rid(r)], delta[rid(r)] = paired_bootstrap_p(a, b), float(a.mean() - b.mean())
        out["vs_baseline_verified"][metric] = {"delta": delta, "p_bootstrap": raw, "p_bootstrap_holm": holm(raw)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--import", dest="imports", nargs="+", metavar="CSV", help="filled sheets, author's first")
    ap.add_argument("--author", default="rater1", help="rater who also built the system (sensitivity analysis)")
    args = ap.parse_args(argv)
    if args.imports:
        import_sheets(args.imports)

    sheets = sorted(RATER_DIR.glob("rater*.csv"), key=lambda p: int(p.stem[5:]))
    if not sheets:
        raise SystemExit("No rater sheets. TODO(human): python -m scripts.synth_qa_agreement --import <filled csvs>")
    with open(SHEET, encoding="utf-8") as f:
        ids = [r["id"] for r in csv.DictReader(f)]
    raters = [p.stem for p in sheets]
    ratings = {c: {} for c in CRITERIA}
    fixed: list[dict] = []
    notes = {}
    for p in sheets:
        with open(p, encoding="utf-8") as f:
            rows = {r["id"]: r for r in csv.DictReader(f)}
        notes[p.stem] = sum(1 for r in rows.values() if r["notes"])
        for c, col in CRITERIA.items():
            ratings[c][p.stem] = {}
            for i in ids:
                raw = rows[i][col]
                ratings[c][p.stem][i] = v = label(raw)
                if v is not None and raw.strip().lower() not in ("y", "yes", "n", "no"):
                    fixed.append({"rater": p.stem, "id": i, "criterion": c, "raw": raw, "read_as": "yes" if v else "no"})

    out: dict = {
        "items": len(ids), "raters": raters, "author": args.author,
        "rule": "pass = strict majority yes (ties fail); verified = passes both criteria; raters weighted equally",
        "labels_normalised": fixed, "raters_with_notes": notes, "criteria": {}, "without_author": {},
    }
    passing = {}
    for c in CRITERIA:
        a = agreement(ratings[c], ids, raters)
        passing[c] = set(a.pop("_pass_ids"))
        out["criteria"][c] = a
        if args.author in raters and len(raters) > 2:
            b = agreement(ratings[c], ids, [r for r in raters if r != args.author])
            b.pop("_pass_ids")
            out["without_author"][c] = {k: b[k] for k in ("n_items", "fleiss_kappa", "gwet_ac1", "pass_strict_majority",
                                                          "ties", "pass_rate")}
    verified = [i for i in ids if all(i in passing[c] for c in CRITERIA)]
    out["verified"] = {"n": len(verified), "rate": wilson(len(verified), len(ids)), "ids": verified}
    out["retrieval_robustness"] = retrieval_on_subset(verified, ids)

    s = get_settings()
    path = s.results_dir / "synth_qa_verification.json"
    path.write_text(json.dumps(out, indent=2))
    for c, a in out["criteria"].items():
        print(f"{c:10s} pass {a['pass_strict_majority']}/{a['n_items']} ties {a['ties']}  "
              f"Fleiss {a['fleiss_kappa'][0]:.3f}  AC1 {a['gwet_ac1'][0]:.3f}")
    print(f"verified (both) {len(verified)}/{len(ids)}  -> {path}")


if __name__ == "__main__":
    main()
