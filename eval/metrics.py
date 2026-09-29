"""Retrieval / classification metrics, bootstrap CIs and paired significance tests."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

# ------------------------------------------------------------------------------ retrieval


def recall_at_k(ranked_rel: Sequence[bool], k: int, n_relevant: int) -> float:
    """Fraction of relevant items found in the top k (binary 'hit' when n_relevant == 1)."""
    if n_relevant <= 0:
        return 0.0
    return min(sum(ranked_rel[:k]), n_relevant) / n_relevant


def hit_at_k(ranked_rel: Sequence[bool], k: int) -> float:
    return float(any(ranked_rel[:k]))


def mrr_at_k(ranked_rel: Sequence[bool], k: int = 10) -> float:
    for i, r in enumerate(ranked_rel[:k]):
        if r:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked_rel: Sequence[bool], k: int, n_relevant: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(ranked_rel[:k]) if r)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(n_relevant, k)))
    return dcg / ideal if ideal > 0 else 0.0


# ------------------------------------------------------------------------------ bootstrap


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
                 stat: Callable = np.mean) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = np.array([stat(v[i]) for i in idx])
    return float(stat(v)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def bootstrap_metric(y_true, y_score, fn: Callable, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """CI for a metric of (y_true, y_score) by resampling examples."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    rng = np.random.default_rng(seed)
    point = fn(y_true, y_score)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(set(y_true[i])) < 2:
            continue
        vals.append(fn(y_true[i], y_score[i]))
    return float(point), float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_bootstrap_p(a: Sequence[float], b: Sequence[float], n_boot: int = 10000, seed: int = 0) -> float:
    """Two-sided paired bootstrap p-value for mean(a) - mean(b) = 0 (Koehn-style, centred)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    obs = d.mean()
    if np.allclose(d, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1) - obs  # centred under H0
    return float((np.abs(boots) >= abs(obs)).mean())


def wilcoxon_p(a: Sequence[float], b: Sequence[float]) -> float:
    from scipy.stats import wilcoxon

    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(a, b, zero_method="wilcox").pvalue)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvals.items(), key=lambda x: x[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


# ------------------------------------------------------------------------------ classification


def binary_report(y_true, y_score, threshold: float) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, float)
    pred = (y_score >= threshold).astype(int)
    out = {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "flag_rate": float(pred.mean()),
    }
    if len(set(y_true)) == 2:
        out["auroc"] = float(roc_auc_score(y_true, y_score))
        out["auprc"] = float(average_precision_score(y_true, y_score))
    return out


def threshold_for_recall(y_true, y_score, target: float) -> float:
    """Largest threshold whose recall on (y_true, y_score) is >= target."""
    y_true = np.asarray(y_true).astype(int)
    pos = np.sort(np.asarray(y_score, float)[y_true == 1])
    if len(pos) == 0:
        return 0.5
    k = int(math.floor((1 - target) * len(pos)))  # we may miss at most k positives
    return float(pos[k])


def threshold_for_precision(y_true, y_score, target: float) -> float:
    """Smallest threshold whose precision is >= target (falls back to the max score)."""
    from sklearn.metrics import precision_recall_curve

    prec, _, thr = precision_recall_curve(np.asarray(y_true).astype(int), np.asarray(y_score, float))
    ok = np.where(prec[:-1] >= target)[0]
    return float(thr[ok[0]]) if len(ok) else float(np.max(y_score))
