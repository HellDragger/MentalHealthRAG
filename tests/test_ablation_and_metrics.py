"""Component-ablation context matching, abstention detection with typographic apostrophes, and Wilson intervals."""

from types import SimpleNamespace

import pytest

from eval.judges import abstained
from mhrag import prompts
from mhrag.pipeline import ABLATION
from scripts.eval_safety import ratio, wilson


def _hit(text, title="T"):
    return SimpleNamespace(chunk=SimpleNamespace(text=text, title=title, section=title))


def count(s):
    return len(s) // 4


def test_context_cut_to_characters_without_scraps():
    hits = [_hit("alpha " * 50), _hit("beta " * 50)]
    ctx, used = prompts.build_context(hits, 10_000, count, max_chars=100)
    assert len(used) == 1 and len(ctx.split("\n", 1)[1]) <= 100        # one passage, cut at a word boundary
    ctx, used = prompts.build_context(hits, 10_000, count, max_chars=len("alpha " * 50) + 10)
    assert len(used) == 1                                                # a 10-character remainder is not added
    ctx, used = prompts.build_context(hits, 10_000, count)
    assert len(used) == 2                                                # unchanged without max_chars


def test_ablation_profile_names():
    assert ABLATION.match("abl_full_naive").groups() == ("full", "naive", None)
    assert ABLATION.match("abl_full_full_matched").group(3) == "_matched"
    assert ABLATION.match("full") is None and ABLATION.match("naive") is None


def test_abstention_accepts_typographic_apostrophes():
    assert abstained("I don’t have information about that in my sources.")
    assert abstained("I don't have information about that in my sources.")
    assert not abstained("The capital of Australia is Canberra.")


def test_wilson_interval():
    lo, hi = wilson(30, 36)
    assert lo == pytest.approx(0.681, abs=1e-3) and hi == pytest.approx(0.921, abs=1e-3)
    assert ratio(3, 30) == "3/30" and ratio(30, 36, ci=True).startswith("30/36 {\\scriptsize [0.68, 0.92]}")


def test_significance_accepts_boolean_metrics():
    from eval.generation_eval import _significance

    def arm(vals):
        return [{"id": f"q{i}", "metrics": {"abstained": v, "faithfulness": 0.5}} for i, v in enumerate(vals)]
    groups = {("m", "a", "oos"): arm([True] * 8 + [False] * 2), ("m", "b", "oos"): arm([False] * 8 + [True] * 2)}
    out = _significance(groups, [("a", "b")])
    assert out["m|oos|abstained"]["delta"]["a>b"] == pytest.approx(0.6)
