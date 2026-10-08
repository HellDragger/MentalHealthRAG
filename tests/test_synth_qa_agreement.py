"""scripts.synth_qa_agreement: label reading, agreement statistics and repair of re-saved sheets."""

import numpy as np
import pytest

import scripts.synth_qa_agreement as sa


def test_labels_accept_variants_and_typos_but_not_nonsense():
    assert [sa.label(v) for v in ("Y", "yes ", "yesy", "yesye", "n", "No", "")] == [1, 1, 1, 1, 0, 0, None]
    with pytest.raises(ValueError):
        sa.label("maybe")


def test_fleiss_and_ac1_hand_computed():
    m = np.array([[1, 1], [1, 1], [1, 0], [0, 0]])  # observed agreement 0.75, yes-rate 5/8
    assert sa.fleiss(m) == pytest.approx((0.75 - 34 / 64) / (1 - 34 / 64))
    assert sa.gwet_ac1(m) == pytest.approx((0.75 - 30 / 64) / (1 - 30 / 64))
    skewed = np.array([[1, 1, 1, 1]] * 19 + [[1, 1, 1, 0]])  # near-unanimous yes: kappa collapses, AC1 does not
    assert sa.fleiss(skewed) < 0 < 0.9 < sa.gwet_ac1(skewed)


def test_strict_majority_ties_fail():
    ids = ["a", "b", "c"]
    r = {"r1": {"a": 1, "b": 1, "c": 0}, "r2": {"a": 1, "b": 1, "c": 0}, "r3": {"a": 1, "b": 0, "c": 1},
         "r4": {"a": 0, "b": 0, "c": 0}}
    out = sa.agreement(r, ids, ["r1", "r2", "r3", "r4"])
    assert out["_pass_ids"] == ["a"] and out["ties"] == 1 and out["pass_strict_majority"] == 1


def test_mixed_encoding_line_is_repaired():
    raw = "ok,“quoted”\n".encode() + b"doesn\x92t,\xe2\x80\x9cx\xe2\x80\x9d\n"  # UTF-8 plus one Windows-1252 byte
    assert sa._decode(raw) == "ok,“quoted”\ndoesn’t,“x”\n"
