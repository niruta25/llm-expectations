"""Calibrating a judge against human labels.

Kappa is the number that decides whether a judge is measuring anything, so the
arithmetic is pinned against hand-computed cases including the degenerate ones.
"""

from __future__ import annotations

import math

import pytest

from llmex import (
    LabelledDecision,
    agreement_rate,
    calibrate_judge,
    cohens_kappa,
)

LABELS = ["password_reset", "billing_dispute"]


def decisions(n=12, judge_is_useful=True):
    """n items, half of them genuinely mislabelled."""
    out = []
    for i in range(n):
        wrong = i % 2 == 0
        if judge_is_useful:
            score = 0.15 if wrong else 0.92  # the judge sees what the human sees
        else:
            score = 0.5  # the judge has no idea
        out.append(
            LabelledDecision(
                doc_id=f"d{i}",
                field_name="jtbd_label",
                score=score,
                predicted_label=LABELS[i % 2],
                gold_label=LABELS[1] if wrong else LABELS[i % 2],
            )
        )
    return out


# -- agreement arithmetic ---------------------------------------------------


def test_agreement_rate_is_the_raw_share():
    assert agreement_rate(["a", "a", "b"], ["a", "b", "b"]) == pytest.approx(2 / 3)
    assert math.isnan(agreement_rate([], []))


def test_kappa_is_one_for_perfect_agreement():
    assert cohens_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == pytest.approx(1.0)


def test_kappa_is_zero_at_chance():
    # Both annotators split 50/50 and agree on exactly half. Chance explains it.
    a = ["x", "x", "y", "y"]
    b = ["x", "y", "x", "y"]
    assert agreement_rate(a, b) == pytest.approx(0.5)
    assert cohens_kappa(a, b) == pytest.approx(0.0)


def test_kappa_goes_negative_below_chance():
    assert cohens_kappa(["a", "b"], ["b", "a"]) < 0


def test_kappa_punishes_agreement_a_skewed_taxonomy_hands_you_for_free():
    """The reason raw agreement is never reported alone.

    Two annotators who both answer the majority label almost always agree
    almost always, while carrying no information at all."""
    a = ["common"] * 19 + ["rare"]
    b = ["common"] * 20
    assert agreement_rate(a, b) == pytest.approx(0.95)  # looks excellent
    assert cohens_kappa(a, b) == pytest.approx(0.0)  # and means nothing


def test_kappa_handles_the_degenerate_single_label_case_without_dividing_by_zero():
    assert cohens_kappa(["a", "a"], ["a", "a"]) == pytest.approx(1.0)
    assert cohens_kappa(["a", "a"], ["b", "b"]) == pytest.approx(0.0)
    assert math.isnan(cohens_kappa([], []))


# -- calibrate_judge --------------------------------------------------------


def test_it_reports_agreement_alongside_the_ranking_metrics():
    cal = calibrate_judge("j1", decisions(), "mock_judge", "mock-judge-1.0", "label_judge")
    assert cal.metrics["auroc"] == pytest.approx(1.0)
    assert cal.metrics["cohens_kappa"] == pytest.approx(1.0)
    assert cal.metrics["judge_human_agreement"] == pytest.approx(1.0)


def test_a_judge_that_knows_nothing_scores_zero_kappa():
    cal = calibrate_judge(
        "j1", decisions(judge_is_useful=False), "mock_judge", "mock-judge-1.0", "label_judge"
    )
    assert cal.metrics["cohens_kappa"] <= 0.0


def test_per_label_thresholds_need_ten_examples_of_that_label():
    # 12 decisions, split evenly: 6 of each label, both under the minimum.
    sparse = calibrate_judge("j1", decisions(12), "p", "m", "s")
    assert sparse.metrics["per_label"] == {}

    # 24 decisions gives 12 of each, which clears it.
    dense = calibrate_judge("j1", decisions(24), "p", "m", "s")
    assert set(dense.metrics["per_label"]) == set(LABELS)
    assert dense.metrics["per_label"][LABELS[0]]["n"] == 12


def test_it_still_produces_a_normal_fingerprinted_calibration():
    cal = calibrate_judge("j1", decisions(), "mock_judge", "mock-judge-1.0", "label_judge")
    assert cal.valid_for("mock_judge", "mock-judge-1.0", "label_judge")
    assert not cal.valid_for("mock_judge", "mock-judge-2.0", "label_judge")
    assert cal.ref.startswith("calibration:j1@")
    assert cal.threshold_for("jtbd_label") is not None


def test_a_decision_knows_whether_it_was_correct():
    d = LabelledDecision("d1", "f", 0.9, "a", "a")
    assert d.is_correct
    assert d.to_labelled_score().is_correct
    assert not LabelledDecision("d1", "f", 0.9, "a", "b").is_correct
