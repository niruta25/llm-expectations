"""Calibration maths, hand-computed. This module must stay at 100% coverage:
a silent bug here produces confidently wrong quality numbers."""

from __future__ import annotations

import math

import pytest

from llmex import Calibration, LabelledScore, auroc, calibrate
from llmex.calibration import confidence_gap, precision_at_k, threshold_for_precision


def test_auroc_is_one_for_perfect_separation():
    # Errors score low, correct extractions score high.
    assert auroc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == pytest.approx(1.0)


def test_auroc_is_zero_for_perfect_inversion():
    assert auroc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == pytest.approx(0.0)


def test_auroc_is_half_when_all_scores_tie():
    # Average-rank tie handling, and no division by zero.
    assert auroc([0.5] * 4, [False, False, True, True]) == pytest.approx(0.5)


def test_auroc_handles_partial_ties_by_average_rank():
    # scores 0.1(err) 0.5(err) 0.5(ok) 0.9(ok)
    # ranks   1        2.5      2.5     4
    # U = (1 + 2.5) - 2*3/2 = 0.5 ; auroc = 1 - 0.5/4 = 0.875
    assert auroc([0.1, 0.5, 0.5, 0.9], [False, False, True, True]) == pytest.approx(0.875)


def test_auroc_is_undefined_with_only_one_class():
    assert math.isnan(auroc([0.1, 0.2], [True, True]))
    assert math.isnan(auroc([0.1, 0.2], [False, False]))


def test_precision_at_k_defaults_to_the_true_error_count():
    scores = [0.1, 0.2, 0.8, 0.9]
    correct = [False, False, True, True]
    assert precision_at_k(scores, correct) == pytest.approx(1.0)
    assert precision_at_k(scores, correct, k=4) == pytest.approx(0.5)


def test_precision_at_k_is_undefined_without_errors():
    assert math.isnan(precision_at_k([0.9], [True]))


def test_confidence_gap_is_scale_sensitive():
    assert confidence_gap([0.1, 0.9], [False, True]) == pytest.approx(0.8)
    assert math.isnan(confidence_gap([0.9], [True]))


def test_threshold_hits_target_precision_and_reports_what_it_costs():
    scores = [0.1, 0.2, 0.3, 0.8, 0.9]
    correct = [False, False, False, True, True]
    t, recall = threshold_for_precision(scores, correct, 1.0)
    assert t > 0.3
    assert recall == pytest.approx(1.0)


def test_threshold_gives_up_rather_than_lying_about_precision():
    # No threshold can flag these errors without dragging in a correct one.
    scores = [0.5, 0.5, 0.5]
    correct = [False, True, True]
    t, recall = threshold_for_precision(scores, correct, 0.9)
    assert (t, recall) == (0.0, 0.0)


def _labels(n_per_field: int, field: str = "vendor") -> list[LabelledScore]:
    out = []
    for i in range(n_per_field):
        out.append(LabelledScore(f"d{i}", field, 0.9, True))
        out.append(LabelledScore(f"d{i}", field, 0.1, False))
    return out


def test_per_field_thresholds_need_ten_labels():
    # Nine labels: no per-field threshold, because there is no defensible number.
    few = calibrate("c", _labels(4), "mock", "mock-1.0", "diverse_ensemble")
    assert "vendor" not in few.thresholds

    enough = calibrate("c", _labels(5), "mock", "mock-1.0", "diverse_ensemble")
    assert "vendor" in enough.thresholds


def test_threshold_for_falls_back_to_the_default():
    cal = Calibration("c", "h", "mock", "mock-1.0", "diverse_ensemble",
                      thresholds={"__default__": 0.5, "vendor": 0.8})
    assert cal.threshold_for("vendor") == 0.8
    assert cal.threshold_for("unknown_field") == 0.5
    assert cal.threshold_for(None) == 0.5


def test_fingerprint_changes_with_any_component():
    base = dict(id="c", gold_set_hash="h", provider_id="mock",
                model_version="mock-1.0", strategy_id="diverse_ensemble")
    a = Calibration(**base)
    assert a.fingerprint() == Calibration(**base).fingerprint()
    assert a.fingerprint() != Calibration(**{**base, "model_version": "mock-2.0"}).fingerprint()
    assert a.fingerprint() != Calibration(**{**base, "gold_set_hash": "h2"}).fingerprint()
    assert a.ref == f"calibration:c@{a.fingerprint()}"


def test_valid_for_rejects_any_drift():
    cal = calibrate("c", _labels(5), "mock", "mock-1.0", "diverse_ensemble")
    assert cal.valid_for("mock", "mock-1.0", "diverse_ensemble")
    assert not cal.valid_for("mock", "mock-2.0", "diverse_ensemble")
    assert not cal.valid_for("other", "mock-1.0", "diverse_ensemble")
    assert not cal.valid_for("mock", "mock-1.0", "single_judge")


def test_calibration_round_trips_through_dict():
    cal = calibrate("c", _labels(5), "mock", "mock-1.0", "diverse_ensemble")
    back = Calibration.from_dict(cal.as_dict())
    assert back.fingerprint() == cal.fingerprint()
    assert back.thresholds == cal.thresholds
    assert back.n_labels == cal.n_labels
