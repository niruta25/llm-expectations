"""Fitting a curve, proving it helps, and refusing it when it cannot."""

from __future__ import annotations

import numpy as np
import pytest

from llm_expectations.calibration import IdentityCalibrator
from llm_expectations.calibration.base import StaleCalibration, fingerprint
from llm_expectations.calibration.platt import (
    MIN_ROWS,
    FieldCalibrator,
    PlattCalibrator,
    fit_sigmoid,
)
from llm_expectations.run import check_calibration_age
from llm_expectations.types import Status, Verdict


def informative(n=1000, slope=0.15, intercept=0.02, seed=0):
    """A score that is informative but badly scaled — Platt's whole use case."""
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    y = rng.random(n) < (slope * x + intercept)
    return x.tolist(), y.tolist()


class TestFitting:
    def test_a_fit_recovers_the_real_error_rate(self):
        x, y = informative()
        calibrator = PlattCalibrator()
        report = calibrator.fit_from_rows(x, y)
        assert report.fitted
        # The raw score claimed 0.9; the truth is nearer 0.16.
        assert calibrator.score(0.9) < 0.35
        assert calibrator.score(0.1) < calibrator.score(0.9)

    def test_the_fitted_curve_is_well_calibrated_out_of_fold(self):
        x, y = informative(n=2000)
        report = PlattCalibrator().fit_from_rows(x, y)
        assert report.ece < 0.05
        assert report.brier < report.brier_base_rate

    def test_quality_is_measured_out_of_fold_and_says_so(self):
        # Fitting on a hundred rows then reporting how well the fit scores
        # those same rows is in-sample performance wearing a lab coat.
        x, y = informative()
        report = PlattCalibrator().fit_from_rows(x, y)
        assert any("out of fold" in note for note in report.notes)

    def test_in_sample_scoring_would_have_flattered_it(self):
        # A score with no signal at all. Out of fold it cannot beat the base
        # rate; the point of cross-fitting is that this shows up.
        rng = np.random.default_rng(1)
        x = rng.random(600).tolist()
        y = (rng.random(600) < 0.3).tolist()
        report = PlattCalibrator().fit_from_rows(x, y)
        assert report.brier >= report.brier_base_rate * 0.98
        assert any("adds nothing" in note for note in report.notes)

    def test_too_few_rows_fits_nothing_and_stays_uncalibrated(self):
        calibrator = PlattCalibrator()
        report = calibrator.fit_from_rows([0.5] * 50, [True] * 25 + [False] * 25)
        assert not report.fitted
        assert not calibrator.is_calibrated
        assert str(MIN_ROWS) in report.notes[0]

    def test_one_class_has_no_boundary_to_fit(self):
        report = PlattCalibrator().fit_from_rows(list(np.random.default_rng(2).random(300)),
                                                  [True] * 300)
        assert not report.fitted
        assert "same side" in report.notes[0]

    def test_an_unfitted_calibrator_passes_the_score_through_untouched(self):
        calibrator = PlattCalibrator()
        assert calibrator.score(0.42) == 0.42
        assert not calibrator.is_calibrated

    def test_the_sigmoid_is_monotone_increasing(self):
        x, y = informative()
        a, b = fit_sigmoid(np.asarray(x), np.asarray(y))
        calibrator = PlattCalibrator(a=a, b=b)
        calibrator._fitted = True
        values = [calibrator.score(v / 20) for v in range(21)]
        assert values == sorted(values)


class TestMonotonicity:
    """One global curve cannot reorder anything, and the report must say so."""

    def test_a_single_curve_leaves_the_order_exactly_as_it_was(self):
        x, y = informative()
        fitted = PlattCalibrator()
        fitted.fit_from_rows(x, y)
        before = sorted(range(len(x)), key=lambda i: -x[i])
        after = sorted(range(len(x)), key=lambda i: -fitted.score(x[i]))
        assert before == after

    def test_one_fitted_field_does_not_count_as_reordering(self):
        x, y = informative()
        one = PlattCalibrator()
        one.fit_from_rows(x, y)
        assert not FieldCalibrator(per_field={"jtbd": one}).reorders

    def test_two_fitted_fields_can_reorder(self):
        x, y = informative()
        first, second = PlattCalibrator(), PlattCalibrator()
        first.fit_from_rows(x, y)
        second.fit_from_rows(x, [not v for v in y])
        assert FieldCalibrator(per_field={"jtbd": first, "outcome": second}).reorders

    def test_a_field_with_no_curve_falls_back_to_the_shared_one(self):
        x, y = informative()
        shared = PlattCalibrator()
        shared.fit_from_rows(x, y)
        calibrator = FieldCalibrator(fallback=shared)
        verdict = Verdict("a", "c", "s-1", "anything", Status.FAIL, 0.9, "r")
        assert calibrator.error_probability(verdict) == pytest.approx(shared.score(0.9))

    def test_with_nothing_fitted_it_returns_the_raw_score(self):
        calibrator = FieldCalibrator()
        verdict = Verdict("a", "c", "s-1", "jtbd", Status.FAIL, 0.9, "r")
        assert calibrator.error_probability(verdict) == pytest.approx(0.9)
        assert not calibrator.is_calibrated


class TestDirection:
    """The probability is of the output being wrong, not the verdict."""

    def test_a_confident_rejection_scores_high_not_low(self):
        x, y = informative()
        fitted = PlattCalibrator()
        fitted.fit_from_rows(x, y)
        rejected = Verdict("a", "c", "s-1", "jtbd", Status.FAIL, 0.95, "wrong")
        approved = Verdict("a", "c", "s-2", "jtbd", Status.PASS, 0.95, "fine")
        # Reading it as "P(the verdict is wrong)" would put the judge's
        # clearest catches at the bottom of the review queue.
        assert fitted.error_probability(rejected) > fitted.error_probability(approved)

    def test_an_unscored_verdict_has_no_probability(self):
        fitted = PlattCalibrator()
        fitted.fit_from_rows(*informative())
        assert fitted.error_probability(
            Verdict("a", "c", "s-1", "jtbd", Status.UNSCORED, None, "")
        ) is None


class TestStaleness:
    def test_the_same_configuration_passes(self):
        stamp = fingerprint(
            judge_id="judge-a", model="m", prompt_hash="p8",
            check_id="label_correct", taxonomy_ref="jtbd@v4",
        )
        check_calibration_age(stamp, stamp, where="calibration.json")

    def test_a_prompt_change_is_refused(self):
        kwargs = dict(judge_id="judge-a", model="m", check_id="label_correct",
                      taxonomy_ref="jtbd@v4")
        with pytest.raises(StaleCalibration, match="fitted against a different"):
            check_calibration_age(
                fingerprint(prompt_hash="p7", **kwargs),
                fingerprint(prompt_hash="p8", **kwargs),
                where="calibration.json",
            )

    def test_a_taxonomy_version_change_is_refused(self):
        kwargs = dict(judge_id="judge-a", model="m", check_id="label_correct",
                      prompt_hash="p8")
        with pytest.raises(StaleCalibration):
            check_calibration_age(
                fingerprint(taxonomy_ref="jtbd@v4", **kwargs),
                fingerprint(taxonomy_ref="jtbd@v5", **kwargs),
                where="calibration.json",
            )

    def test_never_having_seen_one_is_not_staleness(self):
        check_calibration_age(None, "anything", where="calibration.json")


class TestRoundTrip:
    def test_a_fitted_calibrator_survives_a_write_and_a_read(self):
        x, y = informative()
        fitted = PlattCalibrator()
        fitted.fit_from_rows(x, y, fingerprint="abc")
        restored = PlattCalibrator.from_dict(fitted.to_dict())
        assert restored.is_calibrated
        assert restored.score(0.7) == pytest.approx(fitted.score(0.7))
        assert restored.fingerprint == "abc"

    def test_the_per_field_wrapper_round_trips_too(self):
        x, y = informative()
        one = PlattCalibrator()
        one.fit_from_rows(x, y)
        original = FieldCalibrator(per_field={"jtbd": one}, fingerprint="abc")
        restored = FieldCalibrator.from_dict(original.to_dict())
        assert restored.is_calibrated
        verdict = Verdict("a", "c", "s-1", "jtbd", Status.FAIL, 0.8, "r")
        assert restored.error_probability(verdict) == pytest.approx(
            original.error_probability(verdict)
        )


def test_the_identity_calibrator_is_still_not_a_calibration():
    assert not IdentityCalibrator().is_calibrated
