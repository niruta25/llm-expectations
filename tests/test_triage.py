"""Ranking, and the three numbers that are never silently interconverted."""

from __future__ import annotations

import pytest

from llm_expectations.calibration import IdentityCalibrator
from llm_expectations.triage import (
    TriageContext,
    assert_no_labels,
    build_risk_rows,
    resolve_strategy,
)
from llm_expectations.types import Label, Output, Status, Verdict


def verdict(item_id, status, confidence, field="jtbd"):
    return Verdict("judge-a", "label_correct", item_id, field, status, confidence, "because")


def context(**verdicts):
    return TriageContext(
        verdicts={k: tuple(v) for k, v in verdicts.items()},
        outputs={k: Output(k, {}) for k in verdicts},
        fields=("jtbd",),
        calibrator=IdentityCalibrator(),
    )


class TestIdentityCalibrator:
    def test_it_is_not_a_calibration_and_says_so(self):
        calibrator = IdentityCalibrator()
        assert calibrator.is_calibrated is False
        report = calibrator.fit([], [])
        assert report.fitted is False
        assert any("not a probability of error" in note for note in report.notes)

    def test_an_approval_reads_as_one_minus_confidence(self):
        # The design's formula, unchanged, in the case it was written for.
        assert IdentityCalibrator().error_probability(
            verdict("s-1", Status.PASS, 0.9)
        ) == pytest.approx(0.1)

    def test_a_rejection_reads_as_the_confidence_itself(self):
        # A judge that said "incorrect, 0.9" is reporting a 90% chance of
        # error, not a 10% one. Ranking it at 0.1 would put the judge's
        # clearest rejections at the bottom of the queue.
        assert IdentityCalibrator().error_probability(
            verdict("s-1", Status.FAIL, 0.9)
        ) == pytest.approx(0.9)

    @pytest.mark.parametrize(
        "status, confidence", [(Status.UNSCORED, 0.5), (Status.PASS, None)]
    )
    def test_nothing_to_read_produces_none_not_a_stand_in(self, status, confidence):
        assert IdentityCalibrator().error_probability(verdict("s-1", status, confidence)) is None


class TestRanking:
    def test_rejections_outrank_approvals(self):
        ctx = context(
            **{
                "s-1": [verdict("s-1", Status.PASS, 0.95)],
                "s-2": [verdict("s-2", Status.FAIL, 0.8)],
            }
        )
        rows = build_risk_rows(ctx, resolve_strategy("auto", calibrated=False).rank(ctx), "raw")
        assert [row.item_id for row in rows] == ["s-2", "s-1"]

    def test_an_item_is_ranked_by_its_worst_field(self):
        # An item with a correct label and a made-up summary is not a usable
        # item, so the field a reviewer needs to see decides where it sits.
        ctx = context(
            **{
                "s-1": [
                    verdict("s-1", Status.PASS, 0.99, "jtbd"),
                    verdict("s-1", Status.FAIL, 0.85, "outcome"),
                ]
            }
        )
        scores = resolve_strategy("raw_confidence", calibrated=False).rank(ctx)
        assert scores["s-1"] == pytest.approx(0.85)

    def test_an_unjudged_item_gets_a_row_with_no_score(self):
        ctx = context(**{"s-1": [verdict("s-1", Status.UNSCORED, None)]})
        rows = build_risk_rows(ctx, {"s-1": None}, "raw_confidence")
        assert rows[0].triage_score is None
        assert rows[0].item_id == "s-1"

    def test_unjudged_items_sort_after_every_ranked_one(self):
        ctx = context(
            **{
                "s-1": [verdict("s-1", Status.UNSCORED, None)],
                "s-2": [verdict("s-2", Status.PASS, 0.99)],
            }
        )
        rows = build_risk_rows(ctx, resolve_strategy("auto", calibrated=False).rank(ctx), "raw")
        assert [row.item_id for row in rows] == ["s-2", "s-1"]
        assert rows[-1].triage_score is None

    def test_every_row_is_stamped_uncalibrated(self):
        ctx = context(**{"s-1": [verdict("s-1", Status.PASS, 0.9)]})
        row = build_risk_rows(ctx, {"s-1": 0.1}, "raw_confidence")[0]
        assert row.calibrated is False
        assert row.calibration_id is None
        # The calibrated probability is withheld, not filled in from the
        # uncalibrated number wearing its name.
        assert row.calibrated_error_probability is None
        assert row.raw_confidence_mean == pytest.approx(0.9)


class TestStrategyResolution:
    def test_auto_resolves_to_the_honest_option_when_nothing_is_fitted(self):
        assert resolve_strategy("auto", calibrated=False).id == "raw_confidence"

    def test_a_strategy_that_has_not_landed_yet_says_when_it_will(self):
        with pytest.raises(ValueError, match="M4, with Gate 2"):
            resolve_strategy("random", calibrated=False)
        with pytest.raises(ValueError, match="M5b"):
            resolve_strategy("auto", calibrated=True)


class TestLabelLeak:
    """The one runtime assertion behind the structural guarantee."""

    def test_a_clean_context_passes(self):
        assert_no_labels(context(**{"s-1": [verdict("s-1", Status.PASS, 0.9)]}))

    def test_a_label_smuggled_through_extras_is_caught(self):
        ctx = TriageContext(
            verdicts={},
            outputs={},
            fields=(),
            calibrator=IdentityCalibrator(),
            extras={"gold": [Label("s-1", "jtbd", "billing.payment_failed", "ann-1")]},
        )
        with pytest.raises(AssertionError, match="worthless in production"):
            assert_no_labels(ctx)

    def test_the_strategy_itself_runs_the_check(self):
        ctx = TriageContext(
            verdicts={},
            outputs={},
            fields=(),
            calibrator=IdentityCalibrator(),
            extras={"gold": Label("s-1", "jtbd", "x", "ann-1")},
        )
        with pytest.raises(AssertionError):
            resolve_strategy("raw_confidence", calibrated=False).rank(ctx)

    def test_the_context_type_has_no_field_for_labels(self):
        import dataclasses

        assert "labels" not in {f.name for f in dataclasses.fields(TriageContext)}
