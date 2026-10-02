"""Intervals, AUC, and the resampling unit that keeps them honest."""

from __future__ import annotations

import numpy as np
import pytest

from llm_expectations.metrics.ranking import RankingTarget, auc, degenerate_target
from llm_expectations.metrics.stats import Estimate, bootstrap_ci


class TestAUC:
    def test_a_perfect_ranking_scores_one(self):
        assert auc([3, 2, 1, 0], [True, True, False, False]) == 1.0

    def test_an_inverted_ranking_scores_zero(self):
        assert auc([0, 1, 2, 3], [True, True, False, False]) == 0.0

    def test_a_ranker_that_ties_everything_scores_exactly_a_half(self):
        # Not an edge case: an uncalibrated ranker hands the same confidence
        # to hundreds of items at once. A tie-blind implementation would
        # score that pile as if it had been ordered.
        assert auc([1, 1, 1, 1], [True, True, False, False]) == 0.5

    def test_partial_ties_are_split_evenly(self):
        # Pairs (positive, negative): (2,1) wins, (2,0) wins, (1,1) ties for
        # a half, (1,0) wins. 3.5 of 4.
        assert auc([2, 1, 1, 0], [True, False, True, False]) == pytest.approx(0.875)

    def test_one_class_has_no_curve(self):
        assert auc([1, 2, 3], [True, True, True]) is None
        assert auc([1, 2, 3], [False, False, False]) is None

    def test_noise_lands_near_a_half(self):
        rng = np.random.default_rng(0)
        value = auc(rng.random(4000).tolist(), (rng.random(4000) < 0.3).tolist())
        assert 0.45 < value < 0.55


class TestDegenerateTarget:
    def _target(self, errors, total):
        truth = {f"s-{i}": i < errors for i in range(total)}
        return RankingTarget(truth=truth, source="test")

    def test_a_healthy_target_passes(self):
        assert degenerate_target(self._target(100, 500)) is None

    def test_too_few_errors_is_refused_by_count(self):
        reason = degenerate_target(self._target(10, 500))
        assert "10 errors" in reason and "floor is 30" in reason

    def test_too_few_errors_is_refused_by_share(self):
        # 40 errors clears the count floor and fails the 5% share floor.
        assert "4.0%" in degenerate_target(self._target(40, 1000))

    def test_an_empty_target_says_there_is_nothing_to_rank_against(self):
        assert "no items carry a human label" in degenerate_target(self._target(0, 0))

    def test_the_minority_is_whichever_side_is_smaller(self):
        # Almost everything wrong is just as degenerate as almost nothing.
        assert degenerate_target(self._target(495, 500)) is not None


class TestBootstrap:
    def test_the_interval_brackets_the_point_estimate(self):
        by_item = {f"s-{i}": [float(i % 4 != 0)] for i in range(400)}
        estimate = bootstrap_ci(by_item, lambda a: float(a.mean()), resamples=400)
        assert estimate.low < estimate.value < estimate.high

    def test_more_items_give_a_tighter_interval(self):
        def spread(n):
            by_item = {f"s-{i}": [float(i % 4 != 0)] for i in range(n)}
            return bootstrap_ci(by_item, lambda a: float(a.mean()), resamples=400).width

        assert spread(2000) < spread(100)

    def test_resampling_is_over_items_not_over_values(self):
        # Ten fields of one item move together. Treating them as ten
        # independent draws fakes independence and shrinks the interval.
        grouped = {f"s-{i}": [float(i % 2)] * 10 for i in range(40)}
        flat = {f"v-{i}": [float((i // 10) % 2)] for i in range(400)}
        by_item = bootstrap_ci(grouped, lambda a: float(a.mean()), resamples=400)
        by_value = bootstrap_ci(flat, lambda a: float(a.mean()), resamples=400)
        assert by_item.width > by_value.width

    def test_the_same_seed_gives_the_same_interval(self):
        by_item = {f"s-{i}": [float(i % 3)] for i in range(100)}
        first = bootstrap_ci(by_item, lambda a: float(a.mean()), resamples=200, seed=7)
        second = bootstrap_ci(by_item, lambda a: float(a.mean()), resamples=200, seed=7)
        assert (first.low, first.high) == (second.low, second.high)

    def test_a_sample_under_the_floor_is_reported_with_a_plain_warning(self):
        # A wide interval is not a weak result, it is no result — and hiding
        # it would lose the only information the sample carries.
        by_item = {f"s-{i}": [float(i % 2)] for i in range(28)}
        estimate = bootstrap_ci(by_item, lambda a: float(a.mean()), resamples=200, floor=200)
        assert estimate.under_floor
        assert estimate.value is not None
        assert "cannot support a conclusion" in estimate.note

    def test_nothing_to_resample_produces_no_value(self):
        assert bootstrap_ci({}, lambda a: float(a.mean())).value is None

    def test_one_item_gets_a_point_but_no_interval(self):
        estimate = bootstrap_ci({"s-1": [1.0]}, lambda a: float(a.mean()))
        assert estimate.value == 1.0
        assert estimate.low is None
        assert "too few items" in estimate.note


class TestEstimateComparison:
    def test_beating_means_the_interval_clears_the_other_point(self):
        judge = Estimate(value=0.84, low=0.78, high=0.90)
        baseline = Estimate(value=0.50, low=0.44, high=0.56)
        assert judge.beats(baseline)

    def test_a_higher_number_with_an_overlapping_interval_has_not_won(self):
        judge = Estimate(value=0.58, low=0.47, high=0.69)
        baseline = Estimate(value=0.52, low=0.46, high=0.58)
        assert not judge.beats(baseline)

    def test_an_estimate_with_no_interval_cannot_claim_to_beat_anything(self):
        assert not Estimate(value=0.99).beats(Estimate(value=0.5))
