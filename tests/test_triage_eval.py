"""Error Recall@Budget: the question a review budget actually poses."""

from __future__ import annotations

import pytest

from llm_expectations.triage.evaluate import compare, error_recall_at_budget

BUDGETS = (0.01, 0.05, 0.10, 0.20)


def corpus(n=1000, errors=100):
    return {f"s-{i}": i < errors for i in range(n)}


class TestRecallAtBudget:
    def test_a_perfect_ranking_finds_everything_by_the_error_rate(self):
        truth = corpus()
        scores = {f"s-{i}": 1.0 - i / 1000 for i in range(1000)}
        result = error_recall_at_budget(scores, truth, BUDGETS)
        assert result.at(0.01).error_recall == pytest.approx(0.10)
        assert result.at(0.10).error_recall == pytest.approx(1.0)
        assert result.at(0.10).wasted == pytest.approx(0.0)

    def test_an_inverted_ranking_finds_nothing_early(self):
        truth = corpus()
        scores = {f"s-{i}": i / 1000 for i in range(1000)}
        assert error_recall_at_budget(scores, truth, BUDGETS).at(0.05).errors_found == 0

    def test_wasted_review_is_the_other_half_of_the_operating_point(self):
        truth = corpus()
        scores = {f"s-{i}": 1.0 - i / 1000 for i in range(1000)}
        point = error_recall_at_budget(scores, truth, (0.20,)).at(0.20)
        # 200 reviewed, 100 of them errors.
        assert point.precision == pytest.approx(0.5)
        assert point.wasted == pytest.approx(0.5)

    def test_the_budget_is_a_share_of_the_labelled_corpus(self):
        truth = corpus(n=500, errors=50)
        scores = {f"s-{i}": 1.0 - i / 500 for i in range(500)}
        assert error_recall_at_budget(scores, truth, (0.10,)).at(0.10).n_reviewed == 50


class TestTies:
    """A ranker that ties everything has not ordered anything."""

    def test_an_all_tied_ranker_gets_exactly_its_budget_share(self):
        truth = corpus()
        scores = {f"s-{i}": 0.5 for i in range(1000)}
        result = error_recall_at_budget(scores, truth, BUDGETS)
        for budget in BUDGETS:
            assert result.at(budget).error_recall == pytest.approx(budget, abs=0.001)

    def test_item_id_order_cannot_flatter_a_tied_ranker(self):
        # Tie-breaking on id would hand a flat ranker whatever recall the
        # ids happen to produce. Splitting the block proportionally does not.
        errors_first = {f"s-{i}": i < 100 for i in range(1000)}
        errors_last = {f"s-{i}": i >= 900 for i in range(1000)}
        flat = {f"s-{i}": 0.5 for i in range(1000)}
        a = error_recall_at_budget(flat, errors_first, (0.10,)).at(0.10)
        b = error_recall_at_budget(flat, errors_last, (0.10,)).at(0.10)
        assert a.error_recall == pytest.approx(b.error_recall)

    def test_a_tied_block_straddling_the_cutoff_is_split_proportionally(self):
        # Ten items, all tied, five of them errors. A 50% budget should find
        # 2.5 errors in expectation, not 0 or 5.
        truth = {f"s-{i}": i < 5 for i in range(10)}
        scores = {f"s-{i}": 0.5 for i in range(10)}
        assert error_recall_at_budget(scores, truth, (0.5,)).at(0.5).errors_found == 2.5


class TestUnranked:
    def test_items_a_strategy_cannot_score_are_never_reviewed(self):
        truth = corpus(n=100, errors=20)
        scores = {f"s-{i}": (None if i < 20 else 0.5) for i in range(100)}
        result = error_recall_at_budget(scores, truth, (1.0,))
        assert result.unranked == 20
        assert result.unranked_errors == 20
        # Even a full-corpus budget cannot find them.
        assert result.at(1.0).errors_found == 0

    def test_the_unranked_pile_is_counted_because_it_flatters_the_rest(self):
        truth = corpus(n=200, errors=40)
        scores = {f"s-{i}": (None if i < 40 else 1.0 - i / 200) for i in range(200)}
        assert error_recall_at_budget(scores, truth, (0.1,)).unranked_errors == 40


class TestIntervals:
    def test_a_large_corpus_gets_an_interval(self):
        truth = corpus(n=1000, errors=200)
        scores = {f"s-{i}": 1.0 - i / 1000 for i in range(1000)}
        point = error_recall_at_budget(
            scores, truth, (0.10,), resamples=200, min_n=200
        ).at(0.10)
        assert point.ci_low is not None and point.ci_high is not None
        assert point.ci_low <= point.error_recall <= point.ci_high

    def test_a_corpus_under_the_floor_gets_none(self):
        truth = corpus(n=50, errors=10)
        scores = {f"s-{i}": 1.0 - i / 50 for i in range(50)}
        point = error_recall_at_budget(
            scores, truth, (0.10,), resamples=200, min_n=200
        ).at(0.10)
        assert point.ci_low is None


class TestCompare:
    def test_every_strategy_is_scored_on_the_same_target(self):
        truth = corpus()
        ranked = {
            "calibrated_risk": {f"s-{i}": 1.0 - i / 1000 for i in range(1000)},
            "random": {f"s-{i}": (i * 7919 % 1000) / 1000 for i in range(1000)},
        }
        results = {r.strategy: r for r in compare(ranked, truth, BUDGETS, baselines=["random"])}
        assert results["calibrated_risk"].at(0.10).error_recall > results["random"].at(
            0.10
        ).error_recall
        assert results["random"].is_baseline
        assert not results["calibrated_risk"].is_baseline

    def test_a_target_with_no_errors_yields_no_recall(self):
        truth = {f"s-{i}": False for i in range(100)}
        scores = {f"s-{i}": 0.5 for i in range(100)}
        assert error_recall_at_budget(scores, truth, (0.1,)).at(0.1).error_recall is None


class TestFractionalCounts:
    def test_a_split_tie_reports_the_fraction_it_found(self):
        from llm_expectations.report import operating_point_table

        payload = {
            "target_items": 100,
            "target_errors": 10,
            "budgets": [0.1],
            "strategies": [
                {
                    "strategy": "raw_confidence",
                    "baseline": False,
                    "ranked": 100,
                    "unranked": 0,
                    "unranked_errors": 0,
                    "points": [
                        {
                            "budget": 0.1,
                            "n_reviewed": 10,
                            "errors_found": 0.4,
                            "error_recall": 0.04,
                            "precision": 0.04,
                            "wasted": 0.96,
                            "ci_low": None,
                            "ci_high": None,
                        }
                    ],
                }
            ],
        }
        printed = "\n".join(operating_point_table(payload))
        # Rounding 0.4 to "0" beside a recall of 4% reads as a contradiction.
        assert "0.4" in printed
