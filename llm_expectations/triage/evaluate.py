"""Error Recall@Budget — the number a review budget actually poses.

    Error Recall@K = true errors found in the top K reviewed ÷ all true errors

AUC summarises the whole ordering, including the tail nobody will ever open.
This asks the question a person with five hundred review-hours actually has,
and a ranker can win on AUC while losing at every budget you would use.

Needs labels: you cannot count true errors without them. In mode 0 the
ranking is still produced and simply has not been validated, which the report
says rather than implying otherwise.

**Ties are resolved by expectation, not by sort order.** A ranker that gives
every item the same score has not ordered anything, and tie-breaking on item
id would hand it whatever recall the ids happen to produce — flattering or
damning it at random. Splitting a tied block proportionally gives a flat
ranker exactly its budget share, which is the right answer and the same one
random gets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["BudgetPoint", "StrategyEvaluation", "compare", "error_recall_at_budget"]


@dataclass(frozen=True, slots=True)
class BudgetPoint:
    """One strategy at one review budget."""

    budget: float
    n_reviewed: int
    errors_found: float
    total_errors: int
    ci_low: float | None = None
    ci_high: float | None = None

    @property
    def error_recall(self) -> float | None:
        return self.errors_found / self.total_errors if self.total_errors else None

    @property
    def precision(self) -> float | None:
        return self.errors_found / self.n_reviewed if self.n_reviewed else None

    @property
    def wasted(self) -> float | None:
        """The share of reviewed rows that turned out to be fine.

        The other half of the operating point. A budget that finds 71% of
        errors by making a reviewer open five thousand items, 29% of which
        were never wrong, is a real cost and belongs next to the recall.
        """
        precision = self.precision
        return None if precision is None else 1.0 - precision


@dataclass(frozen=True, slots=True)
class StrategyEvaluation:
    """One strategy across every budget, and what it could not rank."""

    strategy: str
    points: tuple[BudgetPoint, ...]
    ranked: int
    unranked: int
    unranked_errors: int
    total_errors: int
    is_baseline: bool = False

    def at(self, budget: float) -> BudgetPoint | None:
        return next((p for p in self.points if p.budget == budget), None)


def _found_in_top(
    scores: Sequence[float], errors: Sequence[bool], k: int
) -> float:
    """Expected true errors in the top k, splitting tied blocks fairly."""
    if k <= 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    ranked_scores = np.asarray(scores, dtype=float)[order]
    ranked_errors = np.asarray(errors, dtype=bool)[order]
    if k >= len(order):
        return float(ranked_errors.sum())

    cutoff = ranked_scores[k - 1]
    above = ranked_scores > cutoff
    at = ranked_scores == cutoff
    taken_above = int(above.sum())
    slots = k - taken_above
    tied = int(at.sum())
    found = float(ranked_errors[above].sum())
    if tied and slots > 0:
        found += float(ranked_errors[at].sum()) * (slots / tied)
    return found


def error_recall_at_budget(
    scores: Mapping[str, float | None],
    truth: Mapping[str, bool],
    budgets: Sequence[float],
    *,
    strategy: str = "",
    is_baseline: bool = False,
    resamples: int = 0,
    seed: int = 0,
    min_n: int = 200,
) -> StrategyEvaluation:
    """Score one strategy at every budget, against the human answers.

    The budget is a share of the **labelled** corpus, so every strategy
    reviews the same number of rows and the comparison is like for like.

    Items the strategy could not score are not reviewed at any budget. How
    many errors sit in that pile is reported: an unranked heap full of
    errors flatters whatever ranked the rest.
    """
    ranked = {i: s for i, s in scores.items() if s is not None and i in truth}
    missing = [i for i in truth if i not in ranked]
    total_errors = sum(truth.values())

    items = list(ranked)
    values = [ranked[i] for i in items]
    outcomes = [truth[i] for i in items]
    n = len(truth)

    points: list[BudgetPoint] = []
    for budget in budgets:
        k = max(1, round(budget * n)) if n else 0
        k = min(k, len(items))
        found = _found_in_top(values, outcomes, k)
        low = high = None
        if resamples and len(items) >= 2 and n >= min_n:
            low, high = _interval(
                values, outcomes, budget, n, resamples=resamples, seed=seed
            )
        points.append(
            BudgetPoint(
                budget=budget,
                n_reviewed=k,
                errors_found=found,
                total_errors=total_errors,
                ci_low=low,
                ci_high=high,
            )
        )

    return StrategyEvaluation(
        strategy=strategy,
        points=tuple(points),
        ranked=len(items),
        unranked=len(missing),
        unranked_errors=sum(1 for i in missing if truth[i]),
        total_errors=total_errors,
        is_baseline=is_baseline,
    )


def _interval(
    values: Sequence[float],
    outcomes: Sequence[bool],
    budget: float,
    n: int,
    *,
    resamples: int,
    seed: int,
) -> tuple[float | None, float | None]:
    """Bootstrap the recall by resampling items — one item, one score."""
    rng = np.random.default_rng(seed)
    size = len(values)
    value_array = np.asarray(values, dtype=float)
    outcome_array = np.asarray(outcomes, dtype=bool)
    draws = []
    for _ in range(resamples):
        picked = rng.integers(0, size, size=size)
        drawn_outcomes = outcome_array[picked]
        total = int(drawn_outcomes.sum())
        if not total:
            continue
        k = min(max(1, round(budget * n)), size)
        found = _found_in_top(value_array[picked].tolist(), drawn_outcomes.tolist(), k)
        draws.append(found / total)
    if not draws:
        return None, None
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def compare(
    ranked: Mapping[str, Mapping[str, float | None]],
    truth: Mapping[str, bool],
    budgets: Sequence[float],
    *,
    baselines: Sequence[str] = (),
    resamples: int = 0,
    seed: int = 0,
    min_n: int = 200,
) -> list[StrategyEvaluation]:
    """Every strategy at every budget, on one target.

    This is the table that settles whether a judge is worth paying for, and
    whether panel disagreement beats it on *your* corpus rather than on the
    corpora somebody else published.
    """
    return [
        error_recall_at_budget(
            scores,
            truth,
            budgets,
            strategy=name,
            is_baseline=name in set(baselines),
            resamples=resamples,
            seed=seed,
            min_n=min_n,
        )
        for name, scores in ranked.items()
    ]
