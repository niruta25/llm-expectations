"""Ranking aggregates, and the guard that stops one being computed on noise.

AUC is secondary here, and deliberately. The question a review budget poses is
"how many of the errors do I find in the first 500 rows I open", and that is
Error Recall@Budget, in ``triage/evaluate.py``. AUC summarises the ordering,
including the part nobody will ever review, and a model can win on it while
losing at every budget you would actually use.

What AUC is good for is Gate 2 — one number per ranker, comparable across
rankers, computed on the same target. That is what it does in this file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["RankingTarget", "auc", "degenerate_target"]


@dataclass(frozen=True, slots=True)
class RankingTarget:
    """What the rankers are being scored against, and whether it can be used.

    The target is "is this item actually wrong", which only human labels can
    answer. ``unrankable`` counts items a strategy could not score — they are
    excluded from the AUC, and how many errors they held is reported, because
    an excluded pile full of errors flatters whatever ranked the rest.
    """

    truth: Mapping[str, bool]
    source: str

    @property
    def positives(self) -> int:
        return sum(self.truth.values())

    @property
    def negatives(self) -> int:
        return len(self.truth) - self.positives


def degenerate_target(
    target: RankingTarget, *, min_items: int = 30, min_share: float = 0.05
) -> str | None:
    """Is the minority class big enough for a ranking claim to mean anything?

    An AUC computed while ranking three negatives looks fine and has a
    tight-looking interval. Returns the reason it cannot be computed, or
    ``None`` when it can.
    """
    total = len(target.truth)
    if total == 0:
        return "no items carry a human label, so there is no target to rank against"
    minority = min(target.positives, target.negatives)
    share = minority / total
    if minority < min_items or share < min_share:
        kind = "errors" if target.positives <= target.negatives else "correct items"
        return (
            f"only {minority} {kind} in {total} labelled items ({share:.1%}) — the floor is "
            f"{min_items} and {min_share:.0%}. Any ranking number off this is noise with a "
            "tight-looking interval around it."
        )
    return None


def auc(scores: Sequence[float], truth: Sequence[bool]) -> float | None:
    """Area under the ROC curve, by ranks, with ties split evenly.

    Ties are not an edge case here: an uncalibrated ranker hands out the same
    confidence to hundreds of items at once, and a tie-blind implementation
    would score that pile as if the ranker had ordered it. Average ranks give
    a tied pair the 0.5 it has earned, which is what makes a flat ranker come
    out at 0.5 instead of looking good by accident.
    """
    truth_array = np.asarray(truth, dtype=bool)
    score_array = np.asarray(scores, dtype=float)
    positives = int(truth_array.sum())
    negatives = int(len(truth_array) - positives)
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(score_array, kind="mergesort")
    ranks = np.empty(len(score_array), dtype=float)
    ranks[order] = np.arange(1, len(score_array) + 1, dtype=float)

    # Average the ranks within each run of equal scores.
    sorted_scores = score_array[order]
    start = 0
    for index in range(1, len(sorted_scores) + 1):
        if index == len(sorted_scores) or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index

    rank_sum = float(ranks[truth_array].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
