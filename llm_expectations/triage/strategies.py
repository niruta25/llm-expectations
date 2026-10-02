"""The ranking strategies. M1 ships the one that needs no labels.

``raw_confidence`` is the default when nothing has been calibrated, and it is
stamped uncalibrated everywhere it appears. The four baselines it has to beat —
random, output length, majority label, panel disagreement — arrive with Gate 2
at M4, and the table that settles which one wins arrives at M5b.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .base import TriageContext, TriageStrategy, assert_no_labels

__all__ = ["STRATEGIES", "RawConfidenceStrategy", "resolve_strategy"]


@dataclass(frozen=True, slots=True)
class RawConfidenceStrategy:
    """Rank by what the judge said, read against the verdict's own direction.

    An item's score is the **worst** of its fields. An item with a correct
    label and a made-up summary is not a usable item, so the field a reviewer
    needs to see decides where the item sits — the same reason item grain is
    always reported next to field grain.
    """

    id: str = "raw_confidence"
    requires: frozenset[str] = frozenset({"verdicts"})

    def rank(self, ctx: TriageContext) -> Mapping[str, float | None]:
        assert_no_labels(ctx)
        scores: dict[str, float | None] = {}
        for item_id in ctx.items():
            probabilities = [
                p
                for p in (
                    ctx.calibrator.error_probability(v) for v in ctx.verdicts.get(item_id, ())
                )
                if p is not None
            ]
            scores[item_id] = max(probabilities) if probabilities else None
        return scores


STRATEGIES: Mapping[str, TriageStrategy] = {
    strategy.id: strategy for strategy in (RawConfidenceStrategy(),)
}


def resolve_strategy(requested: str, *, calibrated: bool) -> TriageStrategy:
    """Pick the strategy, resolving ``auto`` against what is actually fitted.

    ``auto`` means ``calibrated_risk`` when a calibration exists and
    ``raw_confidence`` when one does not. Until M5b there is no fitted
    calibrator, so ``auto`` resolves to the honest option and the report says
    which rather than implying the other.
    """
    if requested == "auto":
        requested = "calibrated_risk" if calibrated else "raw_confidence"
    strategy = STRATEGIES.get(requested)
    if strategy is None:
        arriving = {
            "calibrated_risk": "M5b, with the Platt fit",
            "panel_disagreement": "M4, as a scored baseline",
            "random": "M4, with Gate 2",
            "output_length": "M4, with Gate 2",
            "majority_label": "M4, with Gate 2",
        }.get(requested)
        detail = f" It arrives at {arriving}." if arriving else ""
        raise ValueError(
            f"triage strategy {requested!r} is not available yet.{detail} "
            f"Available now: {', '.join(sorted(STRATEGIES))}."
        )
    return strategy
