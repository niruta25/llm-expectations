"""The ranking strategies: one default, and four that exist to be beaten.

``raw_confidence`` is the default when nothing has been calibrated, and it is
stamped uncalibrated everywhere it appears. The four baselines it has to beat —
random, output length, majority label, panel disagreement — are scored beside
it by Gate 2, and Error Recall@Budget settles which one wins at the budget you
actually review at.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from ..types import Status
from .base import TriageContext, TriageStrategy, assert_no_labels

__all__ = [
    "BASELINES",
    "STRATEGIES",
    "CalibratedRiskStrategy",
    "MajorityLabelStrategy",
    "OutputLengthStrategy",
    "PanelDisagreementStrategy",
    "RandomStrategy",
    "RawConfidenceStrategy",
    "resolve_strategy",
]


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


@dataclass(frozen=True, slots=True)
class RandomStrategy:
    """Shuffle. The floor every other strategy has to clear.

    Seeded, so a run is reproducible and a baseline cannot get lucky twice in
    a row without anyone noticing.
    """

    id: str = "random"
    requires: frozenset[str] = frozenset()

    def rank(self, ctx: TriageContext) -> Mapping[str, float | None]:
        assert_no_labels(ctx)
        rng = random.Random(ctx.seed)  # noqa: S311 — a baseline, not cryptography
        return {item_id: rng.random() for item_id in ctx.items()}


@dataclass(frozen=True, slots=True)
class OutputLengthStrategy:
    """Rank by how much the model wrote.

    A baseline because length is the classic confound: if long outputs are
    the wrong ones on your corpus, a judge that has quietly learned to flag
    long outputs will post a good AUC while knowing nothing. When this one
    wins, the judge is not the thing that is working.
    """

    id: str = "output_length"
    requires: frozenset[str] = frozenset({"outputs"})

    def rank(self, ctx: TriageContext) -> Mapping[str, float | None]:
        assert_no_labels(ctx)
        scores: dict[str, float | None] = {}
        for item_id in ctx.items():
            output = ctx.outputs.get(item_id)
            if output is None:
                scores[item_id] = None
                continue
            scores[item_id] = float(
                sum(len(str(v).split()) for v in output.values.values() if v is not None)
            )
        return scores


@dataclass(frozen=True, slots=True)
class MajorityLabelStrategy:
    """Flag everything that is *not* the most common label.

    The trivial classifier. On a skewed taxonomy it is often surprisingly
    hard to beat, which is exactly why it is here: an accuracy or an AUC that
    cannot clear "assume the biggest label is right" has not earned anything.
    """

    id: str = "majority_label"
    requires: frozenset[str] = frozenset({"outputs"})

    def rank(self, ctx: TriageContext) -> Mapping[str, float | None]:
        assert_no_labels(ctx)
        counts: Counter[tuple[str, str]] = Counter()
        for item_id in ctx.items():
            output = ctx.outputs.get(item_id)
            if output is None:
                continue
            for field in ctx.fields:
                value = output.get(field)
                if isinstance(value, str):
                    counts[(field, value)] += 1

        biggest: dict[str, str] = {}
        for (field, value), n in counts.most_common():
            biggest.setdefault(field, value)
            del n

        scores: dict[str, float | None] = {}
        for item_id in ctx.items():
            output = ctx.outputs.get(item_id)
            if output is None:
                scores[item_id] = None
                continue
            off_majority = sum(
                1
                for field in ctx.fields
                if field in biggest and output.get(field) != biggest[field]
            )
            scores[item_id] = float(off_majority)
        return scores


@dataclass(frozen=True, slots=True)
class PanelDisagreementStrategy:
    """Rank by how much the panel split on the item.

    **A candidate and a baseline at once.** Published results — on invoice
    extraction, and on contested label spaces — put panel disagreement near
    chance and below any single judge, which is why it is not the default.

    But those are other corpora. Asserting the finding holds for yours
    without measuring it would be the same unearned confidence this library
    exists to prevent, so it ships as a selectable strategy *and* a scored
    baseline, and the comparison table settles it on your data. If it wins
    here, use it.
    """

    id: str = "panel_disagreement"
    requires: frozenset[str] = frozenset({"verdicts", "panel"})

    def rank(self, ctx: TriageContext) -> Mapping[str, float | None]:
        assert_no_labels(ctx)
        members = set(ctx.panel_members)
        scores: dict[str, float | None] = {}
        for item_id in ctx.items():
            votes = [
                v.status is Status.PASS
                for v in ctx.verdicts.get(item_id, ())
                if v.judge_id in members and v.status is not Status.UNSCORED
            ]
            if len(votes) < 2:
                scores[item_id] = None
                continue
            approved = sum(votes)
            # Peak disagreement at an even split, zero at unanimity.
            scores[item_id] = 1.0 - abs(2 * approved / len(votes) - 1.0)
        return scores


@dataclass(frozen=True, slots=True)
class CalibratedRiskStrategy:
    """Rank by fitted probability of error. The default once one exists.

    Mechanically close to ``raw_confidence`` — both take the worst field of
    an item — and that closeness is the honest story. A single Platt curve is
    monotone, so it cannot reorder anything; it is the per-field curves that
    can, because a raw 0.7 on one field is not a raw 0.7 on another.

    What the calibration always buys is meaning. ``raw_confidence`` gives you
    an order and nothing else; this gives you numbers you can set a review
    threshold against.
    """

    id: str = "calibrated_risk"
    requires: frozenset[str] = frozenset({"verdicts", "calibration"})

    def rank(self, ctx: TriageContext) -> Mapping[str, float | None]:
        assert_no_labels(ctx)
        scores: dict[str, float | None] = {}
        for item_id in ctx.items():
            values = [
                p
                for p in (
                    ctx.calibrator.error_probability(v) for v in ctx.verdicts.get(item_id, ())
                )
                if p is not None
            ]
            scores[item_id] = max(values) if values else None
        return scores


STRATEGIES: Mapping[str, TriageStrategy] = {
    strategy.id: strategy
    for strategy in (
        RawConfidenceStrategy(),
        CalibratedRiskStrategy(),
        RandomStrategy(),
        OutputLengthStrategy(),
        MajorityLabelStrategy(),
        PanelDisagreementStrategy(),
    )
}

#: The ones that exist to be beaten. Every ranking metric is reported beside
#: all of them, always — a number without its baseline is not a result.
BASELINES: tuple[str, ...] = (
    "random",
    "output_length",
    "majority_label",
    "panel_disagreement",
)


def resolve_strategy(requested: str, *, calibrated: bool) -> TriageStrategy:
    """Pick the strategy, resolving ``auto`` against what is actually fitted.

    ``auto`` means ``calibrated_risk`` when a calibration exists and
    ``raw_confidence`` when one does not. Asking for ``calibrated_risk``
    outright when nothing was fitted is refused rather than quietly downgraded:
    a report that said "calibrated" about a raw ranking would be the lie this
    whole section exists to prevent.
    """
    if requested == "auto":
        requested = "calibrated_risk" if calibrated else "raw_confidence"
    strategy = STRATEGIES.get(requested)
    if strategy is None:
        raise ValueError(
            f"unknown triage strategy {requested!r}. "
            f"Available: {', '.join(sorted(STRATEGIES))}."
        )
    if requested == "calibrated_risk" and not calibrated:
        raise ValueError(
            "strategy 'calibrated_risk' needs a fitted calibration and none was "
            "produced — too few labelled rows, or the fit did not converge. Use 'auto', "
            "which falls back to raw_confidence and stamps the ranking uncalibrated."
        )
    return strategy
