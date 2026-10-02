"""The two gates, and the suppression machinery that gives them teeth.

    GATE 1   Can I trust the measurement?
             If this fails, every number below it is meaningless.

    GATE 2   Is the judge better than nothing?
             If it loses to a trivial baseline, you are paying for noise.

A gate firing does **not** kill the run (DESIGN.md §10). It removes the
numbers it invalidates and says which and why, so a run that is partly
measurable reports the part that is. That is what ``Suppression`` is for: a
guardrail names the metrics it poisons, and anything holding one of those
names is withheld and listed rather than printed with a caveat nobody reads.

Two severities behave differently and the difference is the whole design:

    STOP   this specific number cannot be computed honestly — withheld
    WARN   reported, with a flag attached

Which is distinct again from a number that is merely *underpowered*. A metric
below its sample floor is still printed, with its interval and a line saying
it cannot support a conclusion, because a wide interval is information and
hiding it is not.

This module sits above ``checks/`` rather than inside it on purpose. Gate 2
needs human labels and triage strategies; ``checks/`` is defined as free and
label-blind, and importing either into it would quietly end that guarantee.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .config import RunConfig
from .metrics.ranking import RankingTarget, auc, degenerate_target
from .metrics.stats import Estimate
from .triage.base import TriageContext
from .triage.strategies import BASELINES, STRATEGIES
from .types import Label, Output, Severity, Verdict

__all__ = [
    "Gate",
    "GateResult",
    "GateTwo",
    "Gates",
    "RankerScore",
    "Suppression",
    "build_target",
    "gate_one",
    "gate_two",
]


def _joined(reasons: Sequence[str]) -> str:
    return f"{len(reasons)} guardrails withheld this. First: {reasons[0]}"


@dataclass(frozen=True, slots=True)
class Suppression:
    """A number that cannot be computed honestly, and the reason."""

    metric: str
    reason: str

    def __str__(self) -> str:
        return f"{self.metric}\n{self.reason}"


@dataclass(frozen=True, slots=True)
class GateResult:
    """One guardrail's verdict on the measurement."""

    id: str
    severity: Severity
    message: str
    suppresses: tuple[str, ...] = ()

    @property
    def blocking(self) -> bool:
        return self.severity is Severity.STOP


@dataclass(frozen=True, slots=True)
class Gate:
    """A gate's results, and what passed through it."""

    name: str
    results: tuple[GateResult, ...] = ()
    skipped: str | None = None

    @property
    def passed(self) -> bool:
        """A gate passes when nothing blocking fired.

        A gate that could not run has not passed. ``skipped`` carries why,
        and the report prints that rather than a tick.
        """
        return self.skipped is None and not any(r.blocking for r in self.results)

    @property
    def status(self) -> str:
        if self.skipped is not None:
            return "not run"
        return "PASS" if self.passed else "STOP"

    def suppressions(self) -> tuple[Suppression, ...]:
        return tuple(
            Suppression(metric, result.message)
            for result in self.results
            if result.blocking
            for metric in result.suppresses
        )


def gate_one(
    config: RunConfig,
    *,
    judge_health: Sequence[object],
    items: int,
    verdicts: Sequence[Verdict],
    panel: Mapping[str, object],
) -> Gate:
    """Can the measurement be trusted? Always on, and free.

    Most of this gate was already enforced where it is cheapest to enforce —
    judge approval rates and parse rates at M1, effective votes at M3. This
    assembles those into one verdict and adds the two that need the corpus:
    whether there are enough rows to say anything, and whether the triage
    signal could have seen a label.
    """
    settings = config.settings
    results: list[GateResult] = []

    for row in judge_health:
        for severity, message in getattr(row, "flags", ()):
            judge_id = getattr(row, "judge_id", "?")
            results.append(
                GateResult(
                    id=f"judge_health:{judge_id}",
                    severity=severity,
                    message=message,
                    suppresses=("panel.agreement", "panel.effective_votes")
                    if severity is Severity.STOP
                    else (),
                )
            )

    minimum = int(settings.value("min_n_distribution"))
    if items < minimum:
        results.append(
            GateResult(
                id="sample_size",
                severity=Severity.WARN,
                message=(
                    f"{items} items, below the {minimum} needed for a distribution or drift "
                    "claim. Those numbers are reported with their intervals and cannot "
                    "support a conclusion."
                ),
            )
        )

    warn_below = float(settings.value("effective_votes_warn"))
    effective = panel.get("effective_votes")
    paid = panel.get("votes_paid_for")
    if isinstance(effective, (int, float)) and isinstance(paid, int) and paid:
        if effective / paid < warn_below:
            results.append(
                GateResult(
                    id="effective_votes",
                    severity=Severity.WARN,
                    message=(
                        f"the panel is worth {effective:.1f} of the {paid} opinions you paid "
                        "for. Judges that fail the same way are nearly one judge."
                    ),
                )
            )

    # Structural, not measured: the code that builds triage signals does not
    # accept labels, and `assert_no_labels` confirms it at runtime. Recorded
    # here as a NOTE so the gate's own output states the guarantee rather
    # than leaving a reader to trust it.
    results.append(
        GateResult(
            id="no_label_leakage",
            severity=Severity.NOTE,
            message=(
                "no triage signal can see a human label — the context type has no field "
                "for one and the assertion holds at runtime"
            ),
        )
    )
    return Gate(name="measurement is sound", results=tuple(results))


def build_target(
    labels: Sequence[Label], outputs: Mapping[str, Output], fields: Sequence[str]
) -> RankingTarget:
    """What "this item is wrong" means, from human labels alone.

    An item counts as an error when any labelled field disagrees with its
    human answer. That is the item grain again: an item with a correct label
    and a wrong outcome is not a usable item, so it is a positive here.

    Only the first annotator's answers are used. Pooling two annotators who
    disagree would silently make the target depend on which of them was read
    last; adjudicating them is a different job and mode 2's business.
    """
    primary: dict[tuple[str, str], str] = {}
    for label in sorted(labels, key=lambda label: label.annotator):
        primary.setdefault((label.item_id, label.field), label.label)

    truth: dict[str, bool] = {}
    for (item_id, field_name), answer in primary.items():
        if field_name not in fields or item_id not in outputs:
            continue
        wrong = str(outputs[item_id].get(field_name)) != answer
        truth[item_id] = truth.get(item_id, False) or wrong
    return RankingTarget(truth=truth, source="human labels, first annotator")


@dataclass
class RankerScore:
    """One strategy's AUC against the target, with its interval."""

    strategy: str
    estimate: Estimate
    ranked: int
    unranked: int
    unranked_errors: int = 0
    is_baseline: bool = False
    note: str | None = None

    @property
    def value(self) -> float | None:
        return self.estimate.value


def _score_one(
    strategy_id: str,
    scores: Mapping[str, float | None],
    target: RankingTarget,
    *,
    resamples: int,
    seed: int,
    floor: int,
) -> RankerScore:
    ranked = {
        item_id: score
        for item_id, score in scores.items()
        if score is not None and item_id in target.truth
    }
    missing = [i for i in target.truth if i not in ranked]

    # AUC needs the score and the truth of the same row together, which the
    # generic `bootstrap_ci` statistic signature cannot carry, so the
    # resampling is written out here. It still resamples whole items.
    items = list(ranked)
    values = [ranked[i] for i in items]
    truths = [target.truth[i] for i in items]
    estimate = _auc_ci(
        values, truths, auc(values, truths) if items else None,
        resamples=resamples, seed=seed, floor=floor,
    )
    return RankerScore(
        strategy=strategy_id,
        estimate=estimate,
        ranked=len(ranked),
        unranked=len(missing),
        unranked_errors=sum(1 for i in missing if target.truth[i]),
        is_baseline=strategy_id in BASELINES,
    )


def _auc_ci(
    values: Sequence[float],
    truths: Sequence[bool],
    point: float | None,
    *,
    resamples: int,
    seed: int,
    floor: int,
) -> Estimate:
    """Bootstrap an AUC by resampling items — one item, one score and truth."""
    n = len(values)
    if point is None or n < 2:
        return Estimate(value=point, n=n, floor=floor)
    rng = np.random.default_rng(seed)
    draws = []
    value_array = np.asarray(values, dtype=float)
    truth_array = np.asarray(truths, dtype=bool)
    for _ in range(resamples):
        picked = rng.integers(0, n, size=n)
        drawn = auc(value_array[picked].tolist(), truth_array[picked].tolist())
        if drawn is not None:
            draws.append(drawn)
    if not draws:
        return Estimate(value=point, n=n, floor=floor, note="no resample had both classes")
    low, high = np.percentile(draws, [2.5, 97.5])
    note = None
    if n < floor:
        note = (
            f"n = {n}, below the floor of {floor}. This interval spans too much to support "
            "a conclusion."
        )
    return Estimate(
        value=point, low=float(low), high=float(high), n=n, resamples=len(draws),
        floor=floor, note=note,
    )


@dataclass(frozen=True, slots=True)
class GateTwo:
    """Gate 2's verdict, and the table behind it."""

    gate: Gate
    scores: tuple[RankerScore, ...] = ()
    target_size: int = 0
    target_errors: int = 0

    @property
    def ranker(self) -> RankerScore | None:
        return next((s for s in self.scores if not s.is_baseline), None)

    @property
    def best_baseline(self) -> RankerScore | None:
        baselines = [s for s in self.scores if s.is_baseline and s.value is not None]
        return max(baselines, key=lambda s: s.value or 0.0) if baselines else None


def gate_two(
    config: RunConfig,
    ctx: TriageContext,
    strategy_id: str,
    labels: Sequence[Label],
    outputs: Mapping[str, Output],
) -> GateTwo:
    """Is the judge better than the dumb options, on the same target?

    Every baseline is scored next to the judge, always. A ranking number
    printed without them is not a result — "AUC 0.84" means nothing until you
    know that item length scored 0.48 on the same rows.

    The gate passes only when the judge's **interval** clears the best
    baseline's point estimate. A higher number with an overlapping interval
    has not been shown to beat anything.
    """
    settings = config.settings
    target = build_target(labels, outputs, tuple(config.schema.fields))

    if not target.truth:
        return GateTwo(
            gate=Gate(
                name="judge beats baselines",
                skipped=(
                    "no human labels, so there is no target to rank against. The ranking "
                    "is still produced; it has not been shown to beat guessing, and the "
                    "report says so rather than implying otherwise."
                ),
            )
        )

    reason = degenerate_target(
        target,
        min_items=int(settings.value("minority_class_min_items")),
        min_share=float(settings.value("minority_class_min_share")),
    )
    if reason is not None:
        return GateTwo(
            gate=Gate(
                name="judge beats baselines",
                results=(
                    GateResult(
                        id="degenerate_target",
                        severity=Severity.STOP,
                        message=reason,
                        suppresses=("gate2.auc", "triage.auc"),
                    ),
                ),
            ),
            target_size=len(target.truth),
            target_errors=target.positives,
        )

    resamples = int(settings.value("bootstrap_resamples"))
    floor = int(settings.value("min_n_ranking"))
    wanted = [strategy_id, *(b for b in BASELINES if b != strategy_id)]

    scores: list[RankerScore] = []
    for name in wanted:
        strategy = STRATEGIES.get(name)
        if strategy is None:
            continue
        if "panel" in strategy.requires and not ctx.panel_members:
            continue
        scores.append(
            _score_one(
                name,
                strategy.rank(ctx),
                target,
                resamples=resamples,
                seed=ctx.seed,
                floor=floor,
            )
        )

    outcome = GateTwo(
        gate=Gate(name="judge beats baselines"),
        scores=tuple(scores),
        target_size=len(target.truth),
        target_errors=target.positives,
    )

    ranker, best = outcome.ranker, outcome.best_baseline
    results: list[GateResult] = []
    if ranker is None or ranker.value is None:
        results.append(
            GateResult(
                id="no_ranker",
                severity=Severity.STOP,
                message="the triage strategy produced no usable ranking",
                suppresses=("gate2.auc",),
            )
        )
    elif best is None:
        results.append(
            GateResult(
                id="no_baseline",
                severity=Severity.WARN,
                message="no baseline could be scored, so the judge has nothing to beat",
            )
        )
    elif not ranker.estimate.beats(best.estimate):
        results.append(
            GateResult(
                id="loses_to_baseline",
                severity=Severity.STOP,
                message=(
                    f"{ranker.strategy} scores {ranker.estimate.format()} and {best.strategy} "
                    f"scores {best.estimate.format()} on the same rows. The judge's interval "
                    "does not clear the baseline, so a judge is not buying you anything "
                    "here — and saying that is worth more than a dashboard."
                ),
                suppresses=("triage.validated",),
            )
        )
    if ranker is not None and ranker.unranked_errors:
        results.append(
            GateResult(
                id="unranked_errors",
                severity=Severity.WARN,
                message=(
                    f"{ranker.unranked} labelled items could not be ranked and "
                    f"{ranker.unranked_errors} of them are errors. They are excluded from "
                    "every number above, which flatters whatever ranked the rest."
                ),
            )
        )

    return GateTwo(
        gate=Gate(name="judge beats baselines", results=tuple(results)),
        scores=outcome.scores,
        target_size=outcome.target_size,
        target_errors=outcome.target_errors,
    )


@dataclass(frozen=True, slots=True)
class Gates:
    """Both gates, and everything they withheld."""

    one: Gate
    two: Gate
    table: tuple[RankerScore, ...] = ()
    target_size: int = 0
    target_errors: int = 0
    extra: tuple[Suppression, ...] = field(default_factory=tuple)

    def suppressed(self) -> tuple[Suppression, ...]:
        """One entry per withheld metric, not one per guardrail that hit it.

        Two broken judges suppress the same panel number for the same kind of
        reason. Listing it twice makes the report look like four problems and
        buries the two that are real.
        """
        merged: dict[str, list[str]] = {}
        for suppression in (*self.one.suppressions(), *self.two.suppressions(), *self.extra):
            reasons = merged.setdefault(suppression.metric, [])
            if suppression.reason not in reasons:
                reasons.append(suppression.reason)
        return tuple(
            Suppression(metric, reasons[0] if len(reasons) == 1 else _joined(reasons))
            for metric, reasons in merged.items()
        )

    def is_suppressed(self, metric: str) -> str | None:
        for suppression in self.suppressed():
            if suppression.metric == metric:
                return suppression.reason
        return None
