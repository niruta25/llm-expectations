"""What a ranker is given, and the assertion that it is never given labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..calibration.base import Calibrator
from ..types import Label, Output, RiskRow, Verdict

__all__ = ["TriageContext", "TriageStrategy", "assert_no_labels", "build_risk_rows"]


@dataclass(frozen=True, slots=True)
class TriageContext:
    """Everything a ranking strategy may see.

    Note what is not here, and cannot be added: human labels. A triage signal
    that peeks at the answer key scores beautifully offline and is useless in
    production, so the type that reaches a strategy has no field for one.
    """

    verdicts: Mapping[str, tuple[Verdict, ...]]
    outputs: Mapping[str, Output]
    fields: tuple[str, ...]
    calibrator: Calibrator
    item_ids: tuple[str, ...] = ()
    seed: int = 0
    panel_members: tuple[str, ...] = ()
    extras: Mapping[str, object] = field(default_factory=dict)

    def items(self) -> tuple[str, ...]:
        return self.item_ids or tuple(self.outputs)


@runtime_checkable
class TriageStrategy(Protocol):
    """One way of ordering a corpus for review."""

    @property
    def id(self) -> str: ...

    @property
    def requires(self) -> frozenset[str]: ...

    def rank(self, ctx: TriageContext) -> Mapping[str, float | None]: ...


def assert_no_labels(ctx: object) -> None:
    """The one runtime assertion behind the structural guarantee.

    The signatures already make a label unreachable from here. This catches the
    case where someone widens ``extras`` and posts one through the side door.
    Everything in this library is tunable except this check, because it is not
    a threshold — it is a bug check.
    """
    seen: list[object] = [ctx]
    depth = 0
    while seen and depth < 4:
        depth += 1
        nxt: list[object] = []
        for value in seen:
            if isinstance(value, Label):
                raise AssertionError(
                    "a human label reached the triage context. A signal fitted on the answer "
                    "key ranks beautifully offline and is worthless in production; labels "
                    "belong downstream of the ranker, not inside it."
                )
            if isinstance(value, Mapping):
                nxt.extend(value.values())
            elif isinstance(value, (list, tuple, set, frozenset)):
                nxt.extend(value)
            elif isinstance(value, TriageContext):
                nxt.extend([value.verdicts, value.outputs, value.extras])
        seen = nxt


def build_risk_rows(
    ctx: TriageContext, scores: Mapping[str, float | None], strategy_id: str
) -> tuple[RiskRow, ...]:
    """Assemble the review queue, and say where every place in it came from.

    Ranked items come first, worst first. Items the judge could not decide on
    come last and carry no score — they are not clean, they are unjudged, and
    a file that left them out is how they get forgotten.
    """
    calibrated = ctx.calibrator.is_calibrated
    rows = []
    for item_id in ctx.items():
        verdicts = ctx.verdicts.get(item_id, ())
        confidences = [v.raw_confidence for v in verdicts if v.raw_confidence is not None]
        probabilities = [
            p for p in (ctx.calibrator.error_probability(v) for v in verdicts) if p is not None
        ]
        rows.append(
            RiskRow(
                item_id=item_id,
                triage_score=scores.get(item_id),
                strategy=strategy_id,
                calibrated=calibrated,
                raw_confidence_mean=(sum(confidences) / len(confidences)) if confidences else None,
                calibrated_error_probability=max(probabilities) if probabilities and calibrated
                else None,
                calibration_id=ctx.calibrator.id if calibrated else None,
            )
        )
    return tuple(
        sorted(rows, key=lambda r: (r.triage_score is None, -(r.triage_score or 0.0), r.item_id))
    )


def unranked(rows: Sequence[RiskRow]) -> tuple[RiskRow, ...]:
    return tuple(row for row in rows if row.triage_score is None)
