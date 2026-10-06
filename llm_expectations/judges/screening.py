"""Judge health — free, and the most important check in the run.

A judge approving 94% of everything passes every format check ever written and
quietly destroys anything built on it. This table is the only thing that
catches it, which is why it runs as the verdicts arrive rather than waiting
for the rest of the guardrails (DESIGN.md §12).

Two denominators, on purpose:

**Approval rate is over the verdicts the judge actually scored.** A judge that
answers ``cannot_decide`` half the time and approves everything else has an
approval rate of 100%, and that is the thing worth knowing. Measured over all
calls it would read as 50% and sail through the band.

**Unreadable and cannot-decide rates are over all calls**, because those are
questions about how much of the corpus went unjudged.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..config import Settings
from ..types import Severity, Status, Verdict
from .base import ReplyOutcome

__all__ = ["JudgeHealth", "screen"]


@dataclass(frozen=True, slots=True)
class JudgeHealth:
    """One judge's row in the health table."""

    judge_id: str
    model: str
    calls: int
    approvals: int
    rejections: int
    cannot_decide: int
    unparseable: int
    errors: int
    flags: tuple[tuple[Severity, str], ...] = ()

    @property
    def scored(self) -> int:
        return self.approvals + self.rejections

    @property
    def approval_rate(self) -> float | None:
        return self.approvals / self.scored if self.scored else None

    @property
    def cannot_decide_rate(self) -> float | None:
        return self.cannot_decide / self.calls if self.calls else None

    @property
    def unreadable_rate(self) -> float | None:
        return (self.unparseable + self.errors) / self.calls if self.calls else None

    @property
    def excluded_from_panel(self) -> bool:
        """Past the stop band, this judge's opinions are not aggregated.

        Its raw verdicts are still written to disk — suppressing them would
        hide the evidence for the exclusion.
        """
        return any(severity is Severity.STOP for severity, _ in self.flags)


def screen(verdicts: Iterable[Verdict], settings: Settings) -> tuple[JudgeHealth, ...]:
    """Build the health table, one row per judge."""
    tallies: dict[str, dict[str, int]] = {}

    models: dict[str, str] = {}
    for verdict in verdicts:
        tally = tallies.setdefault(
            verdict.judge_id,
            {"calls": 0, "approvals": 0, "rejections": 0, "cannot_decide": 0,
             "unparseable": 0, "errors": 0},
        )
        tally["calls"] += 1
        models.setdefault(verdict.judge_id, str(verdict.metadata.get("model", "")))
        outcome = verdict.metadata.get("reply")
        if outcome == ReplyOutcome.CANNOT_DECIDE.value:
            tally["cannot_decide"] += 1
        elif outcome == ReplyOutcome.UNPARSEABLE.value:
            tally["unparseable"] += 1
        elif outcome == ReplyOutcome.ERROR.value:
            tally["errors"] += 1
        elif verdict.status is Status.PASS:
            tally["approvals"] += 1
        elif verdict.status is Status.FAIL:
            tally["rejections"] += 1

    rows = []
    for judge_id, tally in sorted(tallies.items()):
        # Built once to get the rates, then stamped with the flags those rates
        # earn. The flags are a function of the counts, never of each other.
        health = JudgeHealth(
            judge_id=judge_id,
            model=models.get(judge_id, ""),
            calls=tally["calls"],
            approvals=tally["approvals"],
            rejections=tally["rejections"],
            cannot_decide=tally["cannot_decide"],
            unparseable=tally["unparseable"],
            errors=tally["errors"],
        )
        rows.append(dataclasses.replace(health, flags=_flags(health, settings)))
    return tuple(rows)


def _flags(health: JudgeHealth, settings: Settings) -> tuple[tuple[Severity, str], ...]:
    flags: list[tuple[Severity, str]] = []
    warn_low, warn_high = settings.value("judge_approval_warn")
    stop_low, stop_high = settings.value("judge_approval_stop")
    rate = health.approval_rate

    if rate is None:
        flags.append(
            (
                Severity.STOP,
                f"{health.judge_id} scored nothing — every call came back as cannot_decide, "
                "unreadable or failed. There is no approval rate to judge it by.",
            )
        )
    elif not stop_low <= rate <= stop_high:
        side = "approves" if rate > stop_high else "rejects"
        flags.append(
            (
                Severity.STOP,
                f"{health.judge_id} {side} {rate:.0%} of what it scores — outside the "
                f"{stop_low:.0%}–{stop_high:.0%} band. A judge this one-sided passes every "
                "format check ever written and carries no information. Excluded from panel "
                "aggregates; its verdicts are still on disk.",
            )
        )
    elif not warn_low <= rate <= warn_high:
        flags.append(
            (
                Severity.WARN,
                f"{health.judge_id} approves {rate:.0%} of what it scores, outside the "
                f"{warn_low:.0%}–{warn_high:.0%} band. Possible rubber stamp.",
            )
        )

    unreadable = health.unreadable_rate or 0.0
    warn_at = settings.value("parse_failure_warn")
    stop_at = settings.value("parse_failure_stop")
    if unreadable > stop_at:
        flags.append(
            (
                Severity.STOP,
                f"{health.judge_id} returned {unreadable:.1%} unreadable replies, over the "
                f"{stop_at:.0%} ceiling. Above this you measured your fallback, not the model.",
            )
        )
    elif unreadable > warn_at:
        flags.append(
            (
                Severity.WARN,
                f"{health.judge_id} returned {unreadable:.1%} unreadable replies, over the "
                f"{warn_at:.0%} warning line.",
            )
        )
    return tuple(flags)


def unreadable_items(
    verdicts: Sequence[Verdict], items_by_id: Mapping[str, object]
) -> dict[str, Any]:
    """Which items failed to parse, and whether they are longer than average.

    Empty replies cluster on hard items. That biases everything downstream, and
    a bare percentage hides it — so the count comes with the comparison.
    """
    failed = [
        v for v in verdicts
        if v.metadata.get("reply") in {ReplyOutcome.UNPARSEABLE.value, ReplyOutcome.ERROR.value}
    ]
    if not failed:
        return {"n": 0}
    lengths = {
        item_id: len(getattr(item, "text", "").split()) for item_id, item in items_by_id.items()
    }
    failed_ids = {v.item_id for v in failed}
    failed_lengths = [lengths[i] for i in failed_ids if i in lengths]
    all_lengths = list(lengths.values())
    return {
        "n": len(failed),
        "item_ids": sorted(failed_ids),
        "mean_words_failed": round(sum(failed_lengths) / len(failed_lengths), 1)
        if failed_lengths
        else None,
        "mean_words_overall": round(sum(all_lengths) / len(all_lengths), 1)
        if all_lengths
        else None,
    }
