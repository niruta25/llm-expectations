"""The text report. Minimal at M1, and honest about being minimal.

Three things it always does (DESIGN.md §11): say which mode each field is in at
the top, print the threshold next to every result, and name what the run cannot
tell you. The third is the one that grows: at M1 almost everything is on that
list, and a report that hid it would read as though a ranking off one judge
with no baselines were a finished answer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .calibration.identity import UNCALIBRATED_NOTICE
from .judges.screening import JudgeHealth
from .schema import Schema
from .types import Item, Mode, RiskRow, Severity, Verdict

__all__ = ["render"]

REVIEW_PREVIEW = 10


def render(
    *,
    run_id: str,
    schema: Schema,
    modes: Mapping[str, Mode],
    label_counts: Mapping[str, tuple[int, int]],
    health: Sequence[JudgeHealth],
    risk_rows: Sequence[RiskRow],
    verdicts: Sequence[Verdict],
    items: Mapping[str, Item],
    metrics: Mapping[str, object],
    cannot_tell: Sequence[str],
    exclusions: Mapping[str, int],
) -> str:
    out: list[str] = [f"llm-expectations   {run_id}", ""]

    cost = metrics.get("cost_usd")
    cost_text = f"${cost:.2f}" if isinstance(cost, (int, float)) else "cost not estimable"
    out.append(
        f"  {metrics.get('items', 0):,} items      {cost_text}      "
        f"{metrics.get('calls', 0):,} calls      {metrics.get('elapsed_s', 0):.1f}s"
    )
    if metrics.get("cache_hits"):
        out.append(f"  {metrics['cache_hits']:,} verdicts reused from cache — not re-paid for")
    out.append("")

    out.append("  MODE")
    width = max((len(name) for name in schema.fields), default=8)
    for name in schema.fields:
        mode = modes.get(name, Mode.NO_LABELS)
        count, double = label_counts.get(name, (0, 0))
        detail = f"{count} labels" if count else "no labels"
        if double:
            detail += f" ({double} with a second annotator)"
        out.append(f"    {name:<{width}}  MODE {int(mode)}    {detail}")
    out.append("")

    out.append("  ⚠ " + UNCALIBRATED_NOTICE.replace("\n", "\n  "))
    out.append("")

    if health:
        out.append("  JUDGE HEALTH")
        out.append(
            f"    {'judge':<12} {'model':<22} {'approves':>9} {'cannot_decide':>14} "
            f"{'unreadable':>11}"
        )
        for row in health:
            out.append(
                f"    {row.judge_id:<12} {row.model[:22]:<22} {_pct(row.approval_rate):>9} "
                f"{_pct(row.cannot_decide_rate):>14} {_pct(row.unreadable_rate):>11}"
            )
        out.append(f"    {'':12} {'':22} {'band 15–85%':>9}")
        for row in health:
            for severity, message in row.flags:
                marker = "✗" if severity is Severity.STOP else "⚠"
                out.append(f"    {marker} {message}")
        out.append("")

    ranked = [row for row in risk_rows if row.triage_score is not None]
    reasons = _reasons(verdicts)
    if ranked:
        shown = ranked[:REVIEW_PREVIEW]
        out.append(f"  REVIEW FIRST   (top {len(shown)} of {len(ranked)} ranked)")
        for position, entry in enumerate(shown, start=1):
            why = reasons.get(entry.item_id, "")
            score = entry.triage_score or 0.0
            out.append(f"    {position:>2}  {entry.item_id:<8} {score:.2f}   {why[:70]}")
        out.append("")

    unranked = [row for row in risk_rows if row.triage_score is None]
    if unranked:
        noun = "item" if len(unranked) == 1 else "items"
        out.append(f"  NOT RANKED   {len(unranked)} {noun}")
        out.append("    These are not clean. They are unjudged, and they have no place in")
        out.append("    the order — so they are listed rather than sorted to the bottom.")
        for entry in unranked[:REVIEW_PREVIEW]:
            out.append(f"    {entry.item_id:<8} {reasons.get(entry.item_id, '')[:66]}")
        out.append("")

    if cannot_tell:
        out.append("  ┌ WHAT THIS RUN CANNOT TELL YOU " + "─" * 41 + "┐")
        for line in cannot_tell:
            head, _, tail = line.partition("\n")
            out.append(f"  │  ✗ {head}")
            for extra in tail.splitlines():
                out.append(f"  │      {extra}")
        out.append("  └" + "─" * 72 + "┘")
        out.append("")

    if exclusions:
        out.append("  EXCLUSIONS")
        for reason, count in sorted(exclusions.items(), key=lambda kv: -kv[1]):
            out.append(f"    {count:>5}   {reason}")
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}" if value >= 0.01 or value == 0 else "<1%"


def _reasons(verdicts: Sequence[Verdict]) -> dict[str, str]:
    """The judge's own sentence, which is what a reviewer actually reads."""
    worst: dict[str, tuple[int, str]] = {}
    rank = {"fail": 0, "unscored": 1, "pass": 2}
    for verdict in verdicts:
        score = rank.get(verdict.status.value, 3)
        current = worst.get(verdict.item_id)
        if current is None or score < current[0]:
            worst[verdict.item_id] = (score, f"{verdict.field}: {verdict.reason}")
    return {item_id: reason for item_id, (_, reason) in worst.items()}
