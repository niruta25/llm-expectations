"""The text report.

Three things it always does (DESIGN.md §11):

**Says which mode each field is in, at the top.** You should never have to
guess whether a number is backed by human answers.

**Prints the threshold next to every result**, and where that threshold came
from. A number without its bar is not a result, and a bar without its source
is not arguable.

**Names what the run cannot tell you.** That list is long at M2 and shrinks as
milestones land. A report that hid it would read as a finished answer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .calibration.identity import UNCALIBRATED_NOTICE
from .checks.base import Skipped
from .judges.screening import JudgeHealth
from .schema import Schema
from .taxonomy import Taxonomy
from .types import Finding, Grain, Item, Mode, RiskRow, Severity, Status, Verdict

__all__ = ["render"]

REVIEW_PREVIEW = 10
EVIDENCE_PREVIEW = 4
RULE = 74

#: How each check reads in the report, and the order the ladder runs in.
CHECK_LABELS: Mapping[str, str] = {
    "label_in_taxonomy": "label in taxonomy",
    "valid_leaf": "valid leaf",
    "abstention_allowed": "abstention not allowed",
    "abstention_rate": "abstention rate",
    "label_collapse": "largest label share",
    "drift": "drift vs previous run",
    "cross_field_agreement": "agrees with other fields",
    "label_correct": "label is correct",
    "free_text": "defect checks",
}
ORDER = list(CHECK_LABELS)

#: Checks that cost money. Shown in their own block, because "free checks
#: found nothing" and "we paid a judge and it found nothing" are different
#: statements and a reader should not have to know which is which.
PAID = frozenset({"label_correct", "claims_supported"})


def render(
    *,
    run_id: str,
    schema: Schema,
    taxonomies: Mapping[str, Taxonomy],
    modes: Mapping[str, Mode],
    label_counts: Mapping[str, tuple[int, int]],
    health: Sequence[JudgeHealth],
    findings: Sequence[Finding],
    skipped: Sequence[Skipped],
    risk_rows: Sequence[RiskRow],
    verdicts: Sequence[Verdict],
    items: Mapping[str, Item],
    metrics: Mapping[str, Any],
    cannot_tell: Sequence[str],
    exclusions: Mapping[str, int],
) -> str:
    out: list[str] = [f"llm-expectations   {run_id}", ""]
    out += _headline(metrics)
    out += _modes(schema, modes, label_counts)
    out += _grains(metrics)
    out += _judges(health)

    for name, spec in schema.fields.items():
        out += _field_section(name, spec, schema, taxonomies, findings, skipped)

    out += _review(risk_rows, verdicts)
    out += _cannot(cannot_tell)
    out += _exclusions(exclusions)
    return "\n".join(out).rstrip() + "\n"


def _headline(metrics: Mapping[str, Any]) -> list[str]:
    cost = metrics.get("cost_usd")
    cost_text = f"${cost:.2f}" if isinstance(cost, (int, float)) else "cost not estimable"
    lines = [
        f"  {metrics.get('items', 0):,} items      {cost_text}      "
        f"{metrics.get('calls', 0):,} calls      {metrics.get('elapsed_s', 0):.1f}s"
    ]
    if metrics.get("cache_hits"):
        lines.append(f"  {metrics['cache_hits']:,} verdicts reused from cache — not re-paid for")
    return lines + [""]


def _modes(
    schema: Schema, modes: Mapping[str, Mode], counts: Mapping[str, tuple[int, int]]
) -> list[str]:
    width = max((len(name) for name in schema.fields), default=8)
    lines = ["  MODE"]
    for name in schema.fields:
        count, double = counts.get(name, (0, 0))
        detail = f"{count} labels" if count else "no labels"
        if double:
            detail += f" ({double} with a second annotator)"
        mode = int(modes.get(name, Mode.NO_LABELS))
        lines.append(f"    {name:<{width}}  MODE {mode}    {detail}")
    return lines + [""]


def _grains(metrics: Mapping[str, Any]) -> list[str]:
    grains = metrics.get("grains") or {}
    field, item = grains.get("field") or {}, grains.get("item") or {}
    if field.get("pass_rate") is None:
        return []
    lines = [
        "  BOTH GRAINS",
        f"    field grain   {field['pass_rate']:.1%} of {field['scored']:,} scored checks pass",
        f"    item grain    {item['pass_rate']:.1%} of {item['items']:,} items pass "
        "every check on them",
        "    The item number is the worse one, and it is the one a consumer needs:",
        "    an item with a correct label and a bad summary is not a usable item.",
    ]
    if field.get("unscored"):
        lines.append(
            f"    {field['unscored']:,} findings are unscored — not checked, and never "
            "counted as passing."
        )
    return lines + [""]


def _judges(health: Sequence[JudgeHealth]) -> list[str]:
    if not health:
        return []
    lines = [
        "  JUDGE HEALTH",
        f"    {'judge':<12} {'model':<22} {'approves':>9} {'cannot_decide':>14} "
        f"{'unreadable':>11}",
    ]
    for row in health:
        lines.append(
            f"    {row.judge_id:<12} {row.model[:22]:<22} {_pct(row.approval_rate):>9} "
            f"{_pct(row.cannot_decide_rate):>14} {_pct(row.unreadable_rate):>11}"
        )
    lines.append(f"    {'':12} {'':22} {'band 15–85%':>9}")
    for row in health:
        for severity, message in row.flags:
            mark = "✗" if severity is Severity.STOP else "⚠"
            wrapped = _wrapped(message, indent=6)
            lines.append(f"    {mark} {wrapped[0].strip()}")
            lines += [f"      {line.strip()}" for line in wrapped[1:]]
    return lines + [""]


def _field_section(
    name: str,
    spec: Any,
    schema: Schema,
    taxonomies: Mapping[str, Taxonomy],
    findings: Sequence[Finding],
    skipped: Sequence[Skipped],
) -> list[str]:
    detail = spec.kind.value
    if spec.taxonomy is not None:
        detail += f" · {spec.taxonomy}"
    elif spec.style is not None:
        detail += f" · {spec.style.value}"
    header = f"  ── {name} "
    header += "─" * max(3, RULE - len(header) - len(detail) - 4) + f" {detail} ──"

    mine = [f for f in findings if f.field == name]
    if not mine:
        return [header, "    nothing checked this field.", ""]

    grouped: dict[str, list[Finding]] = {}
    for found in mine:
        grouped.setdefault(found.check, []).append(found)
    ordered = sorted(grouped, key=lambda c: (ORDER.index(c) if c in ORDER else 99, c))

    lines = [header, ""]
    free = [c for c in ordered if c not in PAID]
    if free:
        lines.append("  free checks")
        for check in free:
            lines += _check_line(check, grouped[check])
        for entry in skipped:
            if entry.field == name:
                lines.append(
                    f"    {CHECK_LABELS.get(entry.check, entry.check):<28} "
                    f"{'—':>8}   ·  not run"
                )
                lines += _wrapped(entry.reason)

    paid = [c for c in ordered if c in PAID]
    if paid:
        lines.append("")
        lines.append("  judge")
        for check in paid:
            lines += _check_line(check, grouped[check])
    return lines + [""]


def _check_line(check: str, group: Sequence[Finding]) -> list[str]:
    label = CHECK_LABELS.get(check, check)
    scored = [f for f in group if f.status is not Status.UNSCORED]
    failed = [f for f in scored if f.status is Status.FAIL]
    unscored = [f for f in group if f.status is Status.UNSCORED]
    corpus = group[0].grain is Grain.CORPUS

    if corpus:
        found = group[0]
        value = _value(check, found.score)
        mark = {Status.PASS: "✓", Status.FAIL: "✗", Status.UNSCORED: "○"}[found.status]
        lines = [f"    {label:<28} {value:>8}   {mark}  {_bar(check, found)}".rstrip()]
        lines += _wrapped(found.evidence.get("why"))
        if check == "drift" and found.evidence.get("moves"):
            for moved, pp in list(found.evidence["moves"].items())[:3]:
                lines.append(f"        {moved:<34} {pp * 100:+.1f}pp")
        return lines

    if not scored:
        return [
            f"    {label:<28} {'—':>8}   ○  all {len(unscored)} unscored",
            *_wrapped(_why(unscored[0])),
        ]

    rate = sum(f.status is Status.PASS for f in scored) / len(scored)
    mark = "✓" if not failed else "✗"
    tail = f"{len(failed)} of {len(scored)} failed" if failed else _bar(check, group[0])
    lines = [f"    {label:<28} {rate:>7.1%}   {mark}  {tail}".rstrip()]
    if unscored:
        lines.append(f"      {len(unscored)} not checked:")
        lines += _wrapped(_why(unscored[0]), indent=8)
    for found in failed[:EVIDENCE_PREVIEW]:
        lines.append(f"      {found.item_id:<8} {_evidence(found)[:62]}")
    if len(failed) > EVIDENCE_PREVIEW:
        lines.append(f"      … and {len(failed) - EVIDENCE_PREVIEW} more")
    return lines


def _why(found: Finding) -> str:
    """Whatever this finding recorded as its reason, under either key.

    Free checks write ``why``; verdict-backed findings carry the judge's own
    sentence under ``reason``. An empty line here would be the report quietly
    declining to say why something was not checked.
    """
    return str(found.evidence.get("why") or found.evidence.get("reason") or "no reason recorded")


def _wrapped(text: str | None, *, indent: int = 6, width: int = RULE) -> list[str]:
    if not text:
        return []
    import textwrap

    pad = " " * indent
    return [pad + line for line in textwrap.wrap(str(text), width=width - indent)]


def _value(check: str, score: float | None) -> str:
    if score is None:
        return "—"
    if check == "drift":
        return f"{score * 100:+.1f}pp"
    return f"{score:.1%}"


def _bar(check: str, found: Finding) -> str:
    """The threshold, and the layer that set it."""
    if found.threshold is None:
        return ""
    source = f" ({found.threshold_from})" if found.threshold_from else ""
    threshold = found.threshold
    if check == "abstention_rate" and isinstance(threshold, (list, tuple)):
        return f"band {threshold[0]:.0%}–{threshold[1]:.0%}{source}"
    if check == "label_collapse":
        return f"max {threshold:.0%}{source}"
    if check == "drift":
        return f"max {threshold * 100:.0f}pp{source}"
    if check == "label_in_taxonomy":
        return f"expect {threshold:.0%}{source}"
    if check == "cross_field_agreement":
        return f"at least {threshold} shared term{source}"
    return ""


def _evidence(found: Finding) -> str:
    evidence = found.evidence
    if found.check == "label_in_taxonomy":
        nearest = ", ".join(evidence.get("nearest", [])) or "nothing close"
        return f"{evidence.get('label')!r} — did you mean {nearest}?"
    if found.check == "valid_leaf":
        children = ", ".join(evidence.get("children", [])[:2])
        return f"{evidence.get('label')!r} is a parent of {children}"
    if found.check == "cross_field_agreement":
        return f"no overlap with {evidence.get('label')} — one of the two is wrong"
    return _why(found)


def _review(risk_rows: Sequence[RiskRow], verdicts: Sequence[Verdict]) -> list[str]:
    reasons = _reasons(verdicts)
    ranked = [row for row in risk_rows if row.triage_score is not None]
    unranked = [row for row in risk_rows if row.triage_score is None]
    lines: list[str] = []

    if ranked:
        lines.append("  ⚠ " + UNCALIBRATED_NOTICE.replace("\n", "\n  "))
        lines.append("")
        shown = ranked[:REVIEW_PREVIEW]
        lines.append(f"  REVIEW FIRST   (top {len(shown)} of {len(ranked)} ranked)")
        for position, entry in enumerate(shown, start=1):
            score = entry.triage_score or 0.0
            lines.append(
                f"    {position:>2}  {entry.item_id:<8} {score:.2f}   "
                f"{reasons.get(entry.item_id, '')[:66]}"
            )
        lines.append("")

    if unranked:
        noun = "item" if len(unranked) == 1 else "items"
        lines.append(f"  NOT RANKED   {len(unranked)} {noun}")
        lines.append("    Not clean — unjudged. They have no place in the order, so they are")
        lines.append("    listed rather than sorted to the bottom.")
        for entry in unranked[:REVIEW_PREVIEW]:
            lines.append(f"    {entry.item_id:<8} {reasons.get(entry.item_id, '')[:62]}")
        lines.append("")
    return lines


def _cannot(cannot_tell: Sequence[str]) -> list[str]:
    if not cannot_tell:
        return []
    lines = ["  ┌ WHAT THIS RUN CANNOT TELL YOU " + "─" * 41 + "┐"]
    for line in cannot_tell:
        head, _, tail = line.partition("\n")
        lines.append(f"  │  ✗ {head}")
        for extra in tail.splitlines():
            for wrapped in _wrapped(extra, indent=0, width=RULE - 8):
                lines.append(f"  │      {wrapped}")
    return lines + ["  └" + "─" * 72 + "┘", ""]


def _exclusions(exclusions: Mapping[str, int]) -> list[str]:
    if not exclusions:
        return []
    lines = ["  EXCLUSIONS"]
    for reason, count in sorted(exclusions.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {count:>5}   {reason}")
    return lines + [""]


def _pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.0%}" if value >= 0.01 or value == 0 else "<1%"


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
