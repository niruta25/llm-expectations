"""The text report.

Three things it always does (DESIGN.md §11):

**Says which mode each field is in, at the top.** You should never have to
guess whether a number is backed by human answers.

**Prints the threshold next to every result**, and where that threshold came
from. A number without its bar is not a result, and a bar without its source
is not arguable.

**Names what the run cannot tell you.** Every number a guardrail withheld,
every field with no human answers, every question the data cannot settle. A
report that hid that list would read as a finished answer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .calibration.identity import UNCALIBRATED_NOTICE
from .checks.base import Skipped
from .gates import Gates
from .judges.screening import JudgeHealth
from .schema import Schema
from .taxonomy import Taxonomy
from .types import Finding, Grain, Item, Mode, RiskRow, Severity, Status, Verdict

__all__ = ["comparison_report", "operating_point_table", "render"]

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
    "label_correct_panel": "panel says label is correct",
    "label_tree_bucket": "matches the human answer",
    "length_in_bounds": "length in bounds",
    "specificity": "specific, not filler",
    "copy_ratio": "copy ratio",
    "boilerplate": "boilerplate",
    "claims_supported": "claims the item supports",
    "free_text": "defect checks",
}
ORDER = list(CHECK_LABELS)

#: Checks that cost money. Shown in their own block, because "free checks
#: found nothing" and "we paid a judge and it found nothing" are different
#: statements and a reader should not have to know which is which.
PAID = frozenset({"label_correct", "label_correct_panel", "claims_supported"})

#: Free to compute but impossible without human answers. Shown apart from the
#: free checks, which run on every corpus, so a reader can see at a glance
#: which numbers would disappear if the labels did.
NEEDS_LABELS = frozenset({"label_tree_bucket"})


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
    panel: Mapping[str, Any],
    humans: Mapping[str, Any],
    free_text_gate: Mapping[str, Any],
    free_text_vs_humans: Mapping[str, Any],
    stability: Mapping[str, Any],
    calibration: Mapping[str, Any],
    triage_eval: Mapping[str, Any],
    gates: Gates,
    risk_rows: Sequence[RiskRow],
    verdicts: Sequence[Verdict],
    items: Mapping[str, Item],
    metrics: Mapping[str, Any],
    cannot_tell: Sequence[str],
    exclusions: Mapping[str, int],
) -> str:
    out: list[str] = [f"llm-expectations   {run_id}", ""]
    out += _headline(metrics)
    out += _gates(gates)
    out += _modes(schema, modes, label_counts)
    out += _grains(metrics)
    out += _judges(health)
    out += _panel(panel)

    for name, spec in schema.fields.items():
        out += _field_section(name, spec, schema, taxonomies, findings, skipped)
        out += _vs_humans((humans.get("fields") or {}).get(name), name)
        out += _defects(free_text_vs_humans.get(name))
    out += _judges_vs_humans(humans.get("judges") or [])

    out += _stability(stability)
    out += _free_text_gate(free_text_gate)
    out += _calibration(calibration)
    out += operating_point_table(triage_eval)
    out += _review(
        risk_rows,
        verdicts,
        ranked_by=str(metrics.get("ranked_by") or ""),
        strategy=str(metrics.get("strategy") or ""),
        calibrated=bool(metrics.get("calibrated")),
    )
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


def _gates(gates: Gates) -> list[str]:
    """Both gates, at the top, before any quality number is reported."""
    marks = {"PASS": "✓", "STOP": "✗", "not run": "○"}
    width = RULE - 4
    lines = ["  ┌ GATES " + "─" * (width - 9) + "┐"]
    for gate in (gates.one, gates.two):
        mark = marks.get(gate.status, "·")
        lines.append(f"  │  {mark} {gate.name:<26} {gate.status}")
    lines.append("  └" + "─" * (width - 1) + "┘")

    if gates.two.skipped:
        lines += _wrapped(gates.two.skipped, indent=5)
    lines += _gate_table(gates)

    for gate in (gates.one, gates.two):
        for result in gate.results:
            if result.severity is Severity.NOTE:
                continue
            mark = "✗" if result.severity is Severity.STOP else "⚠"
            wrapped = _wrapped(result.message, indent=6)
            lines.append(f"    {mark} {wrapped[0].strip()}")
            lines += [f"      {line.strip()}" for line in wrapped[1:]]
    return lines + [""]


def _gate_table(gates: Gates) -> list[str]:
    """Every ranker beside every baseline. A number without one is not a result."""
    if not gates.table:
        return []
    lines = [
        "",
        f"    ranked against {gates.target_errors} known errors in "
        f"{gates.target_size} labelled items",
        f"    {'strategy':<22}{'AUC':>6}  {'95% interval':<18} {'n':>5}",
    ]
    for score in gates.table:
        estimate = score.estimate
        value = "—" if estimate.value is None else f"{estimate.value:.2f}"
        interval = (
            "—"
            if estimate.low is None
            else f"[{estimate.low:.2f}, {estimate.high:.2f}]"
        )
        tag = "  ← baseline" if score.is_baseline else "  ← the judge"
        lines.append(
            f"    {score.strategy:<22}{value:>6}  {interval:<18} {estimate.n:>5}{tag}"
        )
    floored = [s for s in gates.table if s.estimate.under_floor]
    if floored:
        lines += _wrapped(floored[0].estimate.note, indent=4)
    return lines


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


def _panel(panel: Mapping[str, Any]) -> list[str]:
    """The panel's three outputs, and the one number that justifies its cost."""
    if panel.get("collapsed"):
        lines = ["  PANEL"]
        if panel.get("excluded"):
            lines += _wrapped(
                f"excluded from the vote: {', '.join(panel['excluded'])} — outside the "
                "approval band. Their verdicts are still on disk.",
                indent=4,
            )
        lines += _wrapped(str(panel["collapsed"]), indent=4)
        return lines + [""]
    if not panel or not panel.get("members"):
        return []

    members = ", ".join(panel["members"])
    lines = [
        "  PANEL",
        f"    {len(panel['members'])} judges × {panel['sampled_items']:,} items "
        f"({members})",
    ]
    if panel.get("excluded"):
        lines += _wrapped(
            f"excluded from the vote: {', '.join(panel['excluded'])} — outside the "
            "approval band. Their verdicts are still on disk.",
            indent=4,
        )

    correct = panel.get("majority_correct")
    if correct is not None:
        lines.append(f"    majority says correct      {correct:>7.1%}")
    if panel.get("undecided"):
        lines.append(
            f"    undecided                  {panel['undecided']:>7,}   "
            "tied or nobody decided — not a pass"
        )
    if panel.get("agreement") is not None:
        lines.append(f"    judges agree               {panel['agreement']:>7.1%}")

    effective = panel.get("effective_votes")
    if effective is not None:
        lines.append(
            f"    effective votes            {effective:>7.1f}   of "
            f"{panel['votes_paid_for']} paid for"
        )
    if panel.get("warning"):
        lines.append("    ⚠ " + _wrapped(panel["warning"], indent=0)[0])
        lines += [f"      {line}" for line in _wrapped(panel["warning"], indent=0)[1:]]

    for entry in panel.get("leniency", [])[:3]:
        lines.append(
            f"    {entry['lenient']} approves where {entry['strict']} rejects   "
            f"{entry['lenient_approved']} times, against {entry['strict_approved']} "
            "the other way"
        )

    lines += _fuzzy(panel)
    lines.append("    The panel measures. It does not route disagreements to review,")
    lines.append("    auto-approve unanimity, or blend its votes into one score.")
    return lines + [""]


def _fuzzy(panel: Mapping[str, Any]) -> list[str]:
    """Where disagreement concentrates — a taxonomy bug, not a model bug."""
    splits = panel.get("splits") or 0
    if not splits:
        return []
    pairs = panel.get("fuzzy_pairs") or []
    attributed = panel.get("splits_attributed") or 0
    lines = [f"    the judges split on         {splits:>6,} rows"]
    if not pairs:
        lines.append(
            "      no pair can be named: no dissenting judge said which label it would"
        )
        lines.append(
            "      assign instead, and guessing one from the taxonomy's shape would be"
        )
        lines.append("      an inference, not a finding")
        return lines
    for pair in pairs:
        share = pair["splits"] / splits
        lines.append(
            f"      {pair['field']}: {pair['left']} ↔ {pair['right']}   "
            f"{pair['splits']} ({share:.0%} of splits)"
        )
    if attributed < splits:
        lines.append(
            f"      {splits - attributed} split(s) unattributed — the dissenter named "
            "no alternative"
        )
    if panel.get("splits_unusable"):
        lines += _wrapped(
            f"{panel['splits_unusable']} dissenter(s) named a label that field's taxonomy "
            "does not define — a judge-health problem, not a boundary",
            indent=6,
        )
    lines.append("      disagreement concentrated on a boundary is a taxonomy bug")
    return lines


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
    free = [c for c in ordered if c not in PAID and c not in NEEDS_LABELS]
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

    labelled = [c for c in ordered if c in NEEDS_LABELS]
    if labelled:
        lines.append("")
        lines.append("  with labels")
        for check in labelled:
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
        if check == "boilerplate":
            evidence = found.evidence
            mark = "✗" if found.status is Status.FAIL else "✓"
            count = f"{evidence.get('near_duplicates', 0)} of {evidence.get('of', 0)}"
            lines = [f"    {label:<28} {count:>8}   {mark}  near-identical"]
            return lines + _wrapped(evidence.get("why"))
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
    if found.check == "claims_supported":
        unsupported = evidence.get("unsupported") or []
        first = unsupported[0] if unsupported else ""
        return f"{len(unsupported)} invented: {first!r}" if first else _why(found)
    if found.check == "specificity":
        return "nothing here is specific to this item"
    if found.check == "copy_ratio":
        return (
            f"{evidence.get('longest_run')} of {evidence.get('words')} words are one "
            "lifted run"
        )
    if found.check == "label_tree_bucket":
        return (
            f"{found.evidence.get('assigned')!r} vs {found.evidence.get('human')!r} — "
            f"{found.evidence.get('why', '')}"
        )
    if found.check == "label_correct_panel":
        votes = evidence.get("votes") or {}
        dissent = evidence.get("dissent") or {}
        split = ", ".join(f"{j}:{v[:3]}" for j, v in sorted(votes.items()))
        quoted = next(iter(dissent.values()), "")
        return f"{evidence.get('agreement', '')} ({split})" + (f" — {quoted}" if quoted else "")
    return _why(found)


def _vs_humans(scored: Mapping[str, Any] | None, field: str) -> list[str]:
    """What the human answers say about one field."""
    if not scored:
        return []
    lines = [f"  vs humans   {scored['n']} labelled items", ""]

    rows = scored.get("labels") or []
    present = [r for r in rows if r["support"]]
    thin = [r for r in present if r["under_floor"]]

    macro = scored.get("macro_f1")
    if macro is not None:
        lines.append(f"    macro F1                   {macro:>7.2f}   the headline")
        if thin:
            # Macro F1 averages per-label F1s. If most of those are built on
            # a handful of rows each, so is the average, and it inherits
            # their uncertainty without inheriting their visible n.
            lines += _wrapped(
                f"built from {len(present)} per-label scores, {len(thin)} of which sit "
                "below the per-label floor. The average cannot support a conclusion "
                "that its parts cannot.",
                indent=6,
            )
    accuracy, baseline = scored.get("accuracy"), scored.get("majority_baseline")
    if accuracy is not None:
        tail = (
            f"   majority-label baseline {baseline:.1%}" if baseline is not None else ""
        )
        mark = "✓" if scored.get("beats_majority") else "✗"
        lines.append(f"    accuracy                   {accuracy:>7.1%}   {mark}{tail}")
        if scored.get("beats_majority") is False:
            lines += _wrapped(
                "this does not beat always guessing the biggest label, which knows "
                "nothing. Accuracy is never a result on its own.",
                indent=6,
            )

    buckets = scored.get("buckets") or {}
    total = sum(buckets.values())
    if total:
        order = ("exact", "right_parent", "too_shallow", "wrong", "abstained", "unknown")
        shown = " / ".join(
            f"{buckets.get(k, 0) / total:.0%}" for k in order if buckets.get(k)
        )
        names = " / ".join(k.replace("_", " ") for k in order if buckets.get(k))
        lines.append(f"    {names}")
        lines.append(f"    {shown}")
        lines += _bucket_advice(buckets, total)

    weak = scored.get("weak_labels") or []
    if weak:
        rows = {row["label"]: row for row in scored.get("labels") or []}
        for name in weak[:3]:
            row = rows[name]
            lines.append(
                f"    weakest label              {name}   recall "
                f"{row['recall']:.2f}  n={row['support']}"
            )
    lines += _direction(scored.get("confusion_direction") or [])
    lines += _annotators(scored.get("annotators"))
    return lines + [""]


def _bucket_advice(buckets: Mapping[str, int], total: int) -> list[str]:
    """Each bucket points at a different fix, so say which."""
    advice = {
        "right_parent": "siblings confused — two definitions need sharpening",
        "too_shallow": "stopped at a parent — the model is hedging, not misreading",
        "wrong": "a different branch — the model is not reading the item",
        "unknown": "labels that are not in the taxonomy at all",
    }
    out = []
    for key, text in advice.items():
        share = buckets.get(key, 0) / total
        if share >= 0.05:
            out.append(f"      {share:.0%} {text}")
    return out


def _direction(pairs: Sequence[Mapping[str, Any]]) -> list[str]:
    """Symmetric confusion is a taxonomy bug; one-way is a prompt bug."""
    if not pairs:
        return []
    lines = ["    confusable pairs"]
    for pair in pairs:
        forward, backward = pair["forward"], pair["backward"]
        lines.append(f"      {forward['from']} → {forward['to']}")
        lines.append(
            f"        {forward['n']} this way, {backward['n']} back — {pair['shape']}"
        )
        lines += _wrapped(pair["verdict"], indent=8)
    return lines


def _annotators(humans: Mapping[str, Any] | None) -> list[str]:
    """Two people disagreeing is the strongest evidence a taxonomy is wrong."""
    if not humans or not humans.get("compared"):
        return []
    lines = [
        f"    two annotators             {humans['agreement']:.1%} agree over "
        f"{humans['compared']} double-labelled items"
    ]
    concentration = humans.get("concentration")
    pairs = humans.get("pairs") or []
    if pairs and concentration is not None:
        top = pairs[0]
        lines.append(f"      {top['left']} ↔ {top['right']}")
        lines.append(
            f"        {top['n']} of {humans['disagreements']} disagreements "
            f"({concentration:.0%})"
        )
        if concentration >= 0.5:
            lines += _wrapped(
                "your annotators cannot separate these two either. That is not a model "
                "problem — merge them or rewrite both definitions.",
                indent=6,
            )
    return lines


def _defects(scored: Mapping[str, Any] | None) -> list[str]:
    """Each defect box against the check that looks for it."""
    if not scored or not scored.get("defects"):
        return []
    rows = scored["defects"]
    # The denominator is per defect, not per field: each box is compared
    # against a different check, and those checks do not all reach the same
    # rows. One header count would be wrong for every row but the first.
    lines = [
        "  vs human defect ratings",
        "",
        f"    {'defect':<20}{'n':>5}{'agreement':>11}{'tool only':>11}{'human only':>12}",
    ]
    for row in rows:
        lines.append(
            f"    {row['defect']:<20}{row['compared']:>5}{_maybe(row['agreement']):>11}"
            f"{row['tool_only']:>11}{row['human_only']:>12}"
        )
    weakest = min(rows, key=lambda r: r["agreement"] if r["agreement"] is not None else 1.0)
    if weakest["agreement"] is not None and weakest["agreement"] < 0.7:
        lines += _wrapped(
            f"{weakest['defect']} is the weak one. That is the expected outcome rather "
            "than a surprise — it is hard for the judge and hard for the person, and a "
            "number that says so is worth more than one that hides it.",
            indent=4,
        )
    for row in rows:
        if row["leaning"]:
            lines.append(f"    {row['check']} {row['leaning']} against people.")
    return lines + [""]


def _judges_vs_humans(judges: Sequence[Mapping[str, Any]]) -> list[str]:
    """Which way each judge fails, and whether it beats doing nothing."""
    if not judges:
        return []
    lines = [
        "  ── judges vs humans " + "─" * 51,
        "",
        f"    {'judge':<12}{'accuracy':>9}{'vs always-approve':>19}"
        f"{'approves wrong':>16}{'rejects right':>15}",
    ]
    for row in judges:
        accuracy = row.get("accuracy")
        baseline = row.get("always_approve_accuracy")
        mark = "✓" if row.get("beats_always_approve") else "✗"
        lines.append(
            f"    {row['judge']:<12}{_maybe(accuracy):>9}"
            f"{mark + ' ' + _maybe(baseline):>19}"
            f"{_maybe(row.get('approves_wrong')):>16}{_maybe(row.get('rejects_right')):>15}"
        )
    for row in judges:
        if row.get("beats_always_approve") is False:
            lines += _wrapped(
                f"{row['judge']} does not beat approving everything. It is costing money "
                "and adding nothing.",
                indent=4,
            )
        if row.get("leaning") in {"lenient", "strict"}:
            lines.append(f"    {row['judge']} fails {row['leaning']}.")
    lines += _wrapped(
        "A panel of judges that all fail the same way is nearly one judge. What helps "
        "is one that fails the other way.",
        indent=4,
    )
    lines += _confidence(judges)
    return lines + [""]


def _confidence(judges: Sequence[Mapping[str, Any]]) -> list[str]:
    """Does 0.9 mean 90%?"""
    lines: list[str] = []
    for row in judges:
        curve = row.get("confidence") or {}
        if curve.get("ece") is None:
            continue
        mark = "✗" if curve.get("broken") else "✓"
        lines.append(
            f"    {row['judge']} confidence   ECE {curve['ece']:.3f} {mark}   "
            f"Brier {curve['brier']:.3f} vs {curve['brier_base_rate']:.3f} base rate"
        )
        if curve.get("direction"):
            lines += _wrapped(curve["direction"], indent=6)
        if curve.get("beats_base_rate") is False:
            lines += _wrapped(
                "the confidences add nothing — one number for every item would score "
                "the same.",
                indent=6,
            )
    return lines


def _maybe(value: float | None, places: int = 2) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def _stability(stability: Mapping[str, Any]) -> list[str]:
    """Serving stability and decision stability, never blended into one."""
    if not stability or not stability.get("fields"):
        return []
    kind = stability["kind"]
    heading = "SERVING STABILITY" if kind == "serving" else "DECISION STABILITY"
    lines = [
        f"  {heading}",
        f"    {stability['runs']} generations at temperature {stability['temperature']} "
        f"over {stability['sampled']} sampled items",
    ]
    if kind == "serving":
        lines.append("    expect ~100% — a deterministic request should answer once")
    for row in stability["fields"]:
        if row["stability"] is None:
            continue
        lines.append(
            f"    {row['field']:<22} {row['stability']:>7.1%}   {row['agreed']} of "
            f"{row['compared']} agreed"
        )
        for flip in row["flipped"][:2]:
            lines.append(f"      {flip['item_id']:<8} {' / '.join(flip['answers'])[:52]}")
    if stability.get("reads_as_broken"):
        lines += _wrapped(str(stability["reads_as_broken"]), indent=4)
    if stability.get("note"):
        lines += _wrapped(str(stability["note"]), indent=4)
    return lines + [""]


def _free_text_gate(gate: Mapping[str, Any]) -> list[str]:
    """What the free checks saved, and what they cost in missed defects."""
    if not gate or not gate.get("fields"):
        return []
    flagged, audited = gate.get("flagged") or {}, gate.get("audited") or {}
    judged = flagged.get("judged", 0) + audited.get("judged", 0)
    if not judged:
        return []

    lines = ["  FREE-TEXT GATE"]
    for name, counts in sorted(gate["fields"].items()):
        lines.append(
            f"    {name:<22} {counts['flagged']} flagged by a free check, "
            f"{counts['audited']} audited"
        )
    if gate.get("precision") is not None:
        lines.append(
            f"    of the flagged rows        {gate['precision']:.0%} had an unsupported claim"
        )
    if gate.get("gate_note"):
        lines += _wrapped(str(gate["gate_note"]), indent=4)
    if gate.get("why"):
        lines += _wrapped(str(gate["why"]), indent=4)
    return lines + [""]


def _calibration(calibration: Mapping[str, Any]) -> list[str]:
    """What was fitted, whether it helped, and whether it changed anything."""
    if not calibration:
        return []
    lines = ["  CALIBRATION"]
    if not calibration.get("fitted"):
        lines += _wrapped(str(calibration.get("why", "nothing was fitted")), indent=4)
        for name, fit in sorted((calibration.get("fields") or {}).items()):
            for note in fit.get("notes") or []:
                lines += _wrapped(f"{name}: {note}", indent=6)
        return lines + [""]

    for name, fit in sorted((calibration.get("fields") or {}).items()):
        if not fit.get("fitted"):
            lines.append(f"    {name:<22} not fitted")
            for note in fit.get("notes") or []:
                lines += _wrapped(note, indent=6)
            continue
        ece, brier, base = fit.get("ece"), fit.get("brier"), fit.get("brier_base_rate")
        mark = "✗" if (ece or 0) > 0.10 else "✓"
        lines.append(
            f"    {name:<22} n={fit['n']}   ECE {ece:.3f} {mark}   "
            f"Brier {brier:.3f} vs {base:.3f} base rate"
        )
        for note in fit.get("notes") or []:
            lines += _wrapped(note, indent=6)
    if calibration.get("why"):
        lines += _wrapped(str(calibration["why"]), indent=4)
    return lines + [""]


def operating_point_table(evaluation: Mapping[str, Any]) -> list[str]:
    """The deliverable: what a review budget actually buys you.

    DESIGN.md §7 is explicit that this, not AUC, is the primary metric — it
    is the question a person with five hundred review-hours actually has.
    """
    if not evaluation:
        return []
    if evaluation.get("skipped"):
        return ["  OPERATING POINT", *_wrapped(str(evaluation["skipped"]), indent=4), ""]

    strategies = evaluation.get("strategies") or []
    if not strategies:
        return []
    lines = [
        "  OPERATING POINT   Error Recall@Budget",
        f"    {evaluation['target_errors']} known errors in "
        f"{evaluation['target_items']} labelled items",
        "",
        f"    {'strategy':<20}{'budget':>7}{'reviewed':>10}{'found':>8}"
        f"{'recall':>9}{'wasted':>9}",
    ]
    for entry in strategies:
        tag = "" if entry["baseline"] else "  ← the judge"
        for point in entry["points"]:
            recall = point["error_recall"]
            wasted = point["wasted"]
            lines.append(
                f"    {entry['strategy']:<20}{point['budget']:>6.1%}"
                f"{point['n_reviewed']:>10,}{_found(point['errors_found']):>8}"
                f"{_pct_or(recall):>9}{_pct_or(wasted):>9}{tag}"
            )
            tag = ""
        lines.append("")

    judge = next((e for e in strategies if not e["baseline"]), None)
    if judge and judge["unranked_errors"]:
        lines += _wrapped(
            f"{judge['unranked']} labelled items could not be ranked and "
            f"{judge['unranked_errors']} of them are errors — never reviewed at any "
            "budget, and excluded from every recall above.",
            indent=4,
        )
    return lines + [""]


def _pct_or(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _found(value: float) -> str:
    """A tied block split across the cutoff yields a fraction of an error.

    Rounding 0.4 to "0" beside a recall of 8% reads as a contradiction; it
    is the expected count under fair tie-breaking, and showing the decimal
    is what makes the two agree.
    """
    return f"{value:.0f}" if float(value).is_integer() else f"{value:.1f}"


def _review(
    risk_rows: Sequence[RiskRow],
    verdicts: Sequence[Verdict],
    *,
    ranked_by: str = "",
    strategy: str = "",
    calibrated: bool = False,
) -> list[str]:
    reasons = _reasons(verdicts, ranked_by=ranked_by)
    ranked = [row for row in risk_rows if row.triage_score is not None]
    unranked = [row for row in risk_rows if row.triage_score is None]
    lines: list[str] = []

    if ranked:
        if calibrated:
            lines.append(
                f"  ranked by {strategy} — a fitted probability of error, not a feeling"
            )
        else:
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


def _cannot(cannot_tell: Sequence[str], *, subject: str = "RUN") -> list[str]:
    if not cannot_tell:
        return []
    title = f"  ┌ WHAT THIS {subject} CANNOT TELL YOU "
    lines = [title + "─" * max(3, RULE + 2 - len(title)) + "┐"]
    for line in cannot_tell:
        head, _, tail = line.partition("\n")
        wrapped_head = _wrapped(head, indent=0, width=RULE - 8) or [head]
        lines.append(f"  │  ✗ {wrapped_head[0]}")
        for extra in wrapped_head[1:]:
            lines.append(f"  │    {extra}")
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


def _reasons(verdicts: Sequence[Verdict], *, ranked_by: str = "") -> dict[str, str]:
    """The sentence behind the score, which is what a reviewer actually reads.

    Only the ranking judge's. A panel member's reason sitting next to a
    triage score explains a number nobody computed — and the two can
    contradict each other outright, which reads as the tool confusing itself.
    The panel's view of the same item is in its own findings.
    """
    worst: dict[str, tuple[int, str]] = {}
    rank = {"fail": 0, "unscored": 1, "pass": 2}
    for verdict in verdicts:
        if ranked_by and verdict.judge_id != ranked_by:
            continue
        score = rank.get(verdict.status.value, 3)
        current = worst.get(verdict.item_id)
        if current is None or score < current[0]:
            worst[verdict.item_id] = (score, f"{verdict.field}: {verdict.reason}")
    return {item_id: reason for item_id, (_, reason) in worst.items()}


def comparison_report(comparison: Any) -> list[str]:
    """Two runs side by side, and the conclusion this command will not draw."""
    a, b = comparison.a, comparison.b
    lines = [
        "llm-expectations compare",
        "",
        f"  A  {a.run_id:<32} {a.items:>7,} items   prompt {a.prompt or '—'}",
        f"  B  {b.run_id:<32} {b.items:>7,} items   prompt {b.prompt or '—'}",
        "",
    ]
    for note in comparison.notes:
        lines.append("  ⚠ " + _wrapped(note, indent=0)[0])
        lines += [f"    {line}" for line in _wrapped(note, indent=0)[1:]]
    if comparison.notes:
        lines.append("")

    for entry in comparison.fields:
        header = f"  ── {entry.field} "
        lines.append(header + "─" * max(3, RULE - len(header)))
        lines.append("")
        if entry.deltas:
            lines.append(f"    {'':<26}{'A':>9}{'B':>9}{'change':>11}")
        for delta in entry.deltas:
            lines.append(
                f"    {delta.name:<26}{_cell(delta.before, delta.kind):>9}"
                f"{_cell(delta.after, delta.kind):>9}{_change(delta):>11}"
            )
        if entry.deltas:
            lines.append("")

        if entry.moved_labels:
            lines.append("    labels that moved")
            for label, move in entry.moved_labels:
                if abs(move) < 0.0001:
                    continue
                lines.append(f"      {label:<36} {move * 100:+.1f}pp")
            lines.append("")

        if entry.compared:
            lines.append(f"    items that moved          {entry.compared:,} shared")
            lines.append(f"      improved                {len(entry.improved):>6,}")
            lines.append(f"      regressed               {len(entry.regressed):>6,}")
            lines.append(f"      unchanged               {entry.unchanged:>6,}")
            if entry.only_in_a or entry.only_in_b:
                lines.append(
                    f"      only in one run         {entry.only_in_a:,} in A, "
                    f"{entry.only_in_b:,} in B — not compared"
                )
            if entry.untranslatable:
                lines.append(
                    f"      no equivalent label     {entry.untranslatable} excluded by "
                    "the migration"
                )
            lines.append("")

    lines += _cannot(comparison.cannot_tell, subject="COMPARISON")
    return lines


def _cell(value: float | None, kind: str) -> str:
    if value is None:
        return "—"
    return f"{value:.1%}" if kind == "share" else f"{value:.2f}"


def _change(delta: Any) -> str:
    change = delta.change
    if change is None:
        return "—"
    return f"{change * 100:+.1f}pp" if delta.kind == "share" else f"{change:+.2f}"
