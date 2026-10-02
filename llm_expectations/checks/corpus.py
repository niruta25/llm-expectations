"""Checks on the batch, not on any row in it.

These are the ones that only exist across a corpus. An abstention rate, a
label that swallowed everything, a distribution that moved since last week —
none of them are visible one item at a time, and all of them are free.

Drift is the one that catches a prompt change nobody told you about, which is
why it refuses to run across a taxonomy version change rather than drawing you
a line between two things that are not comparable.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from ..schema import FieldKind, FieldSpec
from ..types import Finding, Grain, Status
from .base import CheckContext, finding, register

__all__ = ["abstention_rate", "distribution", "drift", "label_shares"]

ASSIGNED = (FieldKind.ASSIGNED,)


def label_shares(ctx: CheckContext, field: str) -> dict[str, float]:
    """The label distribution over rows that actually carry a label.

    Abstentions are excluded from the denominator on purpose: they have their
    own check, and leaving them in would make a taxonomy gap look like every
    label shrinking at once.
    """
    counts = Counter(
        str(output.get(field))
        for output in ctx.outputs.values()
        if not output.is_abstain(field)
    )
    total = sum(counts.values())
    return {label: n / total for label, n in counts.items()} if total else {}


def _enough_rows(spec: FieldSpec, ctx: CheckContext) -> str | None:
    if not ctx.outputs:
        return "no output rows"
    return None


@register("abstention_rate", Grain.CORPUS, ASSIGNED, applies=_enough_rows)
def abstention_rate(spec: FieldSpec, ctx: CheckContext) -> Sequence[Finding]:
    """Is the abstention rate inside the band, and which way did it miss?

    The two directions mean opposite things — too high is a gap in the
    taxonomy, too low is the model forcing a label onto unclear items — so a
    bare "out of range" would hide the actionable half.
    """
    if not ctx.resolve("allow_abstain", spec).value:
        return []
    threshold = ctx.resolve("abstain_rate", spec)
    low, high = threshold.value
    abstained = [i for i, o in ctx.outputs.items() if o.is_abstain(spec.name)]
    total = len(ctx.outputs)
    rate = len(abstained) / total
    evidence: dict[str, Any] = {"n": len(abstained), "of": total}

    if rate > high:
        status = Status.FAIL
        evidence["why"] = "too high — your taxonomy has a gap and these items had nowhere to go"
    elif rate >= low:
        status = Status.PASS
    elif 1 / total > low:
        # The floor is unreachable at this size: zero abstentions and one
        # abstention straddle the whole band, so nothing here distinguishes a
        # model that correctly never abstains from one that is forcing labels.
        # Failing it would be reporting an artefact of the corpus as a finding.
        status = Status.UNSCORED
        evidence["why"] = (
            f"the {low:.0%} floor cannot be tested on {total} rows — a single abstention "
            f"would already be {1 / total:.0%}, so nothing distinguishes a model that "
            "correctly never abstains from one that is forcing labels"
        )
    else:
        status = Status.FAIL
        evidence["why"] = (
            "too low — the model is forcing a label onto items that do not support one"
        )

    return [
        finding(
            ctx,
            check="abstention_rate",
            grain=Grain.CORPUS,
            status=status,
            field=spec.name,
            score=rate,
            threshold=threshold,
            evidence=evidence,
        )
    ]


@register("label_collapse", Grain.CORPUS, ASSIGNED, applies=_enough_rows)
def distribution(spec: FieldSpec, ctx: CheckContext) -> Sequence[Finding]:
    """Did one label swallow the batch?"""
    shares = label_shares(ctx, spec.name)
    if not shares:
        return []
    threshold = ctx.resolve("max_label_share", spec)
    label, share = max(shares.items(), key=lambda kv: kv[1])
    return [
        finding(
            ctx,
            check="label_collapse",
            grain=Grain.CORPUS,
            status=Status.PASS if share <= threshold.value else Status.FAIL,
            field=spec.name,
            score=share,
            threshold=threshold,
            evidence={
                "largest_label": label,
                "distinct_labels": len(shares),
                "shares": {
                    k: round(v, 4) for k, v in sorted(shares.items(), key=lambda kv: -kv[1])
                },
            },
        )
    ]


def _has_previous(spec: FieldSpec, ctx: CheckContext) -> str | None:
    if not ctx.previous:
        return "no previous run to compare against"
    if spec.name not in (ctx.previous.get("distributions") or {}):
        return "the previous run did not record a distribution for this field"
    return None


@register("drift", Grain.CORPUS, ASSIGNED, applies=_has_previous)
def drift(spec: FieldSpec, ctx: CheckContext) -> Sequence[Finding]:
    """How far did each label's share move since the last run?

    Two runs across a taxonomy version change are not comparable, and two runs
    across a prompt change are not either. The first is refused outright; the
    second is reported beside the number rather than silently trended over.
    """
    previous: Mapping[str, Any] = ctx.previous
    taxonomy = ctx.taxonomy_for(spec.name)
    reference = str(taxonomy.ref) if taxonomy else None
    seen = previous.get("taxonomies") or {}

    if reference is not None and reference not in seen:
        return [
            finding(
                ctx,
                check="drift",
                grain=Grain.CORPUS,
                status=Status.UNSCORED,
                field=spec.name,
                evidence={
                    "why": f"the previous run used {', '.join(sorted(seen)) or 'no taxonomy'} "
                    f"and this one uses {reference}. Without a migration mapping the two "
                    "distributions are not the same measurement.",
                    "previous_run": previous.get("run_id"),
                },
            )
        ]

    before: Mapping[str, float] = (previous.get("distributions") or {}).get(spec.name, {})
    after = label_shares(ctx, spec.name)
    threshold = ctx.resolve("max_drift_pp", spec)
    moves = {
        label: round(after.get(label, 0.0) - before.get(label, 0.0), 4)
        for label in sorted(set(before) | set(after))
    }
    worst = max(moves.items(), key=lambda kv: abs(kv[1])) if moves else ("", 0.0)
    prompts = (previous.get("prompt"), ctx.previous.get("current_prompt"))
    evidence: dict[str, Any] = {
        "previous_run": previous.get("run_id"),
        "moves": {k: v for k, v in sorted(moves.items(), key=lambda kv: -abs(kv[1]))},
        "largest_move": {"label": worst[0], "pp": round(worst[1] * 100, 1)},
    }
    if prompts[0] and prompts[1] and prompts[0] != prompts[1]:
        evidence["why"] = (
            f"the prompt changed from {prompts[0]} to {prompts[1]}. A distribution that "
            "moved across a prompt change is not drift — it is the change you made."
        )
    return [
        finding(
            ctx,
            check="drift",
            grain=Grain.CORPUS,
            status=Status.PASS if abs(worst[1]) <= threshold.value else Status.FAIL,
            field=spec.name,
            score=abs(worst[1]),
            threshold=threshold,
            evidence=evidence,
        )
    ]
