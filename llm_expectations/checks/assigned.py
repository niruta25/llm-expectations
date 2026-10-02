"""Field-grain checks for assigned labels.

These catch *format* problems, not *wrongness*. A perfectly formed label that
is simply the wrong one passes every check in this file — which is the honest
note in DESIGN.md §5 and the reason the triage judge has to see everything.
What they do catch is free, and each one points at a different fix.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..schema import FieldKind, FieldSpec
from ..types import ABSTAIN, Finding, Grain, Status
from .base import CheckContext, finding, register

__all__ = ["abstention_allowed", "label_in_taxonomy", "valid_leaf"]

ASSIGNED = (FieldKind.ASSIGNED,)


def _needs_taxonomy(spec: FieldSpec, ctx: CheckContext) -> str | None:
    if ctx.taxonomy_for(spec.name) is None:
        return f"no taxonomy loaded for {spec.taxonomy}"
    return None


@register("label_in_taxonomy", Grain.FIELD, ASSIGNED, applies=_needs_taxonomy)
def label_in_taxonomy(spec: FieldSpec, ctx: CheckContext) -> Sequence[Finding]:
    """Expect 100%. Anything else means the model invented a label, or the
    taxonomy version moved under the run."""
    taxonomy = ctx.taxonomy_for(spec.name)
    assert taxonomy is not None  # noqa: S101 — guaranteed by `applies`
    findings = []
    for item_id, output in ctx.outputs.items():
        if output.is_abstain(spec.name):
            continue
        value = output.get(spec.name)
        known = isinstance(value, str) and value in taxonomy
        findings.append(
            finding(
                ctx,
                check="label_in_taxonomy",
                grain=Grain.FIELD,
                status=Status.PASS if known else Status.FAIL,
                item_id=item_id,
                field=spec.name,
                threshold=1.0,
                evidence={}
                if known
                else {
                    "label": value,
                    "taxonomy": str(taxonomy.ref),
                    "why": "not a label in this taxonomy at this version — either the model "
                    "invented it, or the taxonomy moved under the run",
                    "nearest": _nearest(str(value), taxonomy),
                },
            )
        )
    return findings


def _nearest(value: str, taxonomy: object) -> list[str]:
    """The labels that do exist nearby, which is what makes the finding usable."""
    import difflib

    return difflib.get_close_matches(value, [node.path for node in taxonomy], n=3, cutoff=0.5)  # type: ignore[attr-defined]


def _leaf_required(spec: FieldSpec, ctx: CheckContext) -> str | None:
    if (missing := _needs_taxonomy(spec, ctx)) is not None:
        return missing
    if not ctx.resolve("require_leaf", spec).value:
        return "require_leaf is off for this field, so a parent label is a valid answer"
    return None


@register("valid_leaf", Grain.FIELD, ASSIGNED, applies=_leaf_required)
def valid_leaf(spec: FieldSpec, ctx: CheckContext) -> Sequence[Finding]:
    """Did it stop at the right depth?

    A label that stopped at the parent is a real label, so it is not a format
    error — it is the model hedging, and that is a different fix from a model
    that is not reading the item.
    """
    taxonomy = ctx.taxonomy_for(spec.name)
    assert taxonomy is not None  # noqa: S101 — guaranteed by `applies`
    threshold = ctx.resolve("require_leaf", spec)
    findings = []
    for item_id, output in ctx.outputs.items():
        if output.is_abstain(spec.name):
            continue
        value = output.get(spec.name)
        if not isinstance(value, str) or value not in taxonomy:
            # label_in_taxonomy owns this row. Failing it twice would make one
            # invented label look like two problems.
            continue
        node = taxonomy.get(value)
        findings.append(
            finding(
                ctx,
                check="valid_leaf",
                grain=Grain.FIELD,
                status=Status.PASS if node.is_leaf else Status.FAIL,
                item_id=item_id,
                field=spec.name,
                threshold=threshold,
                evidence={}
                if node.is_leaf
                else {
                    "label": value,
                    "why": "stopped at a parent. The model hedged rather than misread — "
                    "a different fix from a wrong branch.",
                    "children": list(node.children),
                },
            )
        )
    return findings


def _abstention_forbidden(spec: FieldSpec, ctx: CheckContext) -> str | None:
    if ctx.resolve("allow_abstain", spec).value:
        return "abstention is allowed, so the rate is checked at corpus grain instead"
    return None


@register("abstention_allowed", Grain.FIELD, ASSIGNED, applies=_abstention_forbidden)
def abstention_allowed(spec: FieldSpec, ctx: CheckContext) -> Sequence[Finding]:
    """When the field forbids abstaining, each abstention is its own failure."""
    threshold = ctx.resolve("allow_abstain", spec)
    return [
        finding(
            ctx,
            check="abstention_allowed",
            grain=Grain.FIELD,
            status=Status.FAIL,
            item_id=item_id,
            field=spec.name,
            threshold=threshold,
            evidence={
                "value": output.get(spec.name),
                "why": f"{spec.name} does not allow abstaining, and this row declined to "
                f"pick a label ({ABSTAIN!r} or null)",
            },
        )
        for item_id, output in ctx.outputs.items()
        if output.is_abstain(spec.name)
    ]
