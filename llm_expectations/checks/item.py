"""Cross-field consistency: free, item grain, and underrated.

If the summary talks about logging in and the label says `payment_failed`, one
of them is wrong. The check does not know which, and that is the finding
rather than a failure of it — it tells you to look, and both candidates are in
the evidence.

It uses the taxonomy, which is the fourth reader of that file: a label's
definition, examples and boundary cases are the vocabulary you would expect a
summary about that label to draw on. No NLP library, and nothing trained.

This check belongs to neither field kind. It sits above both and reads
whatever fields the schema names in ``must_agree_with``, which is why it lives
here rather than in `assigned.py` or `free_text.py`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..schema import FieldKind, FieldSpec
from ..taxonomy import Taxonomy
from ..types import Finding, Grain, Status
from .base import CheckContext, finding, register

__all__ = ["cross_field_agreement", "vocabulary"]

_WORD = re.compile(r"[a-z0-9]+")

#: Words too common to carry a signal. Short, and deliberately not a
#: linguistics project: anything longer starts encoding opinions about the
#: domain, and the check is meant to be legible rather than clever.
STOPWORDS = frozenset(
    """
    a an and any anything are as at be been being but by can cannot could did do does
    for from get gets getting go goes had has have how in into is it its just may might
    more most no not of off on one only or other our out over same should so some such
    than that the their them then there these they this those through to too up use used
    using was way we were what when where which who why will with within would you your
    about after all also back because before does etc else ever every from here if into
    like make made many much must never new now often once other own per said say see
    still take than though time under until very want wants well what while
    """.split()
)

MIN_WORD = 3


def vocabulary(taxonomy: Taxonomy, label: str) -> set[str]:
    """The words a summary about this label would plausibly use.

    Drawn from the label's own name, definition and examples, plus its
    ancestors' — a child inherits the vocabulary of its branch, so a summary
    about a declined card is not flagged for failing to say "billing".

    ``not_this`` is deliberately excluded. Those sentences describe what the
    label is *not*, and counting their words as evidence of agreement would
    reward a summary for matching the boundary it was supposed to fall outside.
    """
    words: set[str] = set()
    for path in (*taxonomy.ancestors(label), label):
        node = taxonomy.get(path)
        for text in (node.name, node.path.replace(".", " "), node.definition, *node.examples):
            words |= {
                word
                for word in _WORD.findall(text.lower())
                if len(word) >= MIN_WORD and word not in STOPWORDS
            }
    return words


def _pairs_declared(spec: FieldSpec, ctx: CheckContext) -> str | None:
    if not spec.must_agree_with:
        return "no must_agree_with declared for this field"
    return None


@register(
    "cross_field_agreement",
    Grain.ITEM,
    (FieldKind.ASSIGNED, FieldKind.FREE_TEXT),
    applies=_pairs_declared,
)
def cross_field_agreement(spec: FieldSpec, ctx: CheckContext) -> Sequence[Finding]:
    """Does this field's text draw on the vocabulary of the label it names?"""
    findings: list[Finding] = []
    for other in spec.must_agree_with:
        partner = ctx.schema[other]
        taxonomy = ctx.taxonomy_for(other)
        for item_id, output in ctx.outputs.items():
            findings.append(
                _one(ctx, spec, partner.name, taxonomy, item_id, output)
            )
    return findings


def _one(
    ctx: CheckContext,
    spec: FieldSpec,
    other: str,
    taxonomy: Taxonomy | None,
    item_id: str,
    output: object,
) -> Finding:
    text = output.get(spec.name)  # type: ignore[attr-defined]
    label = output.get(other)  # type: ignore[attr-defined]

    def unscored(why: str) -> Finding:
        return finding(
            ctx,
            check="cross_field_agreement",
            grain=Grain.ITEM,
            status=Status.UNSCORED,
            item_id=item_id,
            field=spec.name,
            evidence={"against": other, "why": why},
        )

    if not isinstance(text, str) or not text.strip():
        return unscored(f"{spec.name} is empty, so there is nothing to compare")
    if output.is_abstain(other):  # type: ignore[attr-defined]
        return unscored(
            f"{other} abstained, so there is no label whose vocabulary to compare against"
        )
    if taxonomy is None or not isinstance(label, str) or label not in taxonomy:
        # An invented label has no definition to draw vocabulary from.
        # label_in_taxonomy owns that row; this check cannot speak to it.
        return unscored(
            f"{label!r} is not a label in {taxonomy.ref if taxonomy else 'any taxonomy'}, "
            "so it carries no vocabulary to compare against"
        )

    expected = vocabulary(taxonomy, label)
    words = {w for w in _WORD.findall(text.lower()) if len(w) >= MIN_WORD and w not in STOPWORDS}
    shared = sorted(words & expected)
    return finding(
        ctx,
        check="cross_field_agreement",
        grain=Grain.ITEM,
        status=Status.PASS if shared else Status.FAIL,
        item_id=item_id,
        field=spec.name,
        score=float(len(shared)),
        threshold=1,
        evidence={"against": other, "label": label, "shared": shared}
        | (
            {}
            if shared
            else {
                "why": f"nothing in {spec.name} draws on the vocabulary of {label}. One of "
                "the two fields is wrong, and this check does not know which — both are "
                "worth opening.",
                "label_vocabulary": sorted(expected)[:12],
            }
        ),
    )
