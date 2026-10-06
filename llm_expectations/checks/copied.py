"""Checks for a value that is supposed to be *in* the document.

The third kind, and the one with a strong free gate. For an assigned label
every free check is about format, so a well-formed wrong label sails through
and the judge has to see everything. Here, **a value that is not in the text
was invented**, and that is free to establish — so you only pay to judge the
suspicious ones (DESIGN.md §5, and §9 for this kind in full).

Two things keep that gate honest.

**Presence is not correctness.** A document mentioning five amounts will
contain the one the model reported whether or not it is the right one. So
``value_in_source`` carries how many candidates it found, and the gate sends
ambiguous rows to a judge even though they passed. Treating a lucky match as
a verified value is the trap this kind invites.

**Matching is by meaning, not by characters.** ``$1,234.50`` and ``1234.5``
are the same amount; ``March 3, 2026`` and ``2026-03-03`` are the same date.
A checker that compared characters would report a model as inventing values
every time it tidied a format, and the resulting grounding rate would measure
formatting rather than fidelity. ``require_verbatim`` is there for the fields
where the characters really do matter — an invoice number, a SKU.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date

from ..schema import FieldKind, FieldSpec, ValueType
from ..types import Finding, Grain, Status
from .base import CheckContext, finding, register

__all__ = [
    "candidates",
    "normalise",
    "parse_date",
    "parse_number",
    "value_in_source",
    "value_shape",
]

COPIED = (FieldKind.COPIED,)

_WS = re.compile(r"\s+")
_NUMBER = re.compile(r"[-+]?\$?\s?\d[\d,. ']*\d|\b\d\b")
_MONTHS = {
    name: index
    for index, names in enumerate(
        [
            ("jan", "january"), ("feb", "february"), ("mar", "march"),
            ("apr", "april"), ("may",), ("jun", "june"), ("jul", "july"),
            ("aug", "august"), ("sep", "sept", "september"), ("oct", "october"),
            ("nov", "november"), ("dec", "december"),
        ],
        start=1,
    )
    for name in names
}
_MONTH_NAMES = "|".join(sorted(_MONTHS, key=len, reverse=True))

#: The date shapes this understands. Anything else is reported as unparseable
#: rather than guessed at — a date silently read the wrong way round is worse
#: than one the tool admits it cannot read.
_DATE_PATTERNS = (
    re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})\b"),
    re.compile(
        rf"\b(?P<mon>{_MONTH_NAMES})\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<y>\d{{4}})\b",
        re.I,
    ),
    re.compile(
        rf"\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<mon>{_MONTH_NAMES})\.?,?\s+(?P<y>\d{{4}})\b",
        re.I,
    ),
)


def normalise(text: str) -> str:
    """Case-folded, whitespace-collapsed. The weakest useful equivalence."""
    return _WS.sub(" ", str(text)).strip().lower()


def parse_number(text: str) -> float | None:
    """Read an amount, ignoring currency symbols and thousands separators."""
    cleaned = re.sub(r"[^\d.\-+]", "", str(text).replace(",", ""))
    if not cleaned or cleaned in {"-", "+", "."}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_date(text: str) -> date | None:
    """Read a date in one of the understood shapes, or admit it cannot."""
    for pattern in _DATE_PATTERNS:
        match = pattern.search(str(text))
        if match is None:
            continue
        parts = match.groupdict()
        month = (
            int(parts["m"])
            if parts.get("m")
            else _MONTHS.get(str(parts.get("mon", "")).lower().rstrip("."), 0)
        )
        try:
            return date(int(parts["y"]), month, int(parts["d"]))
        except ValueError:
            return None
    return None


def candidates(text: str, value_type: ValueType) -> list[str]:
    """Every span of the item that could have been the answer.

    This is what turns "the value is in the document" from proof into
    evidence. One candidate and a match is strong; twelve candidates and a
    match says only that the model picked a number from a page of numbers.
    """
    if value_type is ValueType.NUMBER:
        # Dates are made of numbers. Counting the 3 and the 2026 out of
        # "March 3, 2026" as candidate amounts would inflate the ambiguity
        # of every document that carries a date, which is all of them.
        masked = text
        for pattern in _DATE_PATTERNS:
            masked = pattern.sub(lambda m: " " * len(m.group(0)), masked)
        return [m.group(0).strip() for m in _NUMBER.finditer(masked)]
    if value_type is ValueType.DATE:
        found: list[str] = []
        for pattern in _DATE_PATTERNS:
            found.extend(m.group(0) for m in pattern.finditer(text))
        return found
    return []


def _found(value: str, item_text: str, spec_type: ValueType, verbatim: bool) -> bool | None:
    """Is the value in the text? ``None`` when the comparison cannot be made."""
    if verbatim:
        return value.strip() in item_text

    if spec_type is ValueType.NUMBER:
        wanted = parse_number(value)
        if wanted is None:
            return None
        return any(parse_number(c) == wanted for c in candidates(item_text, spec_type))

    if spec_type is ValueType.DATE:
        wanted_date = parse_date(value)
        if wanted_date is None:
            return None
        return any(parse_date(c) == wanted_date for c in candidates(item_text, spec_type))

    return normalise(value) in normalise(item_text)


def _has_value(output: object, field: str) -> bool:
    value = output.get(field)  # type: ignore[attr-defined]
    return value is not None and str(value).strip() != ""


@register("value_shape", Grain.FIELD, COPIED)
def value_shape(spec: FieldSpec, ctx: CheckContext) -> Sequence[Finding]:
    """Does the value even look like what the field declares it to be?

    A ``number`` field holding "pending" is a different failure from a number
    that is simply absent from the document, and it points somewhere else:
    the model answered a different question rather than misreading this one.
    """
    kind = spec.value_type or ValueType.TEXT
    if kind is ValueType.TEXT:
        return []
    reader = parse_number if kind is ValueType.NUMBER else parse_date
    findings = []
    for item_id, output in ctx.outputs.items():
        if not _has_value(output, spec.name):
            continue
        raw = str(output.get(spec.name))
        readable = reader(raw) is not None
        findings.append(
            finding(
                ctx,
                check="value_shape",
                grain=Grain.FIELD,
                status=Status.PASS if readable else Status.FAIL,
                item_id=item_id,
                field=spec.name,
                evidence={"value": raw, "expected": kind.value}
                | (
                    {}
                    if readable
                    else {
                        "why": f"{raw!r} cannot be read as a {kind.value}. That is not a "
                        "value copied wrongly — it is an answer to a different question."
                    }
                ),
            )
        )
    return findings


@register("value_in_source", Grain.FIELD, COPIED)
def value_in_source(spec: FieldSpec, ctx: CheckContext) -> Sequence[Finding]:
    """Is the value actually in the document it was supposedly copied from?

    The strong free gate this kind gets and the other two do not. A value the
    document does not contain was invented, and establishing that costs
    nothing.

    What it cannot establish is that a value *found* is the **right** one.
    The candidate count rides along in the evidence so the gate can send the
    ambiguous rows to a judge rather than banking a coincidence.
    """
    kind = spec.value_type or ValueType.TEXT
    verbatim = bool(ctx.resolve("require_verbatim", spec).value)
    findings = []
    for item_id, output in ctx.outputs.items():
        item = ctx.items.get(item_id)
        if item is None or not _has_value(output, spec.name):
            continue
        raw = str(output.get(spec.name))
        present = _found(raw, item.text, kind, verbatim)
        if present is None:
            findings.append(
                finding(
                    ctx,
                    check="value_in_source",
                    grain=Grain.FIELD,
                    status=Status.UNSCORED,
                    item_id=item_id,
                    field=spec.name,
                    evidence={
                        "value": raw,
                        "why": f"{raw!r} could not be read as a {kind.value}, so there is "
                        "nothing to look for in the document. value_shape owns this row.",
                    },
                )
            )
            continue
        found = candidates(item.text, kind)
        findings.append(
            finding(
                ctx,
                check="value_in_source",
                grain=Grain.FIELD,
                status=Status.PASS if present else Status.FAIL,
                item_id=item_id,
                field=spec.name,
                score=1.0 if present else 0.0,
                evidence={"value": raw, "candidates": len(found), "verbatim": verbatim}
                | (
                    {}
                    if present
                    else {
                        "why": "the document does not contain this value. A value that is "
                        "not in the source was invented — this is the one defect this "
                        "kind of field can prove for free.",
                        "nearby": found[:5],
                    }
                ),
            )
        )
    return findings


def ambiguous(spec: FieldSpec, ctx: CheckContext, item_id: str) -> bool:
    """Would finding the value in this document have been luck?

    Used by the gate rather than reported as a defect, because a document
    with many candidates is not a fault in the output — it is a limit on
    what the free check can tell you about it.
    """
    item = ctx.items.get(item_id)
    if item is None:
        return False
    ceiling = int(ctx.resolve("ambiguous_above", spec).value)
    return len(candidates(item.text, spec.value_type or ValueType.TEXT)) > ceiling


@register("grounding_rate", Grain.CORPUS, COPIED)
def grounding_rate(spec: FieldSpec, ctx: CheckContext) -> Sequence[Finding]:
    """What share of values the document actually contains.

    The headline for a copied field, and free. It is a floor rather than a
    score: a judge still has to say whether the values that *are* there are
    the right ones.
    """
    rows = [
        output
        for output in ctx.outputs.values()
        if _has_value(output, spec.name)
    ]
    if not rows:
        return []
    kind = spec.value_type or ValueType.TEXT
    verbatim = bool(ctx.resolve("require_verbatim", spec).value)
    threshold = ctx.resolve("min_grounding", spec)

    checked = grounded = 0
    for item_id, output in ctx.outputs.items():
        item = ctx.items.get(item_id)
        if item is None or not _has_value(output, spec.name):
            continue
        present = _found(str(output.get(spec.name)), item.text, kind, verbatim)
        if present is None:
            continue
        checked += 1
        grounded += int(present)
    if not checked:
        return []

    rate = grounded / checked
    return [
        finding(
            ctx,
            check="grounding_rate",
            grain=Grain.CORPUS,
            status=Status.PASS if rate >= threshold.value else Status.FAIL,
            field=spec.name,
            score=rate,
            threshold=threshold,
            evidence={"grounded": grounded, "of": checked}
            | (
                {}
                if rate >= threshold.value
                else {
                    "why": f"{checked - grounded} of {checked} values are not in their "
                    "documents at all. This is a floor, not a score — the ones that are "
                    "there still have to be the right ones."
                }
            ),
        )
    ]
