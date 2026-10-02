"""Free checks on a sentence written *about* an item.

For an assigned field you can hold up the right answer. For free text there
is no single right answer — two summaries can be equally good and share
almost no words — so comparing to one gold summary would score word choice.
The question is not *is it right?* but *is anything wrong with it?*, and these
are the defects you can find without paying anyone (DESIGN.md §8).

Of the five ways a summary goes bad, four are caught here for free. Only
"made up" needs a judge, and it is the one that matters most, so it is worth
paying for.

The surprise is that **free text is cheaper than assigned**, which is the
opposite of what you would guess. Specificity and cross-field agreement are
genuinely predictive — a summary with no detail from its own item, or one
that contradicts its label, is usually the one that invented something — so
unlike the assigned checks, these *do* gate the expensive one.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Sequence

from ..schema import FieldKind, FieldSpec
from ..types import Finding, Grain, Status
from .base import CheckContext, finding, register

__all__ = [
    "boilerplate",
    "copy_ratio",
    "length_in_bounds",
    "longest_shared_run",
    "rare_tokens",
    "specificity",
]

FREE_TEXT = (FieldKind.FREE_TEXT,)
_WORD = re.compile(r"[a-z0-9][a-z0-9'’\-]*")

#: A token in more than this share of items is common enough to be furniture.
#: "Card" in a billing corpus says nothing about *which* session you are
#: reading; "4471" does.
RARE_SHARE = 0.05

#: Words per shingle for near-duplicate detection. Five is long enough that
#: an accidental collision needs five consecutive words in common.
SHINGLE = 5

#: Two outputs sharing this share of their shingles are the same sentence
#: with the names changed.
BOILERPLATE_SIMILARITY = 0.8


def words(text: object) -> list[str]:
    return _WORD.findall(str(text).lower()) if text is not None else []


def _text_of(ctx: CheckContext, field: str) -> dict[str, str]:
    out = {}
    for item_id, output in ctx.outputs.items():
        value = output.get(field)
        if isinstance(value, str) and value.strip():
            out[item_id] = value
    return out


def _no_bounds(spec: FieldSpec, ctx: CheckContext) -> str | None:
    low = ctx.resolve("min_words", spec).value
    high = ctx.resolve("max_words", spec).value
    if low is None and high is None:
        return "neither min_words nor max_words is set for this field"
    return None


@register("length_in_bounds", Grain.FIELD, FREE_TEXT, applies=_no_bounds)
def length_in_bounds(spec: FieldSpec, ctx: CheckContext) -> Sequence[Finding]:
    """Is it the right shape? Three paragraphs where a sentence was asked for."""
    low = ctx.resolve("min_words", spec)
    high = ctx.resolve("max_words", spec)
    findings = []
    for item_id, output in ctx.outputs.items():
        value = output.get(spec.name)
        if value is None or not str(value).strip():
            findings.append(
                finding(
                    ctx,
                    check="length_in_bounds",
                    grain=Grain.FIELD,
                    status=Status.FAIL,
                    item_id=item_id,
                    field=spec.name,
                    score=0.0,
                    evidence={"why": "empty — nothing was written for this field"},
                )
            )
            continue
        n = len(words(value))
        too_short = low.value is not None and n < low.value
        too_long = high.value is not None and n > high.value
        why = None
        if too_short:
            why = f"{n} words, under the {low.value} asked for"
        elif too_long:
            why = f"{n} words, over the {high.value} asked for"
        findings.append(
            finding(
                ctx,
                check="length_in_bounds",
                grain=Grain.FIELD,
                status=Status.FAIL if why else Status.PASS,
                item_id=item_id,
                field=spec.name,
                score=float(n),
                threshold=high if too_long or not too_short else low,
                evidence={"words": n} | ({"why": why} if why else {}),
            )
        )
    return findings


def rare_tokens(items: Iterable[tuple[str, str]], *, share: float = RARE_SHARE) -> set[str]:
    """Tokens that appear in fewer than ``share`` of items.

    No NLP library and nothing trained. A token in most items is furniture —
    it cannot tell you which session you are reading — and the ones that can
    are exactly the ones a specific summary would reach for.
    """
    rows = list(items)
    if not rows:
        return set()
    document_count: Counter[str] = Counter()
    for _, text in rows:
        document_count.update(set(words(text)))
    ceiling = max(1, int(share * len(rows)))
    return {token for token, n in document_count.items() if n <= ceiling}


@register("specificity", Grain.FIELD, FREE_TEXT)
def specificity(spec: FieldSpec, ctx: CheckContext) -> Sequence[Finding]:
    """Does it say anything only true of *this* item, or would it fit any?

    "The customer had a problem and it was resolved" is true of every session
    in the corpus. The check asks whether the output reaches for any of the
    rare tokens its own item offered.
    """
    texts = _text_of(ctx, spec.name)
    rare = rare_tokens((i, item.text) for i, item in ctx.items.items())
    threshold = ctx.resolve("min_specificity", spec)
    findings = []
    for item_id, written in texts.items():
        item = ctx.items.get(item_id)
        if item is None:
            continue
        available = rare & set(words(item.text))
        used = sorted(available & set(words(written)))
        if not available:
            # The item offered nothing distinctive, so a generic summary of it
            # is not the model's fault. Failing it would punish the output for
            # the corpus being repetitive.
            findings.append(
                finding(
                    ctx,
                    check="specificity",
                    grain=Grain.FIELD,
                    status=Status.UNSCORED,
                    item_id=item_id,
                    field=spec.name,
                    evidence={
                        "why": "the item itself holds no token rare in this corpus, so "
                        "there was nothing specific available to use"
                    },
                )
            )
            continue
        findings.append(
            finding(
                ctx,
                check="specificity",
                grain=Grain.FIELD,
                status=Status.PASS if used else Status.FAIL,
                item_id=item_id,
                field=spec.name,
                score=float(len(used)),
                threshold=threshold,
                evidence={"used": used[:5]}
                | (
                    {}
                    if used
                    else {
                        "why": "nothing here is specific to this item — it would fit any "
                        "session in the corpus",
                        "available": sorted(available)[:8],
                    }
                ),
            )
        )
    return findings


def longest_shared_run(source: Sequence[str], written: Sequence[str]) -> int:
    """Longest run of words appearing verbatim in both, in order."""
    if not source or not written:
        return 0
    previous = [0] * (len(written) + 1)
    best = 0
    for token in source:
        current = [0] * (len(written) + 1)
        for j, other in enumerate(written, start=1):
            if token == other:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


@register("copy_ratio", Grain.FIELD, FREE_TEXT)
def copy_ratio(spec: FieldSpec, ctx: CheckContext) -> Sequence[Finding]:
    """Is it summarising, or pasting the source back?

    The longest verbatim run as a share of the output. A summary that is
    mostly one lifted passage has not compressed anything, and it will be
    inside its word bounds while doing it.
    """
    threshold = ctx.resolve("max_copy_ratio", spec)
    findings = []
    for item_id, written in _text_of(ctx, spec.name).items():
        item = ctx.items.get(item_id)
        if item is None:
            continue
        tokens = words(written)
        if not tokens:
            continue
        run = longest_shared_run(words(item.text), tokens)
        ratio = run / len(tokens)
        findings.append(
            finding(
                ctx,
                check="copy_ratio",
                grain=Grain.FIELD,
                status=Status.FAIL if ratio > threshold.value else Status.PASS,
                item_id=item_id,
                field=spec.name,
                score=ratio,
                threshold=threshold,
                evidence={"longest_run": run, "words": len(tokens)}
                | (
                    {}
                    if ratio <= threshold.value
                    else {
                        "why": f"{run} of {len(tokens)} words are one unbroken run lifted "
                        "from the item — that is pasting, not summarising"
                    }
                ),
            )
        )
    return findings


def _shingles(tokens: Sequence[str], size: int = SHINGLE) -> set[str]:
    if len(tokens) < size:
        return {hashlib.blake2b(" ".join(tokens).encode(), digest_size=8).hexdigest()}
    return {
        hashlib.blake2b(" ".join(tokens[i : i + size]).encode(), digest_size=8).hexdigest()
        for i in range(len(tokens) - size + 1)
    }


@register("boilerplate", Grain.CORPUS, FREE_TEXT)
def boilerplate(spec: FieldSpec, ctx: CheckContext) -> Sequence[Finding]:
    """How many outputs are near-duplicates of another output?

    A model that has fallen into a template produces forty-seven summaries
    that are the same sentence with the names changed, and every one of them
    passes length, specificity and copy ratio.

    Found through shared word shingles rather than by comparing every pair,
    so a corpus of half a million does not become a quadratic sweep: two
    outputs only get compared when they already share a run of five words.
    """
    texts = _text_of(ctx, spec.name)
    if len(texts) < 2:
        return []

    by_shingle: dict[str, list[str]] = {}
    sets: dict[str, set[str]] = {}
    for item_id, written in texts.items():
        shingles = _shingles(words(written))
        sets[item_id] = shingles
        for shingle in shingles:
            by_shingle.setdefault(shingle, []).append(item_id)

    candidates: dict[str, set[str]] = {}
    for sharing in by_shingle.values():
        if len(sharing) < 2 or len(sharing) > 200:
            # A shingle shared by hundreds is a stock phrase, not evidence
            # that any particular pair is a duplicate.
            continue
        for left in sharing:
            candidates.setdefault(left, set()).update(x for x in sharing if x != left)

    duplicates: set[str] = set()
    examples: list[dict[str, object]] = []
    for left, others in sorted(candidates.items()):
        for right in sorted(others):
            if left >= right:
                continue
            union = sets[left] | sets[right]
            if not union:
                continue
            similarity = len(sets[left] & sets[right]) / len(union)
            if similarity >= BOILERPLATE_SIMILARITY:
                duplicates.update({left, right})
                if len(examples) < 3:
                    examples.append(
                        {"a": left, "b": right, "similarity": round(similarity, 3)}
                    )

    share = len(duplicates) / len(texts)
    return [
        finding(
            ctx,
            check="boilerplate",
            grain=Grain.CORPUS,
            status=Status.FAIL if duplicates else Status.PASS,
            field=spec.name,
            score=share,
            evidence={"near_duplicates": len(duplicates), "of": len(texts), "pairs": examples}
            | (
                {}
                if not duplicates
                else {
                    "why": f"{len(duplicates)} of {len(texts)} outputs are near-identical "
                    "to another — the model has fallen into a template"
                }
            ),
        )
    ]
