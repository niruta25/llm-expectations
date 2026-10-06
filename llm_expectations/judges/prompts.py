"""What a judge is asked, and how the answer is read.

Four things here are deliberate (DESIGN.md §6):

**The taxonomy definitions go in the prompt.** This is why ``definition`` and
``not_this`` in ``taxonomy.yml`` are load-bearing and not documentation. A
judge shown only label names is guessing at the same boundaries the model was
guessing at.

**A reason is always required.** It is what a reviewer reads when the item
reaches them, and it is how you see *why* two judges split.

**``cannot_decide`` is allowed.** Forcing a verdict on an item that does not
say enough manufactures noise. It maps to unscored — never to "wrong".

A fifth thing, which the design implies rather than states. A judge that
rejects a label is asked which label it would assign instead. DESIGN.md §10
promises fuzzy label *pairs* from panel disagreement with no human labels, and
a yes/no verdict cannot produce a pair — the second half would have to be
guessed from sibling structure, which is the unearned inference this library
exists to catch. A few extra output tokens make the pair real, and the
suggestion is what a reviewer opening the item wants to see anyway.

**Every panel judge gets the byte-identical prompt.** Nothing here varies by
judge, so panel disagreement measures judges rather than prompt differences.

The reply is asked for as one JSON object. The design writes the instruction in
prose; JSON parses far more reliably at a 60-token budget, and an unreadable
reply is expensive here — it is counted, never defaulted, so every one of them
is a hole in the run.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..taxonomy import Taxonomy
from ..types import ABSTAIN, Item, Output, Status
from .base import JudgeRequest, ParsedReply, ReplyOutcome

__all__ = [
    "CLAIM_TOKENS",
    "ClaimSupportTask",
    "GroundednessTask",
    "LabelCorrectTask",
]

_SYSTEM = """You are checking a label another system assigned to an item.
Decide whether the label is correct.
If the item does not say enough to decide, answer cannot_decide rather than guessing.

The permitted labels are:
{labels}

Reply with one JSON object and nothing else:
{{"correct": true | false | "cannot_decide",
 "confidence": <0 to 1>,
 "reason": "<one sentence>",
 "instead": "<the label you would assign, only when correct is false>"}}"""

_USER = """ITEM:
{text}

ASSIGNED LABEL: {label}"""

_JSON = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True, slots=True)
class LabelCorrectTask:
    """Is this assigned label the right one for this item?

    ``require_leaf`` decides which labels are offered as permitted answers. A
    field that demands a leaf should not show the judge the parents, or the
    judge will start approving them.
    """

    check_id: str = "label_correct"
    max_tokens: int = 60
    require_leaf: bool = True

    def build(
        self, item: Item, output: Output, field: str, taxonomy: Taxonomy | None
    ) -> JudgeRequest:
        if taxonomy is None:
            raise ValueError(
                f"{self.check_id} needs a taxonomy for field {field!r}: the definitions are "
                "what the judge is asked to apply, and without them it is guessing at the "
                "same boundaries the model guessed at."
            )
        value = output.get(field)
        return JudgeRequest(
            system=_SYSTEM.format(labels=self._labels(taxonomy)),
            user=_USER.format(text=item.text, label=value),
            max_tokens=self.max_tokens,
            temperature=0.0,
        )

    def _labels(self, taxonomy: Taxonomy) -> str:
        paths = taxonomy.leaves() if self.require_leaf else tuple(n.path for n in taxonomy)
        lines = []
        for path in paths:
            node = taxonomy.get(path)
            lines.append(f"  {path} — {node.definition}")
            for boundary in node.not_this:
                lines.append(f"      not this: {boundary}")
        return "\n".join(lines)

    def parse(self, text: str) -> ParsedReply:
        """Read the reply, or say plainly that it could not be read."""
        match = _JSON.search(text or "")
        if match is None:
            return _unreadable("no JSON object in the reply")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return _unreadable(f"the JSON object did not parse: {exc.msg}")
        if not isinstance(payload, dict):
            return _unreadable("the reply was JSON but not an object")

        correct = payload.get("correct")
        reason = str(payload.get("reason", "")).strip()
        confidence = _confidence(payload.get("confidence"))

        if isinstance(correct, bool):
            # A suggestion only means something on a rejection. Attached to an
            # approval it is the judge contradicting itself, and carrying it
            # forward would put a phantom pair into the fuzzy-pair table.
            instead = payload.get("instead")
            detail = (
                {"instead": instead.strip()}
                if not correct and isinstance(instead, str) and instead.strip()
                else {}
            )
            return ParsedReply(
                outcome=ReplyOutcome.ANSWERED,
                status=Status.PASS if correct else Status.FAIL,
                raw_confidence=confidence,
                reason=reason,
                detail=detail,
            )
        if isinstance(correct, str) and correct.strip().lower() in {
            "cannot_decide",
            "cannot decide",
        }:
            # Unscored, never "wrong". An item nobody can resolve is not
            # evidence that the label on it is bad.
            return ParsedReply(
                outcome=ReplyOutcome.CANNOT_DECIDE,
                status=Status.UNSCORED,
                raw_confidence=confidence,
                reason=reason or "the judge could not decide from the item",
            )
        return _unreadable(f"'correct' was {correct!r}, not true, false or cannot_decide")


def _unreadable(why: str) -> ParsedReply:
    return ParsedReply(
        outcome=ReplyOutcome.UNPARSEABLE,
        status=Status.UNSCORED,
        raw_confidence=None,
        reason=f"the reply could not be read: {why}",
    )


def _confidence(value: object) -> float | None:
    """A confidence outside [0, 1], or missing, becomes None rather than a guess.

    A verdict with no confidence is still a usable verdict — it just cannot be
    ranked, and the triage row says so instead of being handed a number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if 0.0 <= number <= 1.0 else None


def label_is_judgeable(output: Output, field: str) -> bool:
    """Is there anything here for a judge to check?

    An abstention is not a label, so there is no label to be right or wrong
    about. It is a finding for the abstention-rate check, not a judge call —
    and paying to judge it would be paying for a guaranteed ``cannot_decide``.
    """
    value = output.get(field)
    return not (value is None or value == ABSTAIN or str(value).strip() == "")


_CLAIMS_SYSTEM = """You are checking a short text written about an item, against the item itself.

Split the text into its separate factual claims. For each one, decide whether the
item supports it. A claim the item neither states nor implies is NOT supported,
even if it sounds plausible.

Do not judge style, length or word choice. Only whether each claim is grounded.

Reply with one JSON object and nothing else:
{{"claims": [{{"claim": "<quoted from the text>", "supported": true | false}}],
 "missing": "<one important thing the item says that the text leaves out, or empty>"}}"""

_JUDGEMENT_SYSTEM = """You are checking a conclusion drawn about an item, against the item itself.

This is one judgement, not a list of facts. Do not split it up. Decide whether the
item supports the conclusion as a whole.

Reply with one JSON object and nothing else:
{{"claims": [{{"claim": "<the conclusion>", "supported": true | false}}],
 "missing": "<one important thing the item says that the conclusion ignores, or empty>"}}"""

_PROPOSAL_SYSTEM = """You are checking a proposed next step written about an item.

A proposal is about the future, so it is not something the item can state. Do not
check whether the item says it. Decide whether it *follows* from what the item
says — whether a reasonable person reading the item would arrive at it.

Reply with one JSON object and nothing else:
{{"claims": [{{"claim": "<the proposal>", "supported": true | false}}],
 "missing": "<something in the item the proposal overlooks, or empty>"}}"""

_CLAIMS_USER = """ITEM:
{text}

TEXT:
{written}"""

#: A claim list does not fit in the sixty tokens an assigned verdict needs.
CLAIM_TOKENS = 250

_STYLE_PROMPTS = {
    "descriptive": _CLAIMS_SYSTEM,
    "judgement": _JUDGEMENT_SYSTEM,
    "proposal": _PROPOSAL_SYSTEM,
}


@dataclass(frozen=True, slots=True)
class ClaimSupportTask:
    """Does the item actually support what was written about it?

    The one defect a free check cannot reach. Four of the five ways a summary
    goes bad — wrong length, filler, pasting, contradicting another field —
    are caught for nothing. "Made up" is not, and it is the one that matters
    most, so it is worth paying for.

    ``style`` decides the question, because the three kinds of free text are
    grounded differently (DESIGN.md §8):

        descriptive   many small facts  — split into claims, check each
        judgement     one conclusion    — do not split; ask once
        proposal      about the future  — do not ground it at all; ask
                                          whether it follows from the facts

    Asking a proposal whether the item *states* it would fail every single
    one, because a next action is by definition not in the record yet.

    The budget is larger than the assigned judge's — a claim list does not
    fit in sixty tokens — and the list rides back in ``detail`` so a reviewer
    sees which sentence was invented rather than a score.
    """

    check_id: str = "claims_supported"
    max_tokens: int = CLAIM_TOKENS
    style: str = "descriptive"

    def build(
        self, item: Item, output: Output, field: str, taxonomy: Taxonomy | None
    ) -> JudgeRequest:
        written = output.get(field)
        return JudgeRequest(
            system=_STYLE_PROMPTS.get(self.style, _CLAIMS_SYSTEM).format(),
            user=_CLAIMS_USER.format(text=item.text, written=written),
            max_tokens=self.max_tokens,
            temperature=0.0,
        )

    def parse(self, text: str) -> ParsedReply:
        match = _JSON.search(text or "")
        if match is None:
            return _unreadable("no JSON object in the reply")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return _unreadable(f"the JSON object did not parse: {exc.msg}")
        if not isinstance(payload, dict):
            return _unreadable("the reply was JSON but not an object")

        raw = payload.get("claims")
        if not isinstance(raw, list) or not raw:
            return _unreadable("'claims' was missing or empty")

        claims: list[dict[str, Any]] = []
        for entry in raw:
            if not isinstance(entry, dict) or "supported" not in entry:
                continue
            claims.append(
                {
                    "claim": str(entry.get("claim", "")).strip(),
                    "supported": bool(entry.get("supported")),
                }
            )
        if not claims:
            return _unreadable("no claim carried a 'supported' verdict")

        unsupported = [c["claim"] for c in claims if not c["supported"]]
        rate = 1.0 - len(unsupported) / len(claims)
        missing = str(payload.get("missing") or "").strip()
        return ParsedReply(
            outcome=ReplyOutcome.ANSWERED,
            # The threshold decides pass or fail, not the judge. This status
            # records only whether anything at all was ungrounded; the
            # finding applies `min_claim_support` to the rate.
            status=Status.PASS if not unsupported else Status.FAIL,
            # Read in the same direction as every other verdict: higher
            # confidence means more sure of what it just said.
            raw_confidence=rate if not unsupported else 1.0 - rate,
            reason=(
                f"{len(unsupported)} of {len(claims)} claims are not supported by the item"
                if unsupported
                else f"all {len(claims)} claims are supported"
            ),
            detail={
                "claims": claims,
                "unsupported": unsupported,
                "support_rate": rate,
                **({"missing": missing} if missing else {}),
            },
        )


def free_text_is_judgeable(output: Output, field: str) -> bool:
    """Is there anything here for a judge to ground?"""
    value = output.get(field)
    return isinstance(value, str) and bool(value.strip())


_GROUNDED_SYSTEM = """You are checking a value another system extracted from a document.

The value is supposed to have been read out of the document, not inferred or
calculated. Decide two things:

1. Is this value present in the document at all?
2. Is it the value that belongs in this field, rather than a different number,
   date or name that happens to appear nearby?

A document often contains several plausible candidates. Picking the wrong one is
the failure that matters here, and it looks exactly like success from outside.

If the document does not say enough to tell which value belongs in the field,
answer cannot_decide rather than guessing.

Reply with one JSON object and nothing else:
{{"correct": true | false | "cannot_decide",
 "confidence": <0 to 1>,
 "reason": "<one sentence>",
 "instead": "<the value the document actually supports, only when correct is false>"}}"""

_GROUNDED_USER = """DOCUMENT:
{text}

FIELD: {field}{described}
EXTRACTED VALUE: {value}"""


@dataclass(frozen=True, slots=True)
class GroundednessTask:
    """Is this the value the document supports for this field?

    The free check can prove a value is *absent* — that one is invented, and
    it costs nothing. What it cannot prove is that a value it *found* is the
    right one: a document listing a subtotal, a shipping charge and a total
    contains the number the model reported whichever of the three it meant.

    That is the failure this judge exists for, and it is the one that looks
    like success from outside. The reply carries ``instead`` for the same
    reason the label judge does: a reviewer opening the row wants the value
    the document actually supports, not only the news that this one is wrong.
    """

    check_id: str = "value_grounded"
    max_tokens: int = 120
    describe: str = ""

    def build(
        self, item: Item, output: Output, field: str, taxonomy: Taxonomy | None
    ) -> JudgeRequest:
        described = f" — {self.describe}" if self.describe else ""
        return JudgeRequest(
            system=_GROUNDED_SYSTEM.format(),
            user=_GROUNDED_USER.format(
                text=item.text, field=field, described=described, value=output.get(field)
            ),
            max_tokens=self.max_tokens,
            temperature=0.0,
        )

    def parse(self, text: str) -> ParsedReply:
        # The same shape as a label verdict: correct / incorrect /
        # cannot_decide, with an alternative on a rejection. Reusing the
        # parser keeps one definition of what an unreadable reply is.
        return LabelCorrectTask().parse(text)


def copied_is_judgeable(output: Output, field: str) -> bool:
    """Is there a value here for a judge to ground?"""
    value = output.get(field)
    return value is not None and str(value).strip() != ""
