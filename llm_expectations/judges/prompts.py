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

from ..taxonomy import Taxonomy
from ..types import ABSTAIN, Item, Output, Status
from .base import JudgeRequest, ParsedReply, ReplyOutcome

__all__ = ["LabelCorrectTask"]

_SYSTEM = """You are checking a label another system assigned to an item.
Decide whether the label is correct.
If the item does not say enough to decide, answer cannot_decide rather than guessing.

The permitted labels are:
{labels}

Reply with one JSON object and nothing else:
{{"correct": true | false | "cannot_decide", "confidence": <0 to 1>, "reason": "<one sentence>"}}"""

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
            return ParsedReply(
                outcome=ReplyOutcome.ANSWERED,
                status=Status.PASS if correct else Status.FAIL,
                raw_confidence=confidence,
                reason=reason,
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
