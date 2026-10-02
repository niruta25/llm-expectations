"""A scripted provider. The backbone of every test in this repo.

Deterministic, free, and it sees exactly what a real provider sees — a system
prompt and a user prompt, nothing more. Scripting by a substring of the prompt
rather than by item id is what keeps that true: a fake that got handed the item
id would be a different interface from the one under test.

It ships in the package rather than in ``tests/`` so that anyone wiring this
library into their own pipeline can exercise the whole path — cache, findings,
triage, report — without spending anything.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .base import JudgeError, JudgeReply, JudgeRequest

__all__ = ["FakeProvider", "reply"]


def reply(correct: bool | str, confidence: float | None = 0.8, reason: str = "scripted") -> str:
    """Build a well-formed reply body. ``correct`` may be ``"cannot_decide"``."""
    value = "true" if correct is True else "false" if correct is False else f'"{correct}"'
    shown = "null" if confidence is None else confidence
    return f'{{"correct": {value}, "confidence": {shown}, "reason": "{reason}"}}'


@dataclass
class FakeProvider:
    """Answers from a script, matched against the prompt it is given.

    ``rules`` are tried in order; the first whose substring appears in the user
    prompt wins. ``default`` answers everything else. A rule whose value is a
    callable is given the request, which is how a test scripts a provider that
    fails, or one whose behaviour depends on what it was asked.
    """

    model: str = "fake-instruct"
    id: str = "fake"
    rules: Sequence[tuple[str, str | Callable[[JudgeRequest], str]]] = ()
    default: str | Callable[[JudgeRequest], str] = field(default_factory=lambda: reply(True))
    calls: list[JudgeRequest] = field(default_factory=list)

    def complete(self, request: JudgeRequest) -> JudgeReply:
        self.calls.append(request)
        answer = self.default
        for needle, scripted in self.rules:
            if needle in request.user:
                answer = scripted
                break
        text = answer(request) if callable(answer) else answer
        if text is _FAIL:
            raise JudgeError("scripted provider failure")
        return JudgeReply(
            text=text,
            model=self.model,
            # Enough to make cost and token accounting move without pretending
            # to be a tokeniser.
            input_tokens=len(request.system + request.user) // 4,
            output_tokens=len(text) // 4,
            latency_ms=0.0,
        )


_FAIL = "\x00fail"


def _failing(_: JudgeRequest) -> str:
    return _FAIL


FakeProvider.FAILS = staticmethod(_failing)  # type: ignore[attr-defined]
