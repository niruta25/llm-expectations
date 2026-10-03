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

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .base import JudgeError, JudgeReply, JudgeRequest

__all__ = ["FakeProvider", "ScriptedProvider", "claims", "reply"]


def reply(
    correct: bool | str,
    confidence: float | None = 0.8,
    reason: str = "scripted",
    instead: str | None = None,
) -> str:
    """Build a well-formed reply body. ``correct`` may be ``"cannot_decide"``."""
    value = "true" if correct is True else "false" if correct is False else f'"{correct}"'
    shown = "null" if confidence is None else confidence
    suggestion = f', "instead": "{instead}"' if instead else ""
    return f'{{"correct": {value}, "confidence": {shown}, "reason": "{reason}"{suggestion}}}'


def claims(
    supported: Sequence[str] = (),
    unsupported: Sequence[str] = (),
    missing: str = "",
) -> str:
    """Build a well-formed claim-support reply."""
    entries = [{"claim": c, "supported": True} for c in supported]
    entries += [{"claim": c, "supported": False} for c in unsupported]
    if not entries:
        entries = [{"claim": "the text restates the item", "supported": True}]
    return json.dumps({"claims": entries, "missing": missing})


#: How a claim-support prompt is told apart from an assigned-label one. The
#: fake sees exactly what a provider sees — a system and a user string — so it
#: has to recognise the task the same way, from the prompt itself.
CLAIM_MARKER = "\n\nTEXT:\n"


@dataclass
class FakeProvider:
    """Answers from a script, matched against the prompt it is given.

    ``rules`` are tried in order; the first whose substring appears in the user
    prompt wins. ``default`` answers everything else. A rule whose value is a
    callable is given the request, which is how a test scripts a provider that
    fails, or one whose behaviour depends on what it was asked.

    ``claim_default`` answers claim-support prompts, which are a different
    question with a different reply shape. Without it a fake scripted for
    assigned labels would return ``{"correct": ...}`` to a claim request, the
    parser would rightly call that unreadable, and the judge would be
    screened out for a fault in the test rather than in the judge.
    """

    model: str = "fake-instruct"
    id: str = "fake"
    rules: Sequence[tuple[str, str | Callable[[JudgeRequest], str]]] = ()
    default: str | Callable[[JudgeRequest], str] = field(default_factory=lambda: reply(True))
    claim_default: str | Callable[[JudgeRequest], str] | None = None
    calls: list[JudgeRequest] = field(default_factory=list)

    def complete(self, request: JudgeRequest) -> JudgeReply:
        self.calls.append(request)
        asking_about_claims = CLAIM_MARKER in request.user
        answer: str | Callable[[JudgeRequest], str]
        if asking_about_claims:
            answer = self.claim_default if self.claim_default is not None else claims()
        else:
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


@dataclass
class ScriptedProvider:
    """A judge that answers from a hash of the prompt, with no model behind it.

    This is what ``provider: fake`` in ``judges.yml`` builds, and it exists so
    the worked example runs with no API key and no cost. Everything upstream
    of the judge — the free checks, the taxonomy health checks, the gates, the
    report — is real on real data; only the verdicts are invented.

    Deterministic on the prompt, so a run is reproducible and the cache
    behaves exactly as it would with a real judge.

    **A run that uses one is stopped by Gate 1**, loudly, because a report
    full of scripted verdicts looks exactly like a report full of real ones.
    That guard is the only reason this is safe to ship.
    """

    id: str = "fake"
    model: str = "scripted"
    approve_rate: float = 0.75
    calls: list[JudgeRequest] = field(default_factory=list)

    def complete(self, request: JudgeRequest) -> JudgeReply:
        self.calls.append(request)
        # Seeded with the judge id as well as the prompt, so a panel of
        # scripted judges disagrees the way a real one would. Three fakes
        # that answered identically would show an effective vote count of
        # 1.0 and teach the wrong lesson about the panel.
        digest = hashlib.blake2b(
            f"{self.id}\x00{request.system}\x00{request.user}".encode(), digest_size=8
        ).digest()
        roll = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
        confidence = round(0.55 + (int.from_bytes(digest[4:6], "big") / 0xFFFF) * 0.44, 2)

        if CLAIM_MARKER in request.user:
            written = request.user.split(CLAIM_MARKER, 1)[1].strip()
            first = written.split(".")[0].strip() or written[:60]
            text = (
                claims(supported=[first])
                if roll < self.approve_rate
                else claims(supported=[], unsupported=[first])
            )
        elif roll < self.approve_rate:
            text = reply(True, confidence, "scripted: no model was asked")
        else:
            text = reply(False, confidence, "scripted: no model was asked")

        return JudgeReply(
            text=text,
            model=self.model,
            input_tokens=len(request.system + request.user) // 4,
            output_tokens=len(text) // 4,
        )
