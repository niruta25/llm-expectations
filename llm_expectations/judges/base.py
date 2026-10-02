"""The judge seam: a transport, a task, and the one Verdict they produce.

Two things here are load-bearing.

``JudgeTask.build`` takes an item, an output and a taxonomy. It does not take
labels, and there is nowhere in this module to put one. That is the label-leak
guarantee from DESIGN.md §2 expressed as a function signature rather than as a
rule someone has to remember.

Nothing in this module invents a verdict. A provider that fails after its
retries, a reply that cannot be read, and a judge that says ``cannot_decide``
all produce an **unscored** verdict carrying the reason. A coin-flip default is
uncorrelated by construction, which quietly changes every agreement number in
the run, so there are no defaults to be had.
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from ..taxonomy import Taxonomy
from ..types import Item, Output, Status, Verdict

__all__ = [
    "Judge",
    "JudgeError",
    "JudgeReply",
    "JudgeRequest",
    "JudgeTask",
    "ParsedReply",
    "Provider",
    "ReplyOutcome",
]


class JudgeError(RuntimeError):
    """A provider call failed in a way retrying did not fix."""


class ReplyOutcome(str, Enum):
    """What happened to one call, which is not the same as what it decided.

    The health table counts these separately. A judge that says
    ``cannot_decide`` is behaving correctly on an unclear item; a judge whose
    reply could not be parsed is a measurement problem. Both are unscored, and
    collapsing them would hide the second behind the first.
    """

    ANSWERED = "answered"
    CANNOT_DECIDE = "cannot_decide"
    UNPARSEABLE = "unparseable"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class JudgeRequest:
    """One prompt, ready to send."""

    system: str
    user: str
    max_tokens: int
    temperature: float

    @property
    def fingerprint(self) -> str:
        """A hash of the rendered prompt, for the cache key.

        The rendered prompt — not the template — because it carries the
        taxonomy definitions with it. Tighten a definition and every verdict
        that was reached under the old wording stops being a cache hit, which
        is exactly right: it was a different question.
        """
        payload = f"{self.system}\x00{self.user}\x00{self.max_tokens}\x00{self.temperature}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class JudgeReply:
    """What came back, before anyone tries to interpret it."""

    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


@runtime_checkable
class Provider(Protocol):
    """Anything that can complete a prompt."""

    @property
    def id(self) -> str: ...

    @property
    def model(self) -> str: ...

    def complete(self, request: JudgeRequest) -> JudgeReply: ...


@dataclass(frozen=True, slots=True)
class ParsedReply:
    """A reply, read."""

    outcome: ReplyOutcome
    status: Status
    raw_confidence: float | None = None
    reason: str = ""
    detail: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@runtime_checkable
class JudgeTask(Protocol):
    """What to ask, and how to read the answer.

    Note the shape of ``build``: items, outputs and a taxonomy go in. There is
    no parameter for a human label and there will not be one.
    """

    @property
    def check_id(self) -> str: ...

    @property
    def max_tokens(self) -> int: ...

    def build(
        self, item: Item, output: Output, field: str, taxonomy: Taxonomy | None
    ) -> JudgeRequest: ...

    def parse(self, text: str) -> ParsedReply: ...


@dataclass(frozen=True, slots=True)
class Judge:
    """One judge doing one task on one output.

    ``judge_id`` is the configured identity and ``provider`` is how to reach
    it. Both go on every verdict, because "judge-a was lenient" is only
    actionable if you can see which model was behind it.
    """

    judge_id: str
    provider: Provider
    temperature: float
    max_tokens: int

    def ask(
        self,
        task: JudgeTask,
        item: Item,
        output: Output,
        field: str,
        taxonomy: Taxonomy | None = None,
    ) -> tuple[JudgeRequest, Verdict]:
        """Ask once, and return the prompt alongside the verdict.

        The prompt comes back so the caller can key the cache on it without
        rebuilding it. Nothing here writes to disk; the runner owns that.
        """
        request = self.build(task, item, output, field, taxonomy)
        started = time.perf_counter()
        try:
            reply = self.provider.complete(request)
        except JudgeError as exc:
            elapsed = (time.perf_counter() - started) * 1000
            return request, self._verdict(
                task,
                item,
                output,
                field,
                ParsedReply(
                    outcome=ReplyOutcome.ERROR,
                    status=Status.UNSCORED,
                    reason=f"the provider call failed: {exc}",
                ),
                request,
                model=self.provider.model,
                latency_ms=elapsed,
            )

        parsed = task.parse(reply.text)
        return request, self._verdict(
            task,
            item,
            output,
            field,
            parsed,
            request,
            model=reply.model,
            latency_ms=reply.latency_ms,
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            raw_text=reply.text,
        )

    def build(
        self,
        task: JudgeTask,
        item: Item,
        output: Output,
        field: str,
        taxonomy: Taxonomy | None = None,
    ) -> JudgeRequest:
        """Render the prompt this judge would send, without sending it.

        Used by ``plan`` to count tokens, and by the cache to build a key
        before deciding whether a call is needed at all.
        """
        request = task.build(item, output, field, taxonomy)
        return dataclasses.replace(
            request,
            temperature=self.temperature,
            max_tokens=min(request.max_tokens, self.max_tokens) or self.max_tokens,
        )

    def _verdict(
        self,
        task: JudgeTask,
        item: Item,
        output: Output,
        field: str,
        parsed: ParsedReply,
        request: JudgeRequest,
        *,
        model: str,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        raw_text: str | None = None,
    ) -> Verdict:
        metadata: dict[str, Any] = {
            "reply": parsed.outcome.value,
            "model": model,
            "prompt_hash": request.fingerprint,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": round(latency_ms, 1),
            "cache_hit": False,
        }
        if parsed.outcome is ReplyOutcome.UNPARSEABLE and raw_text is not None:
            # Kept so the guardrail can report *which* items failed to parse and
            # what they came back as. Empty replies cluster on hard items, and
            # that biases everything downstream of them.
            metadata["raw_reply"] = raw_text[:500]
        return Verdict(
            judge_id=self.judge_id,
            check_id=task.check_id,
            item_id=item.id,
            field=field,
            status=parsed.status,
            raw_confidence=parsed.raw_confidence,
            reason=parsed.reason,
            detail=parsed.detail,
            metadata=metadata,
        )
