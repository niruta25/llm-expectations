"""Reference providers.

`MockProvider` is deterministic and free. It exists so the whole framework is
testable in CI without a network call or an API key, and so the demo is
reproducible. It scores a field low when the value is not a substring of the
source text, which is a crude stand-in for a real verifier's behaviour.

Because its scoring rule is substring containment, any calibration fitted
against grounding labels will report a flattering AUROC. That is a plumbing
check, not a capability claim.

`HTTPChatProvider` is the shape a real adapter takes. Ship it as a separate
distribution (`llmex-openai`, `llmex-anthropic`) that declares an entry point.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from ..registry import PROVIDERS
from ..types import Capability
from .base import CompletionRequest, CompletionResponse, CostModel, Limits

# The mock's whole scoring rule, stated rather than scattered through the code.
# It is a fixture, not a model: it rewards substring containment and nothing else.
GROUNDED_SCORE = 0.95
NULL_SCORE = 0.85
"""A null is often legitimately absent, so it is not treated as a fabrication."""
UNGROUNDED_BASE = 0.05
UNGROUNDED_JITTER_STEPS = 25
"""Deterministic spread over [BASE, BASE + STEPS/100) so ensemble members
disagree slightly, which is what makes aggregation across them meaningful."""
CHARS_PER_TOKEN = 4
"""Crude token estimate. Real adapters report usage from the API response."""


@PROVIDERS.plugin("mock")
class MockProvider:
    id = "mock"
    model_version = "mock-1.0"
    capabilities = frozenset(
        {Capability.STRUCTURED_OUTPUT, Capability.SYSTEM_PROMPT, Capability.LOGPROBS}
    )
    cost = CostModel(usd_per_1k_in=0.0001, usd_per_1k_out=0.0004)
    limits = Limits(max_concurrency=16)

    def __init__(self, latency_ms: float = 1.0, **_: Any) -> None:
        self._latency = latency_ms

    async def complete(self, req: CompletionRequest) -> CompletionResponse:
        await asyncio.sleep(self._latency / 1000)
        payload = json.loads(req.user)
        source = payload.get("source_text", "")
        extraction = payload.get("extraction", {})

        scores: dict[str, float] = {}
        for name, value in extraction.items():
            if value in (None, "", []):
                scores[name] = NULL_SCORE
            elif str(value) in source:
                scores[name] = GROUNDED_SCORE
            else:
                seed = hashlib.md5(  # noqa: S324 - a fixture seed, not a digest
                    f"{req.tag}:{name}:{value}".encode(), usedforsecurity=False
                ).digest()[0]
                jitter = (seed % UNGROUNDED_JITTER_STEPS) / 100
                scores[name] = round(UNGROUNDED_BASE + jitter, 3)

        body = json.dumps({"field_scores": scores})
        return CompletionResponse(
            text=body,
            parsed={"field_scores": scores},
            tokens_in=max(1, len(req.user) // CHARS_PER_TOKEN),
            tokens_out=max(1, len(body) // CHARS_PER_TOKEN),
            latency_ms=self._latency,
        )


JUDGE_FLOOR = 0.10
JUDGE_RANGE = 0.85
"""A mock judge's score spans [FLOOR, FLOOR + RANGE] as label-token coverage of
the document goes from none to complete."""


@PROVIDERS.plugin("mock_judge")
class MockLabelJudge(MockProvider):
    """A fixture judge for label-valued extractions.

    `MockProvider` scores by substring containment, which is useless for
    classification: the label `password_reset` never appears verbatim in a
    support transcript. This one scores on how much of the label's vocabulary
    the document actually contains.

    That is a real signal and a weak one — exactly what a fixture judge should
    be. It agrees with a human often enough for a calibration to find a
    meaningful threshold, and disagrees often enough that the resulting AUROC
    is a number between 0.5 and 1.0 rather than a flattering 1.00.

    It never reads `gold_label`. A mock that peeked at the answer would make
    every downstream metric a property of this file.
    """

    id = "mock_judge"
    model_version = "mock-judge-1.0"

    _STOPWORDS = frozenset({"a", "an", "the", "of", "to", "and", "or", "for"})

    @classmethod
    def _tokens(cls, value: object) -> list[str]:
        raw = "".join(c if c.isalnum() else " " for c in str(value).casefold())
        return [t for t in raw.split() if t not in cls._STOPWORDS]

    async def complete(self, req: CompletionRequest) -> CompletionResponse:
        await asyncio.sleep(self._latency / 1000)
        payload = json.loads(req.user)
        document = str(payload.get("document", payload.get("source_text", ""))).casefold()
        labels = payload.get("assigned_labels", payload.get("extraction", {}))

        scores: dict[str, float] = {}
        explanations: dict[str, str] = {}
        for name, value in labels.items():
            # Check emptiness before tokenising: str(None) is "none", which
            # would otherwise be scored as a label with one unmatched term.
            tokens = [] if value in (None, "", []) else self._tokens(value)
            if not tokens:
                scores[name] = NULL_SCORE
                explanations[name] = "no label assigned"
                continue
            hits = [t for t in tokens if t in document]
            coverage = len(hits) / len(tokens)
            scores[name] = round(JUDGE_FLOOR + JUDGE_RANGE * coverage, 3)
            explanations[name] = (
                f"{len(hits)}/{len(tokens)} of the label's terms appear in the "
                f"document ({', '.join(hits) if hits else 'none'})"
            )

        body = json.dumps({"field_scores": scores, "explanations": explanations})
        return CompletionResponse(
            text=body,
            parsed={"field_scores": scores, "explanations": explanations},
            tokens_in=max(1, len(req.user) // CHARS_PER_TOKEN),
            tokens_out=max(1, len(body) // CHARS_PER_TOKEN),
            latency_ms=self._latency,
        )


class HTTPChatProvider:
    """Skeleton for a real adapter. Subclass and fill in `_post`.

    No real provider ships in core: transport, auth and retry belong to the
    plugin distribution that owns the endpoint.
    """

    id = "http-chat"
    model_version = "unset"
    capabilities = frozenset({Capability.STRUCTURED_OUTPUT, Capability.SYSTEM_PROMPT})
    cost = CostModel()
    limits = Limits()

    def __init__(
        self, endpoint: str, model: str, api_key: str | None = None, **_: Any
    ) -> None:
        self.endpoint = endpoint
        self.model_version = model
        self.api_key = api_key
        self._sem = asyncio.Semaphore(self.limits.max_concurrency)

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError("supply an HTTP client in the concrete adapter")

    async def complete(self, req: CompletionRequest) -> CompletionResponse:
        async with self._sem:
            body: dict[str, Any] = {
                "model": self.model_version,
                "messages": (
                    ([{"role": "system", "content": req.system}] if req.system else [])
                    + [{"role": "user", "content": req.user}]
                ),
                "temperature": req.temperature,
                "max_tokens": req.max_tokens,
            }
            if req.json_schema:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": req.json_schema,
                }
            raw = await self._post(body)
            text = raw["choices"][0]["message"]["content"]
            usage = raw.get("usage", {})
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            return CompletionResponse(
                text=text,
                parsed=parsed,
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
            )
