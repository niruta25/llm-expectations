"""Model provider contract.

Deliberately thin. A provider knows about transport, auth, tokens and money.
It knows nothing about expectations, scores or thresholds. That separation is
what lets the same scoring strategy run on GPT, Claude, or a local vLLM box.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..types import Capability, Cost


@dataclass(frozen=True)
class CostModel:
    usd_per_1k_in: float = 0.0
    usd_per_1k_out: float = 0.0

    def price(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in / 1000) * self.usd_per_1k_in + (tokens_out / 1000) * self.usd_per_1k_out


@dataclass(frozen=True)
class Limits:
    max_concurrency: int = 8
    requests_per_minute: int | None = None
    max_context_tokens: int = 128_000


@dataclass
class CompletionRequest:
    user: str
    system: str | None = None
    json_schema: dict[str, Any] | None = None
    temperature: float = 0.0
    max_tokens: int = 1024
    want_logprobs: bool = False
    # Which strategy template issued this. When an ensemble disagrees, this is
    # how you tell which framing produced which score.
    tag: str = ""


@dataclass
class CompletionResponse:
    text: str = ""
    parsed: dict[str, Any] | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    logprobs: list[float] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ModelProvider(Protocol):
    id: str
    model_version: str
    capabilities: frozenset[Capability]
    cost: CostModel
    limits: Limits

    async def complete(self, req: CompletionRequest) -> CompletionResponse:
        ...


def cost_of(provider: ModelProvider, resp: CompletionResponse) -> Cost:
    return Cost(
        usd=provider.cost.price(resp.tokens_in, resp.tokens_out),
        calls=1,
        tokens_in=resp.tokens_in,
        tokens_out=resp.tokens_out,
        latency_ms=resp.latency_ms,
    )
