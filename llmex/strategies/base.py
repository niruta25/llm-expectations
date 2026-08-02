"""Scoring strategy contract.

A strategy is *how* you interrogate a model; a provider is *which* model.
Keeping them orthogonal is the whole point: you can A/B a cheap verifier
against an expensive one without touching the expectation, and reuse the
five-call ensemble on a local model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..defaults import DEFAULT_AGGREGATOR
from ..providers.base import ModelProvider
from ..types import Capability, Cost


class StrategyError(Exception):
    """The strategy could not produce a score at all.

    Distinct from a degraded score: a strategy that lost some ensemble members
    still returns a `ScoreSet` and reports `n_calls_dropped`. This is raised
    only when there is *no* information, so the expectation converts it into a
    PROVIDER_ERROR skip rather than inventing a number that would read as a pass.
    """


@dataclass
class ScorePayload:
    doc_id: str
    source_text: str
    extraction: dict[str, object]
    schema: dict[str, Any] = field(default_factory=dict)
    instructions: str = ""
    logprobs: dict[str, float] | None = None
    """Per-field token logprobs from the *generator*, when the extraction
    carried them (`ExtractionRecord.meta["logprob"]`). Only `logprob` uses
    them; observe-only means the framework cannot obtain them itself."""


@dataclass
class ScoreSet:
    doc_score: float
    field_scores: dict[str, float]
    explanations: dict[str, str] = field(default_factory=dict)
    cost: Cost = field(default_factory=Cost)
    aggregation: str = DEFAULT_AGGREGATOR
    """How this strategy rolled fields up into `doc_score`. Reported rather
    than assumed downstream, since a strategy may choose differently."""
    n_calls_used: int = 0
    # Timeout-discarded ensemble members. Consistently non-zero means either
    # the timeout is too tight or the provider is degraded — both worth knowing
    # and invisible otherwise.
    n_calls_dropped: int = 0


@runtime_checkable
class ScoringStrategy(Protocol):
    id: str
    required_capabilities: frozenset[Capability]

    def estimate_calls(self, n_fields: int) -> int:
        ...

    async def score(self, payload: ScorePayload, provider: ModelProvider) -> ScoreSet:
        ...
