"""Core value types. No framework logic lives here.

This is the bottom layer: it imports nothing from the rest of the framework.
A circular import into this module is a layering mistake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Kind(str, Enum):
    """Declared by every expectation. The planner routes on this.

    The order of the members is the order the tiers execute in, and that
    ordering is load-bearing: cheap verdicts gate expensive ones, and anything
    derived from other checks has to wait until they have all reported.
    """

    DETERMINISTIC = "deterministic"
    STATISTICAL = "statistical"
    MODEL_BASED = "model_based"
    DERIVED = "derived"
    """Computed from results other checks already produced. Free, and last.

    A check that grades another check — judge-versus-human agreement, say —
    cannot run in the statistical tier, because the results it reads do not
    exist yet at that point.
    """


class Grain(str, Enum):
    """The unit a result describes."""

    FIELD = "field"
    DOCUMENT = "document"
    CORPUS = "corpus"


class Severity(str, Enum):
    ERROR = "error"
    WARN = "warn"
    INFO = "info"


class Capability(str, Enum):
    """Negotiated at plan time, never discovered at run time."""

    STRUCTURED_OUTPUT = "structured_output"
    LOGPROBS = "logprobs"
    SYSTEM_PROMPT = "system_prompt"
    BATCH_API = "batch_api"
    EMBEDDINGS = "embeddings"


class SkipReason(str, Enum):
    """Makes `success=None` auditable. Without a reason it is just a hole."""

    BUDGET_EXHAUSTED = "budget_exhausted"
    SAMPLED_OUT = "sampled_out"
    UPSTREAM_FAILED = "upstream_failed"
    PROVIDER_ERROR = "provider_error"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class Cost:
    """Accumulated across a run. Reported per result and per suite."""

    usd: float = 0.0
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    cache_hits: int = 0

    def __add__(self, other: Cost) -> Cost:
        # Latency takes the max, not the sum: verifier calls run in parallel,
        # so summing would report 5x the wall-clock truth for an ensemble.
        return Cost(
            usd=self.usd + other.usd,
            calls=self.calls + other.calls,
            tokens_in=self.tokens_in + other.tokens_in,
            tokens_out=self.tokens_out + other.tokens_out,
            latency_ms=max(self.latency_ms, other.latency_ms),
            cache_hits=self.cache_hits + other.cache_hits,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "usd": round(self.usd, 6),
            "calls": self.calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "latency_ms": round(self.latency_ms, 1),
            "cache_hits": self.cache_hits,
        }


@dataclass
class Evidence:
    """Why a result came out the way it did. Never just a boolean.

    A red dashboard nobody can action is worse than no dashboard.
    """

    kind: str = "none"
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "detail": self.detail}


@dataclass
class Provenance:
    """Which exact function produced this verdict.

    `prompt_version` is the field teams omit and then cannot debug. Treat
    (model, prompt, schema) as one versioned artifact: change any of them and
    historical rows were produced by a different function.
    """

    expectation_id: str
    expectation_version: str
    provider_id: str | None = None
    model_version: str | None = None
    strategy_id: str | None = None
    prompt_version: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class PlanError(Exception):
    """Raised at plan time so runs fail before spending money."""
