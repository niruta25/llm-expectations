"""Built-in scoring strategies, ordered by cost.

Each occupies a genuinely different point on the cost/accuracy curve. The
ensemble is the one worth understanding: parallel calls with *different*
tasks and formats, no sequential dependency between them, and a timeout that
discards laggards rather than blocking the batch.
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any

from ..aggregate import harmonic
from ..defaults import DEFAULT_ENSEMBLE_CALLS, DEFAULT_ENSEMBLE_TIMEOUT_S
from ..providers.base import CompletionRequest, ModelProvider, cost_of
from ..registry import STRATEGIES
from ..types import Capability, Cost
from .base import ScorePayload, ScoreSet, StrategyError

_SCORE_SCHEMA: dict[str, Any] = {
    "name": "field_scores",
    "schema": {
        "type": "object",
        "properties": {"field_scores": {"type": "object"}},
        "required": ["field_scores"],
    },
}


def _payload_json(payload: ScorePayload) -> str:
    return json.dumps(
        {
            "source_text": payload.source_text,
            "extraction": payload.extraction,
            "instructions": payload.instructions,
        }
    )


async def _one_call(
    provider: ModelProvider, payload: ScorePayload, system: str, tag: str
) -> tuple[dict[str, float], Cost]:
    req = CompletionRequest(
        user=_payload_json(payload),
        system=system,
        json_schema=_SCORE_SCHEMA,
        tag=tag,
    )
    resp = await provider.complete(req)
    parsed = resp.parsed or {}
    scores: dict[str, float] = parsed.get("field_scores", {})
    return scores, cost_of(provider, resp)


@STRATEGIES.plugin("single_judge")
class SingleJudge:
    """One call, whole document. Cheapest model-based option and the weakest:
    a holistic judge systematically overlooks mistakes in individual fields."""

    id = "single_judge"
    required_capabilities = frozenset({Capability.STRUCTURED_OUTPUT})

    def __init__(self, **_: Any) -> None:
        pass

    def estimate_calls(self, n_fields: int) -> int:
        return 1

    async def score(self, payload: ScorePayload, provider: ModelProvider) -> ScoreSet:
        scores, cost = await _one_call(
            provider,
            payload,
            "Assess whether this extraction is correct and complete.",
            "single",
        )
        doc = harmonic(list(scores.values())) if scores else 1.0
        return ScoreSet(doc, scores, cost=cost, n_calls_used=1)


@STRATEGIES.plugin("per_field_judge")
class PerFieldJudge:
    """One call per field. Best per-field recall, but N calls per document
    hits token rate limits hard on schemas with fifty-plus fields."""

    id = "per_field_judge"
    required_capabilities = frozenset({Capability.STRUCTURED_OUTPUT})

    def __init__(self, **_: Any) -> None:
        pass

    def estimate_calls(self, n_fields: int) -> int:
        return max(1, n_fields)

    async def score(self, payload: ScorePayload, provider: ModelProvider) -> ScoreSet:
        async def one(name: str) -> tuple[str, float | None, Cost]:
            sub = ScorePayload(
                payload.doc_id,
                payload.source_text,
                {name: payload.extraction[name]},
                payload.schema,
                f"Evaluate only the field '{name}'.",
            )
            scores, cost = await _one_call(
                provider, sub, "Verify this single field.", f"field:{name}"
            )
            # None, not a stand-in value: a field the judge declined to score
            # is unscored, and the expectation reports it as such.
            return name, scores.get(name), cost

        pairs = await asyncio.gather(*(one(n) for n in payload.extraction))
        field_scores = {n: s for n, s, _ in pairs if s is not None}
        total = Cost()
        for _, _, c in pairs:
            total = total + c
        return ScoreSet(
            harmonic(list(field_scores.values())),
            field_scores,
            cost=total,
            n_calls_used=len(pairs),
        )


@STRATEGIES.plugin("diverse_ensemble")
class DiverseEnsemble:
    """Several parallel verifier calls with deliberately different framings,
    aggregated. Diversity is the mechanism: identical prompts would just
    reproduce the same blind spot five times.

    Empirically, members beyond about five yield under 1% improvement. Do not
    add more without measuring.
    """

    id = "diverse_ensemble"
    required_capabilities = frozenset({Capability.STRUCTURED_OUTPUT})

    TEMPLATES: list[tuple[str, str]] = [
        ("holistic", "Argue why this extraction might be wrong, then rate each field."),
        ("strict", "Be strict. Treat unverifiable or partially correct values as wrong."),
        ("grounding", "For each field, decide whether the value appears in the source."),
        (
            "omission",
            "Look for information present in the source but missing from the extraction.",
        ),
        ("format", "Check each value's type, format and normalisation against the schema."),
    ]

    def __init__(
        self,
        n_calls: int = DEFAULT_ENSEMBLE_CALLS,
        timeout_s: float = DEFAULT_ENSEMBLE_TIMEOUT_S,
        **_: Any,
    ) -> None:
        self.templates = self.TEMPLATES[: max(1, min(n_calls, len(self.TEMPLATES)))]
        self.timeout_s = timeout_s

    def estimate_calls(self, n_fields: int) -> int:
        return len(self.templates)

    async def score(self, payload: ScorePayload, provider: ModelProvider) -> ScoreSet:
        # No sequential dependencies: wall-clock is one call, not five.
        tasks = [
            asyncio.create_task(_one_call(provider, payload, system, tag))
            for tag, system in self.templates
        ]
        done, pending = await asyncio.wait(tasks, timeout=self.timeout_s)
        for t in pending:
            t.cancel()

        per_template: list[dict[str, float]] = []
        total = Cost()
        cause: Exception | None = None
        for t in done:
            try:
                scores, cost = t.result()
            except Exception as exc:  # one bad member must not kill the ensemble
                cause = exc
                continue
            per_template.append(scores)
            total = total + cost

        if not per_template:
            # Every member failed or timed out. Returning a middling score here
            # would clear a middling threshold and read as a pass, which is the
            # exact failure the unscored state exists to prevent.
            why = f": {cause}" if cause else f" (timeout={self.timeout_s}s)"
            raise StrategyError(
                f"{self.id}: all {len(tasks)} ensemble members failed or timed out"
                f"{why}; no score available"
            ) from cause

        # Two-stage aggregation: arithmetic mean across templates for each field
        # (noisy estimates of the same quantity), then harmonic across fields for
        # the document (failure is the union of component failures).
        field_scores: dict[str, float] = {}
        for name in payload.extraction:
            observed = [t[name] for t in per_template if name in t]
            if observed:
                field_scores[name] = sum(observed) / len(observed)
            # A field no surviving template scored is left out entirely rather
            # than filled in, so it surfaces as unscored instead of as a pass.

        return ScoreSet(
            doc_score=harmonic(list(field_scores.values())),
            field_scores=field_scores,
            cost=total,
            n_calls_used=len(per_template),
            n_calls_dropped=len(pending),
        )


@STRATEGIES.plugin("logprob")
class LogprobScore:
    """Reuses the *generator's* own token probabilities, so it issues no calls
    and costs nothing. Requires LOGPROBS, which several major APIs do not
    expose — which is also what makes it a good demonstration that the
    capability gate fails at plan time rather than mid-run.

    Observe-only means the framework cannot fetch logprobs itself: they must
    arrive on the extraction records as `meta["logprob"]`. Without them there
    is no measurement to report, so this raises rather than inventing one.
    Note that logprobs are confidence, not correctness — a model is routinely
    confident and wrong — so calibrate this harder than you think you need to.
    """

    id = "logprob"
    required_capabilities = frozenset({Capability.LOGPROBS})

    def __init__(self, **_: Any) -> None:
        pass

    def estimate_calls(self, n_fields: int) -> int:
        return 0

    async def score(self, payload: ScorePayload, provider: ModelProvider) -> ScoreSet:
        missing = [f for f in payload.extraction if f not in (payload.logprobs or {})]
        if missing:
            raise StrategyError(
                f"{self.id}: no generator logprobs for {sorted(missing)}. This "
                f"strategy scores from ExtractionRecord.meta['logprob']; supply "
                f"them at ingest or pick a strategy that queries a verifier."
            )
        assert payload.logprobs is not None  # noqa: S101 - narrowed by `missing`
        scores = {
            name: max(0.0, min(1.0, math.exp(payload.logprobs[name])))
            for name in payload.extraction
        }
        return ScoreSet(harmonic(list(scores.values())), scores)
