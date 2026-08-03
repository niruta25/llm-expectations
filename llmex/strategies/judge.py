"""Judge strategies for label-valued extractions.

An LLM judge is a classifier that predicts "this label is wrong." Nothing about
that sentence is specific to any one taxonomy, so the rubric and the label set
are configuration rather than code: the same strategy grades support-session
intents, document types, or sentiment, and the calibration is keyed to whatever
it was fitted on.

What makes this a measurement rather than an opinion is entirely downstream —
the calibration, and the planner's refusal to let an uncalibrated judge block.
This module's only job is to ask the question consistently and report what came
back, including the reasoning, which is the part a human needs to adjudicate a
disagreement.
"""

from __future__ import annotations

import json
from typing import Any

from ..aggregate import harmonic
from ..defaults import DEFAULT_JUDGE_TEMPERATURE
from ..providers.base import CompletionRequest, ModelProvider, cost_of
from ..registry import STRATEGIES
from ..types import Capability
from .base import ScorePayload, ScoreSet, StrategyError

_JUDGE_SCHEMA: dict[str, Any] = {
    "name": "label_verdicts",
    "schema": {
        "type": "object",
        "properties": {
            "field_scores": {
                "type": "object",
                "description": "field name -> confidence in [0,1] that the label is correct",
            },
            "explanations": {
                "type": "object",
                "description": "field name -> one sentence of reasoning",
            },
        },
        "required": ["field_scores"],
    },
}

_SYSTEM = """You are grading labels that another system assigned to a document.

For each field, judge whether the assigned label is the correct one for this \
document, and return a confidence in [0,1] that it is correct. A value near 0 \
means the label is wrong; a value near 1 means it is right.

Judge only what the document supports. If the document is ambiguous or lacks \
the evidence needed to decide, score near 0.5 rather than guessing — a \
confident wrong verdict is worse than an honest uncertain one."""


@STRATEGIES.plugin("label_judge")
class LabelJudge:
    """One call per document, grading every label on it.

    Config:
      `rubric`     task-specific guidance appended to the system prompt
      `label_set`  the closed taxonomy, listed for the judge
      `temperature` defaults to 0 — a measuring instrument should not sample
    """

    id = "label_judge"
    required_capabilities = frozenset(
        {Capability.STRUCTURED_OUTPUT, Capability.SYSTEM_PROMPT}
    )

    def __init__(
        self,
        rubric: str = "",
        label_set: list[str] | None = None,
        temperature: float = DEFAULT_JUDGE_TEMPERATURE,
        **_: Any,
    ) -> None:
        self.rubric = rubric
        self.label_set = list(label_set or [])
        self.temperature = temperature

    def estimate_calls(self, n_fields: int) -> int:
        return 1

    def system_prompt(self) -> str:
        parts = [_SYSTEM]
        if self.label_set:
            parts.append(
                "The permitted labels are:\n"
                + "\n".join(f"  - {label}" for label in self.label_set)
                + "\nA label outside this set is wrong by definition."
            )
        if self.rubric:
            parts.append(self.rubric.strip())
        return "\n\n".join(parts)

    async def score(self, payload: ScorePayload, provider: ModelProvider) -> ScoreSet:
        req = CompletionRequest(
            user=json.dumps(
                {
                    "document": payload.source_text,
                    "assigned_labels": payload.extraction,
                    "instructions": payload.instructions,
                }
            ),
            system=self.system_prompt(),
            json_schema=_JUDGE_SCHEMA,
            temperature=self.temperature,
            tag="label_judge",
        )
        resp = await provider.complete(req)
        parsed = resp.parsed or {}
        raw = parsed.get("field_scores")
        if not isinstance(raw, dict) or not raw:
            raise StrategyError(
                f"{self.id}: judge returned no field_scores "
                f"(text={resp.text[:120]!r}). Nothing was measured."
            )

        # Only fields the judge actually spoke about. A field it skipped is
        # left out so the expectation reports it unscored rather than passed.
        scores = {
            name: max(0.0, min(1.0, float(raw[name])))
            for name in payload.extraction
            if name in raw
        }
        if not scores:
            raise StrategyError(
                f"{self.id}: judge scored none of the requested fields "
                f"{sorted(payload.extraction)}."
            )

        explanations_raw = parsed.get("explanations") or {}
        explanations = {
            name: str(explanations_raw[name])
            for name in scores
            if name in explanations_raw
        }

        return ScoreSet(
            doc_score=harmonic(list(scores.values())),
            field_scores=scores,
            explanations=explanations,
            cost=cost_of(provider, resp),
            n_calls_used=1,
        )
