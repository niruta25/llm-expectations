"""Planning.

Everything that can fail should fail here, before a single token is spent:

  1. Alias resolution     provider/strategy alias not defined in the suite
  2. Capability gap       strategy needs LOGPROBS, provider does not expose them
  3. Calibration guard    a MODEL_BASED check cannot block on a hand-typed number
  4. Staleness            the calibration was fitted on a different model
  5. Correlated verifier  judging a model's output with the same model (warning)
  6. Budget               estimated spend exceeds the cap (warning)

The calibration guard is the reason this is a quality framework and not a
wrapper around an API. It makes trust laundering a config-time error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .batch import Batch
from .defaults import (
    DEFAULT_PROVIDER_ALIAS,
    DEFAULT_STRATEGY_ALIAS,
    ESTIMATED_TOKENS_IN,
    ESTIMATED_TOKENS_OUT,
)
from .expectation import Expectation
from .suite import Suite
from .types import Kind, PlanError, Severity


@dataclass
class Step:
    expectation: Expectation
    kind: Kind
    estimated_calls: int = 0
    estimated_usd: float = 0.0


@dataclass
class Plan:
    steps: list[Step] = field(default_factory=list)
    estimated_calls: int = 0
    estimated_usd: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def ordered(self) -> list[Step]:
        """Cheap and deterministic first: their results gate the expensive tier.

        This ordering is load-bearing, not cosmetic.
        """
        rank = {Kind.DETERMINISTIC: 0, Kind.STATISTICAL: 1, Kind.MODEL_BASED: 2}
        return sorted(self.steps, key=lambda s: rank[s.kind])

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": [
                {
                    "expectation_id": s.expectation.id,
                    "kind": s.kind.value,
                    "estimated_calls": s.estimated_calls,
                    "estimated_usd": round(s.estimated_usd, 6),
                }
                for s in self.ordered()
            ],
            "estimated_calls": self.estimated_calls,
            "estimated_usd": round(self.estimated_usd, 6),
            "warnings": self.warnings,
        }


class Planner:
    def __init__(
        self,
        estimated_tokens_in: int = ESTIMATED_TOKENS_IN,
        estimated_tokens_out: int = ESTIMATED_TOKENS_OUT,
    ) -> None:
        """The token counts shape the plan-time cost estimate. They are a guess
        by construction — the planner has not read the documents — so override
        them once you have measured your own corpus. The estimate lands in the
        manifest next to the actual spend, which is how you find out it was wrong.
        """
        self.estimated_tokens_in = estimated_tokens_in
        self.estimated_tokens_out = estimated_tokens_out

    def plan(self, suite: Suite, batch: Batch) -> Plan:
        plan = Plan()

        for exp in suite.expectations:
            step = Step(expectation=exp, kind=exp.kind)

            if exp.kind is Kind.MODEL_BASED:
                provider = self._resolve_provider(suite, exp)
                strategy = self._resolve_strategy(suite, exp)

                missing = strategy.required_capabilities - provider.capabilities
                if missing:
                    raise PlanError(
                        f"{exp.id}: strategy '{strategy.id}' requires "
                        f"{sorted(c.value for c in missing)} but provider "
                        f"'{provider.id}' ({provider.model_version}) does not expose them. "
                        f"Pick a different strategy or provider."
                    )

                cal_id = exp.config.get("calibration")
                cal = suite.calibrations.get(cal_id) if cal_id else None
                if exp.severity is Severity.ERROR and cal is None:
                    raise PlanError(
                        f"{exp.id}: severity=error requires a calibration. "
                        f"An uncalibrated model score is an opinion, not a threshold. "
                        f"Either attach one or drop to severity=warn."
                    )
                if cal is not None and not cal.valid_for(
                    provider.id, provider.model_version, strategy.id
                ):
                    raise PlanError(
                        f"{exp.id}: calibration '{cal.id}' was fitted on "
                        f"{cal.provider_id}/{cal.model_version}/{cal.strategy_id} "
                        f"but this run uses "
                        f"{provider.id}/{provider.model_version}/{strategy.id}. "
                        f"Recalibrate."
                    )

                # A warning rather than an error: self-verification is sometimes
                # the only option. But it must be loud, because a self-graded
                # score reads systematically high.
                generators = {r.generator_model for r in batch.records if r.generator_model}
                if provider.model_version in generators:
                    plan.warnings.append(
                        f"{exp.id}: verifier model '{provider.model_version}' also "
                        f"generated these extractions. Errors will be correlated and "
                        f"the score will read high. Use a different verifier."
                    )

                exp.config["_strategy_calls"] = strategy.estimate_calls(
                    max(1, len(batch.field_names))
                )
                step.estimated_calls = exp.estimate_calls(batch)
                step.estimated_usd = step.estimated_calls * self._unit_cost(provider)

            plan.steps.append(step)
            plan.estimated_calls += step.estimated_calls
            plan.estimated_usd += step.estimated_usd

        if plan.estimated_usd > suite.budget.max_usd:
            plan.warnings.append(
                f"estimated ${plan.estimated_usd:.2f} exceeds budget "
                f"${suite.budget.max_usd:.2f}; checks will be marked unscored once "
                f"the cap is hit rather than passing silently."
            )
        return plan

    @staticmethod
    def _resolve_provider(suite: Suite, exp: Expectation) -> Any:
        alias = exp.config.get("provider", DEFAULT_PROVIDER_ALIAS)
        if alias not in suite.providers:
            raise PlanError(
                f"{exp.id}: no provider aliased '{alias}'. defined: {sorted(suite.providers)}"
            )
        return suite.providers[alias]

    @staticmethod
    def _resolve_strategy(suite: Suite, exp: Expectation) -> Any:
        alias = exp.config.get("strategy", DEFAULT_STRATEGY_ALIAS)
        if alias not in suite.strategies:
            raise PlanError(
                f"{exp.id}: no strategy aliased '{alias}'. defined: {sorted(suite.strategies)}"
            )
        return suite.strategies[alias]

    def _unit_cost(self, provider: Any) -> float:
        price: float = provider.cost.price(
            self.estimated_tokens_in, self.estimated_tokens_out
        )
        return price
