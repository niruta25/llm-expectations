"""Execution.

Async core. The only synchronous surface is `run_sync`, for notebooks and
scripts that have no event loop of their own.

Tier order matters: deterministic checks run first and their verdicts are
threaded into `ctx.prior`, which is what lets the model tier escalate exactly
the documents that already look suspect instead of paying for all of them.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from .aggregate import get as get_aggregator
from .batch import Batch, Context
from .defaults import RUN_ID_LENGTH
from .planner import Plan, Planner
from .result import Result, RunResult
from .sinks import Sink
from .suite import Suite
from .types import Cost, Evidence, Grain, Kind


class Runner:
    def __init__(self, planner: Planner | None = None, sinks: list[Sink] | None = None) -> None:
        self.planner = planner or Planner()
        self.sinks: list[Sink] = sinks or []

    async def run(self, suite: Suite, batch: Batch, plan: Plan | None = None) -> RunResult:
        plan = plan or self.planner.plan(suite, batch)
        run_id = uuid.uuid4().hex[:RUN_ID_LENGTH]

        ctx = Context(
            providers=suite.providers,
            strategies=suite.strategies,
            calibrations=suite.calibrations,
            budget=suite.budget,
            prior={},
            run_id=run_id,
        )

        results: list[Result] = []
        total = Cost()

        for step in plan.ordered():
            produced = await step.expectation.validate(batch, ctx)
            if step.kind is not Kind.MODEL_BASED:
                # M6: deterministic steps are mutually independent and should
                # fan out with asyncio.gather, merging ctx.prior after.
                self._merge_prior(ctx, produced)

            results.extend(produced)
            for r in produced:
                total = total + r.cost

        results.extend(self._rollup(suite, results))

        run = RunResult(
            run_id=run_id,
            results=results,
            cost=total,
            manifest={
                "run_id": run_id,
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "suite": suite.fingerprint(),
                "estimated_calls": plan.estimated_calls,
                "estimated_usd": round(plan.estimated_usd, 4),
                "actual": total.as_dict(),
                "n_records": len(batch),
                "n_documents": len(batch.doc_ids),
                "warnings": list(plan.warnings),
            },
            warnings=list(plan.warnings),
        )
        for sink in self.sinks:
            await sink.emit(run)
        return run

    def run_sync(self, suite: Suite, batch: Batch, plan: Plan | None = None) -> RunResult:
        return asyncio.run(self.run(suite, batch, plan))

    @staticmethod
    def _merge_prior(ctx: Context, produced: list[Result]) -> None:
        """A field is 'clean' only if every cheap check has passed it."""
        for r in produced:
            if r.grain is not Grain.FIELD or r.success is None:
                continue
            key = (r.doc_id, r.field_name)
            ctx.prior[key] = ctx.prior.get(key, True) and r.success

    @staticmethod
    def _rollup(suite: Suite, results: list[Result]) -> list[Result]:
        """Deterministic field results also get a document verdict, so both
        grains are reportable for every check, not only the model-based one.

        The `already_scored` guard is mandatory: model-based expectations emit
        their own document row, and without it they get two contradictory ones.
        """
        agg = get_aggregator(suite.aggregator)
        already_scored = {
            (r.expectation_id, r.doc_id) for r in results if r.grain is Grain.DOCUMENT
        }
        buckets: dict[tuple[str, str], list[Result]] = {}
        for r in results:
            if r.grain is not Grain.FIELD or r.score is None or r.success is None:
                continue
            if (r.expectation_id, r.doc_id) in already_scored:
                continue  # the expectation owns its own document verdict
            buckets.setdefault((r.expectation_id, r.doc_id), []).append(r)

        rolled: list[Result] = []
        for (exp_id, doc_id), group in buckets.items():
            scores = {g.field_name: g.score for g in group}
            try:
                doc_score = agg(scores, suite.field_weights)
            except TypeError:  # a third-party aggregator taking one argument
                doc_score = agg(list(scores.values()))
            thresholds = [g.threshold for g in group if g.threshold is not None]
            weakest = min(scores, key=lambda k: scores[k] or 0.0) if scores else None
            rolled.append(
                Result(
                    expectation_id=exp_id,
                    grain=Grain.DOCUMENT,
                    doc_id=doc_id,
                    success=all(g.success for g in group),
                    score=doc_score,
                    threshold=min(thresholds) if thresholds else None,
                    threshold_source=group[0].threshold_source,
                    severity=group[0].severity,
                    evidence=Evidence(
                        "rollup",
                        {
                            "aggregation": suite.aggregator,
                            "n_fields": len(group),
                            "weakest_field": weakest,
                        },
                    ),
                    provenance=group[0].provenance,
                )
            )
        return rolled
