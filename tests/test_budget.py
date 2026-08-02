"""Budget: exhaustion marks results unscored, and never converts to a pass."""

from __future__ import annotations

import asyncio

from conftest import cfg, make_tiny_batch

from llmex import Budget, Cost, LabelledScore, Runner, SkipReason, Suite, calibrate


def _cal():
    return calibrate(
        "c1",
        [LabelledScore("d1", "vendor", 0.9, True), LabelledScore("d1", "vendor", 0.1, False)],
        provider_id="mock",
        model_version="mock-1.0",
        strategy_id="diverse_ensemble",
    )


def _model_suite(budget: Budget) -> Suite:
    c = cfg(
        expectations=[
            {
                "type": "expect_field_trustworthy",
                "provider": "v",
                "audit_rate": 1.0,
                "calibration": "c1",
                "severity": "error",
            }
        ]
    )
    suite = Suite.from_dict(c, calibrations={"c1": _cal()})
    suite.budget = budget
    return suite


async def test_reserve_refuses_once_the_cap_is_reached():
    b = Budget(max_calls=5)
    assert await b.reserve(5)
    await b.record(Cost(calls=5))
    assert not await b.reserve(1)
    assert b.exhausted


async def test_reserve_refuses_on_money_as_well_as_calls():
    b = Budget(max_usd=1.0)
    await b.record(Cost(usd=0.9))
    assert await b.reserve(1, est_usd=0.05)
    assert not await b.reserve(1, est_usd=0.5)
    assert b.remaining_usd() < 0.11


async def test_concurrent_records_do_not_lose_updates():
    b = Budget()
    await asyncio.gather(*(b.record(Cost(usd=0.01, calls=1)) for _ in range(100)))
    assert b.spent.calls == 100


def test_exhaustion_marks_unscored_never_passes():
    """P5. Budget exhaustion must never convert a run to green."""
    run = Runner().run_sync(_model_suite(Budget(max_calls=0)), make_tiny_batch())
    rows = [r for r in run.results if r.expectation_id == "expect_field_trustworthy"]
    assert rows
    assert all(r.success is None for r in rows)
    assert all(r.skip_reason is SkipReason.BUDGET_EXHAUSTED for r in rows)
    assert run.summary()["blocking_failures"] == 0
    assert len(run.unscored()) == len(rows)


def test_a_generous_budget_lets_the_check_run():
    run = Runner().run_sync(_model_suite(Budget(max_usd=10.0)), make_tiny_batch())
    rows = [r for r in run.results if r.expectation_id == "expect_field_trustworthy"]
    assert all(r.success is not None for r in rows)
    assert run.cost.calls > 0


def test_a_field_the_verifier_did_not_score_is_unscored_not_passed():
    """A missing field score used to default to 0.5 against a 0.5 threshold,
    which compares as a pass. An absent measurement is not a passing one."""
    from llmex import PROVIDERS
    from llmex.providers.mock import MockProvider

    class PartialScorer(MockProvider):
        """Answers about `vendor` and stays silent about everything else."""

        id = "partial"

        async def complete(self, req):
            resp = await super().complete(req)
            kept = {k: v for k, v in (resp.parsed or {})["field_scores"].items()
                    if k == "vendor"}
            resp.parsed = {"field_scores": kept}
            return resp

    PROVIDERS.register("partial", PartialScorer)
    c = cfg(
        providers={"v": {"plugin": "partial"}},
        expectations=[
            {
                "type": "expect_field_trustworthy",
                "provider": "v",
                "audit_rate": 1.0,
                "severity": "warn",
            }
        ],
    )
    run = Runner().run_sync(Suite.from_dict(c), make_tiny_batch())
    rows = {
        r.field_name: r
        for r in run.results
        if r.expectation_id == "expect_field_trustworthy" and r.field_name
    }
    assert rows["vendor"].success is True
    assert rows["total_amount"].success is None
    assert rows["total_amount"].skip_reason is SkipReason.NOT_APPLICABLE


def test_a_dead_provider_becomes_a_skip_not_a_crash():
    from llmex import PROVIDERS
    from llmex.providers.mock import MockProvider

    class Exploding(MockProvider):
        id = "exploding"

        async def complete(self, req):
            raise RuntimeError("provider is down")

    PROVIDERS.register("exploding", Exploding)
    c = cfg(
        providers={"v": {"plugin": "exploding"}},
        expectations=[
            {
                "type": "expect_field_trustworthy",
                "provider": "v",
                "audit_rate": 1.0,
                "severity": "warn",
            }
        ],
    )
    run = Runner().run_sync(Suite.from_dict(c), make_tiny_batch())
    rows = [r for r in run.results if r.expectation_id == "expect_field_trustworthy"]
    assert rows
    assert all(r.skip_reason is SkipReason.PROVIDER_ERROR for r in rows)
    assert "provider is down" in rows[0].evidence.detail["detail"]
