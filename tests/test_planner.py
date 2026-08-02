"""One test per guard. Everything that can fail must fail before a token is spent."""

from __future__ import annotations

import pytest
from conftest import cfg, make_tiny_batch

from llmex import (
    Batch,
    Capability,
    ExtractionRecord,
    Kind,
    LabelledScore,
    PlanError,
    Planner,
    SourceDoc,
    Suite,
    calibrate,
)


def _cal(model_version: str = "mock-1.0", strategy_id: str = "diverse_ensemble"):
    return calibrate(
        "c1",
        [
            LabelledScore("d1", "vendor", 0.9, True),
            LabelledScore("d1", "total_amount", 0.1, False),
        ],
        provider_id="mock",
        model_version=model_version,
        strategy_id=strategy_id,
    )


def _model_cfg(**over):
    spec = {"type": "expect_field_trustworthy", "provider": "v"}
    spec.update(over)
    return cfg(expectations=[spec])


# -- guards -----------------------------------------------------------------


def test_uncalibrated_model_check_cannot_block():
    with pytest.raises(PlanError, match="requires a calibration"):
        Planner().plan(Suite.from_dict(_model_cfg(severity="error")), make_tiny_batch())


def test_uncalibrated_model_check_may_warn():
    # The same config at severity=warn plans cleanly.
    plan = Planner().plan(Suite.from_dict(_model_cfg(severity="warn")), make_tiny_batch())
    assert len(plan.steps) == 1


def test_capability_gap_fails_at_plan_time():
    from llmex import PROVIDERS

    class NoLogprobs(PROVIDERS.get("mock")):
        id = "nolp"
        model_version = "nolp-1"
        capabilities = frozenset({Capability.STRUCTURED_OUTPUT})

    PROVIDERS.register("nolp", NoLogprobs)
    c = cfg(
        providers={"v": {"plugin": "nolp"}},
        expectations=[
            {
                "type": "expect_field_trustworthy",
                "provider": "v",
                "strategy": "logprob",
                "severity": "warn",
            }
        ],
    )
    with pytest.raises(PlanError, match="logprobs"):
        Planner().plan(Suite.from_dict(c), make_tiny_batch())


def test_stale_calibration_is_rejected():
    c = _model_cfg(calibration="c1", severity="error")
    suite = Suite.from_dict(c, calibrations={"c1": _cal(model_version="OLD-VERSION")})
    with pytest.raises(PlanError, match="Recalibrate"):
        Planner().plan(suite, make_tiny_batch())


def test_calibration_fitted_on_another_strategy_is_rejected():
    c = _model_cfg(calibration="c1", severity="error", strategy="single_judge")
    suite = Suite.from_dict(c, calibrations={"c1": _cal()})
    with pytest.raises(PlanError, match="Recalibrate"):
        Planner().plan(suite, make_tiny_batch())


def test_unknown_provider_alias_lists_the_defined_ones():
    c = cfg(expectations=[{"type": "expect_field_trustworthy", "provider": "typo",
                           "severity": "warn"}])
    with pytest.raises(PlanError, match="no provider aliased 'typo'"):
        Planner().plan(Suite.from_dict(c), make_tiny_batch())


def test_unknown_strategy_alias_lists_the_defined_ones():
    c = _model_cfg(strategy="typo", severity="warn")
    with pytest.raises(PlanError, match="no strategy aliased 'typo'"):
        Planner().plan(Suite.from_dict(c), make_tiny_batch())


def test_self_verification_warns_but_does_not_block():
    # There are legitimate reasons to self-verify. It must be loud, because a
    # self-graded score reads systematically high.
    recs = [ExtractionRecord("d1", "vendor", "X", generator_model="mock-1.0")]
    b = Batch(recs, lambda d: SourceDoc(d, "text"))
    suite = Suite.from_dict(_model_cfg(calibration="c1", severity="error"),
                            calibrations={"c1": _cal()})
    plan = Planner().plan(suite, b)
    assert any("correlated" in w for w in plan.warnings)


def test_over_budget_warns_that_checks_will_be_unscored():
    c = _model_cfg(calibration="c1", severity="error", audit_rate=1.0)
    c["budget"] = {"max_usd": 0.0}
    suite = Suite.from_dict(c, calibrations={"c1": _cal()})
    plan = Planner().plan(suite, make_tiny_batch())
    assert any("exceeds budget" in w for w in plan.warnings)


# -- ordering and estimation ------------------------------------------------


def test_tier_ordering_is_cheap_first():
    c = cfg(
        expectations=[
            {"type": "expect_field_trustworthy", "provider": "v", "severity": "warn"},
            {"type": "expect_field_null_rate_between", "severity": "warn"},
            {"type": "expect_field_grounded_in_source"},
        ]
    )
    plan = Planner().plan(Suite.from_dict(c), make_tiny_batch())
    assert [s.kind for s in plan.ordered()] == [
        Kind.DETERMINISTIC,
        Kind.STATISTICAL,
        Kind.MODEL_BASED,
    ]


def test_deterministic_steps_cost_nothing_to_plan():
    plan = Planner().plan(Suite.from_dict(cfg()), make_tiny_batch())
    assert plan.estimated_calls == 0
    assert plan.estimated_usd == 0.0


def test_the_cost_estimate_token_shape_is_overridable():
    """It is a guess by construction — the planner has not read the documents.
    Whoever has measured their own corpus must be able to say so."""
    suite = Suite.from_dict(_model_cfg(severity="warn", audit_rate=1.0))
    default = Planner().plan(suite, make_tiny_batch())
    bigger = Planner(estimated_tokens_in=20_000, estimated_tokens_out=2_000).plan(
        suite, make_tiny_batch()
    )
    assert bigger.estimated_usd == pytest.approx(default.estimated_usd * 10)
    assert bigger.estimated_calls == default.estimated_calls  # calls are unaffected


def test_model_steps_are_costed_before_running():
    plan = Planner().plan(
        Suite.from_dict(_model_cfg(severity="warn", audit_rate=1.0)), make_tiny_batch()
    )
    assert plan.estimated_calls > 0
    assert plan.estimated_usd > 0.0
    assert plan.as_dict()["steps"][0]["expectation_id"] == "expect_field_trustworthy"
