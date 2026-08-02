"""Value-type semantics. Small file, load-bearing assertions."""

from __future__ import annotations

import json

import pytest

from llmex import Cost, Evidence, Provenance, Result, RunResult, Severity
from llmex.types import Grain, SkipReason


def test_cost_sums_tokens_but_takes_max_latency():
    # Verifier calls run in parallel: summing latency would report 5x the
    # wall-clock truth for an ensemble.
    a = Cost(usd=0.1, calls=1, tokens_in=10, tokens_out=2, latency_ms=50.0)
    b = Cost(usd=0.2, calls=1, tokens_in=20, tokens_out=4, latency_ms=80.0)
    total = a + b
    assert total.usd == pytest.approx(0.3)
    assert total.calls == 2
    assert total.tokens_in == 30
    assert total.tokens_out == 6
    assert total.latency_ms == 80.0


def test_cost_identity():
    assert (Cost() + Cost()).as_dict()["calls"] == 0


def test_provenance_omits_unset_fields():
    p = Provenance("expect_x", "1", prompt_version="v3")
    d = p.as_dict()
    assert d == {"expectation_id": "expect_x", "expectation_version": "1",
                 "prompt_version": "v3"}


def test_result_blocking_failure_requires_error_severity():
    kw = dict(expectation_id="e", grain=Grain.FIELD, doc_id="d", field_name="f")
    assert Result(**kw, success=False, severity=Severity.ERROR).blocking_failure
    assert not Result(**kw, success=False, severity=Severity.WARN).blocking_failure
    assert not Result(**kw, success=None, severity=Severity.ERROR).blocking_failure
    assert not Result(**kw, success=True, severity=Severity.ERROR).blocking_failure


def test_result_round_trips_through_json():
    r = Result(
        expectation_id="e", grain=Grain.FIELD, doc_id="d", field_name="f",
        success=None, skip_reason=SkipReason.SAMPLED_OUT,
        evidence=Evidence("skipped", {"reason": "sampled_out"}),
        provenance=Provenance("e", "1"),
    )
    back = json.loads(json.dumps(r.as_dict(), default=str))
    assert back["skip_reason"] == "sampled_out"
    assert back["success"] is None
    assert back["threshold_source"] == "manual"


def test_run_result_summary_counts_three_states():
    def row(success):
        return Result(expectation_id="e", grain=Grain.FIELD, doc_id="d",
                      field_name="f", success=success)

    run = RunResult("r1", [row(True), row(False), row(None)], Cost(), {})
    s = run.summary()
    assert s["by_grain"]["field"] == {"pass": 1, "fail": 1, "unscored": 1}
    assert len(run.failures()) == 1
    assert len(run.unscored()) == 1
