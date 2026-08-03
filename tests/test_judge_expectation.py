"""The judge-backed expectation and the check that grades the judge."""

from __future__ import annotations

import json

import pytest

from llmex import (
    Batch,
    ExtractionRecord,
    Grain,
    Kind,
    LabelledDecision,
    PlanError,
    Planner,
    Runner,
    SkipReason,
    SourceDoc,
    Suite,
    calibrate_judge,
)
from llmex.expectations import ExpectExtractionLabelTrustworthy, ExpectFieldTrustworthy

SESSIONS = {
    "s1": "User asked us to reset the password on their locked account.",
    "s2": "Customer disputed a duplicate billing charge on the invoice.",
    "s3": "Caller wanted to cancel the subscription before the renewal date.",
    "s4": "User could not export their data and asked for a data export.",
}

# Placeholder taxonomy. Swap for the real JTBD categories in one place.
LABELS = ["password_reset", "billing_dispute", "cancel_subscription", "data_export"]

TRUTH = {
    "s1": "password_reset",
    "s2": "billing_dispute",
    "s3": "cancel_subscription",
    "s4": "data_export",
}
PREDICTED = {**TRUTH, "s3": "billing_dispute"}  # one seeded mislabel


def session_batch(with_gold=True):
    records = []
    for doc_id in SESSIONS:
        meta = {"gold_label": TRUTH[doc_id]} if with_gold else {}
        records.append(
            ExtractionRecord(doc_id, "jtbd_label", PREDICTED[doc_id], meta=meta)
        )
    return Batch(records, lambda d: SourceDoc(d, SESSIONS[d]))


def _cal():
    decisions = [
        LabelledDecision(f"c{i}", "jtbd_label", 0.15 if i % 2 else 0.92,
                         LABELS[i % 4], LABELS[(i + (1 if i % 2 else 0)) % 4])
        for i in range(20)
    ]
    return calibrate_judge(
        "jtbd_v1", decisions, "mock_judge", "mock-judge-1.0", "label_judge"
    )


def judge_cfg(**over):
    spec = {
        "type": "expect_extraction_label_trustworthy",
        "fields": ["jtbd_label"],
        "provider": "j",
        "audit_rate": 1.0,
        "calibration": "jtbd_v1",
        "severity": "error",
    }
    spec.update(over)
    return {
        "suite": "jtbd",
        "providers": {"j": {"plugin": "mock_judge"}},
        "strategies": {"label_judge": {"plugin": "label_judge", "label_set": LABELS}},
        "expectations": [spec],
    }


# -- it really is a thin subclass -------------------------------------------


def test_it_inherits_the_parent_rather_than_reimplementing_it():
    assert issubclass(ExpectExtractionLabelTrustworthy, ExpectFieldTrustworthy)
    # The whole point: no overridden validate(), so escalation, budget,
    # skip paths and threshold provenance cannot drift from the parent's.
    assert "validate" not in ExpectExtractionLabelTrustworthy.__dict__


def test_it_only_overrides_the_question_being_asked():
    exp = ExpectExtractionLabelTrustworthy()
    assert exp.default_strategy == "label_judge"
    assert exp.field_evidence_kind == "label_verdict"
    assert exp.kind is Kind.MODEL_BASED


def test_it_inherits_the_uncalibrated_guard():
    c = judge_cfg(calibration=None, severity="error")
    c["expectations"][0].pop("calibration")
    with pytest.raises(PlanError, match="requires a calibration"):
        Planner().plan(Suite.from_dict(c), session_batch())


def test_it_inherits_the_staleness_guard():
    stale = _cal()
    stale.model_version = "mock-judge-9.9"
    with pytest.raises(PlanError, match="Recalibrate"):
        Planner().plan(
            Suite.from_dict(judge_cfg(), calibrations={"jtbd_v1": stale}), session_batch()
        )


# -- behaviour --------------------------------------------------------------


def test_it_emits_both_grains_with_label_specific_evidence():
    run = Runner().run_sync(
        Suite.from_dict(judge_cfg(), calibrations={"jtbd_v1": _cal()}), session_batch()
    )
    rows = [r for r in run.results if r.expectation_id == "expect_extraction_label_trustworthy"]
    fields = [r for r in rows if r.grain is Grain.FIELD]
    docs = [r for r in rows if r.grain is Grain.DOCUMENT]
    assert len(fields) == 4
    assert len(docs) == 4  # exactly one per document, no rollup duplicate
    assert fields[0].evidence.kind == "label_verdict"
    assert docs[0].evidence.kind == "label_verdict_doc"


def test_verification_cost_is_reported_once_per_document_not_once_per_row():
    """One call scored the whole document. The runner sums cost across every
    result row, so attributing the same ScoreSet to each field row as well
    would report the spend two or three times over."""
    run = Runner().run_sync(
        Suite.from_dict(judge_cfg(), calibrations={"jtbd_v1": _cal()}), session_batch()
    )
    rows = [r for r in run.results if r.expectation_id == "expect_extraction_label_trustworthy"]
    assert run.cost.calls == 4  # four sessions, one judge call each
    assert sum(r.cost.calls for r in rows) == 4
    assert all(r.cost.calls == 0 for r in rows if r.grain is Grain.FIELD)


def test_it_only_grades_the_fields_it_was_configured_for():
    """A verifier bills for every field handed to it, so a config naming one
    field on a two-field document must not quietly pay for both."""
    b = Batch(
        [
            ExtractionRecord("s1", "jtbd_label", "password_reset"),
            ExtractionRecord("s1", "outcome", "resolved"),
        ],
        lambda d: SourceDoc(d, SESSIONS["s1"]),
    )
    run = Runner().run_sync(
        Suite.from_dict(judge_cfg(), calibrations={"jtbd_v1": _cal()}), b
    )
    graded = {
        r.field_name
        for r in run.results
        if r.expectation_id == "expect_extraction_label_trustworthy" and r.field_name
    }
    assert graded == {"jtbd_label"}


def test_the_judge_explains_itself():
    run = Runner().run_sync(
        Suite.from_dict(judge_cfg(), calibrations={"jtbd_v1": _cal()}), session_batch()
    )
    row = next(
        r for r in run.results
        if r.expectation_id == "expect_extraction_label_trustworthy" and r.field_name
    )
    assert row.evidence.detail["explanation"]  # empty for every other strategy


def test_the_threshold_carries_its_calibration_provenance():
    cal = _cal()
    run = Runner().run_sync(
        Suite.from_dict(judge_cfg(), calibrations={"jtbd_v1": cal}), session_batch()
    )
    row = next(
        r for r in run.results
        if r.expectation_id == "expect_extraction_label_trustworthy" and r.field_name
    )
    assert row.threshold_source == cal.ref
    assert row.threshold_source.startswith("calibration:jtbd_v1@")


def test_the_judge_never_sees_the_gold_label():
    """Structural guarantee, asserted end to end: gold rides on
    ExtractionRecord.meta and the payload handed to a strategy has no field
    for it, so a judge cannot grade itself against the answer key."""
    seen: list[str] = []

    from llmex import PROVIDERS
    from llmex.providers.mock import MockLabelJudge

    class Recording(MockLabelJudge):
        # Same id and model_version as the fixture judge: changing either would
        # trip the staleness guard, which is a different test.
        async def complete(self, req):
            seen.append(json.dumps({"system": req.system, "user": req.user}))
            return await super().complete(req)

    PROVIDERS.register("recording", Recording)
    c = judge_cfg()
    c["providers"] = {"j": {"plugin": "recording"}}
    Runner().run_sync(Suite.from_dict(c, calibrations={"jtbd_v1": _cal()}), session_batch())

    assert seen
    for sent in seen:
        assert "gold_label" not in sent
        assert "cancel_subscription" not in sent or "s3" not in sent


# -- expect_judge_agrees_with_gold ------------------------------------------


def agreement_cfg(**over):
    c = judge_cfg()
    spec = {
        "type": "expect_judge_agrees_with_gold",
        "fields": ["jtbd_label"],
        "min_kappa": 0.4,
        "severity": "warn",
    }
    spec.update(over)
    c["expectations"].append(spec)
    return c


def test_it_runs_after_the_judge_it_grades():
    plan = Planner().plan(
        Suite.from_dict(agreement_cfg(), calibrations={"jtbd_v1": _cal()}), session_batch()
    )
    kinds = [s.kind for s in plan.ordered()]
    assert kinds == [Kind.MODEL_BASED, Kind.DERIVED]  # derived is always last


def test_it_reports_kappa_at_corpus_grain():
    run = Runner().run_sync(
        Suite.from_dict(agreement_cfg(), calibrations={"jtbd_v1": _cal()}), session_batch()
    )
    row = next(r for r in run.results if r.expectation_id == "expect_judge_agrees_with_gold")
    assert row.grain is Grain.CORPUS
    assert row.evidence.kind == "judge_agreement"
    assert row.evidence.detail["n"] == 4
    assert "cohens_kappa" in row.evidence.detail
    assert "raw_agreement" in row.evidence.detail
    assert row.score is not None


def test_without_gold_it_is_unscored_rather_than_passing():
    """"We could not check the judge" and "the judge is fine" are different
    statements, and only one of them is green."""
    run = Runner().run_sync(
        Suite.from_dict(agreement_cfg(), calibrations={"jtbd_v1": _cal()}),
        session_batch(with_gold=False),
    )
    row = next(r for r in run.results if r.expectation_id == "expect_judge_agrees_with_gold")
    assert row.success is None
    assert row.skip_reason is SkipReason.NOT_APPLICABLE


def test_it_costs_nothing_because_it_reads_results_that_already_exist():
    run = Runner().run_sync(
        Suite.from_dict(agreement_cfg(), calibrations={"jtbd_v1": _cal()}), session_batch()
    )
    row = next(r for r in run.results if r.expectation_id == "expect_judge_agrees_with_gold")
    assert row.cost.calls == 0
    assert row.cost.usd == 0.0
