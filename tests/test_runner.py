"""Runner: tier ordering, prior merging, rollup dedupe, the grain gap."""

from __future__ import annotations

import json

from conftest import GROUNDED_FIELDS, cfg, make_tiny_batch

from llmex import Grain, JsonlSink, LabelledScore, Runner, Suite, calibrate


def _cal():
    return calibrate(
        "c1",
        [LabelledScore("d1", "vendor", 0.9, True), LabelledScore("d1", "vendor", 0.1, False)],
        provider_id="mock",
        model_version="mock-1.0",
        strategy_id="diverse_ensemble",
    )


GOLDEN_CFG = cfg(
    expectations=[
        {"type": "expect_field_grounded_in_source", "fields": GROUNDED_FIELDS},
    ]
)


# -- both grains ------------------------------------------------------------


def test_both_grains_are_emitted_for_every_check(golden_batch):
    run = Runner().run_sync(Suite.from_dict(GOLDEN_CFG), golden_batch)
    grains = {r.grain for r in run.results}
    assert Grain.FIELD in grains and Grain.DOCUMENT in grains


def test_document_pass_rate_is_worse_than_field_pass_rate(golden_batch):
    """P4. If this ever inverts, the rollup is broken."""
    run = Runner().run_sync(Suite.from_dict(GOLDEN_CFG), golden_batch)
    field_rows = [r for r in run.results if r.grain is Grain.FIELD and r.success is not None]
    doc_rows = [r for r in run.results if r.grain is Grain.DOCUMENT and r.success is not None]
    field_rate = sum(1 for r in field_rows if r.success) / len(field_rows)
    doc_rate = sum(1 for r in doc_rows if r.success) / len(doc_rows)
    assert field_rate > doc_rate


def test_rollup_uses_the_configured_aggregator(golden_batch):
    c = dict(GOLDEN_CFG)
    c["aggregate"] = {"field_to_document": "weighted_harmonic",
                      "field_weights": {"vendor": 3.0}}
    run = Runner().run_sync(Suite.from_dict(c), golden_batch)
    doc = next(r for r in run.results if r.grain is Grain.DOCUMENT and r.doc_id == "g2")
    assert doc.evidence.kind == "rollup"
    assert doc.evidence.detail["aggregation"] == "weighted_harmonic"
    assert doc.evidence.detail["weakest_field"] == "vendor"


def test_a_self_scoring_expectation_gets_exactly_one_document_row(golden_batch):
    """The rollup dedupe. Without it, model-based checks get two contradictory rows."""
    c = cfg(
        expectations=[
            {"type": "expect_field_grounded_in_source", "fields": GROUNDED_FIELDS},
            {
                "type": "expect_field_trustworthy",
                "provider": "v",
                "audit_rate": 1.0,
                "calibration": "c1",
                "severity": "error",
            },
        ]
    )
    run = Runner().run_sync(Suite.from_dict(c, calibrations={"c1": _cal()}), golden_batch)
    keys = [(r.expectation_id, r.doc_id) for r in run.results if r.grain is Grain.DOCUMENT]
    assert len(keys) == len(set(keys))
    assert ("expect_field_trustworthy", "g1") in keys


# -- prior threading --------------------------------------------------------


def test_cheap_failures_escalate_a_document_past_the_audit_rate(golden_batch):
    """P10: a document that failed a free check is verified even at audit_rate=0."""
    c = cfg(
        expectations=[
            {"type": "expect_field_grounded_in_source", "fields": GROUNDED_FIELDS},
            {
                "type": "expect_field_trustworthy",
                "provider": "v",
                "audit_rate": 0.0,
                "calibration": "c1",
                "severity": "warn",
            },
        ]
    )
    run = Runner().run_sync(Suite.from_dict(c, calibrations={"c1": _cal()}), golden_batch)
    scored = {
        r.doc_id
        for r in run.results
        if r.expectation_id == "expect_field_trustworthy" and r.success is not None
    }
    # g2, g3 and g8 hold the grounding-detectable errors.
    assert scored == {"g2", "g3", "g8"}


def test_clean_documents_are_sampled_out_not_passed(golden_batch):
    from llmex import SkipReason

    c = cfg(
        expectations=[
            {"type": "expect_field_grounded_in_source", "fields": GROUNDED_FIELDS},
            {
                "type": "expect_field_trustworthy",
                "provider": "v",
                "audit_rate": 0.0,
                "calibration": "c1",
                "severity": "warn",
            },
        ]
    )
    run = Runner().run_sync(Suite.from_dict(c, calibrations={"c1": _cal()}), golden_batch)
    skipped = [
        r
        for r in run.results
        if r.expectation_id == "expect_field_trustworthy" and r.success is None
    ]
    assert skipped
    assert all(r.skip_reason is SkipReason.SAMPLED_OUT for r in skipped)


def test_corpus_and_unscored_rows_are_not_folded_into_the_prior(golden_batch):
    """Only field-grain verdicts gate the model tier. A corpus-grain null-rate
    row carries a field name but describes the whole batch, and an unscored row
    carries no verdict at all — neither may mark a document suspect."""
    from llmex import Context
    from llmex.runner import Runner as R

    c = cfg(
        expectations=[
            {"type": "expect_field_null_rate_between", "max_rate": 0.0, "severity": "warn"},
            {
                "type": "expect_field_trustworthy",
                "provider": "v",
                "audit_rate": 0.0,
                "calibration": "c1",
                "severity": "warn",
            },
        ]
    )
    run = Runner().run_sync(Suite.from_dict(c, calibrations={"c1": _cal()}), golden_batch)
    scored = [
        r
        for r in run.results
        if r.expectation_id == "expect_field_trustworthy" and r.success is not None
    ]
    assert scored == []  # the corpus-grain failure escalated nothing

    ctx = Context()
    R._merge_prior(ctx, [r for r in run.results if r.success is None])
    assert ctx.prior == {}


def test_rollup_tolerates_a_single_argument_aggregator(golden_batch):
    """A third-party aggregator may not accept the weights argument."""
    from llmex import AGGREGATORS

    @AGGREGATORS.plugin("p10_unweighted")
    def tenth_percentile(scores):
        vals = sorted(scores.values() if hasattr(scores, "values") else scores)
        return vals[max(0, int(len(vals) * 0.10) - 1)] if vals else 1.0

    c = dict(GOLDEN_CFG)
    c["aggregate"] = {"field_to_document": "p10_unweighted"}
    run = Runner().run_sync(Suite.from_dict(c), golden_batch)
    doc = next(r for r in run.results if r.grain is Grain.DOCUMENT and r.doc_id == "g2")
    assert doc.score is not None


# -- manifest and sinks -----------------------------------------------------


def test_manifest_pins_the_run(golden_batch):
    run = Runner().run_sync(Suite.from_dict(GOLDEN_CFG), golden_batch)
    m = run.manifest
    assert m["suite"]["expectations"][0]["id"] == "expect_field_grounded_in_source"
    assert m["n_documents"] == 9
    assert m["n_records"] == 63
    json.dumps(m)  # must round-trip


def test_jsonl_sink_writes_a_row_per_result_and_a_manifest(tmp_path, golden_batch):
    out = tmp_path / "results.jsonl"
    sink = JsonlSink(str(out))
    run = Runner(sinks=[sink]).run_sync(Suite.from_dict(GOLDEN_CFG), golden_batch)

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == len(run.results)
    assert rows[0]["run_id"] == run.run_id
    manifest = json.loads(out.with_suffix(".manifest.json").read_text())
    assert manifest["run_id"] == run.run_id


def test_console_sink_prints_a_summary(capsys, golden_batch):
    from llmex import ConsoleSink

    Runner(sinks=[ConsoleSink()]).run_sync(Suite.from_dict(GOLDEN_CFG), golden_batch)
    out = capsys.readouterr().out
    assert "blocking failures" in out
    assert "expect_field_grounded_in_source" in out


async def test_the_async_surface_produces_the_same_verdicts_as_the_sync_one(golden_batch):
    suite = Suite.from_dict(GOLDEN_CFG)
    first = await Runner().run(suite, golden_batch)
    second = await Runner().run(suite, golden_batch)
    assert first.run_id != second.run_id  # each run is its own artifact
    assert [(r.doc_id, r.field_name, r.success) for r in first.results] == [
        (r.doc_id, r.field_name, r.success) for r in second.results
    ]


def test_run_sync_needs_no_event_loop_of_its_own():
    run = Runner().run_sync(Suite.from_dict(GOLDEN_CFG), make_tiny_batch())
    assert run.results
