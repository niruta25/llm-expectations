"""Classification metrics and the two free label checks."""

from __future__ import annotations

import pytest

from llmex import (
    Batch,
    Context,
    ExtractionRecord,
    Grain,
    SkipReason,
    SourceDoc,
    confusion_matrix,
    normalise_label,
    report,
    report_from_batch,
)
from llmex.expectations import ExpectFieldMatchesGold, ExpectLabelDistributionStable


def label_batch(rows, field="jtbd_label"):
    """rows: (doc_id, predicted, gold_or_None)"""
    records = []
    for doc_id, pred, gold in rows:
        meta = {} if gold is None else {"gold_label": gold}
        records.append(ExtractionRecord(doc_id, field, pred, meta=meta))
    return Batch(records, lambda d: SourceDoc(d, f"session {d}"))


# -- normalisation ----------------------------------------------------------


@pytest.mark.parametrize(
    "raw", ["password_reset", "Password Reset", "password-reset", "  PASSWORD   reset "]
)
def test_one_label_written_four_ways_is_one_label(raw):
    # A taxonomy that survives several annotators contains all of these.
    assert normalise_label(raw) == "password reset"


# -- metrics ----------------------------------------------------------------


def test_confusion_matrix_counts_pairs():
    m = confusion_matrix([("a", "a"), ("a", "b"), ("b", "b"), ("a", "a")])
    assert m == {("a", "a"): 2, ("a", "b"): 1, ("b", "b"): 1}


def test_report_computes_accuracy_and_per_label_metrics():
    #  gold a: 2 right, 1 called b   |  gold b: 1 right
    r = report([("a", "a"), ("a", "a"), ("a", "b"), ("b", "b")])
    assert r.n == 4
    assert r.accuracy == pytest.approx(0.75)
    assert r.per_label["a"].recall == pytest.approx(2 / 3)
    assert r.per_label["a"].precision == pytest.approx(1.0)
    assert r.per_label["b"].precision == pytest.approx(0.5)
    assert r.per_label["b"].recall == pytest.approx(1.0)


def test_macro_f1_refuses_to_let_a_rare_label_hide_behind_volume():
    # 18 of one label all correct, 2 of another all wrong. Accuracy reads 90%.
    pairs = [("common", "common")] * 18 + [("rare", "common")] * 2
    r = report(pairs)
    assert r.accuracy == pytest.approx(0.9)
    assert r.macro_f1 < 0.55  # the honest number
    assert r.worst_labels(1)[0].label == "rare"


def test_report_survives_an_empty_corpus():
    r = report([])
    assert r.n == 0 and r.accuracy == 1.0


def test_report_is_json_serialisable():
    import json

    d = report([("a", "b")]).as_dict()
    assert json.loads(json.dumps(d))["matrix"] == {"a|b": 1}


def test_report_from_batch_ignores_records_without_gold():
    b = label_batch([("d1", "a", "a"), ("d2", "b", None), ("d3", "b", "a")])
    r = report_from_batch(b, "jtbd_label")
    assert r.n == 2  # d2 is not evidence either way
    assert r.accuracy == pytest.approx(0.5)


# -- expect_field_matches_gold ----------------------------------------------


def test_gold_comparison_flags_the_mismatches():
    b = label_batch([("d1", "password_reset", "password_reset"),
                     ("d2", "billing_dispute", "cancel_subscription")])
    rows = {r.doc_id: r for r in ExpectFieldMatchesGold().check(b, Context())}
    assert rows["d1"].success is True
    assert rows["d1"].evidence.kind == "gold_match"
    assert rows["d2"].success is False
    assert rows["d2"].evidence.detail == {
        "expected": "cancel_subscription", "got": "billing_dispute", "normalised": True
    }


def test_gold_comparison_normalises_by_default_and_can_be_told_not_to():
    b = label_batch([("d1", "Password Reset", "password_reset")])
    assert ExpectFieldMatchesGold().check(b, Context())[0].success is True
    strict = ExpectFieldMatchesGold(normalise=False)
    assert strict.check(b, Context())[0].success is False


def test_an_unlabelled_record_is_unscored_not_passed():
    """A record nobody labelled is not evidence of correctness. Letting it read
    green is how a gold set silently stops covering its corpus."""
    b = label_batch([("d1", "password_reset", None)])
    row = ExpectFieldMatchesGold().check(b, Context())[0]
    assert row.success is None
    assert row.skip_reason is SkipReason.NOT_APPLICABLE
    assert "gold_label" in row.evidence.detail["detail"]


def test_gold_key_is_configurable():
    recs = [ExtractionRecord("d1", "lbl", "a", meta={"human": "b"})]
    b = Batch(recs, lambda d: SourceDoc(d, ""))
    assert ExpectFieldMatchesGold(gold_key="human").check(b, Context())[0].success is False


# -- expect_label_distribution_stable ---------------------------------------


def test_collapse_onto_one_label_is_caught_even_when_accuracy_is_fine():
    rows = [(f"d{i}", "password_reset", "password_reset") for i in range(19)]
    rows.append(("d19", "billing_dispute", "billing_dispute"))
    b = label_batch(rows)

    # Every single prediction is correct...
    assert report_from_batch(b, "jtbd_label").accuracy == pytest.approx(1.0)
    # ...and the distribution still says the classifier stopped discriminating.
    row = ExpectLabelDistributionStable(max_share=0.9).check(b, Context())[0]
    assert row.success is False
    assert row.evidence.detail["commonest"] == "password reset"
    assert row.evidence.detail["commonest_share"] == pytest.approx(0.95)


def test_a_healthy_spread_passes_at_corpus_grain():
    rows = [(f"d{i}", ["a", "b", "c"][i % 3], None) for i in range(9)]
    row = ExpectLabelDistributionStable().check(label_batch(rows), Context())[0]
    assert row.success is True
    assert row.grain is Grain.CORPUS
    assert row.doc_id == "__corpus__"
    assert row.evidence.detail["distinct"] == 3


def test_a_single_distinct_label_fails_on_min_distinct():
    rows = [(f"d{i}", "only_one", None) for i in range(4)]
    row = ExpectLabelDistributionStable(max_share=1.0).check(label_batch(rows), Context())[0]
    assert row.success is False
    assert any("distinct" in f for f in row.evidence.detail["failures"])


def test_drift_from_a_baseline_is_measured_as_total_variation_distance():
    rows = [(f"d{i}", "a" if i < 8 else "b", None) for i in range(10)]
    b = label_batch(rows)

    steady = ExpectLabelDistributionStable(
        baseline={"a": 0.8, "b": 0.2}, max_share=1.0
    ).check(b, Context())[0]
    assert steady.success is True
    assert steady.evidence.detail["baseline_shift"] == pytest.approx(0.0)

    moved = ExpectLabelDistributionStable(
        baseline={"a": 0.4, "b": 0.6}, max_share=1.0, max_shift=0.25
    ).check(b, Context())[0]
    assert moved.success is False
    assert moved.evidence.detail["baseline_shift"] == pytest.approx(0.4)


def test_nulls_are_excluded_from_the_distribution():
    rows = [("d1", "a", None), ("d2", None, None), ("d3", "b", None)]
    row = ExpectLabelDistributionStable(max_share=1.0).check(label_batch(rows), Context())[0]
    assert row.evidence.detail["n"] == 2
