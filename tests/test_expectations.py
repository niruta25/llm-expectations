"""Expectations against the seeded golden corpus, plus the sync shims."""

from __future__ import annotations

import pytest
from conftest import GROUNDED_FIELDS, GROUNDING_DETECTABLE, cfg, make_tiny_batch

from llmex import Context, Evidence, Grain, Runner, Suite, field_check
from llmex.expectations import (
    ExpectFieldGroundedInSource,
    ExpectFieldNullRateBetween,
    ExpectFieldsToSatisfy,
    ExpectFieldType,
)

CROSS_FIELD_RULE = (
    "subtotal + tax - total_num < 0.01 and total_num - subtotal - tax < 0.01"
)


# -- grounding --------------------------------------------------------------


def test_grounding_flags_exactly_the_seeded_errors_it_can_see(golden_batch, golden_answers):
    """The acceptance criterion: no more, no fewer.

    Grounding catches values that are absent from the source. It cannot catch
    `wrong_instance` or `misattribution`, because those values really are in
    the document — that gap is the argument for the model tier.
    """
    exp = ExpectFieldGroundedInSource(fields=GROUNDED_FIELDS)
    flagged = {
        (r.doc_id, r.field_name)
        for r in exp.check(golden_batch, Context())
        if r.success is False
    }
    expected = {
        k for k, a in golden_answers.items() if a["error_type"] in GROUNDING_DETECTABLE
    }
    assert flagged == expected


def test_grounding_evidence_names_the_branch_that_decided(golden_batch):
    exp = ExpectFieldGroundedInSource(fields=GROUNDED_FIELDS)
    by_key = {(r.doc_id, r.field_name): r for r in exp.check(golden_batch, Context())}

    assert by_key[("g1", "vendor")].evidence.kind == "exact_match"
    assert by_key[("g7", "currency")].evidence.kind == "null_value"

    fabricated = by_key[("g2", "vendor")]
    assert fabricated.evidence.kind == "fuzzy_match"
    assert "closest_source_text" in fabricated.evidence.detail
    assert fabricated.score < 0.92


def test_grounding_uses_the_declared_span_when_there_is_no_exact_match(golden_docs):
    from llmex import Batch, ExtractionRecord, SourceDoc

    # Value absent verbatim but the declared span quotes something close.
    rec = ExtractionRecord("g6", "vendor", "Lakeview Print Company", span=(0, 17))
    b = Batch([rec], lambda d: SourceDoc(d, golden_docs[d]))
    r = ExpectFieldGroundedInSource(fields=["vendor"]).check(b, Context())[0]
    assert r.evidence.kind == "span_match"
    assert r.evidence.detail["source_text"] == "Lakeview Print Co"


def test_grounding_can_be_told_that_nulls_are_not_acceptable(golden_batch):
    strict = ExpectFieldGroundedInSource(fields=["currency"], allow_null=False)
    nulls = [r for r in strict.check(golden_batch, Context()) if r.doc_id == "g7"]
    assert nulls[0].success is False


def test_grounding_carries_prompt_version_into_provenance(golden_batch):
    r = ExpectFieldGroundedInSource(fields=["vendor"]).check(golden_batch, Context())[0]
    assert r.provenance.prompt_version == "invoice-extract@v3"


# -- type -------------------------------------------------------------------


def test_type_check_flags_the_wrong_python_type(golden_batch):
    exp = ExpectFieldType(types={"vendor": "string", "subtotal": "string"})
    bad = {r.field_name for r in exp.check(golden_batch, Context()) if r.success is False}
    assert bad == {"subtotal"}  # it is a float, not a string


def test_type_check_respects_nullable(golden_batch):
    lenient = ExpectFieldType(types={"currency": "string"}, nullable=["currency"])
    strict = ExpectFieldType(types={"currency": "string"})
    assert all(r.success for r in lenient.check(golden_batch, Context()))
    assert any(r.success is False for r in strict.check(golden_batch, Context()))


def test_type_check_ignores_fields_it_was_not_given(golden_batch):
    exp = ExpectFieldType(types={"vendor": "string"})
    assert {r.field_name for r in exp.check(golden_batch, Context())} == {"vendor"}


# -- cross-field rule -------------------------------------------------------


def test_cross_field_rule_catches_the_derived_error(golden_batch):
    exp = ExpectFieldsToSatisfy(expression=CROSS_FIELD_RULE, label="total_reconciles")
    failed = {r.doc_id for r in exp.check(golden_batch, Context()) if r.success is False}
    assert failed == {"g9"}  # inputs right, total wrong — and it cost nothing


def test_cross_field_rule_emits_document_grain(golden_batch):
    exp = ExpectFieldsToSatisfy(expression=CROSS_FIELD_RULE)
    rows = exp.check(golden_batch, Context())
    assert all(r.grain is Grain.DOCUMENT for r in rows)
    assert all(r.field_name is None for r in rows)


def test_a_broken_rule_fails_loudly_rather_than_erroring_the_run(golden_batch):
    exp = ExpectFieldsToSatisfy(expression="no_such_field > 1")
    rows = exp.check(golden_batch, Context())
    assert all(r.success is False for r in rows)
    assert rows[0].evidence.kind == "rule_error"
    assert "no_such_field" in rows[0].evidence.detail["error"]


# -- statistical ------------------------------------------------------------


def test_null_rate_catches_the_omission(golden_batch):
    exp = ExpectFieldNullRateBetween(max_rate=0.0)
    failed = {r.field_name for r in exp.check(golden_batch, Context()) if r.success is False}
    assert failed == {"currency"}


def test_null_rate_reports_at_corpus_grain(golden_batch):
    rows = ExpectFieldNullRateBetween(max_rate=0.2).check(golden_batch, Context())
    assert all(r.grain is Grain.CORPUS and r.doc_id == "__corpus__" for r in rows)
    currency = next(r for r in rows if r.field_name == "currency")
    assert currency.score == pytest.approx(1 / 9)
    assert currency.success is True


# -- sync shims -------------------------------------------------------------


def test_a_field_check_function_is_instantiable_and_registered():
    # The generated class must carry `check` in its namespace at creation;
    # assigning it afterwards leaves __abstractmethods__ populated.
    @field_check(id="test_non_empty", fields=["vendor"])
    def non_empty(rec, batch, ctx):
        return bool(rec.value), Evidence("emptiness")

    from llmex import EXPECTATIONS

    assert "test_non_empty" in EXPECTATIONS.names()
    instance = EXPECTATIONS.get("test_non_empty")()
    assert instance.check is not None

    run = Runner().run_sync(
        Suite.from_dict(cfg(expectations=[{"type": "test_non_empty"}])),
        make_tiny_batch(vendor=""),
    )
    assert [r.success for r in run.results if r.grain is Grain.FIELD] == [False]


def test_a_field_check_may_return_a_bare_bool():
    @field_check(id="test_bare_bool", fields=["vendor"])
    def truthy(rec, batch, ctx):
        return bool(rec.value)

    from llmex import EXPECTATIONS

    rows = EXPECTATIONS.get("test_bare_bool")().check(make_tiny_batch(), Context())
    assert rows[0].success is True
    assert rows[0].evidence.kind == "none"


async def test_blocking_expectations_are_offloaded_to_a_thread():
    @field_check(id="test_blocking", fields=["vendor"], blocking=True)
    def slow(rec, batch, ctx):
        return True

    from llmex import EXPECTATIONS

    exp = EXPECTATIONS.get("test_blocking")()
    rows = await exp.validate(make_tiny_batch(), Context())
    assert [r.success for r in rows] == [True]
