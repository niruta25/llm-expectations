"""The third field kind: a value that is supposed to be *in* the document."""

from __future__ import annotations

from datetime import date

import pytest

from llm_expectations.checks import CheckContext, run_checks
from llm_expectations.checks.copied import (
    candidates,
    normalise,
    parse_date,
    parse_number,
)
from llm_expectations.config import settings_from_mapping
from llm_expectations.judges.prompts import GroundednessTask
from llm_expectations.schema import ValueType, schema_from_mapping
from llm_expectations.types import Item, Output, Status

INVOICE = (
    "Northwind Supply invoice INV-3001 dated March 3, 2026. "
    "Subtotal $1,200.00, shipping $34.50, total $1,234.50. Terms net 30."
)


def build(rows, schema=None):
    """``rows`` maps an item id to (document text, extracted values)."""
    schema = schema or {
        "item": "invoice",
        "fields": {
            "total": {"kind": "copied", "value_type": "number"},
            "issued": {"kind": "copied", "value_type": "date"},
            "vendor": {"kind": "copied"},
        },
    }
    return CheckContext(
        run_id="t",
        schema=schema_from_mapping(schema),
        settings=settings_from_mapping({}),
        items={k: Item(k, text) for k, (text, _) in rows.items()},
        outputs={k: Output(k, values) for k, (_, values) in rows.items()},
        taxonomies={},
    )


def results(ctx, check):
    findings, _ = run_checks(ctx)
    return [f for f in findings if f.check == check]


class TestParsing:
    @pytest.mark.parametrize(
        "text, expected",
        [("$1,234.50", 1234.5), ("1234.5", 1234.5), ("1 234,50", None), ("pending", None)],
    )
    def test_numbers_ignore_currency_and_separators(self, text, expected):
        got = parse_number(text)
        if expected is None:
            assert got is None or got != 1234.5
        else:
            assert got == expected

    @pytest.mark.parametrize(
        "text",
        ["2026-03-03", "March 3, 2026", "Mar. 3, 2026", "3 March 2026", "3rd March 2026"],
    )
    def test_the_understood_date_shapes_all_read_the_same_day(self, text):
        assert parse_date(text) == date(2026, 3, 3)

    def test_an_ambiguous_date_is_refused_rather_than_guessed(self):
        # 03/03 is unambiguous but 04/05 is not, and a parser that read one
        # would read the other. A date silently taken the wrong way round is
        # worse than one the tool admits it cannot read.
        assert parse_date("04/05/2026") is None

    def test_an_impossible_date_is_not_invented(self):
        assert parse_date("February 30, 2026") is None

    def test_normalise_folds_case_and_whitespace(self):
        assert normalise("  Northwind   SUPPLY ") == "northwind supply"


class TestCandidates:
    def test_numbers_inside_a_date_are_not_candidate_amounts(self):
        # Otherwise every document carrying a date looks more ambiguous than
        # it is, and the gate pays for rows it did not need to.
        found = candidates(INVOICE, ValueType.NUMBER)
        assert "2026" not in found
        assert "3" not in found
        assert {"$1,200.00", "$34.50", "$1,234.50"} <= set(found)

    def test_dates_are_found_in_any_understood_shape(self):
        assert candidates(INVOICE, ValueType.DATE) == ["March 3, 2026"]

    def test_text_fields_have_no_enumerable_candidates(self):
        assert candidates(INVOICE, ValueType.TEXT) == []


class TestValueInSource:
    def test_a_value_the_document_contains_passes(self):
        ctx = build({"i-1": (INVOICE, {"total": "1234.50"})})
        assert results(ctx, "value_in_source")[0].status is Status.PASS

    def test_a_value_the_document_does_not_contain_is_invented(self):
        found = results(build({"i-1": (INVOICE, {"total": "9999.00"})}), "value_in_source")[0]
        assert found.status is Status.FAIL
        assert "was invented" in found.evidence["why"]

    def test_a_reformatted_amount_still_counts_as_present(self):
        # The document says "$1,234.50"; the model said "1234.5". Matching
        # characters would call that an invention and the grounding rate
        # would measure formatting rather than fidelity.
        ctx = build({"i-1": (INVOICE, {"total": "1234.5"})})
        assert results(ctx, "value_in_source")[0].status is Status.PASS

    def test_a_reformatted_date_still_counts_as_present(self):
        ctx = build({"i-1": (INVOICE, {"issued": "2026-03-03"})})
        found = next(f for f in results(ctx, "value_in_source") if f.field == "issued")
        assert found.status is Status.PASS

    def test_require_verbatim_demands_the_characters(self):
        schema = {
            "item": "invoice",
            "fields": {
                "total": {"kind": "copied", "value_type": "number", "require_verbatim": True}
            },
        }
        ctx = build({"i-1": (INVOICE, {"total": "1234.5"})}, schema=schema)
        assert results(ctx, "value_in_source")[0].status is Status.FAIL

    def test_the_candidate_count_rides_along_as_evidence(self):
        # Presence is weak evidence in a document full of amounts, and the
        # gate needs to know that.
        found = results(build({"i-1": (INVOICE, {"total": "1234.50"})}), "value_in_source")[0]
        assert found.evidence["candidates"] >= 3

    def test_an_unreadable_value_is_unscored_here_not_failed(self):
        # value_shape owns that row; failing it twice would make one defect
        # look like two.
        found = results(build({"i-1": (INVOICE, {"total": "pending"})}), "value_in_source")
        assert found[0].status is Status.UNSCORED


class TestValueShape:
    def test_a_value_that_is_not_a_number_at_all_is_a_different_failure(self):
        found = results(build({"i-1": (INVOICE, {"total": "pending"})}), "value_shape")[0]
        assert found.status is Status.FAIL
        assert "a different question" in found.evidence["why"]

    def test_a_text_field_has_no_shape_to_check(self):
        schema = {"item": "invoice", "fields": {"vendor": {"kind": "copied"}}}
        ctx = build({"i-1": (INVOICE, {"vendor": "Northwind Supply"})}, schema=schema)
        assert results(ctx, "value_shape") == []


class TestGroundingRate:
    def test_it_is_the_share_of_values_their_documents_contain(self):
        rows = {
            f"i-{i}": (INVOICE, {"total": "1234.50" if i < 9 else "9999.00"})
            for i in range(10)
        }
        found = next(f for f in results(build(rows), "grounding_rate") if f.field == "total")
        assert found.score == pytest.approx(0.9)

    def test_it_is_called_a_floor_rather_than_a_score(self):
        rows = {f"i-{i}": (INVOICE, {"total": "9999.00"}) for i in range(10)}
        found = next(f for f in results(build(rows), "grounding_rate") if f.field == "total")
        assert found.status is Status.FAIL
        assert "floor, not a score" in found.evidence["why"]


class TestTheGroundednessJudge:
    def test_it_is_asked_which_value_belongs_in_the_field(self):
        request = GroundednessTask().build(
            Item("i-1", INVOICE), Output("i-1", {"total": "34.50"}), "total", None
        )
        assert "EXTRACTED VALUE: 34.50" in request.user
        assert "rather than a different number" in request.system

    def test_it_catches_the_failure_presence_cannot(self):
        # 34.50 IS in the document. It is the shipping charge.
        parsed = GroundednessTask().parse(
            '{"correct": false, "confidence": 0.9, '
            '"reason": "that is the shipping charge", "instead": "$1,234.50"}'
        )
        assert parsed.status is Status.FAIL
        assert parsed.detail["instead"] == "$1,234.50"

    def test_an_unreadable_reply_is_unscored(self):
        assert GroundednessTask().parse("sure thing").status is Status.UNSCORED


class TestTheGate:
    def _ctx_and_plan(self, rows):
        from llm_expectations.run import flagged_by_free_checks, gated_plan

        ctx = build(rows)
        findings, _ = run_checks(ctx)

        class _Config:
            """Only what the gate reads, so the test needs no project on disk."""

            schema = ctx.schema
            settings = ctx.settings

        plan = gated_plan(
            _Config(), ctx.items, ctx.outputs, flagged_by_free_checks(findings), ctx=ctx
        )
        return ctx, plan

    def test_an_invented_value_is_sent_to_the_judge(self):
        _, plan = self._ctx_and_plan({"i-1": (INVOICE, {"total": "9999.00"})})
        assert "i-1" in plan["total"][0]

    def test_an_ambiguous_document_is_sent_even_though_it_passed(self):
        # The invoice holds several amounts, so finding the right one there
        # is a coincidence until a judge says otherwise.
        _, plan = self._ctx_and_plan({"i-1": (INVOICE, {"total": "1234.50"})})
        assert "i-1" in plan["total"][0]

    def test_a_document_with_one_candidate_is_not_sent(self):
        single = "Receipt from Cobalt Print. Amount due $88.00."
        _, plan = self._ctx_and_plan({"i-1": (single, {"total": "88.00"})})
        # Nothing flagged it and nothing was ambiguous, so it falls to the
        # audit sample rather than being paid for outright.
        assert "i-1" not in plan["total"][0]
