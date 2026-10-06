"""The shipped example projects, run end to end.

Every README in ``examples/`` quotes numbers from a real run. The quickest
way for those to become fiction is for a default to move, a check to be
renamed, or a corpus to be regenerated with a different seed, and for nobody
to notice because the examples are documentation rather than code.

So each one is run here. Not to pin exact figures — a test that broke every
time a threshold moved would be deleted within a month — but to hold the
claims the READMEs are built on: which gates fire, roughly how wrong each
corpus is, and which field kind each project is actually demonstrating.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_expectations.config import load_run
from llm_expectations.run import analyse_run, load_dataset, run
from llm_expectations.schema import FieldKind

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
PROJECTS = ("jtbd", "ticket-routing", "release-notes", "invoice-extraction")


def offline(name: str) -> Path:
    return EXAMPLES / name / "run-offline.yml"


@pytest.fixture(scope="module", params=PROJECTS)
def finished(request, tmp_path_factory):
    """Each project, run once offline and shared across the tests below."""
    config = load_run(offline(request.param))
    out = tmp_path_factory.mktemp(request.param)
    result = run(config, out=out, ask=lambda _: "y")
    assert result is not None
    return request.param, config, result


class TestEveryExampleRuns:
    def test_it_produces_a_report_and_findings(self, finished):
        _, _, result = finished
        assert result.findings
        assert "GATES" in result.report

    def test_a_scripted_judge_always_stops_gate_one(self, finished):
        # The one thing every offline run must say. A report built on
        # scripted verdicts is laid out exactly like a real one.
        _, _, result = finished
        gate = result.metrics["gates"]["measurement_sound"]
        assert gate["status"] == "STOP"
        assert any("scripted" in r["message"] for r in gate["results"])

    def test_re_analysing_costs_nothing_and_keeps_the_gate_measurement(self, finished):
        """`analyse` is advertised as free, and as giving the same answer.

        The gate's own audit is the part that used to go missing here: which
        rows were flagged and which were sampled lived in verdict metadata,
        which does not survive the cache. It is recomputed from the plan now,
        and this is what says so.
        """
        _, _, result = finished
        again = analyse_run(result.directory)
        assert again.metrics["calls"] == 0
        assert again.metrics["cost_usd"] == 0.0
        assert again.metrics["free_text_gate"] == result.metrics["free_text_gate"]

    def test_every_declared_field_is_reported_on(self, finished):
        _, config, result = finished
        reported = {f.field for f in result.findings}
        assert set(config.schema.fields) <= reported


class TestTheThreeKindsAreActuallyDemonstrated:
    """Each project is the worked example for one kind, and only by being it."""

    def test_ticket_routing_is_assigned_only(self):
        schema = load_run(offline("ticket-routing")).schema
        assert {s.kind for s in schema.fields.values()} == {FieldKind.ASSIGNED}

    def test_release_notes_carries_the_free_text_field(self):
        schema = load_run(offline("release-notes")).schema
        assert schema["note"].kind is FieldKind.FREE_TEXT
        # The assigned companion is not decoration: cross-field agreement is
        # the cheapest check the note gets, and it needs a partner.
        assert schema["note"].must_agree_with == ("area",)

    def test_invoice_extraction_is_copied_only_and_uses_all_three_value_types(self):
        schema = load_run(offline("invoice-extraction")).schema
        assert {s.kind for s in schema.fields.values()} == {FieldKind.COPIED}
        assert {s.value_type.value for s in schema.fields.values()} == {
            "text", "date", "number",
        }


class TestTheClaimsTheReadmesMake:
    def test_routing_clears_the_gate_two_floors(self):
        """examples/ticket-routing exists to get *past* the floors jtbd fails."""
        config = load_run(offline("ticket-routing"))
        dataset = load_dataset(config)
        labelled = {label.item_id for label in dataset.labels if label.field == "queue"}
        assert len(labelled) >= 200, "the ranking floor is 200 labelled items"

    def test_notes_rates_enough_to_rank(self):
        config = load_run(offline("release-notes"))
        dataset = load_dataset(config)
        rated = {label.item_id for label in dataset.labels if label.field == "note"}
        assert len(rated) >= 200

    def test_a_free_text_defect_rating_is_a_ranking_target(self):
        """The fix the free-text project forced.

        A free-text label is a defect rating, so it cannot be compared
        against the sentence — but "the rater ticked a box" is a perfectly
        good definition of an error, and without it a project whose main
        field is free text could never build a target at all.
        """
        from llm_expectations.gates import build_target
        from llm_expectations.types import Label, Output

        outputs = {"pr-1": Output("pr-1", {"note": "a line"}),
                   "pr-2": Output("pr-2", {"note": "another"})}
        labels = [
            Label("pr-1", "note", "made_up", "ann-1"),
            Label("pr-2", "note", "none", "ann-1"),
        ]
        # Named as a gold-valued field, the box name never equals the
        # sentence, so everything is an error. That is the trap.
        assert build_target(labels, outputs, ("note",)).positives == 2
        # Named as what it is, exactly the rated row is.
        assert build_target(labels, outputs, (), rated=("note",)).positives == 1

    def test_invoices_are_grounded_more_often_than_they_are_right(self):
        """The whole point of the copied kind, in one assertion.

        If these two ever converge on this corpus, either the gap was
        removed from the fixture or `value_in_source` has started doing
        something it cannot do.
        """
        config = load_run(offline("invoice-extraction"))
        dataset = load_dataset(config)
        result = _run_once(config)
        grounded = _rate(result, "value_in_source", "total")
        correct = _rate(result, "value_matches_human", "total")
        assert grounded is not None and correct is not None
        assert grounded > correct + 0.03, (
            "a total that is in the document is not thereby the right total"
        )
        assert dataset.labels, "the gap needs human answers to be visible at all"


def _run_once(config):
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        result = run(config, out=Path(directory), ask=lambda _: "y")
        assert result is not None
        return result


def _rate(result, check: str, field: str) -> float | None:
    rows = [f for f in result.findings if f.check == check and f.field == field]
    scored = [f for f in rows if f.status.name != "UNSCORED"]
    return (
        sum(f.status.name == "PASS" for f in scored) / len(scored) if scored else None
    )
