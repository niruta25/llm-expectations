"""The two absences in the type layer are the point, so they are pinned here."""

from __future__ import annotations

import dataclasses

import pytest

from llm_expectations.types import Item, Label, Mode, Output, RiskRow, Status, Verdict


def _fields(cls: type) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


def test_unscored_is_not_a_pass():
    # Without a third state, a run that quietly stopped checking reports green.
    assert Status.PASS.is_pass
    assert not Status.FAIL.is_pass
    assert not Status.UNSCORED.is_pass
    assert not Status.UNSCORED.is_scored


def test_verdict_carries_no_error_probability_and_no_triage_score():
    # A judge does not produce either. A calibrator computes one later, from a
    # verdict plus a fitted model. A field here would invite writing it at
    # judge time, which is the bug the whole calibration section prevents.
    assert "calibrated_error_probability" not in _fields(Verdict)
    assert "triage_score" not in _fields(Verdict)
    assert "raw_confidence" in _fields(Verdict)


def test_the_derived_row_carries_them_instead_and_says_which():
    assert {"calibrated_error_probability", "triage_score", "calibrated"} <= _fields(RiskRow)


def test_an_output_has_nowhere_to_put_a_human_answer():
    # Labels travel as their own type on a separate path, so judge-calling code
    # cannot be handed an answer key by accident.
    assert "label" not in _fields(Output)
    assert _fields(Label) == {"item_id", "field", "label", "annotator"}


def test_modes_are_ordered_so_capability_is_a_comparison():
    assert Mode.NO_LABELS < Mode.LABELLED < Mode.DOUBLE_LABELLED
    assert int(Mode.NO_LABELS) == 0


def test_abstain_reads_as_abstain_whether_null_or_named():
    assert Output("s-1", {"jtbd": None}).is_abstain("jtbd")
    assert Output("s-1", {"jtbd": "abstain"}).is_abstain("jtbd")
    assert not Output("s-1", {"jtbd": "billing"}).is_abstain("jtbd")


def test_value_types_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Item("s-1", "text").id = "s-2"  # type: ignore[misc]
