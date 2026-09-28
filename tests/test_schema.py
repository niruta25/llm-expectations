"""Field kinds, and the rule that an unread setting is an error."""

from __future__ import annotations

import pytest
import yaml

from llm_expectations.schema import (
    FIELD_SETTINGS,
    FieldKind,
    SchemaError,
    TextStyle,
    schema_from_mapping,
)


def build(**fields):
    return schema_from_mapping({"item": "support_session", "fields": fields}, where="schema.yml")


ASSIGNED = {"kind": "assigned", "taxonomy": "jtbd@v4"}


def test_a_schema_knows_its_fields_and_its_taxonomies():
    schema = build(
        jtbd=ASSIGNED,
        summary={"kind": "free_text", "must_agree_with": ["jtbd"]},
        outcome={"kind": "assigned", "taxonomy": "outcomes@v1"},
    )
    assert len(schema) == 3
    assert [f.name for f in schema.of_kind(FieldKind.ASSIGNED)] == ["jtbd", "outcome"]
    assert [str(r) for r in schema.taxonomy_refs()] == ["jtbd@v4", "outcomes@v1"]


def test_free_text_defaults_to_descriptive():
    assert build(summary={"kind": "free_text"})["summary"].style is TextStyle.DESCRIPTIVE


def test_an_assigned_field_needs_a_pinned_taxonomy():
    with pytest.raises(SchemaError, match="pinned taxonomy"):
        build(jtbd={"kind": "assigned"})


def test_copied_fields_say_they_are_v1_rather_than_doing_nothing():
    with pytest.raises(SchemaError, match="v1"):
        build(amount={"kind": "copied"})


def test_an_unknown_key_is_an_error_because_a_threshold_that_is_not_read_does_nothing():
    with pytest.raises(SchemaError, match="max_labl_share"):
        build(jtbd={**ASSIGNED, "max_labl_share": 0.5})


def test_a_setting_from_the_other_kind_is_refused_with_the_kind_it_belongs_to():
    with pytest.raises(SchemaError, match="applies to free_text"):
        build(jtbd={**ASSIGNED, "max_copy_ratio": 0.5})
    with pytest.raises(SchemaError, match="applies to assigned"):
        build(summary={"kind": "free_text", "require_leaf": True})


@pytest.mark.parametrize(
    "override, message",
    [
        ({"max_label_share": "half"}, "must be a number"),
        ({"max_label_share": 1.5}, "outside the permitted range"),
        ({"require_leaf": "yes"}, "must be true or false"),
        ({"abstain_rate": 0.2}, "two-element band"),
        ({"abstain_rate": [0.2, 0.01]}, "descending"),
        ({"abstain_rate": [0.01, 0.2, 0.5]}, "exactly two"),
    ],
)
def test_a_mistyped_value_is_caught_before_anything_runs(override, message):
    with pytest.raises(SchemaError, match=message):
        build(jtbd={**ASSIGNED, **override})


def test_bands_come_back_as_tuples_whatever_the_yaml_said():
    assert build(jtbd={**ASSIGNED, "abstain_rate": [0.05, 0.3]})["jtbd"].overrides[
        "abstain_rate"
    ] == (0.05, 0.3)


def test_must_agree_with_has_to_name_a_real_field():
    with pytest.raises(SchemaError, match="not\na field|not a field"):
        build(summary={"kind": "free_text", "must_agree_with": ["nope"]})
    with pytest.raises(SchemaError, match="agree with itself"):
        build(summary={"kind": "free_text", "must_agree_with": ["summary"]})


def test_only_declared_overrides_are_kept_so_provenance_survives():
    # Merging with the defaults here would lose which layer set what, and the
    # report has to print that next to the number.
    spec = build(jtbd={**ASSIGNED, "require_leaf": False})["jtbd"]
    assert spec.overrides == {"require_leaf": False}
    assert "max_label_share" not in spec.overrides


def test_every_setting_declares_which_kinds_may_use_it():
    assert all(s.kinds for s in FIELD_SETTINGS.values())
    assert all(s.doc for s in FIELD_SETTINGS.values())


def test_the_worked_example_schema_loads(tmp_path):
    from .conftest import EXAMPLE

    schema = schema_from_mapping(yaml.safe_load((EXAMPLE / "schema.yml").read_text()))
    assert schema.item == "support_session"
    assert schema["summary"].must_agree_with == ("jtbd",)
