"""The tree, the hash that stops silent redefinition, and the static checks."""

from __future__ import annotations

import pytest
import yaml

from llm_expectations.taxonomy import (
    TaxonomyChangedError,
    TaxonomyError,
    check_recorded_hash,
    load_taxonomy,
    parse_ref,
    taxonomy_from_mapping,
)
from llm_expectations.types import Severity

from .conftest import EXAMPLE, MINIMAL_TAXONOMY


def build(body: str):
    return taxonomy_from_mapping(yaml.safe_load(body))


@pytest.fixture
def jtbd():
    return build(MINIMAL_TAXONOMY)


def test_parse_ref_requires_a_version():
    assert parse_ref("jtbd@v4") == parse_ref(" jtbd@v4 ")
    with pytest.raises(TaxonomyError, match="id@vN"):
        parse_ref("jtbd")


def test_the_tree_knows_its_own_shape(jtbd):
    assert jtbd.roots() == ("billing",)
    assert set(jtbd.leaves()) == {"billing.payment_failed", "billing.refund_request"}
    assert not jtbd.is_leaf("billing")
    assert jtbd.parent_of("billing.payment_failed") == "billing"
    assert jtbd.ancestors("billing.payment_failed") == ("billing",)
    assert jtbd.siblings("billing.payment_failed") == ("billing.refund_request",)
    assert "billing.nope" not in jtbd


def test_a_label_without_a_definition_is_rejected_at_parse_time():
    # Not a health warning: the definition is what goes into the judge prompt
    # and in front of the annotator, so a label without one is unusable.
    with pytest.raises(TaxonomyError, match="no definition"):
        build("id: x\nversion: 1\nlabels:\n  billing: {examples: ['a']}\n")


def test_abstain_cannot_be_a_label():
    with pytest.raises(TaxonomyError, match="reserved"):
        build("id: x\nversion: 1\nlabels:\n  abstain: {definition: 'unclear'}\n")


def test_a_name_cannot_contain_the_path_separator():
    with pytest.raises(TaxonomyError, match="separator"):
        build("id: x\nversion: 1\nlabels:\n  'a.b': {definition: 'd'}\n")


def test_a_mistyped_node_key_is_an_error_not_a_silently_ignored_line():
    with pytest.raises(TaxonomyError, match="unknown key"):
        build("id: x\nversion: 1\nlabels:\n  a: {definition: 'd', not_these: ['b']}\n")


def test_version_must_be_a_number_not_a_string():
    with pytest.raises(TaxonomyError, match="'version'"):
        build("id: x\nversion: v1\nlabels:\n  a: {definition: 'd'}\n")


class TestContentHash:
    def test_reformatting_does_not_move_it(self, jtbd):
        reordered = build(
            """
            # a comment nobody should be punished for
            version: 4
            id: jtbd
            labels:
              billing:
                children:
                  refund_request:
                    definition: "The customer wants money returned to them."
                    not_this: ["a charge that never succeeded"]
                    examples: ["wants the duplicate charge reversed"]
                  payment_failed:
                    not_this: ["asking for money back"]
                    examples: ["the renewal never went through"]
                    definition: "A charge was attempted and did not go through."
                definition: "Anything about money moving, or failing to move."
                not_this: ["cannot reach the page at all"]
            """
        )
        assert reordered.content_hash == jtbd.content_hash

    def test_tightening_a_definition_moves_it(self, jtbd):
        edited = build(MINIMAL_TAXONOMY.replace("did not go through", "was refused"))
        assert edited.content_hash != jtbd.content_hash

    def test_adding_a_boundary_case_moves_it(self, jtbd):
        edited = build(
            MINIMAL_TAXONOMY.replace(
                'not_this: ["asking for money back"]',
                'not_this: ["asking for money back", "the issuer refused it"]',
            )
        )
        assert edited.content_hash != jtbd.content_hash


class TestVersionGuard:
    """Editing a taxonomy without bumping the version is an error."""

    def test_unchanged_content_passes(self, jtbd):
        check_recorded_hash(jtbd, {"jtbd@v4": jtbd.content_hash})

    def test_a_reference_never_seen_before_passes(self, jtbd):
        check_recorded_hash(jtbd, {})

    def test_changed_content_under_a_fixed_version_is_refused(self, jtbd):
        edited = build(MINIMAL_TAXONOMY.replace("did not go through", "was refused"))
        with pytest.raises(TaxonomyChangedError) as caught:
            check_recorded_hash(edited, {"jtbd@v4": jtbd.content_hash})
        message = str(caught.value)
        assert "jtbd@v4 has changed" in message
        assert "Bump to v5" in message


class TestHealth:
    def test_the_worked_example_is_clean(self):
        assert load_taxonomy(EXAMPLE / "taxonomy.yml").health() == []

    def test_missing_examples_are_reported_but_only_as_a_warning(self):
        issues = build(
            "id: x\nversion: 1\nlabels:\n  a: {definition: 'first thing'}\n"
        ).health()
        codes = {i.check: i.severity for i in issues}
        assert codes["taxonomy_examples"] is Severity.WARN

    def test_require_examples_off_silences_only_that_check(self):
        taxonomy = build("id: x\nversion: 1\nlabels:\n  a: {definition: 'first thing'}\n")
        assert not [i for i in taxonomy.health(require_examples=False)
                    if i.check == "taxonomy_examples"]

    def test_a_name_reused_across_branches_is_flagged(self):
        issues = build(
            """
            id: x
            version: 1
            labels:
              billing:
                definition: "money"
                children: {other: {definition: "billing, but none of the above"}}
              access:
                definition: "getting in"
                children: {other: {definition: "access, but not any of the listed ones"}}
            """
        ).health()
        duplicates = [i for i in issues if i.check == "taxonomy_duplicate_names"]
        assert [set(i.paths) for i in duplicates] == [{"billing.other", "access.other"}]

    def test_two_near_identical_definitions_are_flagged_as_one_label(self):
        issues = build(
            """
            id: x
            version: 1
            labels:
              a: {definition: "A charge was attempted and did not go through"}
              b: {definition: "A charge was attempted and did not go thru"}
            """
        ).health()
        assert any(i.check == "taxonomy_similar_definitions" for i in issues)

    def test_uneven_leaf_depth_is_a_note_not_a_warning(self):
        issues = build(
            """
            id: x
            version: 1
            labels:
              a:
                definition: "first"
                children: {a1: {definition: "first child"}}
              b: {definition: "second"}
            """
        ).health()
        depth = next(i for i in issues if i.check == "taxonomy_leaf_depth")
        assert depth.severity is Severity.NOTE
        assert depth.paths == ("b",)

    def test_siblings_without_boundaries_are_noted(self):
        issues = build(
            """
            id: x
            version: 1
            labels:
              a: {definition: "first thing entirely"}
              b: {definition: "wholly different second"}
            """
        ).health()
        assert any(i.check == "taxonomy_boundaries" for i in issues)

    def test_a_lone_root_has_no_boundary_to_draw(self):
        issues = build("id: x\nversion: 1\nlabels:\n  a: {definition: 'only thing'}\n").health()
        assert not [i for i in issues if i.check == "taxonomy_boundaries"]


def test_a_missing_file_says_so_plainly():
    with pytest.raises(TaxonomyError, match="no taxonomy file"):
        load_taxonomy("nowhere.yml")
