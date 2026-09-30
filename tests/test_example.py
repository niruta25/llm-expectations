"""The fixture has to agree with its own manifest.

`expected.yml` is the acceptance contract for every milestone from M2 on. A
manifest that has drifted from the data would quietly weaken every test that
reads it, so it is checked against the files here, where there is nothing else
to hide behind.
"""

from __future__ import annotations

import pytest
import yaml

from llm_expectations.config import load_run
from llm_expectations.read import index_items, read_labels, read_outputs
from llm_expectations.types import ABSTAIN

from .conftest import EXAMPLE


@pytest.fixture(scope="module")
def example():
    config = load_run(EXAMPLE / "run.yml")
    return {
        "config": config,
        "manifest": yaml.safe_load((EXAMPLE / "expected.yml").read_text(encoding="utf-8")),
        "items": index_items(config.items),
        "outputs": {o.item_id: o for o in read_outputs(config.outputs)},
        "labels": list(read_labels(config.labels)),
    }


def test_the_manifest_counts_what_is_actually_there(example):
    assert example["manifest"]["corpus"]["items"] == len(example["items"])
    assert example["manifest"]["corpus"]["fields"] == list(example["config"].schema.fields)


def test_every_item_is_either_planted_or_clean_and_none_is_both(example):
    manifest = example["manifest"]
    planted = {plant["item"] for plant in manifest["plants"]}
    clean = set(manifest["clean"])
    assert not planted & clean
    assert planted | clean == set(example["items"])


def test_each_plant_names_a_field_that_exists_and_a_check_that_will_own_it(example):
    fields = set(example["config"].schema.fields)
    for plant in example["manifest"]["plants"]:
        assert plant["field"] in fields, plant
        assert plant["caught_by"], plant
        assert plant["milestone"].startswith("M"), plant


def test_each_plant_describes_the_output_that_is_actually_in_the_file(example):
    for plant in example["manifest"]["plants"]:
        if "output" not in plant:
            continue
        assert example["outputs"][plant["item"]].get(plant["field"]) == plant["output"], plant


def test_each_declared_truth_matches_the_human_label(example):
    by_key = {(label.item_id, label.field): label.label
              for label in example["labels"] if label.annotator == "ann-1"}
    for plant in example["manifest"]["plants"]:
        if "truth" not in plant:
            continue
        assert by_key[(plant["item"], plant["field"])] == plant["truth"], plant


def test_the_clean_items_really_do_carry_the_human_answer(example):
    by_key = {(label.item_id, label.field): label.label
              for label in example["labels"] if label.annotator == "ann-1"}
    for item_id in example["manifest"]["clean"]:
        for field in ("jtbd", "outcome"):
            assert example["outputs"][item_id].get(field) == by_key[(item_id, field)]


def test_the_abstention_is_recorded_as_correct_behaviour_not_as_a_plant(example):
    manifest = example["manifest"]
    correct = {(entry["item"], entry["field"]) for entry in manifest["correct_but_unusual"]}
    assert ("s-13", "jtbd") in correct
    assert example["outputs"]["s-13"].is_abstain("jtbd")
    # ann-1 abstained; ann-2 forced a label onto it. That disagreement is the
    # taxonomy gap showing up in the data, not a labelling mistake.
    by_annotator = {(label.item_id, label.field, label.annotator): label.label
                    for label in example["labels"]}
    assert by_annotator[("s-13", "jtbd", "ann-1")] == ABSTAIN
    assert by_annotator[("s-13", "jtbd", "ann-2")] != ABSTAIN


def test_every_non_abstaining_label_is_a_real_leaf_of_its_taxonomy(example):
    # The human answers are the answer key. If one of them is not in the tree,
    # every metric computed against it is measuring the wrong thing.
    config = example["config"]
    for label in example["labels"]:
        if label.label == ABSTAIN:
            continue
        taxonomy = config.taxonomy_for(label.field)
        assert label.label in taxonomy, label
        assert taxonomy.is_leaf(label.label), label


def test_the_fuzzy_pair_is_where_the_two_annotators_actually_disagree(example):
    by_key: dict[tuple[str, str], set[str]] = {}
    for label in example["labels"]:
        by_key.setdefault((label.item_id, label.field), set()).add(label.label)
    disputed = {key for key, values in by_key.items() if len(values) > 1}
    pairs = {frozenset(by_key[key]) for key in disputed if key[1] == "jtbd"}
    assert frozenset({"billing.card_declined", "billing.payment_failed"}) in pairs


def test_the_fixture_is_deliberately_far_below_every_sample_floor(example):
    # A corpus this size proves checks fire on the right rows and nothing at
    # all about whether a metric means anything. If this example ever reports
    # a confident ranking number, the guardrails are broken.
    settings = example["config"].settings
    assert len(example["items"]) < settings.value("min_n_ranking")
    assert len(example["items"]) < settings.value("min_n_distribution")


def test_the_example_taxonomies_pass_their_static_health_checks(example):
    for taxonomy in example["config"].taxonomies.values():
        assert taxonomy.health() == [], taxonomy.ref
