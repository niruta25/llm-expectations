"""The free checks: what each one catches, and what it refuses to claim."""

from __future__ import annotations

import pytest

from llm_expectations.checks import CheckContext, checks_for, run_checks
from llm_expectations.checks.item import vocabulary
from llm_expectations.config import load_run, settings_from_mapping
from llm_expectations.schema import FieldKind, schema_from_mapping
from llm_expectations.taxonomy import load_taxonomy
from llm_expectations.types import Grain, Item, Output, Status

from .conftest import EXAMPLE


@pytest.fixture(scope="module")
def jtbd():
    return load_taxonomy(EXAMPLE / "taxonomy.yml")


def build_ctx(outputs, *, taxonomy, schema=None, settings=None, previous=None, items=None):
    schema = schema or schema_from_mapping(
        {
            "item": "s",
            "fields": {
                "jtbd": {"kind": "assigned", "taxonomy": "jtbd@v4"},
                "summary": {"kind": "free_text", "must_agree_with": ["jtbd"]},
            },
        }
    )
    outs = {k: Output(k, v) for k, v in outputs.items()}
    return CheckContext(
        run_id="t",
        schema=schema,
        settings=settings or settings_from_mapping({}),
        items=items or {k: Item(k, "some session text") for k in outs},
        outputs=outs,
        taxonomies={"jtbd": taxonomy},
        previous=previous or {},
    )


def results(ctx, check):
    findings, _ = run_checks(ctx)
    return [f for f in findings if f.check == check]


def test_the_runner_never_branches_on_field_kind():
    # Adding a kind means registering against it, not editing the runner.
    import inspect

    from llm_expectations.checks import base

    source = inspect.getsource(base.run_checks)
    assert "FieldKind" not in source
    assert "assigned" not in source and "free_text" not in source


def test_checks_are_registered_against_the_kinds_they_read():
    assigned = {c.name for c in checks_for(FieldKind.ASSIGNED)}
    free_text = {c.name for c in checks_for(FieldKind.FREE_TEXT)}
    assert {"label_in_taxonomy", "valid_leaf", "label_collapse"} <= assigned
    assert "label_in_taxonomy" not in free_text
    # The item check belongs to neither kind — it sits above both.
    assert "cross_field_agreement" in assigned & free_text


class TestLabelInTaxonomy:
    def test_a_real_label_passes(self, jtbd):
        ctx = build_ctx({"s-1": {"jtbd": "billing.card_declined"}}, taxonomy=jtbd)
        assert results(ctx, "label_in_taxonomy")[0].status is Status.PASS

    def test_an_invented_label_fails_and_suggests_what_does_exist(self, jtbd):
        ctx = build_ctx({"s-1": {"jtbd": "access.login_broken"}}, taxonomy=jtbd)
        found = results(ctx, "label_in_taxonomy")[0]
        assert found.status is Status.FAIL
        assert "access.sso_issue" in found.evidence["nearest"]

    def test_an_abstention_is_not_a_missing_label(self, jtbd):
        ctx = build_ctx({"s-1": {"jtbd": None}}, taxonomy=jtbd)
        assert results(ctx, "label_in_taxonomy") == []


class TestValidLeaf:
    def test_a_parent_label_fails_and_names_its_children(self, jtbd):
        ctx = build_ctx({"s-1": {"jtbd": "billing"}}, taxonomy=jtbd)
        found = results(ctx, "valid_leaf")[0]
        assert found.status is Status.FAIL
        assert "billing.payment_failed" in found.evidence["children"]
        assert "hedged" in found.evidence["why"]

    def test_an_invented_label_is_not_failed_twice(self, jtbd):
        # label_in_taxonomy owns that row. Failing it here as well would make
        # one invented label look like two separate problems.
        ctx = build_ctx({"s-1": {"jtbd": "nope.nothing"}}, taxonomy=jtbd)
        assert results(ctx, "valid_leaf") == []

    def test_the_check_does_not_run_when_a_parent_is_a_valid_answer(self, jtbd):
        schema = schema_from_mapping(
            {
                "item": "s",
                "fields": {
                    "jtbd": {"kind": "assigned", "taxonomy": "jtbd@v4", "require_leaf": False}
                },
            }
        )
        ctx = build_ctx({"s-1": {"jtbd": "billing"}}, taxonomy=jtbd, schema=schema)
        findings, skipped = run_checks(ctx)
        assert not [f for f in findings if f.check == "valid_leaf"]
        assert any(s.check == "valid_leaf" and "require_leaf is off" in s.reason for s in skipped)


class TestAbstention:
    def _ctx(self, abstentions, total, jtbd, **settings):
        outputs = {
            f"s-{i}": {"jtbd": None if i < abstentions else "billing.card_declined"}
            for i in range(total)
        }
        return build_ctx(outputs, taxonomy=jtbd, settings=settings_from_mapping(settings))

    def test_a_rate_inside_the_band_passes(self, jtbd):
        found = results(self._ctx(10, 100, jtbd), "abstention_rate")[0]
        assert found.status is Status.PASS
        assert found.score == pytest.approx(0.10)

    def test_too_high_points_at_a_gap_in_the_taxonomy(self, jtbd):
        found = results(self._ctx(50, 100, jtbd), "abstention_rate")[0]
        assert found.status is Status.FAIL
        assert "gap" in found.evidence["why"]

    def test_too_low_points_at_the_model_forcing_labels(self, jtbd):
        found = results(self._ctx(0, 1000, jtbd), "abstention_rate")[0]
        assert found.status is Status.FAIL
        assert "forcing" in found.evidence["why"]

    def test_a_floor_the_corpus_cannot_reach_is_unscored_not_failed(self, jtbd):
        # On 13 rows the smallest non-zero rate is 7.7%, so a 1% floor cannot
        # distinguish "correctly never abstains" from "forcing". Failing it
        # would report an artefact of the corpus as a finding.
        found = results(self._ctx(0, 13, jtbd), "abstention_rate")[0]
        assert found.status is Status.UNSCORED
        assert "cannot be tested on 13 rows" in found.evidence["why"]

    def test_a_high_rate_still_fails_on_a_small_corpus(self, jtbd):
        # The ceiling is reachable at any size; only the floor is not.
        found = results(self._ctx(6, 13, jtbd), "abstention_rate")[0]
        assert found.status is Status.FAIL

    def test_a_field_that_forbids_abstaining_fails_each_row(self, jtbd):
        schema = schema_from_mapping(
            {
                "item": "s",
                "fields": {
                    "jtbd": {"kind": "assigned", "taxonomy": "jtbd@v4", "allow_abstain": False}
                },
            }
        )
        ctx = build_ctx(
            {"s-1": {"jtbd": None}, "s-2": {"jtbd": "billing.card_declined"}},
            taxonomy=jtbd,
            schema=schema,
        )
        found = results(ctx, "abstention_allowed")
        assert len(found) == 1
        assert (found[0].item_id, found[0].status) == ("s-1", Status.FAIL)


class TestCollapse:
    def test_one_label_swallowing_the_batch_fails(self, jtbd):
        outputs = {f"s-{i}": {"jtbd": "billing.card_declined"} for i in range(10)}
        outputs["s-99"] = {"jtbd": "access.sso_issue"}
        found = results(build_ctx(outputs, taxonomy=jtbd), "label_collapse")[0]
        assert found.status is Status.FAIL
        assert found.evidence["largest_label"] == "billing.card_declined"
        assert found.threshold == 0.5

    def test_abstentions_are_out_of_the_denominator(self, jtbd):
        # Leaving them in would make a taxonomy gap look like every label
        # shrinking at once.
        outputs = {"s-1": {"jtbd": "billing.card_declined"}, "s-2": {"jtbd": None}}
        found = results(build_ctx(outputs, taxonomy=jtbd), "label_collapse")[0]
        assert found.score == 1.0


class TestDrift:
    def _previous(self, shares, taxonomy="jtbd@v4", run_id="earlier", prompt="p7"):
        return {
            "run_id": run_id,
            "prompt": prompt,
            "taxonomies": {taxonomy: "hash"},
            "distributions": {"jtbd": shares},
        }

    def test_no_previous_run_means_the_check_does_not_run(self, jtbd):
        ctx = build_ctx({"s-1": {"jtbd": "billing.card_declined"}}, taxonomy=jtbd)
        _, skipped = run_checks(ctx)
        assert any(s.check == "drift" and "no previous run" in s.reason for s in skipped)

    def test_a_stable_distribution_passes(self, jtbd):
        ctx = build_ctx(
            {"s-1": {"jtbd": "billing.card_declined"}},
            taxonomy=jtbd,
            previous=self._previous({"billing.card_declined": 1.0}),
        )
        assert results(ctx, "drift")[0].status is Status.PASS

    def test_a_label_that_moved_fails_and_names_it(self, jtbd):
        ctx = build_ctx(
            {"s-1": {"jtbd": "billing.card_declined"}},
            taxonomy=jtbd,
            previous=self._previous({"billing.payment_failed": 1.0}),
        )
        found = results(ctx, "drift")[0]
        assert found.status is Status.FAIL
        assert found.evidence["largest_move"]["pp"] == pytest.approx(100.0)

    def test_two_taxonomy_versions_refuse_to_compare(self, jtbd):
        ctx = build_ctx(
            {"s-1": {"jtbd": "billing.card_declined"}},
            taxonomy=jtbd,
            previous=self._previous({"billing.card_declined": 1.0}, taxonomy="jtbd@v5"),
        )
        found = results(ctx, "drift")[0]
        assert found.status is Status.UNSCORED
        assert "not the same measurement" in found.evidence["why"]

    def test_a_prompt_change_is_named_beside_the_number(self, jtbd):
        previous = self._previous({"billing.payment_failed": 1.0}, prompt="p7")
        previous["current_prompt"] = "p8"
        ctx = build_ctx(
            {"s-1": {"jtbd": "billing.card_declined"}}, taxonomy=jtbd, previous=previous
        )
        found = results(ctx, "drift")[0]
        assert "it is the change you made" in found.evidence["why"]


class TestCrossField:
    def test_vocabulary_is_inherited_down_the_branch(self, jtbd):
        words = vocabulary(jtbd, "billing.card_declined")
        assert {"card", "declined", "issuer"} <= words
        assert "money" in words  # from the `billing` parent

    def test_boundary_cases_are_not_counted_as_agreement(self, jtbd):
        # `not_this` says what the label is *not*. Counting those words would
        # reward a summary for matching the boundary it should fall outside.
        assert "refund" not in vocabulary(jtbd, "billing.payment_failed")

    def test_a_summary_that_shares_the_label_vocabulary_passes(self, jtbd):
        ctx = build_ctx(
            {
                "s-1": {
                    "jtbd": "billing.card_declined",
                    "summary": "Her card was declined by the issuer at renewal.",
                }
            },
            taxonomy=jtbd,
        )
        assert results(ctx, "cross_field_agreement")[0].status is Status.PASS

    def test_a_summary_about_something_else_fails_without_saying_which_is_wrong(self, jtbd):
        ctx = build_ctx(
            {
                "s-1": {
                    "jtbd": "access.sso_issue",
                    "summary": "Her card was declined and a new one fixed it.",
                }
            },
            taxonomy=jtbd,
        )
        found = results(ctx, "cross_field_agreement")[0]
        assert found.status is Status.FAIL
        assert found.grain is Grain.ITEM
        assert "does not know which" in found.evidence["why"]

    def test_an_invented_label_leaves_this_unscored_rather_than_failed(self, jtbd):
        # An invented label has no definition to draw vocabulary from, so this
        # check genuinely cannot speak to the row.
        ctx = build_ctx(
            {"s-1": {"jtbd": "access.login_broken", "summary": "anything at all"}},
            taxonomy=jtbd,
        )
        found = results(ctx, "cross_field_agreement")[0]
        assert found.status is Status.UNSCORED
        assert "carries no vocabulary" in found.evidence["why"]

    def test_an_abstention_leaves_this_unscored(self, jtbd):
        ctx = build_ctx({"s-1": {"jtbd": None, "summary": "anything"}}, taxonomy=jtbd)
        assert results(ctx, "cross_field_agreement")[0].status is Status.UNSCORED


@pytest.fixture(scope="module")
def fixture_findings():
    """Every free check, run once over the worked example."""
    from llm_expectations.run import load_dataset

    config = load_run(EXAMPLE / "run.yml")
    dataset = load_dataset(config)
    ctx = CheckContext(
        run_id="t",
        schema=config.schema,
        settings=config.settings,
        items=dataset.items,
        outputs=dataset.outputs,
        taxonomies=config.taxonomies,
    )
    return run_checks(ctx)


class TestAgainstTheFixture:
    def test_no_clean_item_is_ever_flagged(self, fixture_findings):
        # The acceptance test. A check that also flags s-01 is as broken as
        # one that misses s-09.
        import yaml

        manifest = yaml.safe_load((EXAMPLE / "expected.yml").read_text())
        findings, _ = fixture_findings
        flagged = {f.item_id for f in findings if f.status is Status.FAIL and f.item_id}
        assert not flagged & set(manifest["clean"])

    def test_every_flagged_item_is_one_the_manifest_planted(self, fixture_findings):
        import yaml

        manifest = yaml.safe_load((EXAMPLE / "expected.yml").read_text())
        findings, _ = fixture_findings
        flagged = {f.item_id for f in findings if f.status is Status.FAIL and f.item_id}
        assert flagged <= {plant["item"] for plant in manifest["plants"]}

    def test_every_plant_due_by_m2_is_caught_by_the_check_that_owns_it(self, fixture_findings):
        import yaml

        manifest = yaml.safe_load((EXAMPLE / "expected.yml").read_text())
        findings, _ = fixture_findings
        failed = {(f.item_id, f.check) for f in findings if f.status is Status.FAIL}
        for plant in manifest["plants"]:
            if plant["milestone"] != "M2":
                continue
            assert any(
                (plant["item"], check) in failed for check in plant["caught_by"]
            ), f"{plant['item']} ({plant['defect']}) was not caught by {plant['caught_by']}"


class TestTheManifestContract:
    """The three rules `expected.yml` states, checked against the manifest itself."""

    @staticmethod
    def manifest():
        import yaml

        return yaml.safe_load((EXAMPLE / "expected.yml").read_text())

    def test_a_second_check_firing_on_a_planted_row_is_declared(self, fixture_findings):
        findings, _ = fixture_findings
        manifest = self.manifest()
        declared = {
            (plant["item"], check)
            for plant in manifest["plants"]
            for check in [*plant["caught_by"], *plant.get("also_caught_by", [])]
        }
        for found in findings:
            if found.status is Status.FAIL and found.item_id:
                assert (found.item_id, found.check) in declared, (
                    f"{found.check} flagged {found.item_id} and the manifest does not say so"
                )

    def test_every_milestone_in_the_manifest_is_one_that_exists(self):
        assert {p["milestone"] for p in self.manifest()["plants"]} <= {
            "M1", "M2", "M3", "M4", "M5", "M5b", "M6", "M7"
        }
