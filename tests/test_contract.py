"""The design's own acceptance list, one test each.

DESIGN.md §12 names the mistakes that are easy to make and hard to see. Each
is pinned somewhere in the suite already; this module gathers them in the
design's order so the contract can be read in one place and a regression
names the rule it broke rather than a helper function.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from llm_expectations.types import Status


def test_1_unscored_never_counts_as_a_pass():
    """Without a third state, a run that stopped checking reports green."""
    assert not Status.UNSCORED.is_pass
    assert not Status.UNSCORED.is_scored
    # And nothing in the library may define its own notion of passing.
    assert Status.PASS.is_pass and not Status.FAIL.is_pass


def test_2_a_judge_never_receives_a_label():
    """Asserted at the type level, not left to discipline."""
    from llm_expectations.judges.base import Judge, JudgeTask
    from llm_expectations.run import collect
    from llm_expectations.triage.base import TriageContext

    for function in (JudgeTask.build, Judge.ask, Judge.build, collect):
        params = inspect.signature(function).parameters
        assert not any("Label" in str(p.annotation) for p in params.values()), function
        assert "labels" not in params, function
    assert "labels" not in {f.name for f in dataclasses.fields(TriageContext)}


def test_3_a_rubber_stamp_leaves_the_aggregates_but_not_the_disk(
    example, tmp_path, scripted
):
    from llm_expectations.judges.fake import FakeProvider, reply
    from llm_expectations.run import run

    strict = FakeProvider(model="strict", default=reply(False, 0.9, "wrong"))
    rubber = FakeProvider(model="rubber", default=reply(True, 0.99, "fine"))
    by_id = {"judge-a": strict, "judge-b": rubber, "judge-c": strict}
    result = run(
        example, out=tmp_path / "out", provider_factory=lambda spec: by_id[spec.id]
    )
    assert "judge-b" in result.metrics["panel"]["excluded"]
    assert any(v.judge_id == "judge-b" for v in result.verdicts)


def test_4_unreadable_replies_are_counted_never_defaulted(example, tmp_path, scripted):
    from llm_expectations.run import run

    provider = scripted(default="not json at all", claim_default="not json at all")
    result = run(example, out=tmp_path / "out", provider_factory=lambda spec: provider)
    unreadable = [v for v in result.verdicts if v.metadata.get("reply") == "unparseable"]
    assert unreadable
    # A coin-flip default is uncorrelated by construction and would quietly
    # move every agreement number in the run.
    assert all(v.status is Status.UNSCORED for v in unreadable)
    assert all(v.raw_confidence is None for v in unreadable)


class TestRule5SuppressVersusFlag:
    """§10 Tier 1 withholds; Tier 2 reports and flags. Different cases."""

    def test_an_invalidated_number_is_withheld_and_named(self, example, provider, tmp_path):
        from llm_expectations.run import run

        result = run(
            example, out=tmp_path / "out", provider_factory=lambda spec: provider
        )
        withheld = {row["metric"] for row in result.metrics["suppressed"]}
        assert "gate2.auc" in withheld
        assert result.metrics["gates"]["beats_baselines"]["table"] == []

    def test_an_underpowered_number_is_printed_with_its_caveat(
        self, example, provider, tmp_path
    ):
        from llm_expectations.run import run

        # A wide interval is not a weak result, it is no result — and hiding
        # it would lose the only information the sample carries.
        report = run(
            example, out=tmp_path / "out", provider_factory=lambda spec: provider
        ).report
        assert "macro F1" in report
        assert "cannot support a conclusion" in report


def test_6_analyse_makes_zero_model_calls(example, provider, tmp_path):
    import llm_expectations.run as run_module
    from llm_expectations.run import analyse_run, run

    first = run(example, out=tmp_path / "out", provider_factory=lambda spec: provider)
    calls = len(provider.calls)

    def explode(spec):
        raise AssertionError("analyse constructed a provider")

    original = run_module.build_provider
    run_module.build_provider = explode
    try:
        again = analyse_run(first.directory)
    finally:
        run_module.build_provider = original
    assert again.metrics["calls"] == 0
    assert len(provider.calls) == calls


def test_7_editing_a_taxonomy_without_bumping_is_an_error():
    import yaml

    from llm_expectations.taxonomy import (
        TaxonomyChangedError,
        check_recorded_hash,
        taxonomy_from_mapping,
    )

    from .conftest import MINIMAL_TAXONOMY

    before = taxonomy_from_mapping(yaml.safe_load(MINIMAL_TAXONOMY))
    after = taxonomy_from_mapping(
        yaml.safe_load(MINIMAL_TAXONOMY.replace("did not go through", "was refused"))
    )
    check_recorded_hash(before, {"jtbd@v4": before.content_hash})
    with pytest.raises(TaxonomyChangedError, match="Bump to v5"):
        check_recorded_hash(after, {"jtbd@v4": before.content_hash})


def test_8_item_grain_is_never_rosier_than_field_grain(example, provider, tmp_path):
    from llm_expectations.run import run

    grains = run(
        example, out=tmp_path / "out", provider_factory=lambda spec: provider
    ).metrics["grains"]
    # An item with a correct label and a bad summary is not a usable item.
    assert grains["item"]["pass_rate"] <= grains["field"]["pass_rate"]


def test_9_two_taxonomy_versions_refuse_to_compare(two_runs):
    from llm_expectations.compare import ComparisonError, compare_runs, load_run_directory

    _, a, b = two_runs(n=200, taxonomy_of_b="jtbd@v5")
    with pytest.raises(ComparisonError, match="migration mapping"):
        compare_runs(load_run_directory(a), load_run_directory(b))


def test_10_a_reply_in_the_wrong_shape_is_unreadable():
    """An assigned reply to a claim question is not an answer to it."""
    from llm_expectations.judges.fake import claims, reply
    from llm_expectations.judges.prompts import ClaimSupportTask, LabelCorrectTask

    assert ClaimSupportTask().parse(reply(True, 0.9, "fine")).status is Status.UNSCORED
    assert LabelCorrectTask().parse(claims(supported=["a"])).status is Status.UNSCORED
