"""The three things the report always does, and the drift state behind one of them."""

from __future__ import annotations

import pytest

from llm_expectations.config import load_run
from llm_expectations.judges.fake import reply
from llm_expectations.run import grain_rates, run
from llm_expectations.types import Finding, Grain, Status

from .conftest import EXAMPLE


def finding(check, status, item_id=None, field="jtbd", **kw):
    return Finding(
        run_id="t", check=check, grain=Grain.FIELD, status=status,
        item_id=item_id, field=field, **kw
    )


@pytest.fixture
def report(no_confirm, scripted, tmp_path):
    def build(**kwargs):
        config = no_confirm(load_run(EXAMPLE / "run.yml"), max_usd=5.0)
        provider = scripted(default=reply(True, 0.9, "the label matches"))
        return run(
            config,
            out=kwargs.pop("out", tmp_path / "out"),
            provider_factory=lambda spec: provider,
            **kwargs,
        )

    return build


class TestAlwaysDoesThree:
    def test_it_says_which_mode_each_field_is_in_at_the_top(self, report):
        lines = report().report.splitlines()
        head = "\n".join(lines[:12])
        assert "MODE" in head
        assert "MODE 0" in head and "no labels" in head

    def test_it_prints_the_threshold_and_its_source_beside_the_number(self, report):
        text = report().report
        assert "band 1%–20% (default)" in text
        assert "max 50% (default)" in text

    def test_it_names_what_it_cannot_conclude(self, report):
        text = report().report
        assert "WHAT THIS RUN CANNOT TELL YOU" in text
        assert "better than guessing" in text


class TestLayout:
    def test_free_checks_and_the_judge_are_separate_blocks(self, report):
        # "free checks found nothing" and "we paid a judge and it found
        # nothing" are different statements.
        text = report().report
        assert "  free checks" in text
        assert "\n  judge\n" in text

    def test_a_check_that_did_not_run_says_so_rather_than_vanishing(self, report):
        text = report().report
        assert "not run" in text
        assert "no must_agree_with declared" in text

    def test_both_grains_appear_and_the_item_one_is_worse(self, report):
        result = report()
        grains = result.metrics["grains"]
        assert grains["item"]["pass_rate"] <= grains["field"]["pass_rate"]
        assert "BOTH GRAINS" in result.report

    def test_nothing_wraps_past_eighty_columns(self, report):
        assert [line for line in report().report.splitlines() if len(line) > 80] == []

    def test_every_field_gets_a_section(self, report):
        text = report().report
        for name in ("jtbd", "summary", "outcome"):
            assert f"── {name} " in text


class TestGrainRollup:
    def test_an_item_passes_only_if_every_check_on_it_passes(self):
        rates = grain_rates(
            [
                finding("a", Status.PASS, "s-1"),
                finding("b", Status.FAIL, "s-1"),
                finding("a", Status.PASS, "s-2"),
                finding("b", Status.PASS, "s-2"),
            ]
        )
        assert rates["field"]["pass_rate"] == 0.75
        assert rates["item"]["pass_rate"] == 0.5

    def test_unscored_findings_are_counted_apart_and_never_absorbed(self):
        rates = grain_rates(
            [finding("a", Status.PASS, "s-1"), finding("b", Status.UNSCORED, "s-1")]
        )
        assert rates["field"]["pass_rate"] == 1.0
        assert rates["field"]["unscored"] == 1
        # An item whose only other check was not run still passes on what was
        # checked — the unscored row is reported, not folded in either way.
        assert rates["item"]["pass_rate"] == 1.0

    def test_nothing_scored_produces_no_rate_rather_than_a_hundred_percent(self):
        rates = grain_rates([finding("a", Status.UNSCORED, "s-1")])
        assert rates["field"]["pass_rate"] is None
        assert rates["item"]["pass_rate"] is None


class TestDriftAcrossRuns:
    def test_the_first_run_records_a_distribution_for_the_next_one(self, report, tmp_path):
        import json

        result = report(out=tmp_path / "out")
        lines = (tmp_path / "out" / "index.jsonl").read_text().splitlines()
        index = [json.loads(line) for line in lines]
        assert index[-1]["distributions"]["jtbd"]
        assert result.metrics["distributions"]["jtbd"]

    def test_a_second_run_compares_against_the_first(self, report, tmp_path):
        out = tmp_path / "out"
        report(out=out)
        second = report(out=out)
        drift = [f for f in second.findings if f.check == "drift"]
        assert drift
        assert all(f.status is Status.PASS for f in drift)  # same corpus, no movement

    def test_a_shifted_distribution_is_reported_in_percentage_points(
        self, no_confirm, scripted, tmp_path
    ):
        import dataclasses
        import json

        out = tmp_path / "out"
        config = no_confirm(load_run(EXAMPLE / "run.yml"), max_usd=5.0)
        provider = scripted(default=reply(True, 0.9, "ok"))
        run(config, out=out, provider_factory=lambda spec: provider)

        rows = [json.loads(line) for line in config.outputs.read_text().splitlines()]
        for row in rows[:5]:
            row["jtbd"] = "billing.card_declined"
        shifted = tmp_path / "shifted.jsonl"
        shifted.write_text("\n".join(json.dumps(r) for r in rows))

        second = run(
            dataclasses.replace(config, outputs=shifted),
            out=out,
            provider_factory=lambda spec: provider,
        )
        drift = next(f for f in second.findings if f.check == "drift" and f.field == "jtbd")
        assert drift.status is Status.FAIL
        assert drift.evidence["largest_move"]["label"] == "billing.card_declined"
        assert drift.evidence["largest_move"]["pp"] > 10

    def test_re_analysis_compares_against_the_same_baseline_not_itself(
        self, report, tmp_path
    ):
        from llm_expectations.run import analyse_run

        out = tmp_path / "out"
        report(out=out)
        second = report(out=out)
        again = analyse_run(second.directory)
        before = {f.item_id: f.score for f in second.findings if f.check == "drift"}
        after = {f.item_id: f.score for f in again.findings if f.check == "drift"}
        assert before == after


class TestReviewReasons:
    """The sentence beside a score must be the one that produced it."""

    def test_only_the_ranking_judge_explains_the_queue(self, no_confirm, tmp_path):
        from llm_expectations.judges.fake import FakeProvider

        config = no_confirm(load_run(EXAMPLE / "run.yml"), max_usd=5.0)
        ranker = FakeProvider(model="ranker", default=reply(True, 0.9, "the ranker's view"))
        other = FakeProvider(model="other", default=reply(True, 0.9, "a panel member's view"))
        by_id = {"judge-a": ranker, "judge-b": other, "judge-c": other}
        result = run(
            config, out=tmp_path / "out", provider_factory=lambda spec: by_id[spec.id]
        )
        review = result.report[result.report.index("REVIEW FIRST") :]
        review = review[: review.index("WHAT THIS RUN")]
        assert "the ranker's view" in review
        # A panel member's reason beside a triage score explains a number
        # nobody computed, and the two can contradict each other outright.
        assert "a panel member's view" not in review
