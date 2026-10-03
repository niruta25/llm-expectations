"""Free checks on a sentence written about an item."""

from __future__ import annotations

import pytest

from llm_expectations.checks import CheckContext, run_checks
from llm_expectations.checks.free_text import longest_shared_run, rare_tokens, words
from llm_expectations.config import settings_from_mapping
from llm_expectations.schema import schema_from_mapping
from llm_expectations.taxonomy import load_taxonomy
from llm_expectations.types import Item, Output, Status

from .conftest import EXAMPLE

SCHEMA = {
    "item": "s",
    "fields": {
        "jtbd": {"kind": "assigned", "taxonomy": "jtbd@v4"},
        "summary": {"kind": "free_text", "min_words": 5, "max_words": 30},
    },
}


@pytest.fixture(scope="module")
def jtbd():
    return load_taxonomy(EXAMPLE / "taxonomy.yml")


def build(rows, jtbd, schema=SCHEMA, **settings):
    """``rows`` maps an item id to (item text, summary)."""
    items = {k: Item(k, text) for k, (text, _) in rows.items()}
    outputs = {
        k: Output(k, {"summary": summary, "jtbd": "billing.card_declined"})
        for k, (_, summary) in rows.items()
    }
    return CheckContext(
        run_id="t",
        schema=schema_from_mapping(schema),
        settings=settings_from_mapping(settings),
        items=items,
        outputs=outputs,
        taxonomies={"jtbd": jtbd},
    )


def results(ctx, check):
    findings, _ = run_checks(ctx)
    return [f for f in findings if f.check == check]


class TestLength:
    def test_a_summary_in_bounds_passes(self, jtbd):
        ctx = build({"s-1": ("x " * 50, "the card was declined at renewal time")}, jtbd)
        assert results(ctx, "length_in_bounds")[0].status is Status.PASS

    def test_three_paragraphs_where_a_sentence_was_asked_for_fails(self, jtbd):
        ctx = build({"s-1": ("x " * 50, "word " * 60)}, jtbd)
        found = results(ctx, "length_in_bounds")[0]
        assert found.status is Status.FAIL
        assert "over the 30" in found.evidence["why"]

    def test_too_short_fails_the_other_way(self, jtbd):
        ctx = build({"s-1": ("x " * 50, "declined")}, jtbd)
        assert "under the 5" in results(ctx, "length_in_bounds")[0].evidence["why"]

    def test_an_empty_summary_is_a_failure_not_an_omission(self, jtbd):
        ctx = build({"s-1": ("x " * 50, "   ")}, jtbd)
        found = results(ctx, "length_in_bounds")[0]
        assert found.status is Status.FAIL
        assert "empty" in found.evidence["why"]

    def test_without_bounds_the_check_does_not_run(self, jtbd):
        schema = {
            "item": "s",
            "fields": {
                "jtbd": {"kind": "assigned", "taxonomy": "jtbd@v4"},
                "summary": {"kind": "free_text"},
            },
        }
        ctx = build({"s-1": ("x " * 50, "anything")}, jtbd, schema=schema)
        _, skipped = run_checks(ctx)
        assert any(s.check == "length_in_bounds" for s in skipped)


class TestSpecificity:
    def _corpus(self, special):
        """Twenty near-identical sessions so common words are common."""
        rows = {
            f"s-{i}": (
                f"the customer had a billing problem and it was handled case {i}",
                "the customer had a problem and it was resolved",
            )
            for i in range(20)
        }
        rows["s-99"] = special
        return rows

    def test_filler_true_of_every_session_is_flagged(self, jtbd):
        ctx = build(
            self._corpus((
                "Maya's card ending 4471 was declined at renewal",
                "the customer had a problem and it was resolved",
            )),
            jtbd,
        )
        found = next(f for f in results(ctx, "specificity") if f.item_id == "s-99")
        assert found.status is Status.FAIL
        assert "would fit any session" in found.evidence["why"]

    def test_a_rare_detail_from_the_item_passes(self, jtbd):
        ctx = build(
            self._corpus((
                "Maya's card ending 4471 was declined at renewal",
                "card 4471 was declined at renewal",
            )),
            jtbd,
        )
        found = next(f for f in results(ctx, "specificity") if f.item_id == "s-99")
        assert found.status is Status.PASS
        assert "4471" in found.evidence["used"]

    def test_an_item_with_nothing_distinctive_is_unscored_not_failed(self, jtbd):
        # Punishing the output for the corpus being repetitive would blame
        # the model for the data.
        rows = {
            f"s-{i}": ("the same words every time", "the same words every time")
            for i in range(30)
        }
        ctx = build(rows, jtbd)
        statuses = {f.status for f in results(ctx, "specificity")}
        assert statuses == {Status.UNSCORED}

    def test_rare_tokens_are_the_ones_in_few_items(self):
        rows = [(f"s-{i}", "common words here") for i in range(100)]
        rows.append(("s-99", "common words here 4471"))
        rare = rare_tokens(rows)
        assert "4471" in rare
        assert "common" not in rare


class TestCopyRatio:
    def test_a_paraphrase_passes(self, jtbd):
        ctx = build(
            {
                "s-1": (
                    "Maya wrote in March. Her card ending 4471 was declined when the "
                    "subscription auto-renewed, and a replacement card worked.",
                    "Maya's 4471 card failed at renewal; a replacement cleared it.",
                )
            },
            jtbd,
        )
        assert results(ctx, "copy_ratio")[0].status is Status.PASS

    def test_pasting_the_source_back_fails(self, jtbd):
        source = "her card ending 4471 was declined when the subscription auto renewed"
        ctx = build({"s-1": (f"Maya wrote in March. {source}.", source)}, jtbd)
        found = results(ctx, "copy_ratio")[0]
        assert found.status is Status.FAIL
        assert found.score == pytest.approx(1.0)
        assert "pasting, not summarising" in found.evidence["why"]

    def test_the_longest_run_is_what_counts_not_total_overlap(self):
        # Scattered shared words are what a summary is made of; one unbroken
        # run is what pasting looks like.
        source = "a b c d e f g h".split()
        assert longest_shared_run(source, "a c e g".split()) == 1
        assert longest_shared_run(source, "d e f g".split()) == 4

    def test_a_short_summary_inside_its_bounds_can_still_be_pasted(self, jtbd):
        # Length cannot catch this one, which is why copy ratio exists.
        source = "the export crashes on any file over twenty thousand rows"
        ctx = build({"s-1": (source + " since the update.", source)}, jtbd)
        length = results(ctx, "length_in_bounds")[0]
        assert length.status is Status.PASS
        assert results(ctx, "copy_ratio")[0].status is Status.FAIL


class TestBoilerplate:
    def test_a_template_across_the_batch_is_found(self, jtbd):
        rows = {
            f"s-{i}": (
                f"session {i} about something happening to a customer",
                "the issue was investigated and a resolution was provided to the customer",
            )
            for i in range(10)
        }
        found = results(build(rows, jtbd), "boilerplate")[0]
        assert found.status is Status.FAIL
        assert found.evidence["near_duplicates"] == 10
        assert "fallen into a template" in found.evidence["why"]

    def test_genuinely_different_summaries_pass(self, jtbd):
        rows = {
            "s-1": ("a", "the card was declined by the issuer at renewal time"),
            "s-2": ("b", "the password reset email bounced off a corporate mail server"),
            "s-3": ("c", "the export crashes above twenty thousand rows since the update"),
        }
        assert results(build(rows, jtbd), "boilerplate")[0].status is Status.PASS

    def test_a_pair_is_named_so_it_can_be_read(self, jtbd):
        rows = {
            "s-1": ("a", "the issue was investigated and a resolution was provided here"),
            "s-2": ("b", "the issue was investigated and a resolution was provided here"),
            "s-3": ("c", "something else entirely different happened in this third session"),
        }
        found = results(build(rows, jtbd), "boilerplate")[0]
        assert found.evidence["pairs"]
        assert {found.evidence["pairs"][0]["a"], found.evidence["pairs"][0]["b"]} == {
            "s-1", "s-2"
        }

    def test_one_output_cannot_duplicate_anything(self, jtbd):
        assert results(build({"s-1": ("a", "only one summary here at all")}, jtbd),
                       "boilerplate") == []


def test_words_lowercases_and_keeps_digits():
    assert words("Card 4471 Was Declined!") == ["card", "4471", "was", "declined"]


@pytest.fixture(scope="module")
def findings():
    """Every free check, run once over the worked example."""
    from llm_expectations.config import load_run
    from llm_expectations.run import check_context, load_dataset

    config = load_run(EXAMPLE / "run.yml")
    dataset = load_dataset(config)
    return run_checks(check_context(config, dataset, "t"))[0]


class TestAgainstTheFixture:
    def test_the_filler_summary_is_caught(self, findings):
        assert any(
            f.check == "specificity" and f.item_id == "s-11" and f.status is Status.FAIL
            for f in findings
        )

    def test_the_copied_summary_is_caught(self, findings):
        assert any(
            f.check == "copy_ratio" and f.item_id == "s-13" and f.status is Status.FAIL
            for f in findings
        )

    def test_no_clean_item_is_flagged_by_a_free_text_check(self, findings):
        import yaml

        clean = set(yaml.safe_load((EXAMPLE / "expected.yml").read_text())["clean"])
        text_checks = {"specificity", "copy_ratio", "length_in_bounds", "boilerplate"}
        flagged = {
            f.item_id
            for f in findings
            if f.check in text_checks and f.status is Status.FAIL and f.item_id
        }
        assert not flagged & clean


class TestTheClaimJudgeOnTheFixture:
    """The one plant nothing free can reach."""

    @staticmethod
    def scripted_for_s10():
        from llm_expectations.judges.fake import FakeProvider, claims, reply

        def answer(request):
            if "ASSIGNED LABEL:" in request.user:
                return reply(True, 0.9, "the label matches")
            if "issued a refund" in request.user:
                return claims(
                    supported=["Noor's Amex was declined twice at renewal"],
                    unsupported=["we issued a refund of $49"],
                )
            return claims(supported=["the text restates the item"])

        return FakeProvider(model="m", default=answer, claim_default=answer)

    def test_the_invented_claim_is_found_when_the_judge_is_asked(self, tmp_path):
        import dataclasses

        from llm_expectations.config import Budget, load_run
        from llm_expectations.run import run

        # The free gate does not flag s-10 — correctly, every free check
        # passes it — so a full audit is what asks the judge about it. The
        # gate's cost is measured in the report, not assumed away.
        config = load_run(EXAMPLE / "run.yml")
        config = dataclasses.replace(
            config,
            budget=Budget(None, False),
            settings=type(config.settings)(
                project={**config.settings.project, "free_text_audit_rate": 1.0}
            ),
        )
        result = run(
            config, out=tmp_path / "out", provider_factory=lambda spec: self.scripted_for_s10()
        )
        failed = {
            f.item_id for f in result.findings
            if f.check == "claims_supported" and f.status is Status.FAIL
        }
        assert failed == {"s-10"}
        found = next(f for f in result.findings if f.check == "claims_supported"
                     and f.item_id == "s-10")
        assert "refund" in found.evidence["unsupported"][0]

    def test_the_gates_cost_is_measured_rather_than_assumed(self, tmp_path):
        import dataclasses

        from llm_expectations.config import Budget, load_run
        from llm_expectations.run import run

        config = load_run(EXAMPLE / "run.yml")
        config = dataclasses.replace(
            config,
            budget=Budget(None, False),
            settings=type(config.settings)(
                project={**config.settings.project, "free_text_audit_rate": 1.0}
            ),
        )
        result = run(
            config, out=tmp_path / "out", provider_factory=lambda spec: self.scripted_for_s10()
        )
        gate = result.metrics["free_text_gate"]
        # s-10 is in the audited pile and it is a real defect, so the gate's
        # miss rate is non-zero and the report says so.
        assert gate["miss_rate"] > 0
        assert "what the free gate is missing, measured rather than assumed" in gate["why"]


class TestDefectRatings:
    """Mode 1 for free text: people mark boxes, not better summaries."""

    @staticmethod
    def agreement(marks, flagged):
        from llm_expectations.metrics.classification import defect_agreement
        from llm_expectations.types import Label

        labels = [
            Label(item, "summary", mark, "ann-1") for item, marks_ in marks.items()
            for mark in marks_
        ]
        return {row.defect: row for row in defect_agreement("summary", labels, flagged)}

    def test_a_check_and_a_person_agreeing_scores_one(self):
        rows = self.agreement(
            {"s-1": ["made_up"], "s-2": ["none"]},
            {"claims_supported": {"s-1": True, "s-2": False}},
        )
        assert rows["made_up"].agreement == 1.0

    def test_a_check_that_fires_where_people_do_not_is_named_as_over_flagging(self):
        rows = self.agreement(
            {f"s-{i}": ["none"] for i in range(10)},
            {"specificity": {f"s-{i}": True for i in range(10)}},
        )
        assert rows["too_generic"].agreement == 0.0
        assert rows["too_generic"].leaning == "over-flags"

    def test_a_check_that_misses_what_people_catch_is_named_as_under_flagging(self):
        rows = self.agreement(
            {f"s-{i}": ["made_up"] for i in range(10)},
            {"claims_supported": {f"s-{i}": False for i in range(10)}},
        )
        assert rows["made_up"].leaning == "under-flags"

    def test_an_unrated_row_is_not_counted_as_agreement(self):
        # Treating unrated rows as clean would inflate every number here as
        # a corpus grew.
        rows = self.agreement(
            {"s-1": ["none"]},
            {"specificity": {"s-1": False, "s-2": False, "s-3": False}},
        )
        assert rows["too_generic"].compared == 1

    def test_a_defect_with_no_matching_check_is_absent_rather_than_zero(self):
        rows = self.agreement({"s-1": ["made_up"]}, {})
        assert rows == {}

    def test_rated_clean_is_distinguishable_from_unrated(self):
        from llm_expectations.metrics.classification import RATED_CLEAN

        rated = self.agreement(
            {"s-1": [RATED_CLEAN]}, {"specificity": {"s-1": False, "s-2": False}}
        )
        unrated = self.agreement({}, {"specificity": {"s-1": False, "s-2": False}})
        assert rated["too_generic"].compared == 1
        assert unrated == {}

    def test_the_fixture_reports_an_agreement_per_box(self, tmp_path):
        from llm_expectations.cli import main

        assert main(
            ["run", str(EXAMPLE / "run-offline.yml"), "--out", str(tmp_path), "--yes"]
        ) == 0
        directory = next(p for p in tmp_path.iterdir() if p.is_dir())
        report = (directory / "report.md").read_text()
        assert "vs human defect ratings" in report
        for defect in ("made_up", "too_generic", "contradicts", "missing"):
            assert defect in report

    def test_each_box_carries_its_own_denominator(self, tmp_path):
        import json

        from llm_expectations.cli import main

        # Each box is compared against a different check, and those checks do
        # not all reach the same rows. One shared header count would be wrong
        # for every row but the first.
        main(["run", str(EXAMPLE / "run-offline.yml"), "--out", str(tmp_path), "--yes"])
        directory = next(p for p in tmp_path.iterdir() if p.is_dir())
        metrics = json.loads((directory / "metrics.json").read_text())
        rows = metrics["free_text_vs_humans"]["summary"]["defects"]
        assert len({row["compared"] for row in rows}) > 1
