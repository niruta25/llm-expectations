"""The pipeline end to end, and the promises it makes about money."""

from __future__ import annotations

import json

import pytest

from llm_expectations.budget import BudgetGuard
from llm_expectations.cache import VerdictCache, cache_key
from llm_expectations.config import load_run
from llm_expectations.judges.fake import reply
from llm_expectations.plan import estimate_cost, plan_run
from llm_expectations.run import analyse_run, detect_modes, load_dataset, run
from llm_expectations.types import Mode, Status, Verdict

from .conftest import EXAMPLE

# Keyed on text unique to the item, because that is all a provider ever sees.
FIXTURE_SCRIPT = (
    ("fraud alert", reply(False, 0.81, "the issuer refused the card")),
    ("year up front", reply(False, 0.74, "billing is a parent, not a leaf")),
    ("reset loop", reply(False, 0.88, "access.login_broken is not a permitted label")),
    ("Okta", reply("cannot_decide", 0.4, "the item does not say enough")),
    ("blank panel", "not json at all"),
)


def execute(config, out, provider, **kwargs):
    return run(config, out=out, provider_factory=lambda spec: provider, **kwargs)


class TestEndToEnd:
    def test_the_run_writes_every_file_the_design_names(self, example, provider, tmp_path):
        result = execute(example, tmp_path / "out", provider)
        written = {path.name for path in result.directory.iterdir()}
        assert written == {
            "verdicts.jsonl", "findings.jsonl", "risk.jsonl",
            "metrics.json", "run.json", "report.md",
        }
        assert (tmp_path / "out" / "index.jsonl").exists()

    def test_the_planted_errors_it_can_see_come_out_on_top(self, example, provider, tmp_path):
        # s-07 sibling confusion, s-08 too shallow, s-09 invented label. The
        # other five plants need checks that do not exist yet; M1 is not
        # expected to find them, and does not pretend to.
        result = execute(example, tmp_path / "out", provider)
        ranked = [row.item_id for row in result.risk_rows if row.triage_score is not None]
        assert set(ranked[:3]) == {"s-07", "s-08", "s-09"}

    def test_an_abstention_is_not_judged_and_not_scored_as_wrong(self, example, provider, tmp_path):
        result = execute(example, tmp_path / "out", provider)
        judged = {(v.item_id, v.field) for v in result.verdicts}
        assert ("s-13", "jtbd") not in judged
        unscored = [
            f for f in result.findings
            if (f.item_id, f.field) == ("s-13", "jtbd") and f.status is Status.UNSCORED
        ]
        assert unscored and "abstention" in unscored[0].evidence["reason"]

    def test_the_free_text_defect_checks_are_unscored_with_a_reason_not_absent(
        self, example, provider, tmp_path
    ):
        # Cross-field agreement scores `summary` from M2 on, because it is an
        # item check and belongs to neither kind. The defect checks that are
        # specific to free text are still unscored, and say which milestone
        # they wait for rather than quietly not appearing.
        result = execute(example, tmp_path / "out", provider)
        defects = [f for f in result.findings if f.field == "summary" and f.check == "free_text"]
        assert len(defects) == len(load_dataset(example).outputs)
        assert all(f.status is Status.UNSCORED for f in defects)
        assert all("M6" in f.evidence["reason"] for f in defects)

    def test_an_unreadable_reply_leaves_the_item_unranked_rather_than_clean(
        self, example, provider, tmp_path
    ):
        result = execute(example, tmp_path / "out", provider)
        unranked = {row.item_id for row in result.risk_rows if row.triage_score is None}
        assert "s-13" in unranked  # its only judged field came back as prose

    def test_the_report_leads_with_the_uncalibrated_stamp(self, example, provider, tmp_path):
        report = execute(example, tmp_path / "out", provider).report
        assert "UNCALIBRATED" in report
        assert "not a probability of error" in report
        assert "WHAT THIS RUN CANNOT TELL YOU" in report

    def test_the_report_says_what_it_cannot_conclude(self, example, provider, tmp_path):
        report = execute(example, tmp_path / "out", provider).report
        assert "Error Recall@Budget" in report
        assert "withheld" in report
        # And it is specific about free text rather than claiming the field is
        # untouched — cross-field agreement does read it.
        assert "invented anything, or is filler" in report
        assert "cross-field agreement is the only check reading these fields" in report

    def test_item_grain_is_never_rosier_than_field_grain(self, example, provider, tmp_path):
        result = execute(example, tmp_path / "out", provider)
        scored = [f for f in result.findings if f.status is not Status.UNSCORED]
        by_item: dict[str, list[bool]] = {}
        for finding in scored:
            by_item.setdefault(str(finding.item_id), []).append(finding.status is Status.PASS)
        field_rate = sum(f.status is Status.PASS for f in scored) / len(scored)
        item_rate = sum(all(v) for v in by_item.values()) / len(by_item)
        assert item_rate <= field_rate


class TestModes:
    def test_no_labels_is_mode_zero(self, project):
        config = load_run(project())
        assert detect_modes(config, ()) == {"jtbd": Mode.NO_LABELS, "summary": Mode.NO_LABELS}

    def test_the_example_reports_one_mode_per_field(self, example):
        modes = detect_modes(example, load_dataset(example).labels)
        assert modes["jtbd"] is Mode.DOUBLE_LABELLED  # ann-2 second-opinions four items
        assert modes["outcome"] is Mode.LABELLED
        assert modes["summary"] is Mode.NO_LABELS

    def test_mode_reports_the_data_shape_and_the_floors_do_the_refusing(
        self, example, provider, tmp_path
    ):
        # Thirteen labels is mode 1 by shape, so the classification metrics
        # are computed — and every one of them carries the caveat its sample
        # size earns. Hiding them would lose the only information thirteen
        # rows actually hold; printing them bare would be the lie.
        result = execute(example, tmp_path / "out", provider)
        assert result.metrics["modes"]["outcome"] == 1
        assert "macro F1" in result.report
        assert "cannot support a conclusion" in result.report
        # And no ranking claim is made off thirteen rows.
        assert "gate2.auc" in result.report


class TestCache:
    def test_a_second_run_reusing_the_first_makes_no_calls(self, example, provider, tmp_path):
        first = execute(example, tmp_path / "out", provider)
        before = len(provider.calls)
        second = execute(example, tmp_path / "out2", provider, reuse=first.directory)
        assert len(provider.calls) == before
        assert second.metrics["calls"] == 0
        assert second.metrics["cost_usd"] == 0.0

    def test_reused_verdicts_are_marked_and_cost_nothing(self, example, provider, tmp_path):
        first = execute(example, tmp_path / "out", provider)
        second = execute(example, tmp_path / "out2", provider, reuse=first.directory)
        assert all(v.metadata["cache_hit"] for v in second.verdicts)
        assert sum(f.cost_usd for f in second.findings) == 0.0

    def test_a_changed_definition_is_a_miss_because_it_was_a_different_question(
        self, project, scripted, no_confirm, tmp_path
    ):
        from .conftest import MINIMAL_TAXONOMY

        config = no_confirm(load_run(project()))
        judge = scripted()
        first = execute(config, tmp_path / "out", judge)
        edited = MINIMAL_TAXONOMY.replace("did not go through", "was refused by the issuer")
        changed = no_confirm(load_run(project(**{"taxonomy.yml": edited})))
        before = len(judge.calls)
        execute(changed, tmp_path / "out2", judge, reuse=first.directory)
        assert len(judge.calls) > before

    def test_a_verdict_is_on_disk_the_moment_it_comes_back(self, tmp_path):
        # Not buffered: a run that dies halfway has still paid for every call
        # it made, and the point of this file is never paying twice.
        path = tmp_path / "verdicts.jsonl"
        with VerdictCache(path) as cache:
            cache.put(
                cache_key(
                    judge_id="a", model="m", prompt_fingerprint="f",
                    item_id="s-1", field="jtbd", output_value="x",
                ),
                Verdict("a", "label_correct", "s-1", "jtbd", Status.PASS, 0.9, "ok"),
            )
            assert path.read_text().count("\n") == 1


class TestAnalyse:
    def test_analyse_issues_zero_model_calls(self, example, provider, tmp_path):
        first = execute(example, tmp_path / "out", provider)

        def explode(spec):
            raise AssertionError("analyse must never construct a provider")

        import llm_expectations.run as run_module

        original = run_module.build_provider
        run_module.build_provider = explode
        try:
            again = analyse_run(first.directory)
        finally:
            run_module.build_provider = original

        assert again.metrics["calls"] == 0
        assert len(again.verdicts) == len(first.verdicts)
        assert [r.item_id for r in again.risk_rows] == [r.item_id for r in first.risk_rows]

    def test_analyse_reloads_the_config_the_run_actually_used(
        self, project, scripted, no_confirm, tmp_path
    ):
        from .conftest import MINIMAL_RUN

        # Not "run.yml" — a run started from another filename has to be
        # re-analysable, and guessing the name would reload the wrong project.
        root = project(**{"experiment.yml": MINIMAL_RUN})
        config = no_confirm(load_run(root.parent / "experiment.yml"))
        result = execute(config, tmp_path / "out", scripted())
        manifest = json.loads((result.directory / "run.json").read_text())
        assert manifest["run_yml"].endswith("experiment.yml")
        assert analyse_run(result.directory).metrics["calls"] == 0


class TestBudget:
    def test_declining_the_prompt_spends_nothing(self, provider, tmp_path):
        config = load_run(EXAMPLE / "run.yml")  # confirm: true
        assert run(config, out=tmp_path / "out", provider_factory=lambda s: provider,
                   ask=lambda _: "n") is None
        assert provider.calls == []

    def test_work_past_the_cap_is_unscored_never_passed(
        self, example, provider, tmp_path, no_confirm
    ):
        capped = no_confirm(example, max_usd=0.0001)
        result = execute(capped, tmp_path / "out", provider)
        capped_rows = [
            v for v in result.verdicts if "budget was reached" in v.reason
        ]
        assert capped_rows
        assert all(v.status is Status.UNSCORED for v in capped_rows)

    def test_an_unpriced_model_produces_no_estimate_rather_than_a_plausible_one(self, project):
        from .conftest import MINIMAL_JUDGES

        # judge-b is gpt-5-mini, which has no entry in the price table. A made
        # -up cost is the same species of error as a made-up confidence.
        judges = MINIMAL_JUDGES.replace("triage:\n  judge: judge-a", "triage:\n  judge: judge-b")
        config = load_run(project(**{"judges.yml": judges}))
        dataset = load_dataset(config)
        plan = plan_run(config, dataset.items, dataset.outputs)
        assert plan.usd is None
        assert plan.unpriced_models == ("gpt-5-mini",)
        assert estimate_cost("gpt-5-mini", 1000, 100) is None

    def test_a_cap_that_cannot_be_enforced_says_so(self, capsys):
        import io

        from llm_expectations.plan import Plan, PlannedJob

        plan = Plan(
            jobs=(PlannedJob("triage", "judge-b", "some-model", 10, 0, 100, 60, None),),
            skipped={},
            unpriced_models=("some-model",),
        )
        stream = io.StringIO()
        BudgetGuard(max_usd=5.0, confirm=False).approve(plan, stream=stream)
        assert "cannot be enforced" in stream.getvalue()


def test_the_run_module_is_not_shadowed_by_a_function_of_the_same_name():
    # `from .run import run` in __init__ would rebind the package attribute,
    # so `import llm_expectations.run` would hand back a function. Callers
    # reaching for the module would get a confusing AttributeError instead.
    import types

    import llm_expectations
    import llm_expectations.run as module

    assert isinstance(module, types.ModuleType)
    assert not hasattr(llm_expectations, "run") or isinstance(
        llm_expectations.run, types.ModuleType
    )


class TestRunDirectories:
    """Two runs in one minute must not merge into one."""

    def test_a_second_run_in_the_same_minute_gets_its_own_directory(self, tmp_path):
        from datetime import datetime, timezone

        from llm_expectations.write import run_directory

        now = datetime(2026, 9, 29, 14, 32, tzinfo=timezone.utc)
        first = run_directory(tmp_path, "jtbd-p8", now=now)
        second = run_directory(tmp_path, "jtbd-p8", now=now)
        assert first != second
        assert first.name == "2026-09-29_1432_jtbd-p8"
        assert second.name == "2026-09-29_1432-2_jtbd-p8"

    def test_two_runs_do_not_share_a_verdict_file(self, example, provider, tmp_path):
        first = execute(example, tmp_path / "out", provider)
        second = execute(example, tmp_path / "out", provider)
        assert first.directory != second.directory
        # Appending into the first run's cache would silently merge two runs.
        assert len(first.verdicts) == len(second.verdicts)


class TestPanel:
    """The panel measures. The tests below pin what it is not allowed to do."""

    @pytest.fixture
    def three_judges(self):
        from llm_expectations.judges.fake import FakeProvider

        strict = FakeProvider(
            model="strict-1",
            rules=(
                ("fraud alert", reply(False, 0.85, "the issuer refused it",
                                      instead="billing.card_declined")),
                ("Amex was declined", reply(False, 0.7, "reads as a payment failure",
                                            instead="billing.payment_failed")),
            ),
            default=reply(True, 0.9, "matches"),
        )
        middling = FakeProvider(
            model="middling-1",
            rules=(
                ("fraud alert", reply(False, 0.6, "the bank refused it",
                                      instead="billing.card_declined")),
                ("Okta", reply("cannot_decide", 0.4, "not enough detail")),
            ),
            default=reply(True, 0.88, "fine"),
        )
        soft = FakeProvider(model="soft-1", default=reply(True, 0.95, "looks fine"))
        return {"judge-a": strict, "judge-b": soft, "judge-c": middling}

    def run_panel(self, example, tmp_path, three_judges, **kw):
        return run(
            example,
            out=tmp_path / "out",
            provider_factory=lambda spec: three_judges[spec.id],
            **kw,
        )

    def test_every_member_is_asked_the_byte_identical_prompt(
        self, example, tmp_path, three_judges
    ):
        # Otherwise panel disagreement measures prompt differences, not judges.
        self.run_panel(example, tmp_path, three_judges)
        prompts = {
            judge_id: {(r.system, r.user) for r in provider.calls}
            for judge_id, provider in three_judges.items()
        }
        first = next(iter(prompts.values()))
        assert all(seen == first for seen in prompts.values())

    def test_a_panel_finding_shows_the_votes_and_quotes_the_dissent(
        self, example, tmp_path, three_judges
    ):
        result = self.run_panel(example, tmp_path, three_judges)
        panel = [f for f in result.findings if f.check == "label_correct_panel"]
        assert panel
        # A real split: two judges that both decided, and decided differently.
        # `correct` against `cannot_decide` is not dissent — an abstention is
        # not a vote, and quoting one as a dissenting opinion would be the
        # panel inventing disagreement.
        split = next(
            f
            for f in panel
            if len({v for v in f.evidence["votes"].values() if v in {"correct", "incorrect"}}) > 1
        )
        assert split.evidence["dissent"]
        assert split.evidence["agreement"]

    def test_excluding_a_judge_can_leave_every_split_a_tie_and_it_says_so(
        self, example, tmp_path, three_judges
    ):
        # Dropping the rubber stamp leaves two voters, so a disagreement is
        # 1-1: unscored, not a pass. That is the honest consequence of a
        # three-judge panel with one broken member, and it is an argument for
        # replacing the judge rather than for breaking the tie.
        result = self.run_panel(example, tmp_path, three_judges)
        panel = [f for f in result.findings if f.check == "label_correct_panel"]
        tied = [f for f in panel if "tied" in f.evidence["agreement"]]
        assert tied
        assert all(f.status is Status.UNSCORED for f in tied)

    def test_an_abstention_is_never_quoted_as_dissent(
        self, example, tmp_path, three_judges
    ):
        result = self.run_panel(example, tmp_path, three_judges)
        for found in result.findings:
            if found.check != "label_correct_panel":
                continue
            abstained = {
                j for j, vote in found.evidence["votes"].items() if vote == "cannot_decide"
            }
            assert not (abstained & set(found.evidence["dissent"]))

    def test_the_panel_never_feeds_the_ranking(self, example, tmp_path, three_judges):
        # The first of the three things DESIGN.md §6 says it must not do.
        # Only the triage judge's verdicts may move an item up the queue.
        result = self.run_panel(example, tmp_path, three_judges)
        triage_only = run(
            example,
            out=tmp_path / "solo",
            provider_factory=lambda spec: three_judges["judge-a"],
        )
        with_panel = {r.item_id: r.triage_score for r in result.risk_rows}
        without = {r.item_id: r.triage_score for r in triage_only.risk_rows}
        assert with_panel == without

    def test_unanimity_is_not_treated_as_proof(self, example, tmp_path, three_judges):
        # A unanimous pass is still only a pass on the panel check; it does
        # not short-circuit anything else or mark the item exempt.
        result = self.run_panel(example, tmp_path, three_judges)
        unanimous = [
            f
            for f in result.findings
            if f.check == "label_correct_panel"
            and set(f.evidence["votes"].values()) == {"correct"}
        ]
        assert unanimous
        flagged = {f.item_id for f in result.findings if f.status is Status.FAIL}
        # s-09's label is invented; the panel agreeing cannot clear it.
        assert "s-09" in flagged

    def test_a_rubber_stamp_is_dropped_from_the_vote_but_kept_on_disk(
        self, example, tmp_path, three_judges
    ):
        result = self.run_panel(example, tmp_path, three_judges)
        panel = result.metrics["panel"]
        assert "judge-b" in panel["excluded"]
        assert "judge-b" not in panel["members"]
        assert any(v.judge_id == "judge-b" for v in result.verdicts)

    def test_effective_votes_are_reported_against_what_was_paid_for(
        self, example, tmp_path, three_judges
    ):
        panel = self.run_panel(example, tmp_path, three_judges).metrics["panel"]
        assert panel["effective_votes"] <= panel["votes_paid_for"]

    def test_a_judge_in_both_jobs_is_asked_once(self, example, tmp_path, three_judges):
        # judge-a does triage and sits on the panel. The cache key is the
        # question, not the job that wanted the answer.
        self.run_panel(example, tmp_path, three_judges)
        asked = [(r.system, r.user) for r in three_judges["judge-a"].calls]
        assert len(asked) == len(set(asked))

    def test_the_report_carries_the_panel_block(self, example, tmp_path, three_judges):
        report = self.run_panel(example, tmp_path, three_judges).report
        assert "  PANEL" in report
        assert "effective votes" in report
        assert "does not route disagreements to review" in report

    def test_without_a_panel_the_report_says_what_that_costs(
        self, example, provider, tmp_path, no_confirm
    ):
        import dataclasses

        from llm_expectations.config import Judges

        judges = config_without_panel = example.judges
        stripped = dataclasses.replace(
            example, judges=Judges(judges=judges.judges, panel=None, triage=judges.triage)
        )
        assert config_without_panel is judges
        report = run(
            stripped, out=tmp_path / "out", provider_factory=lambda spec: provider
        ).report
        assert "no panel is wired" in report


class TestFindingsAreNotPooledAcrossJudges:
    """One judge's verdicts decide `label_correct`; the panel has its own check."""

    def test_the_rate_is_per_item_not_per_verdict(self, example, tmp_path, scripted):
        from llm_expectations.judges.fake import reply

        result = run(
            example,
            out=tmp_path / "out",
            provider_factory=lambda spec: scripted(default=reply(True, 0.9, "fine")),
        )
        per_field: dict[str, set[str]] = {}
        for found in result.findings:
            if found.check == "label_correct":
                key = (found.item_id, found.field)
                assert key not in per_field.setdefault(str(found.field), set())
                per_field[str(found.field)].add(key)

    def test_a_rubber_stamp_on_the_panel_cannot_lift_the_judge_rate(
        self, example, tmp_path, scripted
    ):
        from llm_expectations.judges.fake import FakeProvider, reply

        strict = FakeProvider(model="strict", default=reply(False, 0.9, "wrong"))
        soft = FakeProvider(model="soft", default=reply(True, 0.95, "fine"))
        by_id = {"judge-a": strict, "judge-b": soft, "judge-c": soft}
        result = run(
            example, out=tmp_path / "out", provider_factory=lambda spec: by_id[spec.id]
        )
        scored = [
            f
            for f in result.findings
            if f.check == "label_correct" and f.status is not Status.UNSCORED
        ]
        # judge-a rejected everything it was asked. Two lenient panel members
        # must not drag that towards a pass.
        assert scored
        assert all(f.status is Status.FAIL for f in scored)


class TestTheManifestCoversTheWholePipeline:
    """`expected.yml` must account for every check that fires, not just free ones."""

    @staticmethod
    def manifest():
        import yaml

        from .conftest import EXAMPLE

        return yaml.safe_load((EXAMPLE / "expected.yml").read_text())

    def test_no_clean_item_is_flagged_by_anything_in_the_full_run(
        self, example, provider, tmp_path
    ):
        result = run(
            example, out=tmp_path / "out", provider_factory=lambda spec: provider
        )
        flagged = {f.item_id for f in result.findings if f.status is Status.FAIL and f.item_id}
        assert not flagged & set(self.manifest()["clean"])

    def test_every_flagged_item_is_one_the_manifest_planted(
        self, example, provider, tmp_path
    ):
        result = run(
            example, out=tmp_path / "out", provider_factory=lambda spec: provider
        )
        flagged = {f.item_id for f in result.findings if f.status is Status.FAIL and f.item_id}
        planted = {plant["item"] for plant in self.manifest()["plants"]}
        assert flagged <= planted, f"unaccounted: {sorted(flagged - planted)}"

    def test_every_plant_due_by_now_is_caught_by_the_check_that_owns_it(
        self, example, provider, tmp_path
    ):
        landed = {"M1", "M2", "M3", "M4", "M5"}
        result = run(
            example, out=tmp_path / "out", provider_factory=lambda spec: provider
        )
        failed = {(f.item_id, f.check) for f in result.findings if f.status is Status.FAIL}
        for plant in self.manifest()["plants"]:
            if plant["milestone"] not in landed:
                continue
            assert any((plant["item"], check) in failed for check in plant["caught_by"]), (
                f"{plant['item']} ({plant['defect']}) was not caught by {plant['caught_by']}"
            )

    def test_the_abstention_is_never_scored_as_a_wrong_answer(
        self, example, provider, tmp_path
    ):
        # s-13 abstains because jtbd@v4 has no leaf for it. That is the right
        # answer, and counting it as a model error would turn a taxonomy gap
        # into a quality number.
        result = run(
            example, out=tmp_path / "out", provider_factory=lambda spec: provider
        )
        rows = [
            f
            for f in result.findings
            if f.item_id == "s-13" and f.field == "jtbd" and f.check == "label_tree_bucket"
        ]
        assert rows and all(f.status is Status.UNSCORED for f in rows)
