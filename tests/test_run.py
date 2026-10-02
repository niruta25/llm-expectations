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


@pytest.fixture
def example(no_confirm):
    return no_confirm(load_run(EXAMPLE / "run.yml"), max_usd=5.0)


@pytest.fixture
def provider(scripted):
    return scripted(rules=FIXTURE_SCRIPT, default=reply(True, 0.92, "the label matches"))


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
        assert "better than guessing" in report
        assert "Error Recall@Budget" in report
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
        # Thirteen labels is mode 1 by shape and far below every floor. The
        # report must not turn it into a quality number.
        result = execute(example, tmp_path / "out", provider)
        assert result.metrics["modes"]["outcome"] == 1
        assert "macro F1" not in result.report
        assert "accuracy" not in result.report.lower()


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
