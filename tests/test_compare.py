"""Two runs head to head, and the conclusion compare refuses to draw."""

from __future__ import annotations

import json

import pytest

from llm_expectations.compare import (
    ComparisonError,
    compare_runs,
    load_and_compare,
    load_run_directory,
)
from llm_expectations.report import comparison_report
from llm_expectations.taxonomy import TaxonomyError, load_migration

MIGRATION = """
from: jtbd@v4
to: jtbd@v5
labels:
  billing.payment_failed:  billing.charge_failed
  billing.card_declined:   billing.charge_failed
  billing.refund_request:  billing.refund_request
  access.password_reset:   access.password_reset
  access.sso_issue:        access.sso_issue
  product.bug_report:      product.bug_report
  product.feature_request: null
"""


class TestLoading:
    def test_a_directory_that_is_not_a_run_says_so(self, tmp_path):
        with pytest.raises(ComparisonError, match="no run.json"):
            load_run_directory(tmp_path)

    def test_a_finished_run_loads_its_metrics_and_findings(self, two_runs):
        _, a, _ = two_runs(n=200)
        loaded = load_run_directory(a)
        assert loaded.items == 200
        assert loaded.prompt == "p7"
        assert loaded.findings


class TestComparing:
    def test_the_metrics_line_up_with_their_change(self, two_runs):
        _, a, b = two_runs(n=400, rates=(0.30, 0.15))
        comparison = compare_runs(load_run_directory(a), load_run_directory(b))
        jtbd = next(f for f in comparison.fields if f.field == "jtbd")
        macro = next(d for d in jtbd.deltas if d.name == "macro F1")
        assert macro.before < macro.after
        assert macro.change == pytest.approx(macro.after - macro.before)

    def test_items_are_counted_as_improved_regressed_or_unchanged(self, two_runs):
        _, a, b = two_runs(n=400, rates=(0.30, 0.15))
        jtbd = next(
            f
            for f in compare_runs(load_run_directory(a), load_run_directory(b)).fields
            if f.field == "jtbd"
        )
        assert jtbd.compared == 400
        assert jtbd.improved and jtbd.regressed
        # Most items tie in a real A/B, which is the whole reason a net
        # delta is not a result.
        assert jtbd.unchanged > len(jtbd.improved)

    def test_a_prompt_change_is_named_as_a_confound(self, two_runs):
        _, a, b = two_runs(n=200)
        notes = compare_runs(load_run_directory(a), load_run_directory(b)).notes
        assert any("p7 → p8" in note for note in notes)
        assert any("more than one possible cause" in note for note in notes)

    def test_labels_that_moved_are_ranked_by_how_far(self, two_runs):
        _, a, b = two_runs(n=400, rates=(0.4, 0.05))
        jtbd = next(
            f
            for f in compare_runs(load_run_directory(a), load_run_directory(b)).fields
            if f.field == "jtbd"
        )
        moves = [abs(move) for _, move in jtbd.moved_labels]
        assert moves == sorted(moves, reverse=True)


class TestItRefusesToConclude:
    """The one thing M7's compare will not do."""

    def test_it_never_says_which_run_is_better(self, two_runs):
        _, a, b = two_runs(n=400, rates=(0.40, 0.05))
        report = "\n".join(
            comparison_report(compare_runs(load_run_directory(a), load_run_directory(b)))
        )
        lowered = report.lower()
        for claim in ("b is better", "a is better", "significant", "wins"):
            assert claim not in lowered

    def test_it_states_the_net_and_then_withholds_the_verdict(self, two_runs):
        _, a, b = two_runs(n=400, rates=(0.40, 0.05))
        comparison = compare_runs(load_run_directory(a), load_run_directory(b))
        assert comparison.cannot_tell
        said = " ".join(comparison.cannot_tell)
        assert "not a result without a test" in said
        assert "That test is v1" in said

    def test_the_box_is_headed_for_a_comparison_not_a_run(self, two_runs):
        _, a, b = two_runs(n=200)
        report = "\n".join(
            comparison_report(compare_runs(load_run_directory(a), load_run_directory(b)))
        )
        assert "WHAT THIS COMPARISON CANNOT TELL YOU" in report


class TestVersions:
    def test_two_taxonomy_versions_refuse_to_compare(self, two_runs):
        _, a, b = two_runs(n=200, taxonomy_of_b="jtbd@v5")
        with pytest.raises(ComparisonError, match="Without a migration mapping"):
            compare_runs(load_run_directory(a), load_run_directory(b))

    def test_a_migration_makes_them_comparable(self, two_runs, tmp_path):
        root, a, b = two_runs(n=200, taxonomy_of_b="jtbd@v5")
        path = tmp_path / "m.yml"
        path.write_text(MIGRATION, encoding="utf-8")
        comparison = load_and_compare(a, b, migration_path=path)
        assert any("jtbd@v4 → jtbd@v5" in note for note in comparison.notes)
        assert root  # the project the runs came from

    def test_a_migration_between_the_wrong_versions_is_refused(self, two_runs, tmp_path):
        _, a, b = two_runs(n=200, taxonomy_of_b="jtbd@v5")
        path = tmp_path / "m.yml"
        path.write_text(MIGRATION.replace("to: jtbd@v5", "to: jtbd@v9"), encoding="utf-8")
        with pytest.raises(ComparisonError, match="wrong pair"):
            load_and_compare(a, b, migration_path=path)

    def test_the_same_version_with_different_content_is_refused(self, two_runs):
        _, a, b = two_runs(n=200)
        manifest = json.loads((b / "run.json").read_text())
        manifest["taxonomies"] = {"jtbd@v4": "someotherhash"}
        (b / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ComparisonError, match="content differs"):
            compare_runs(load_run_directory(a), load_run_directory(b))

    def test_merged_and_dropped_labels_are_reported(self, two_runs, tmp_path):
        _, a, b = two_runs(n=200, taxonomy_of_b="jtbd@v5")
        path = tmp_path / "m.yml"
        path.write_text(MIGRATION, encoding="utf-8")
        notes = load_and_compare(a, b, migration_path=path).notes
        assert any("no equivalent" in note for note in notes)
        assert any("labels merged" in note for note in notes)


class TestMigrationFiles:
    def test_a_mapping_must_name_its_own_endpoints(self, tmp_path):
        path = tmp_path / "m.yml"
        path.write_text("labels:\n  a: b\n", encoding="utf-8")
        with pytest.raises(TaxonomyError, match="wrong pair|'from' is required"):
            load_migration(path)

    def test_a_rename_a_merge_and_a_drop_all_parse(self, tmp_path):
        path = tmp_path / "m.yml"
        path.write_text(MIGRATION, encoding="utf-8")
        migration = load_migration(path)
        assert migration.translate("billing.payment_failed") == "billing.charge_failed"
        assert migration.translate("product.feature_request") is None
        assert migration.dropped == ("product.feature_request",)
        assert migration.merged["billing.charge_failed"] == (
            "billing.card_declined",
            "billing.payment_failed",
        )

    def test_a_non_string_target_is_refused(self, tmp_path):
        path = tmp_path / "m.yml"
        path.write_text("from: a@v1\nto: a@v2\nlabels:\n  x: 3\n", encoding="utf-8")
        with pytest.raises(TaxonomyError, match="Use a label name, or null"):
            load_migration(path)

    def test_a_missing_file_says_so(self, tmp_path):
        with pytest.raises(TaxonomyError, match="no migration file"):
            load_migration(tmp_path / "nope.yml")


class TestTheCompareCommand:
    def test_it_reads_from_disk_and_makes_no_calls(self, two_runs, capsys):
        from llm_expectations.cli import main

        _, a, b = two_runs(n=200)
        assert main(["compare", str(a), str(b)]) == 0
        printed = capsys.readouterr().out
        assert "Zero model calls" in printed
        assert "items that moved" in printed

    def test_a_version_mismatch_is_a_message_not_a_traceback(self, two_runs, capsys):
        from llm_expectations.cli import main

        _, a, b = two_runs(n=200, taxonomy_of_b="jtbd@v5")
        assert main(["compare", str(a), str(b)]) == 2
        assert "migration mapping" in capsys.readouterr().err

    def test_the_migration_flag_unblocks_it(self, two_runs, tmp_path, capsys):
        from llm_expectations.cli import main

        _, a, b = two_runs(n=200, taxonomy_of_b="jtbd@v5")
        path = tmp_path / "m.yml"
        path.write_text(MIGRATION, encoding="utf-8")
        assert main(["compare", str(a), str(b), "--migration", str(path)]) == 0
        assert "jtbd@v4 → jtbd@v5" in capsys.readouterr().out
