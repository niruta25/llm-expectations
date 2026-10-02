"""The two gates: what they refuse, what they withhold, and what they prove."""

from __future__ import annotations

import dataclasses

import pytest

from llm_expectations.config import Budget, load_run
from llm_expectations.gates import Gate, GateResult, Gates, build_target
from llm_expectations.run import run
from llm_expectations.types import Label, Output, Severity


@pytest.fixture
def big(big_corpus, oracle_judge):
    """A corpus Gate 2 can actually run on, plus a judge of tunable quality."""

    def build(*, catches=0.85, false_alarms=0.03, n=400, error_rate=0.2, **kw):
        root, truth = big_corpus(n=n, error_rate=error_rate)
        config = dataclasses.replace(
            load_run(root / "run.yml"), budget=Budget(None, False)
        )
        provider = oracle_judge(truth, catches=catches, false_alarms=false_alarms, **kw)
        return config, provider, root

    return build


class TestTarget:
    def test_an_item_is_an_error_if_any_labelled_field_is_wrong(self):
        outputs = {
            "s-1": Output("s-1", {"jtbd": "a", "outcome": "x"}),
            "s-2": Output("s-2", {"jtbd": "a", "outcome": "x"}),
        }
        labels = [
            Label("s-1", "jtbd", "a", "ann-1"),
            Label("s-1", "outcome", "y", "ann-1"),  # wrong
            Label("s-2", "jtbd", "a", "ann-1"),
            Label("s-2", "outcome", "x", "ann-1"),
        ]
        target = build_target(labels, outputs, ("jtbd", "outcome"))
        assert target.truth == {"s-1": True, "s-2": False}

    def test_only_the_first_annotator_defines_the_target(self):
        # Pooling two who disagree would make the target depend on which was
        # read last. Adjudicating them is mode 2's business.
        outputs = {"s-1": Output("s-1", {"jtbd": "a"})}
        labels = [Label("s-1", "jtbd", "b", "ann-2"), Label("s-1", "jtbd", "a", "ann-1")]
        assert build_target(labels, outputs, ("jtbd",)).truth == {"s-1": False}

    def test_fields_outside_the_schema_are_ignored(self):
        outputs = {"s-1": Output("s-1", {"jtbd": "a"})}
        labels = [Label("s-1", "notafield", "z", "ann-1")]
        assert build_target(labels, outputs, ("jtbd",)).truth == {}


class TestGateOne:
    def test_the_example_passes_when_no_judge_is_broken(self, example, provider, tmp_path):
        result = run(
            example, out=tmp_path / "out", provider_factory=lambda spec: provider
        )
        assert result.metrics["gates"]["measurement_sound"]["status"] == "PASS"

    def test_a_rubber_stamp_stops_the_gate_and_withholds_the_panel_numbers(
        self, example, tmp_path, scripted
    ):
        from llm_expectations.judges.fake import reply

        always = scripted(default=reply(True, 0.95, "fine with everything"))
        result = run(example, out=tmp_path / "out", provider_factory=lambda spec: always)
        gate = result.metrics["gates"]["measurement_sound"]
        assert gate["status"] == "STOP"
        withheld = {s["metric"] for s in result.metrics["suppressed"]}
        assert {"panel.agreement", "panel.effective_votes"} <= withheld

    def test_a_small_corpus_is_a_warning_not_a_stop(self, example, provider, tmp_path):
        # Thirteen items cannot support a distribution claim. That is worth
        # saying; it is not a reason to withhold the whole run.
        result = run(
            example, out=tmp_path / "out", provider_factory=lambda spec: provider
        )
        results = result.metrics["gates"]["measurement_sound"]["results"]
        sample = next(r for r in results if r["id"] == "sample_size")
        assert sample["severity"] == "warn"

    def test_the_label_leak_guarantee_is_stated_in_the_gate(
        self, example, provider, tmp_path
    ):
        result = run(
            example, out=tmp_path / "out", provider_factory=lambda spec: provider
        )
        results = result.metrics["gates"]["measurement_sound"]["results"]
        assert any(r["id"] == "no_label_leakage" for r in results)


class TestGateTwo:
    def test_with_no_labels_it_does_not_run_and_says_why(self, project, scripted, no_confirm,
                                                          tmp_path):
        config = no_confirm(load_run(project()))
        result = run(config, out=tmp_path / "out", provider_factory=lambda spec: scripted())
        gate = result.metrics["gates"]["beats_baselines"]
        assert gate["status"] == "not run"
        assert "no human labels" in gate["skipped"]
        # The ranking is still produced; it just has not been validated.
        assert result.risk_rows

    def test_a_degenerate_target_withholds_the_auc_rather_than_printing_it(
        self, example, provider, tmp_path
    ):
        result = run(
            example, out=tmp_path / "out", provider_factory=lambda spec: provider
        )
        gate = result.metrics["gates"]["beats_baselines"]
        assert gate["status"] == "STOP"
        withheld = {s["metric"] for s in result.metrics["suppressed"]}
        assert "gate2.auc" in withheld
        assert gate["table"] == []

    def test_a_good_judge_clears_every_baseline(self, big, tmp_path):
        config, provider, _ = big(catches=0.85, false_alarms=0.03)
        result = run(config, out=tmp_path / "out", provider_factory=lambda spec: provider)
        gate = result.metrics["gates"]["beats_baselines"]
        assert gate["status"] == "PASS"
        table = {row["strategy"]: row for row in gate["table"]}
        assert table["raw_confidence"]["auc"] > 0.8
        assert all(
            table["raw_confidence"]["ci_low"] > table[name]["auc"]
            for name in table
            if table[name]["baseline"] and table[name]["auc"] is not None
        )

    def test_every_baseline_is_scored_beside_the_judge_always(self, big, tmp_path):
        config, provider, _ = big()
        result = run(config, out=tmp_path / "out", provider_factory=lambda spec: provider)
        table = {row["strategy"] for row in result.metrics["gates"]["beats_baselines"]["table"]}
        assert {"random", "output_length", "majority_label"} <= table

    def test_a_judge_no_better_than_chance_fails_the_gate(self, big, tmp_path):
        # The honest outcome: "a judge is not buying you anything here" is
        # worth more than a dashboard.
        config, provider, _ = big(catches=0.2, false_alarms=0.2)
        result = run(config, out=tmp_path / "out", provider_factory=lambda spec: provider)
        gate = result.metrics["gates"]["beats_baselines"]
        assert gate["status"] == "STOP"
        assert any("not buying you anything" in r["message"] for r in gate["results"])

    def test_random_lands_near_chance(self, big, tmp_path):
        config, provider, _ = big()
        result = run(config, out=tmp_path / "out", provider_factory=lambda spec: provider)
        table = {r["strategy"]: r for r in result.metrics["gates"]["beats_baselines"]["table"]}
        assert 0.4 < table["random"]["auc"] < 0.6

    def test_panel_disagreement_is_scored_as_a_baseline_not_used_as_a_ranker(
        self, big, tmp_path
    ):
        config, provider, _ = big()
        result = run(config, out=tmp_path / "out", provider_factory=lambda spec: provider)
        gate = result.metrics["gates"]["beats_baselines"]
        names = {r["strategy"]: r for r in gate["table"]}
        assert names["panel_disagreement"]["baseline"] is True
        assert result.metrics["strategy"] == "raw_confidence"

    def test_the_table_carries_an_interval_for_every_row(self, big, tmp_path):
        config, provider, _ = big()
        result = run(config, out=tmp_path / "out", provider_factory=lambda spec: provider)
        for row in result.metrics["gates"]["beats_baselines"]["table"]:
            assert row["ci_low"] is not None and row["ci_high"] is not None

    def test_unranked_errors_are_flagged_because_excluding_them_flatters(
        self, big, tmp_path, big_corpus, oracle_judge
    ):
        from llm_expectations.judges.fake import FakeProvider, reply

        root, truth = big_corpus(n=400, error_rate=0.2)
        config = dataclasses.replace(load_run(root / "run.yml"), budget=Budget(None, False))

        # A judge that abstains on everything it would have flagged: the
        # unjudged pile is then full of errors and the ranked rest looks
        # spotless.
        def answer(request):
            index = request.user.split("session ")[1].split(" ")[0]
            assigned, real = truth[f"s-{index}"]
            if assigned != real:
                return reply("cannot_decide", 0.5, "not sure")
            return reply(True, 0.9, "fine")

        result = run(
            config,
            out=tmp_path / "out",
            provider_factory=lambda spec: FakeProvider(model="m", default=answer),
        )
        gate = result.metrics["gates"]["beats_baselines"]
        assert any(r["id"] == "unranked_errors" for r in gate["results"])


class TestSuppression:
    def test_a_stop_withholds_its_metrics_and_a_warn_does_not(self):
        stop = Gate(
            "g",
            (
                GateResult("a", Severity.STOP, "broken", ("x.y",)),
                GateResult("b", Severity.WARN, "iffy", ("z.w",)),
            ),
        )
        assert [s.metric for s in stop.suppressions()] == ["x.y"]

    def test_two_guardrails_hitting_one_metric_are_merged(self):
        # Listing it twice makes the report look like two problems.
        gates = Gates(
            one=Gate(
                "g",
                (
                    GateResult("a", Severity.STOP, "first", ("panel.agreement",)),
                    GateResult("b", Severity.STOP, "second", ("panel.agreement",)),
                ),
            ),
            two=Gate("h"),
        )
        withheld = gates.suppressed()
        assert len(withheld) == 1
        assert "2 guardrails withheld this" in withheld[0].reason

    def test_a_gate_that_did_not_run_has_not_passed(self):
        assert not Gate("g", skipped="no labels").passed
        assert Gate("g", skipped="no labels").status == "not run"

    def test_a_note_never_blocks(self):
        gate = Gate("g", (GateResult("n", Severity.NOTE, "fyi", ("x",)),))
        assert gate.passed
        assert gate.suppressions() == ()

    def test_lookup_returns_the_reason(self):
        gates = Gates(
            one=Gate("g", (GateResult("a", Severity.STOP, "because", ("m",)),)),
            two=Gate("h"),
        )
        assert gates.is_suppressed("m") == "because"
        assert gates.is_suppressed("other") is None


class TestGatesInTheReport:
    def test_both_gates_print_at_the_top_with_their_status(self, big, tmp_path):
        config, provider, _ = big()
        report = run(
            config, out=tmp_path / "out", provider_factory=lambda spec: provider
        ).report
        head = report[: report.index("MODE")]
        assert "measurement is sound" in head and "judge beats baselines" in head

    def test_the_comparison_table_prints_every_baseline(self, big, tmp_path):
        config, provider, _ = big()
        report = run(
            config, out=tmp_path / "out", provider_factory=lambda spec: provider
        ).report
        for name in ("raw_confidence", "random", "output_length", "majority_label"):
            assert name in report
        assert "← the judge" in report and "← baseline" in report

    def test_nothing_wraps_past_eighty_columns(self, big, tmp_path):
        config, provider, _ = big()
        report = run(
            config, out=tmp_path / "out", provider_factory=lambda spec: provider
        ).report
        assert [line for line in report.splitlines() if len(line) > 80] == []


class TestAgainstHumans:
    """What mode 1 and mode 2 unlock, end to end."""

    def test_the_classification_block_appears_once_labels_exist(self, big, tmp_path):
        config, provider, _ = big()
        result = run(config, out=tmp_path / "out", provider_factory=lambda spec: provider)
        scored = result.metrics["vs_humans"]["fields"]["jtbd"]
        assert scored["macro_f1"] is not None
        assert scored["accuracy"] is not None
        assert scored["majority_baseline"] is not None

    def test_accuracy_never_appears_without_its_baseline(self, big, tmp_path):
        config, provider, _ = big()
        report = run(
            config, out=tmp_path / "out", provider_factory=lambda spec: provider
        ).report
        for line in report.splitlines():
            if line.strip().startswith("accuracy"):
                assert "majority-label baseline" in line

    def test_the_tree_buckets_add_up_to_the_labelled_corpus(self, big, tmp_path):
        config, provider, _ = big(n=400)
        result = run(config, out=tmp_path / "out", provider_factory=lambda spec: provider)
        scored = result.metrics["vs_humans"]["fields"]["jtbd"]
        assert sum(scored["buckets"].values()) == scored["n"] == 400

    def test_a_sibling_heavy_corpus_surfaces_that_boundary(self, big_corpus, oracle_judge,
                                                            tmp_path):
        import dataclasses

        root, truth = big_corpus(n=600, error_rate=0.3, sibling_bias=0.9)
        config = dataclasses.replace(load_run(root / "run.yml"), budget=Budget(None, False))
        result = run(
            config,
            out=tmp_path / "out",
            provider_factory=lambda spec: oracle_judge(truth),
        )
        pairs = result.metrics["vs_humans"]["fields"]["jtbd"]["confusion_direction"]
        top = pairs[0]
        assert set(top["pair"]) == {"billing.payment_failed", "billing.card_declined"}
        assert top["shape"] == "symmetric"
        assert "taxonomy" in top["verdict"]

    def test_two_annotators_unlock_the_strongest_evidence(self, big_corpus, oracle_judge,
                                                           tmp_path):
        import dataclasses

        root, truth = big_corpus(n=400, error_rate=0.2, second_annotator=200, sibling_bias=0.9)
        config = dataclasses.replace(load_run(root / "run.yml"), budget=Budget(None, False))
        result = run(
            config, out=tmp_path / "out", provider_factory=lambda spec: oracle_judge(truth)
        )
        humans = result.metrics["vs_humans"]["fields"]["jtbd"]["annotators"]
        assert humans["compared"] == 200
        assert humans["agreement"] < 1.0
        top = humans["pairs"][0]
        assert {top["left"], top["right"]} == {
            "billing.payment_failed",
            "billing.card_declined",
        }

    def test_a_judge_that_cannot_beat_approving_everything_is_named(
        self, big_corpus, oracle_judge, tmp_path
    ):
        import dataclasses

        root, truth = big_corpus(n=400, error_rate=0.1)
        config = dataclasses.replace(load_run(root / "run.yml"), budget=Budget(None, False))
        # Catches almost nothing and false-alarms often: worse than silence.
        provider = oracle_judge(truth, catches=0.1, false_alarms=0.3)
        result = run(config, out=tmp_path / "out", provider_factory=lambda spec: provider)
        judges = {row["judge"]: row for row in result.metrics["vs_humans"]["judges"]}
        assert judges["judge-a"]["beats_always_approve"] is False
        assert "does not beat approving everything" in result.report

    def test_judge_direction_is_reported_per_judge(self, big, tmp_path):
        config, provider, _ = big()
        result = run(config, out=tmp_path / "out", provider_factory=lambda spec: provider)
        for row in result.metrics["vs_humans"]["judges"]:
            assert row["approves_wrong"] is not None
            assert row["rejects_right"] is not None
            assert row["leaning"] in {"lenient", "strict", "balanced", "unknown"}

    def test_a_field_with_no_labels_gets_no_classification_block(
        self, example, provider, tmp_path
    ):
        result = run(
            example, out=tmp_path / "out", provider_factory=lambda spec: provider
        )
        fields = result.metrics["vs_humans"]["fields"]
        assert "summary" not in fields  # free text, and unlabelled
        assert "jtbd" in fields

    def test_labels_still_never_reach_a_judge(self, big, tmp_path):
        # M5 is the milestone where labels finally get used. They are used
        # downstream of every call, and the prompts must still be clean.
        config, provider, _ = big()
        run(config, out=tmp_path / "out", provider_factory=lambda spec: provider)
        for request in provider.calls:
            assert "ann-1" not in request.user and "ann-1" not in request.system
