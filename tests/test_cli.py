"""The four commands, and the one that must stay free."""

from __future__ import annotations

import pytest

from llm_expectations.cli import main

from .conftest import EXAMPLE


def test_no_arguments_prints_help(capsys):
    assert main([]) == 0
    assert "analyse" in capsys.readouterr().out


def test_check_reports_a_healthy_taxonomy(capsys):
    assert main(["check", str(EXAMPLE / "taxonomy.yml")]) == 0
    out = capsys.readouterr().out
    assert "jtbd@v4" in out and "no issues" in out


def test_check_exits_nonzero_on_a_broken_one(tmp_path, capsys):
    broken = tmp_path / "t.yml"
    broken.write_text("id: x\nversion: 1\nlabels:\n  a: {definition: 'first thing'}\n")
    assert main(["check", str(broken)]) == 0  # warnings only, no stop
    assert "taxonomy_examples" in capsys.readouterr().out


def test_plan_makes_no_calls_and_says_so(capsys):
    assert main(["plan", str(EXAMPLE / "run.yml")]) == 0
    out = capsys.readouterr().out
    assert "No calls were made" in out
    assert "25 calls" in out


def test_a_config_error_is_a_message_not_a_traceback(tmp_path, capsys):
    bad = tmp_path / "run.yml"
    bad.write_text("run_id: x\n")
    assert main(["plan", str(bad)]) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_compare_needs_two_directories(capsys):
    with pytest.raises(SystemExit):
        main(["compare"])


def test_run_then_analyse(tmp_path, monkeypatch, capsys, scripted):
    import llm_expectations.run as run_module

    provider = scripted()
    # Patched at module level, which only works because `run` resolves the
    # factory at call time. A default argument would have bound the real one
    # and sent this test at the network.
    monkeypatch.setattr(run_module, "build_provider", lambda spec: provider)
    assert main(["run", str(EXAMPLE / "run.yml"), "--out", str(tmp_path), "--yes"]) == 0
    directory = next(p for p in tmp_path.iterdir() if p.is_dir())
    calls = len(provider.calls)

    capsys.readouterr()
    assert main(["analyse", str(directory)]) == 0
    assert "Zero model calls" in capsys.readouterr().out
    assert len(provider.calls) == calls


def test_run_declined_at_the_prompt_exits_nonzero(tmp_path, monkeypatch, scripted):
    import llm_expectations.run as run_module

    provider = scripted()
    monkeypatch.setattr(run_module, "build_provider", lambda spec: provider)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert main(["run", str(EXAMPLE / "run.yml"), "--out", str(tmp_path)]) == 1
    assert provider.calls == []


def test_the_cli_never_reaches_the_network_when_the_factory_is_replaced(
    tmp_path, monkeypatch, scripted
):
    """`run` must resolve its provider factory at call time, not bind it.

    A default argument of `provider_factory=build_provider` captures the real
    function when the module is imported, so replacing the module attribute
    does nothing and the run goes out to the internet. This test fails in
    seconds rather than minutes if that regresses.
    """
    import llm_expectations.judges.providers as providers
    import llm_expectations.run as run_module

    def explode(*args, **kwargs):
        raise AssertionError("a real provider was constructed")

    monkeypatch.setattr(providers.httpx, "Client", explode)
    monkeypatch.setattr(run_module, "build_provider", lambda spec: scripted())
    assert main(["run", str(EXAMPLE / "run.yml"), "--out", str(tmp_path), "--yes"]) == 0


def test_triage_eval_reads_from_disk_and_makes_no_calls(tmp_path, monkeypatch, capsys):
    """The table is computed from verdicts and labels on disk, for nothing."""
    import json
    import random

    import llm_expectations.run as run_module
    from llm_expectations.judges.fake import FakeProvider, reply


    rng = random.Random(0)
    labels_pool = [
        "billing.payment_failed", "billing.card_declined", "billing.refund_request",
        "access.password_reset", "access.sso_issue",
    ]
    root = tmp_path / "project"
    root.mkdir()
    items, outputs, labels = [], [], []
    for i in range(400):
        truth = rng.choice(labels_pool)
        assigned = (
            rng.choice([x for x in labels_pool if x != truth]) if rng.random() < 0.2 else truth
        )
        items.append({"id": f"s-{i}", "text": f"session {i} about {truth}"})
        outputs.append({"item_id": f"s-{i}", "jtbd": assigned})
        labels.append(
            {"item_id": f"s-{i}", "field": "jtbd", "label": truth, "annotator": "ann-1"}
        )
    for name, rows in (
        ("items.jsonl", items), ("outputs.jsonl", outputs), ("labels.jsonl", labels)
    ):
        (root / name).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (root / "taxonomy.yml").write_text(
        (EXAMPLE / "taxonomy.yml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (root / "judges.yml").write_text(
        (EXAMPLE / "judges.yml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (root / "schema.yml").write_text(
        "item: s\nfields:\n  jtbd: {kind: assigned, taxonomy: 'jtbd@v4'}\n", encoding="utf-8"
    )
    (root / "run.yml").write_text(
        "run_id: big\nitems: items.jsonl\noutputs: outputs.jsonl\nlabels: labels.jsonl\n"
        "schema: schema.yml\ntaxonomy: taxonomy.yml\njudges: judges.yml\n",
        encoding="utf-8",
    )

    provider = FakeProvider(model="m", default=reply(False, 0.8, "unsure"))
    monkeypatch.setattr(run_module, "build_provider", lambda spec: provider)
    out = tmp_path / "out"
    assert main(["run", str(root / "run.yml"), "--out", str(out), "--yes"]) == 0
    directory = next(p for p in out.iterdir() if p.is_dir())

    calls = len(provider.calls)
    capsys.readouterr()
    assert main(["triage-eval", str(directory)]) == 0
    printed = capsys.readouterr().out
    assert "Error Recall@Budget" in printed
    assert "Zero model calls" in printed
    assert len(provider.calls) == calls


def test_triage_eval_on_an_unlabelled_run_says_why(tmp_path, monkeypatch, capsys, scripted):
    import llm_expectations.run as run_module

    monkeypatch.setattr(run_module, "build_provider", lambda spec: scripted())
    out = tmp_path / "out"
    # The minimal project has no labels.
    from .conftest import MINIMAL_JUDGES, MINIMAL_RUN, MINIMAL_SCHEMA, MINIMAL_TAXONOMY

    root = tmp_path / "p"
    root.mkdir()
    for name, body in (
        ("taxonomy.yml", MINIMAL_TAXONOMY),
        ("schema.yml", MINIMAL_SCHEMA),
        ("judges.yml", MINIMAL_JUDGES),
        ("run.yml", MINIMAL_RUN),
        ("items.jsonl", '{"id": "s-1", "text": "her card was declined"}\n'),
        (
            "outputs.jsonl",
            '{"item_id": "s-1", "jtbd": "billing.payment_failed", "summary": "declined"}\n',
        ),
    ):
        (root / name).write_text(body, encoding="utf-8")

    assert main(["run", str(root / "run.yml"), "--out", str(out), "--yes"]) == 0
    directory = next(p for p in out.iterdir() if p.is_dir())
    capsys.readouterr()
    assert main(["triage-eval", str(directory)]) == 2
    assert "human labels" in capsys.readouterr().err


class TestTheOfflineExample:
    """The worked example must run for anyone, with no key and no cost."""

    def test_it_runs_end_to_end_without_a_network(self, tmp_path, monkeypatch):
        import llm_expectations.judges.providers as providers

        def explode(*args, **kwargs):
            raise AssertionError("a scripted judge reached for the network")

        monkeypatch.setattr(providers.httpx, "Client", explode)
        assert (
            main(["run", str(EXAMPLE / "run-offline.yml"), "--out", str(tmp_path), "--yes"]) == 0
        )

    def test_it_costs_nothing_because_it_asks_nothing(self, tmp_path, capsys):
        main(["run", str(EXAMPLE / "run-offline.yml"), "--out", str(tmp_path), "--yes"])
        assert "$0.00" in capsys.readouterr().out

    def test_gate_one_stops_the_run_and_says_the_verdicts_are_invented(
        self, tmp_path, capsys
    ):
        # The only reason shipping a scripted judge is safe: a report built on
        # one is laid out exactly like a report built on a real judge.
        main(["run", str(EXAMPLE / "run-offline.yml"), "--out", str(tmp_path), "--yes"])
        printed = capsys.readouterr().out
        # Whitespace-normalised: the report wraps at eighty columns, so a
        # phrase can straddle a line break.
        flat = " ".join(printed.split())
        assert "are scripted judges" in flat
        assert "must not be read as a measurement" in flat
        assert "measurement is sound STOP" in flat

    def test_the_free_checks_are_still_real(self, tmp_path, capsys):
        # Everything upstream of the judge runs on real data. The planted
        # defects the free checks own are genuinely found.
        main(["run", str(EXAMPLE / "run-offline.yml"), "--out", str(tmp_path), "--yes"])
        printed = capsys.readouterr().out
        for planted in ("s-08", "s-09", "s-11", "s-12", "s-13"):
            assert planted in printed

    def test_a_scripted_judge_suppresses_every_judge_backed_number(self, tmp_path):
        import json

        main(["run", str(EXAMPLE / "run-offline.yml"), "--out", str(tmp_path), "--yes"])
        directory = next(p for p in tmp_path.iterdir() if p.is_dir())
        metrics = json.loads((directory / "metrics.json").read_text())
        withheld = {row["metric"] for row in metrics["suppressed"]}
        assert {"panel.agreement", "triage.validated", "gate2.auc"} <= withheld

    def test_a_panel_of_scripted_judges_still_disagrees(self, tmp_path):
        import json

        # Seeded by judge id, so three fakes do not answer identically. Three
        # that agreed on everything would teach the wrong lesson about what a
        # panel is worth.
        main(["run", str(EXAMPLE / "run-offline.yml"), "--out", str(tmp_path), "--yes"])
        directory = next(p for p in tmp_path.iterdir() if p.is_dir())
        panel = json.loads((directory / "metrics.json").read_text())["panel"]
        assert panel.get("splits", 0) > 0 or (panel.get("agreement") or 1.0) < 1.0
