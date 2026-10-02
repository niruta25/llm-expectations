"""The four commands, and the one that must stay free."""

from __future__ import annotations

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


def test_compare_says_when_it_arrives(capsys):
    assert main(["compare"]) == 1
    assert "M7" in capsys.readouterr().out


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
