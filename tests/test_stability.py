"""Does the model give the same answer twice — and which question is that?"""

from __future__ import annotations

import dataclasses
import json

import pytest

from llm_expectations.config import Budget, ConfigError, load_run
from llm_expectations.judges.base import JudgeReply, JudgeRequest
from llm_expectations.run import run
from llm_expectations.stability import measure_stability, regenerate_sample

from .conftest import EXAMPLE, MINIMAL_RUN


class Replayer:
    """A producing model that answers from a script, per call."""

    id = "regenerate"
    model = "claude-sonnet-5"

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls: list[JudgeRequest] = []

    def complete(self, request: JudgeRequest) -> JudgeReply:
        self.calls.append(request)
        text = self.answers[(len(self.calls) - 1) % len(self.answers)]
        return JudgeReply(text=text, model=self.model)


def project(tmp_path, *, temperature=0.0, sample=5, runs=2):
    """A loadable project with a regenerate hook."""
    root = tmp_path / "p"
    root.mkdir(exist_ok=True)
    for name in ("taxonomy.yml", "schema.yml", "judges.yml", "items.jsonl", "outputs.jsonl"):
        (root / name).write_text(
            (EXAMPLE / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
    (root / "outcomes.yml").write_text(
        (EXAMPLE / "outcomes.yml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (root / "prompt.txt").write_text("Classify the session.", encoding="utf-8")
    (root / "run.yml").write_text(
        "run_id: stability\nitems: items.jsonl\noutputs: outputs.jsonl\n"
        "schema: schema.yml\ntaxonomy: [taxonomy.yml, outcomes.yml]\njudges: judges.yml\n"
        "produced_by:\n  model: claude-sonnet-5\n  prompt_version: p8\n"
        "  regenerate:\n    provider: anthropic\n    model: claude-sonnet-5\n"
        "    api_key_env: ANTHROPIC_API_KEY\n    prompt_file: prompt.txt\n"
        f"    temperature: {temperature}\n    sample: {sample}\n    runs: {runs}\n",
        encoding="utf-8",
    )
    return dataclasses.replace(load_run(root / "run.yml"), budget=Budget(None, False))


class TestConfig:
    def test_no_hook_means_no_stability_check(self, tmp_path):
        root = tmp_path / "q"
        root.mkdir()
        for name, body in (
            ("taxonomy.yml", (EXAMPLE / "taxonomy.yml").read_text()),
            ("schema.yml", "item: s\nfields:\n  jtbd: {kind: assigned, taxonomy: 'jtbd@v4'}\n"),
            ("items.jsonl", '{"id": "s-1", "text": "declined"}\n'),
            ("outputs.jsonl", '{"item_id": "s-1", "jtbd": "billing.card_declined"}\n'),
            ("run.yml", MINIMAL_RUN.replace("judges: judges.yml\n", "")),
        ):
            (root / name).write_text(body, encoding="utf-8")
        config = load_run(root / "run.yml")
        assert config.produced_by is None or config.produced_by.regenerate is None
        assert measure_stability(config, {}, []) is None

    def test_one_run_cannot_disagree_with_itself(self, tmp_path):
        with pytest.raises(ConfigError, match="at least 2"):
            project(tmp_path, runs=1)

    def test_a_sample_is_required_because_this_is_a_diagnostic(self, tmp_path):
        with pytest.raises(ConfigError, match="never a re-run"):
            project(tmp_path, sample=0)

    def test_the_prompt_file_cannot_be_guessed(self, tmp_path):
        config = project(tmp_path)
        text = (config.source).read_text().replace("prompt_file: prompt.txt\n", "")
        config.source.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigError, match="prompt_file"):
            load_run(config.source)

    def test_temperature_decides_which_number_you_get(self, tmp_path):
        assert project(tmp_path, temperature=0.0).produced_by.regenerate.measures == "serving"
        assert project(tmp_path, temperature=0.7).produced_by.regenerate.measures == "decision"


class TestMeasuring:
    def _rows(self, config, answers):
        provider = Replayer(answers)
        from llm_expectations.run import load_dataset

        dataset = load_dataset(config)
        rows, unreadable = regenerate_sample(
            config, dataset.items, provider, fields=tuple(config.schema.fields)
        )
        return dataset, rows, unreadable, provider

    def test_a_deterministic_model_is_fully_stable(self, tmp_path):
        config = project(tmp_path, temperature=0.0)
        from llm_expectations.run import load_dataset

        dataset = load_dataset(config)
        # Always echo back the output already on disk.
        answers = [
            json.dumps({"jtbd": o.get("jtbd"), "outcome": o.get("outcome")})
            for o in dataset.outputs.values()
        ]
        rows, _ = regenerate_sample(
            config, dataset.items, Replayer(answers), fields=tuple(config.schema.fields)
        )
        report = measure_stability(config, dataset.outputs, rows)
        assert report.kind == "serving"
        assert report.reads_as_broken is None or "0%" in report.reads_as_broken

    def test_a_flipping_model_is_reported_as_unstable(self, tmp_path):
        config = project(tmp_path, temperature=0.0)
        dataset, rows, _, _ = self._rows(
            config,
            [
                json.dumps({"jtbd": "billing.card_declined", "outcome": "resolved"}),
                json.dumps({"jtbd": "billing.payment_failed", "outcome": "pending"}),
            ],
        )
        report = measure_stability(config, dataset.outputs, rows)
        jtbd = next(f for f in report.fields if f.field == "jtbd")
        assert jtbd.stability < 1.0
        assert jtbd.flipped

    def test_at_temperature_zero_instability_points_at_the_serving_stack(self, tmp_path):
        config = project(tmp_path, temperature=0.0)
        dataset, rows, _, _ = self._rows(
            config,
            [
                json.dumps({"jtbd": "billing.card_declined"}),
                json.dumps({"jtbd": "billing.payment_failed"}),
            ],
        )
        said = measure_stability(config, dataset.outputs, rows).reads_as_broken
        assert "nondeterministic" in said

    def test_above_zero_instability_points_at_the_boundary(self, tmp_path):
        config = project(tmp_path, temperature=0.7)
        dataset, rows, _, _ = self._rows(
            config,
            [
                json.dumps({"jtbd": "billing.card_declined"}),
                json.dumps({"jtbd": "billing.payment_failed"}),
            ],
        )
        said = measure_stability(config, dataset.outputs, rows).reads_as_broken
        assert "how fragile the label is" in said
        assert "nondeterministic" not in said

    def test_it_never_looks_past_the_sample(self, tmp_path):
        config = project(tmp_path, sample=3, runs=2)
        _, _, _, provider = self._rows(config, ['{"jtbd": "billing.card_declined"}'])
        # Three items, twice. Never the corpus.
        assert len(provider.calls) == 6

    def test_an_unreadable_reply_is_counted_not_defaulted(self, tmp_path):
        config = project(tmp_path)
        dataset, rows, unreadable, _ = self._rows(config, ["not json at all"])
        assert rows == []
        assert unreadable > 0
        report = measure_stability(config, dataset.outputs, rows, unreadable=unreadable)
        assert "would invent stability" in report.note


class TestEndToEnd:
    def test_the_regenerated_answers_go_to_their_own_file(self, tmp_path):
        from llm_expectations.judges.fake import FakeProvider, reply

        config = project(tmp_path, sample=4)
        before = (config.root / "outputs.jsonl").read_text()
        result = run(
            config,
            out=tmp_path / "out",
            provider_factory=lambda spec: (
                Replayer(['{"jtbd": "billing.card_declined", "outcome": "resolved"}'])
                if spec.id == "regenerate"
                else FakeProvider(model="m", default=reply(True, 0.9, "ok"))
            ),
        )
        assert (result.directory / "regenerated.jsonl").exists()
        # The outputs under test are the one thing that must survive the test.
        assert (config.root / "outputs.jsonl").read_text() == before

    def test_the_report_names_which_stability_it_measured(self, tmp_path):
        from llm_expectations.judges.fake import FakeProvider, reply

        config = project(tmp_path, temperature=0.0, sample=4)
        report = run(
            config,
            out=tmp_path / "out",
            provider_factory=lambda spec: (
                Replayer(['{"jtbd": "billing.card_declined", "outcome": "resolved"}'])
                if spec.id == "regenerate"
                else FakeProvider(model="m", default=reply(True, 0.9, "ok"))
            ),
        ).report
        assert "SERVING STABILITY" in report
        assert "DECISION STABILITY" not in report


def test_a_stability_of_zero_is_the_worst_result_not_a_missing_one():
    """`value or default` is wrong on a float, and 0.0 is the case that bites."""
    from llm_expectations.stability import FieldStability, StabilityReport

    report = StabilityReport(
        kind="serving",
        temperature=0.0,
        runs=2,
        sampled=10,
        fields=(FieldStability(field="jtbd", compared=10, agreed=0),),
    )
    assert report.fields[0].stability == 0.0
    assert report.reads_as_broken is not None
    assert "100%" in report.reads_as_broken
