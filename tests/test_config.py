"""The three-layer merge, and everything that must validate before spending."""

from __future__ import annotations

import pytest

from llm_expectations.config import (
    ConfigError,
    judges_from_mapping,
    load_run,
    settings_from_mapping,
)
from llm_expectations.schema import schema_from_mapping

from .conftest import EXAMPLE, MINIMAL_JUDGES, MINIMAL_RUN, MINIMAL_TAXONOMY


class TestSettingsLayering:
    def setup_method(self):
        self.schema = schema_from_mapping(
            {
                "item": "s",
                "fields": {
                    "jtbd": {"kind": "assigned", "taxonomy": "j@v1", "max_label_share": 0.3}
                },
            }
        )
        self.field = self.schema["jtbd"]

    def test_the_field_block_wins_and_says_so(self):
        settings = settings_from_mapping({"max_label_share": 0.4})
        resolved = settings.resolve("max_label_share", self.field)
        assert (resolved.value, resolved.source) == (0.3, "schema.yml:jtbd")

    def test_settings_yml_wins_over_the_default_and_says_so(self):
        settings = settings_from_mapping({"max_label_share": 0.4})
        assert settings.resolve("max_label_share").value == 0.4
        assert settings.resolve("max_label_share").source == "settings.yml"

    def test_untouched_settings_fall_through_to_the_default(self):
        resolved = settings_from_mapping({}).resolve("abstain_rate", self.field)
        assert resolved.value == (0.01, 0.20)
        assert resolved.source == "default"

    def test_a_field_scoped_setting_may_be_set_project_wide(self):
        # That is the middle layer's whole job.
        assert settings_from_mapping({"min_claim_support": 0.99}).value("min_claim_support") == 0.99

    def test_an_unknown_setting_is_an_error(self):
        with pytest.raises(ConfigError, match="unknown setting"):
            settings_from_mapping({"bootstrap_resample": 500})

    def test_a_mistyped_value_is_caught_here_too(self):
        with pytest.raises(ConfigError, match="outside the permitted range"):
            settings_from_mapping({"parse_failure_warn": 2})

    def test_the_label_leak_assertion_is_not_a_setting(self):
        # Everything is tunable except this. It is not a threshold, it is a bug
        # check, and offering a knob for it would be offering a way to turn the
        # bug check off.
        with pytest.raises(ConfigError, match="not a threshold, it is a bug check"):
            settings_from_mapping({"label_leak_assertion": False})


class TestJudgeWiring:
    def test_the_example_wires_two_jobs(self):
        judges = judges_from_mapping(
            {
                "judges": [
                    {"id": "a", "provider": "anthropic", "model": "m", "api_key_env": "K"},
                    {"id": "b", "provider": "openai", "model": "m", "api_key_env": "K"},
                ],
                "panel": {"members": ["a", "b"], "sample": 50},
                "triage": {"judge": "a", "budgets": [0.05, 0.01, 0.01]},
            }
        )
        assert judges.panel.sample == 50
        assert judges.triage.budgets == (0.01, 0.05)  # sorted and deduped
        assert judges.triage.strategy == "auto"

    def test_an_api_key_pasted_where_its_name_belongs_is_refused(self):
        # This file gets committed. The key never should.
        with pytest.raises(ConfigError, match="takes the NAME"):
            judges_from_mapping(
                {"judges": [{"id": "a", "provider": "anthropic", "model": "m",
                             "api_key_env": "sk-ant-api03-abcdef"}]}
            )

    def test_a_local_endpoint_is_required_for_an_openai_compatible_judge(self):
        with pytest.raises(ConfigError, match="needs an 'endpoint'"):
            judges_from_mapping(
                {"judges": [{"id": "a", "provider": "openai_compatible", "model": "m"}]}
            )

    def test_two_judges_cannot_share_an_id(self):
        with pytest.raises(ConfigError, match="share the id"):
            judges_from_mapping(
                {"judges": [{"id": "a", "provider": "openai", "model": "m"}] * 2}
            )

    def test_a_job_can_only_name_a_declared_judge(self):
        declared = {"judges": [{"id": "a", "provider": "openai", "model": "m"}]}
        with pytest.raises(ConfigError, match="not a declared judge"):
            judges_from_mapping({**declared, "triage": {"judge": "z"}})
        with pytest.raises(ConfigError, match="not a declared judge"):
            judges_from_mapping({**declared, "panel": {"members": ["a", "z"]}})

    def test_a_panel_of_one_is_refused_because_every_panel_number_is_undefined(self):
        with pytest.raises(ConfigError, match="panel of one"):
            judges_from_mapping(
                {
                    "judges": [{"id": "a", "provider": "openai", "model": "m"}],
                    "panel": {"members": ["a"]},
                }
            )

    def test_an_unknown_triage_strategy_is_caught_before_any_calls(self):
        with pytest.raises(ConfigError, match="unknown triage strategy"):
            judges_from_mapping(
                {
                    "judges": [{"id": "a", "provider": "openai", "model": "m"}],
                    "triage": {"judge": "a", "strategy": "confidence"},
                }
            )

    def test_ranking_by_disagreement_needs_something_to_disagree(self):
        with pytest.raises(ConfigError, match="needs a panel"):
            judges_from_mapping(
                {
                    "judges": [{"id": "a", "provider": "openai", "model": "m"}],
                    "triage": {"judge": "a", "strategy": "panel_disagreement"},
                }
            )

    def test_a_budget_outside_zero_to_one_is_not_a_review_budget(self):
        with pytest.raises(ConfigError, match="share of the corpus"):
            judges_from_mapping(
                {
                    "judges": [{"id": "a", "provider": "openai", "model": "m"}],
                    "triage": {"judge": "a", "budgets": [5]},
                }
            )


class TestLoadRun:
    def test_paths_resolve_against_the_run_file_not_the_working_directory(self, project, tmp_path):
        config = load_run(project())
        assert config.items == (tmp_path / "items.jsonl").resolve()
        assert config.labels is None

    def test_a_schema_pin_must_match_the_file_it_points_at(self, project):
        bumped = MINIMAL_TAXONOMY.replace("version: 4", "version: 5")
        with pytest.raises(ConfigError, match="this file is version 5"):
            load_run(project(**{"taxonomy.yml": bumped}))

    def test_a_pin_with_no_file_behind_it_is_refused(self, project):
        schema = "item: s\nfields:\n  jtbd: {kind: assigned, taxonomy: other@v1}\n"
        with pytest.raises(ConfigError, match="no taxonomy file declares id 'other'"):
            load_run(project(**{"schema.yml": schema}))

    def test_two_files_declaring_one_id_is_refused(self, project):
        run = MINIMAL_RUN.replace(
            "taxonomy: taxonomy.yml", "taxonomy: [taxonomy.yml, copy.yml]"
        )
        with pytest.raises(ConfigError, match="both declare id"):
            load_run(project(**{"run.yml": run, "copy.yml": MINIMAL_TAXONOMY}))

    def test_a_prompt_version_is_required_once_produced_by_is_given(self, project):
        run = MINIMAL_RUN + "produced_by:\n  model: claude-sonnet-5\n"
        with pytest.raises(ConfigError, match="prompt_version"):
            load_run(project(**{"run.yml": run}))

    def test_an_unknown_run_key_is_an_error(self, project):
        with pytest.raises(ConfigError, match="unknown key"):
            load_run(project(**{"run.yml": MINIMAL_RUN + "outputz: o.jsonl\n"}))

    def test_judges_are_optional_because_the_free_checks_are_not(self, project):
        run = MINIMAL_RUN.replace("judges: judges.yml\n", "")
        config = load_run(project(**{"run.yml": run}))
        assert config.judges.judges == {}
        assert config.judges.triage is None

    def test_a_missing_file_names_the_file(self, project):
        run = MINIMAL_RUN.replace("schema.yml", "nope.yml")
        with pytest.raises(ConfigError, match="no file at"):
            load_run(project(**{"run.yml": run}))


def test_the_worked_example_loads_whole():
    config = load_run(EXAMPLE / "run.yml")
    assert config.run_id == "jtbd-p8"
    assert sorted(config.taxonomies) == ["jtbd", "outcomes"]
    assert config.taxonomy_for("jtbd").ref.version == 4
    assert config.taxonomy_for("summary") is None
    assert config.produced_by.prompt_version == "p8"
    assert config.budget.max_usd == 5.0
    # The three layers, all present and all distinguishable.
    max_words = config.settings.resolve("max_words", config.schema["summary"])
    assert max_words.source == "schema.yml:summary"
    assert config.settings.resolve("min_n_ranking").source == "settings.yml"
    assert config.settings.resolve("bootstrap_resamples").source == "default"


def test_the_example_judges_file_parses_the_shape_the_design_shows():
    config = load_run(EXAMPLE / "run.yml")
    assert set(config.judges.judges) == {"judge-a", "judge-b", "judge-c"}
    assert config.judges.panel.members == ("judge-a", "judge-b", "judge-c")
    assert config.judges.triage.judge == "judge-a"
    assert config.judges.judges["judge-c"].endpoint == "http://gpu-01:1234/v1"
    assert MINIMAL_JUDGES  # the fixture is exercised above
