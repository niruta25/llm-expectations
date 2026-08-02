"""Suite: YAML round-trip, alias resolution, fingerprint stability."""

from __future__ import annotations

import pytest
import yaml

from llmex import Severity, Suite

YAML_SUITE = """
suite: invoice_extraction
version: 3

providers:
  cheap_verifier:
    plugin: mock
    latency_ms: 2

strategies:
  ensemble:
    plugin: diverse_ensemble
    n_calls: 3
    timeout_s: 5

budget:
  max_usd: 40
  max_calls: 20000

aggregate:
  field_to_document: weighted_harmonic
  field_weights:
    total_amount: 3.0

expectations:
  - type: expect_field_grounded_in_source
    fields: ["vendor"]
    min_ratio: 0.9
    severity: error
  - type: expect_field_null_rate_between
    max_rate: 0.2
    severity: warn
"""


@pytest.fixture
def suite_path(tmp_path):
    p = tmp_path / "suite.yml"
    p.write_text(YAML_SUITE)
    return p


def test_yaml_loads_every_section(suite_path):
    s = Suite.from_yaml(suite_path)
    assert s.name == "invoice_extraction"
    assert s.version == "3"
    assert s.budget.max_usd == 40
    assert s.budget.max_calls == 20000
    assert s.aggregator == "weighted_harmonic"
    assert s.field_weights == {"total_amount": 3.0}
    assert len(s.expectations) == 2


def test_provider_and_strategy_kwargs_reach_the_constructor(suite_path):
    s = Suite.from_yaml(suite_path)
    assert s.providers["cheap_verifier"]._latency == 2
    assert s.strategies["ensemble"].estimate_calls(10) == 3
    assert s.strategies["ensemble"].timeout_s == 5


def test_builtin_strategies_are_auto_aliased_by_their_id(suite_path):
    # `strategy: diverse_ensemble` must work with no `strategies:` block.
    s = Suite.from_dict({"suite": "t", "expectations": []})
    assert "diverse_ensemble" in s.strategies
    assert "single_judge" in s.strategies
    # An explicit alias does not shadow the built-ins.
    assert {"ensemble", "diverse_ensemble"} <= set(Suite.from_yaml(suite_path).strategies)


def test_severity_defaults_to_error_and_is_parsed(suite_path):
    s = Suite.from_yaml(suite_path)
    assert s.expectations[0].severity is Severity.ERROR
    assert s.expectations[1].severity is Severity.WARN
    default = Suite.from_dict(
        {"suite": "t", "expectations": [{"type": "expect_field_type"}]}
    )
    assert default.expectations[0].severity is Severity.ERROR


def test_config_reaches_the_expectation(suite_path):
    s = Suite.from_yaml(suite_path)
    assert s.expectations[0].config == {"fields": ["vendor"], "min_ratio": 0.9}


def test_unknown_expectation_type_names_what_is_available():
    with pytest.raises(KeyError, match="expect_field_grounded_in_source"):
        Suite.from_dict({"suite": "t", "expectations": [{"type": "expect_nonsense"}]})


def test_fingerprint_is_stable_and_moves_with_the_config(suite_path):
    a = Suite.from_yaml(suite_path).fingerprint()
    b = Suite.from_yaml(suite_path).fingerprint()
    assert a == b

    cfg = yaml.safe_load(YAML_SUITE)
    cfg["expectations"][0]["min_ratio"] = 0.5
    assert Suite.from_dict(cfg).fingerprint() != a


def test_fingerprint_records_expectation_versions_and_providers(suite_path):
    fp = Suite.from_yaml(suite_path).fingerprint()
    assert fp["expectations"][0]["version"] == "1"
    assert fp["providers"]["cheap_verifier"]["model_version"] == "mock-1.0"
    assert fp["aggregator"] == "weighted_harmonic"


def test_an_empty_suite_is_valid():
    s = Suite.from_dict({"suite": "empty"})
    assert s.expectations == []
    assert s.budget.max_usd == float("inf")
