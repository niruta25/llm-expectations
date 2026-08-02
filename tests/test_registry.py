"""Registry: local-then-entry-point resolution with a helpful failure."""

from __future__ import annotations

import pytest

from llmex import EXPECTATIONS, PROVIDERS, STRATEGIES
from llmex.registry import Registry


def test_missing_name_lists_what_is_available():
    r = Registry("llmex.nothing")
    r.register("alpha", object())
    with pytest.raises(KeyError) as exc:
        r.get("beta")
    assert "alpha" in str(exc.value)
    assert "llmex.nothing" in str(exc.value)


def test_decorator_form_registers_and_returns_the_object():
    r = Registry("llmex.nothing")

    @r.plugin("thing")
    class Thing:
        pass

    assert r.get("thing") is Thing
    assert r.names() == ["thing"]


def test_broken_entry_points_do_not_break_import():
    # A third-party plugin that fails to load must surface as a missing name,
    # not as an ImportError at `import llmex`.
    r = Registry("llmex.definitely-not-a-real-group")
    assert r.names() == []


def test_builtins_are_registered():
    assert "expect_field_grounded_in_source" in EXPECTATIONS.names()
    assert "diverse_ensemble" in STRATEGIES.names()
    assert "mock" in PROVIDERS.names()
