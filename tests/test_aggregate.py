"""Aggregation semantics: the soft-minimum property and mapping/sequence duality."""

from __future__ import annotations

import pytest

from llmex import arithmetic, harmonic, minimum, weighted_harmonic


def test_harmonic_is_a_soft_minimum():
    # Nineteen good fields and one broken one. The arithmetic mean reads
    # healthy; the harmonic mean reads broken, which it is.
    scores = [0.99] * 19 + [0.02]
    assert arithmetic(scores) > 0.9
    assert harmonic(scores) < 0.35


def test_harmonic_survives_a_zero():
    assert 0.0 < harmonic([0.0, 1.0]) < 0.01


@pytest.mark.parametrize("agg", [harmonic, arithmetic, minimum, weighted_harmonic])
def test_every_aggregator_accepts_a_mapping(agg):
    # The runner passes a mapping so weighted variants can see field names.
    # Naive `for s in scores` over a dict iterates keys and raises.
    assert agg({"a": 1.0, "b": 1.0}) == pytest.approx(1.0)
    assert agg({"a": 1.0, "b": 0.5}) == agg([1.0, 0.5])


@pytest.mark.parametrize("agg", [harmonic, arithmetic, minimum, weighted_harmonic])
def test_empty_input_is_a_pass_not_a_crash(agg):
    assert agg([]) == 1.0
    assert agg({}) == 1.0


@pytest.mark.parametrize("agg", [harmonic, arithmetic, minimum, weighted_harmonic])
def test_none_scores_are_dropped_not_counted(agg):
    assert agg({"a": 1.0, "b": None}) == pytest.approx(1.0)


def test_weights_shift_the_document_score():
    scores = {"total_amount": 0.2, "notes": 1.0}
    unweighted = weighted_harmonic(scores)
    critical = weighted_harmonic(scores, {"total_amount": 3.0})
    assert critical < unweighted  # a wrong total hurts more than wrong notes


def test_minimum_is_the_hard_floor():
    assert minimum({"a": 0.3, "b": 0.9}) == pytest.approx(0.3)


def test_scores_are_clamped_into_range():
    assert harmonic([2.0, 2.0]) == pytest.approx(1.0)
    assert minimum([-5.0]) > 0.0
