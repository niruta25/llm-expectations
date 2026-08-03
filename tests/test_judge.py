"""The label_judge strategy and its fixture provider."""

from __future__ import annotations

import pytest

from llmex import Capability
from llmex.providers.base import CompletionRequest, CompletionResponse
from llmex.providers.mock import MockLabelJudge, MockProvider
from llmex.strategies import LabelJudge, ScorePayload, StrategyError

LABELS = ["password_reset", "billing_dispute", "cancel_subscription"]

SESSION = (
    "User could not sign in and asked us to reset the password on their "
    "account. Sent a reset link and confirmed access was restored."
)


def payload(**labels):
    return ScorePayload(
        doc_id="s1", source_text=SESSION, extraction=labels or {"jtbd_label": "password_reset"}
    )


def judge(**kw):
    return LabelJudge(label_set=LABELS, **kw)


# -- contract ---------------------------------------------------------------


def test_it_conforms_to_the_strategy_protocol():
    from llmex.strategies import ScoringStrategy

    assert isinstance(judge(), ScoringStrategy)
    assert judge().estimate_calls(10) == 1  # one call grades the whole document


def test_it_declares_the_capabilities_it_uses():
    assert judge().required_capabilities == frozenset(
        {Capability.STRUCTURED_OUTPUT, Capability.SYSTEM_PROMPT}
    )


def test_a_judge_does_not_sample_by_default():
    # A measuring instrument with sampling noise is a defective instrument.
    assert judge().temperature == 0.0


def test_the_taxonomy_and_rubric_reach_the_system_prompt():
    prompt = judge(rubric="Prefer the job the user came to get done.").system_prompt()
    for label in LABELS:
        assert label in prompt
    assert "wrong by definition" in prompt  # the closed-set instruction
    assert "Prefer the job the user came to get done." in prompt


def test_the_same_strategy_serves_any_taxonomy():
    other = LabelJudge(label_set=["invoice", "receipt", "statement"])
    assert "invoice" in other.system_prompt()
    assert "password_reset" not in other.system_prompt()


# -- scoring ----------------------------------------------------------------


async def test_it_scores_and_explains_each_label():
    s = await judge().score(payload(), MockLabelJudge())
    assert set(s.field_scores) == {"jtbd_label"}
    assert 0.0 <= s.field_scores["jtbd_label"] <= 1.0
    # No other built-in strategy fills this in, so trust_score evidence has
    # always read "". A judge that cannot say why is not worth adjudicating.
    assert s.explanations["jtbd_label"]
    assert s.n_calls_used == 1
    assert s.cost.calls == 1


async def test_a_right_label_outscores_a_wrong_one():
    right = await judge().score(payload(jtbd_label="password_reset"), MockLabelJudge())
    wrong = await judge().score(payload(jtbd_label="billing_dispute"), MockLabelJudge())
    assert right.field_scores["jtbd_label"] > wrong.field_scores["jtbd_label"]


async def test_the_document_score_is_a_soft_minimum_over_fields():
    s = await judge().score(
        payload(jtbd_label="password_reset", outcome="billing_dispute"), MockLabelJudge()
    )
    values = list(s.field_scores.values())
    arithmetic = sum(values) / len(values)
    # Soft, not hard: pulled hard toward the worst field without going under it.
    assert min(values) <= s.doc_score < arithmetic
    assert s.aggregation == "harmonic"


# -- degradation ------------------------------------------------------------


class SilentJudge(MockProvider):
    async def complete(self, req: CompletionRequest) -> CompletionResponse:
        return CompletionResponse(text="{}", parsed={})


class PartialJudge(MockLabelJudge):
    """Grades the first field and ignores the rest."""

    async def complete(self, req: CompletionRequest) -> CompletionResponse:
        resp = await super().complete(req)
        scores = (resp.parsed or {})["field_scores"]
        first = sorted(scores)[:1]
        resp.parsed = {"field_scores": {k: scores[k] for k in first}, "explanations": {}}
        return resp


async def test_a_judge_that_says_nothing_raises_rather_than_scoring_zero():
    # Silence is not a verdict. Zero would read as a confident failure.
    with pytest.raises(StrategyError, match="Nothing was measured"):
        await judge().score(payload(), SilentJudge())


async def test_fields_the_judge_skipped_are_left_out_not_filled_in():
    s = await judge().score(
        payload(jtbd_label="password_reset", outcome="resolved"), PartialJudge()
    )
    assert set(s.field_scores) == {"jtbd_label"}  # `outcome` surfaces as unscored


async def test_scores_outside_the_unit_interval_are_clamped():
    class OverconfidentJudge(MockProvider):
        async def complete(self, req):
            return CompletionResponse(
                text="", parsed={"field_scores": {"jtbd_label": 7.5}}
            )

    s = await judge().score(payload(), OverconfidentJudge())
    assert s.field_scores["jtbd_label"] == 1.0


# -- the fixture judge ------------------------------------------------------


async def test_the_mock_judge_scores_on_label_vocabulary_not_substring():
    # `password_reset` never appears verbatim in a transcript; the terms do.
    assert "password_reset" not in SESSION
    s = await judge().score(payload(), MockLabelJudge())
    assert s.field_scores["jtbd_label"] > 0.9


def test_the_payload_a_judge_receives_has_nowhere_to_carry_gold():
    """Structural, not behavioural: a judge cannot peek at the answer because
    the contract it is handed has no field for one. Gold lives on
    ExtractionRecord.meta, which never crosses into ScorePayload."""
    assert "gold" not in {f.name for f in __import__("dataclasses").fields(ScorePayload)}


async def test_the_mock_judge_scores_only_from_the_document_and_the_label():
    wrong = await judge().score(
        ScorePayload("s1", SESSION, {"jtbd_label": "billing_dispute"}), MockLabelJudge()
    )
    assert wrong.field_scores["jtbd_label"] < 0.5  # neither term is in the session


async def test_the_mock_judge_is_deterministic():
    a = await judge().score(payload(), MockLabelJudge())
    b = await judge().score(payload(), MockLabelJudge())
    assert a.field_scores == b.field_scores


async def test_an_unassigned_label_is_treated_as_a_plausible_absence():
    s = await judge().score(payload(jtbd_label=None), MockLabelJudge())
    assert s.field_scores["jtbd_label"] == 0.85
