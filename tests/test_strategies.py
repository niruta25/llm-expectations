"""Strategies: concurrency, timeout dropping, exception isolation."""

from __future__ import annotations

import asyncio
import math
import time

import pytest

from llmex import Capability
from llmex.providers.base import CompletionRequest, CompletionResponse, CostModel, Limits
from llmex.providers.mock import MockProvider
from llmex.strategies import DiverseEnsemble, PerFieldJudge, ScorePayload, SingleJudge
from llmex.strategies.builtin import LogprobScore

PAYLOAD = ScorePayload(
    doc_id="d1",
    source_text="Paid 1,530.00 to Brightstone Manufacturing.",
    extraction={"vendor": "Brightstone Manufacturing", "total_amount": "9,999.99"},
)


class SlowProvider(MockProvider):
    """Every call takes `delay` seconds of real time."""

    def __init__(self, delay: float = 0.05, **kw):
        super().__init__(**kw)
        self.delay = delay

    async def complete(self, req: CompletionRequest) -> CompletionResponse:
        await asyncio.sleep(self.delay)
        return await super().complete(req)


class UnevenProvider(MockProvider):
    """Two framings answer immediately, three lag. Lets a timeout land between."""

    FAST = {"holistic", "strict"}

    async def complete(self, req: CompletionRequest) -> CompletionResponse:
        await asyncio.sleep(0.0 if req.tag in self.FAST else 0.5)
        return await super().complete(req)


class ExplodingProvider(MockProvider):
    async def complete(self, req: CompletionRequest) -> CompletionResponse:
        raise RuntimeError("provider is down")


class OneBadTemplate(MockProvider):
    """Fails only for the `strict` framing, so the rest of the ensemble stands."""

    async def complete(self, req: CompletionRequest) -> CompletionResponse:
        if req.tag == "strict":
            raise RuntimeError("this member is broken")
        return await super().complete(req)


# -- shape ------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy,expected",
    [(SingleJudge(), 1), (PerFieldJudge(), 4), (DiverseEnsemble(), 5), (LogprobScore(), 0)],
)
def test_estimate_calls_is_honest(strategy, expected):
    assert strategy.estimate_calls(4) == expected


def test_ensemble_size_is_configurable_and_capped():
    assert DiverseEnsemble(n_calls=3).estimate_calls(10) == 3
    assert DiverseEnsemble(n_calls=99).estimate_calls(10) == 5  # only five framings exist


def test_logprob_declares_the_capability_it_needs():
    assert Capability.LOGPROBS in LogprobScore().required_capabilities


async def test_every_strategy_scores_both_grains():
    for strategy in (SingleJudge(), PerFieldJudge(), DiverseEnsemble()):
        s = await strategy.score(PAYLOAD, MockProvider())
        assert set(s.field_scores) == {"vendor", "total_amount"}
        assert 0.0 <= s.doc_score <= 1.0
        assert s.aggregation == "harmonic"  # reported, not assumed downstream


async def test_logprob_scores_from_the_generators_own_probabilities():
    payload = ScorePayload(
        PAYLOAD.doc_id,
        PAYLOAD.source_text,
        PAYLOAD.extraction,
        logprobs={"vendor": math.log(0.98), "total_amount": math.log(0.30)},
    )
    s = await LogprobScore().score(payload, MockProvider())
    assert s.field_scores["vendor"] == pytest.approx(0.98)
    assert s.field_scores["total_amount"] == pytest.approx(0.30)


async def test_logprob_refuses_to_invent_a_score_it_was_not_given():
    # Observe-only means the framework cannot fetch logprobs itself. A constant
    # standing in for a measurement is the trust laundering this project exists
    # to prevent, so the absence has to be loud.
    from llmex.strategies import StrategyError

    with pytest.raises(StrategyError, match="no generator logprobs"):
        await LogprobScore().score(PAYLOAD, MockProvider())


async def test_the_ensemble_scores_an_ungrounded_value_lower():
    s = await DiverseEnsemble().score(PAYLOAD, MockProvider())
    assert s.field_scores["total_amount"] < s.field_scores["vendor"]
    assert s.n_calls_used == 5


# -- concurrency ------------------------------------------------------------


async def test_ensemble_wall_clock_is_one_call_not_five():
    provider = SlowProvider(delay=0.05)
    start = time.perf_counter()
    s = await DiverseEnsemble().score(PAYLOAD, provider)
    elapsed = time.perf_counter() - start
    assert s.n_calls_used == 5
    assert elapsed < 0.05 * 3  # generous, but nowhere near sequential


async def test_per_field_judge_also_fans_out():
    provider = SlowProvider(delay=0.05)
    payload = ScorePayload("d1", "text", {f"f{i}": "v" for i in range(6)})
    start = time.perf_counter()
    await PerFieldJudge().score(payload, provider)
    assert time.perf_counter() - start < 0.05 * 4


async def test_cost_latency_is_the_max_not_the_sum():
    s = await DiverseEnsemble().score(PAYLOAD, MockProvider(latency_ms=10.0))
    assert s.cost.calls == 5
    assert s.cost.latency_ms == pytest.approx(10.0)


# -- degradation ------------------------------------------------------------


async def test_a_timeout_drops_laggards_and_still_returns_a_score():
    s = await DiverseEnsemble(timeout_s=0.05).score(PAYLOAD, UnevenProvider())
    assert s.n_calls_used == 2
    assert s.n_calls_dropped == 3
    assert set(s.field_scores) == {"vendor", "total_amount"}
    assert s.cost.calls == 2  # you are not billed for what you cancelled


async def test_losing_every_member_raises_rather_than_inventing_a_score():
    # A middling default would clear a middling threshold and read as a pass.
    from llmex.strategies import StrategyError

    with pytest.raises(StrategyError, match="no score available"):
        await DiverseEnsemble(timeout_s=0.001).score(PAYLOAD, SlowProvider(delay=0.2))

    with pytest.raises(StrategyError):
        await DiverseEnsemble().score(PAYLOAD, ExplodingProvider())


async def test_one_broken_ensemble_member_does_not_kill_the_rest():
    s = await DiverseEnsemble().score(PAYLOAD, OneBadTemplate())
    assert s.n_calls_used == 4
    assert set(s.field_scores) == {"vendor", "total_amount"}


async def test_a_dead_provider_propagates_so_the_expectation_can_skip():
    with pytest.raises(RuntimeError):
        await SingleJudge().score(PAYLOAD, ExplodingProvider())


# -- provider contract ------------------------------------------------------


def test_cost_model_prices_input_and_output_separately():
    cm = CostModel(usd_per_1k_in=1.0, usd_per_1k_out=2.0)
    assert cm.price(1000, 500) == pytest.approx(2.0)


def test_mock_provider_declares_its_capabilities_and_limits():
    p = MockProvider()
    assert Capability.STRUCTURED_OUTPUT in p.capabilities
    assert isinstance(p.limits, Limits)


async def test_mock_provider_is_deterministic():
    a = await MockProvider().complete(
        CompletionRequest(user='{"source_text": "x", "extraction": {"f": "y"}}', tag="t")
    )
    b = await MockProvider().complete(
        CompletionRequest(user='{"source_text": "x", "extraction": {"f": "y"}}', tag="t")
    )
    assert a.parsed == b.parsed


async def test_http_provider_skeleton_refuses_to_pretend_it_works():
    from llmex.providers import HTTPChatProvider

    p = HTTPChatProvider(endpoint="http://nowhere", model="m1")
    with pytest.raises(NotImplementedError):
        await p.complete(CompletionRequest(user="{}"))
