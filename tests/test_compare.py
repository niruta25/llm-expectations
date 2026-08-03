"""A/B comparison: significance, position bias, and the boring answer."""

from __future__ import annotations

import pytest
from conftest import cfg

from llmex import (
    Batch,
    Budget,
    Cost,
    ExtractionRecord,
    PairwiseJudge,
    PairwisePayload,
    Runner,
    SourceDoc,
    Suite,
    compare_runs,
    compare_with_judge,
    mcnemar_exact,
)
from llmex.compare import Comparison, DocumentVerdict
from llmex.providers.base import CompletionRequest, CompletionResponse
from llmex.providers.mock import MockPairwiseJudge

DOC = "Paid 1,530.00 to Brightstone Manufacturing on February 12, 2024."


def variant_batch(vendor: str, total: str = "1,530.00", n_docs: int = 1) -> Batch:
    recs = []
    for i in range(n_docs):
        recs.append(ExtractionRecord(f"d{i}", "vendor", vendor))
        recs.append(ExtractionRecord(f"d{i}", "total_amount", total))
    return Batch(recs, lambda d: SourceDoc(d, DOC))


def run_variant(batch: Batch):
    return Runner().run_sync(Suite.from_dict(cfg()), batch)


def comparison(wins_a: int, wins_b: int, ties: int = 0) -> Comparison:
    c = Comparison("a", "b")
    for i in range(wins_a):
        c.verdicts.append(DocumentVerdict(f"a{i}", "a"))
    for i in range(wins_b):
        c.verdicts.append(DocumentVerdict(f"b{i}", "b"))
    for i in range(ties):
        c.verdicts.append(DocumentVerdict(f"t{i}", "tie"))
    return c


# -- significance -----------------------------------------------------------


def test_mcnemar_only_counts_discordant_documents():
    # Documents both variants got right say nothing about which is better.
    assert mcnemar_exact(0, 0) == 1.0
    assert comparison(0, 0, ties=500).p_value == 1.0


def test_mcnemar_matches_hand_computed_values():
    # 1 vs 3 discordant: 2 * (C(4,0) + C(4,1)) / 2^4 = 2 * 5 / 16
    assert mcnemar_exact(1, 3) == pytest.approx(0.625)
    # 0 vs 5: 2 * C(5,0) / 2^5 = 2/32
    assert mcnemar_exact(0, 5) == pytest.approx(0.0625)
    assert mcnemar_exact(0, 6) == pytest.approx(0.03125)
    assert mcnemar_exact(3, 3) == pytest.approx(1.0)


def test_a_p_value_is_never_above_one():
    assert mcnemar_exact(1, 1) <= 1.0
    assert mcnemar_exact(2, 2) <= 1.0


def test_a_small_lopsided_win_is_still_not_significant():
    """The lesson the module exists to teach. Three net wins on a two-dozen
    document corpus is a coin flip, however good the win rate looks."""
    c = comparison(wins_a=1, wins_b=3, ties=20)
    assert c.win_rate_b_decided == pytest.approx(0.75)  # reads like a landslide
    assert not c.significant()
    assert "no significant difference" in c.verdict()
    assert "more net wins" in c.verdict()


def test_a_large_consistent_win_is_significant():
    c = comparison(wins_a=1, wins_b=14, ties=40)
    assert c.significant()
    assert "b beats a" in c.verdict()


def test_ties_are_included_in_the_headline_rate_and_excluded_from_the_other():
    c = comparison(wins_a=1, wins_b=3, ties=20)
    assert c.win_rate_b == pytest.approx(3 / 24)  # the conservative number
    assert c.win_rate_b_decided == pytest.approx(3 / 4)
    assert c.n == 24


def test_identical_variants_say_so_rather_than_picking_one():
    assert "identical on all" in comparison(0, 0, ties=10).verdict()
    assert comparison(0, 0, 0).verdict() == "no comparable documents"


def test_a_comparison_is_json_serialisable():
    import json

    d = json.loads(json.dumps(comparison(1, 3, 20).as_dict()))
    assert d["significant_at_05"] is False
    assert d["n"] == 24


# -- free comparison over two runs ------------------------------------------


def test_the_variant_that_fails_fewer_checks_wins_the_document():
    a = run_variant(variant_batch(vendor="Northwind Traders Ltd"))  # hallucinated
    b = run_variant(variant_batch(vendor="Brightstone Manufacturing"))  # grounded
    c = compare_runs(a, b, a_label="baseline", b_label="candidate")
    assert c.wins_b == 1
    assert c.wins_a == 0
    assert c.verdicts[0].a_failures > c.verdicts[0].b_failures


def test_identical_runs_tie_on_every_document():
    batch = variant_batch(vendor="Brightstone Manufacturing", n_docs=3)
    c = compare_runs(run_variant(batch), run_variant(batch))
    assert c.n == 3
    assert c.ties == 3
    assert c.p_value == 1.0


def test_only_documents_present_in_both_runs_are_compared():
    a = run_variant(variant_batch("Brightstone Manufacturing", n_docs=3))
    b = run_variant(variant_batch("Brightstone Manufacturing", n_docs=2))
    c = compare_runs(a, b)
    assert c.n == 2
    assert any("present in both runs" in note for note in c.notes)


def test_unscored_results_are_not_evidence_for_either_side():
    from llmex import Grain, Result

    a = run_variant(variant_batch("Brightstone Manufacturing"))
    b = run_variant(variant_batch("Brightstone Manufacturing"))
    # A skipped check on one side must not hand the document to the other.
    b.results.append(
        Result(
            expectation_id="expect_field_grounded_in_source",
            grain=Grain.FIELD, doc_id="d0", field_name="vendor", success=None,
        )
    )
    assert compare_runs(a, b).ties == 1


def test_on_restricts_the_comparison_to_the_checks_under_test():
    a = run_variant(variant_batch("Northwind Traders Ltd"))
    b = run_variant(variant_batch("Brightstone Manufacturing"))
    assert compare_runs(a, b, on=["expect_field_grounded_in_source"]).wins_b == 1
    assert compare_runs(a, b, on=["expect_field_type"]).n == 0


# -- pairwise judging -------------------------------------------------------


class AlwaysPrefersFirst(MockPairwiseJudge):
    """A judge with total position bias — the documented pairwise failure mode."""

    async def complete(self, req: CompletionRequest) -> CompletionResponse:
        import json

        body = json.dumps({"winner": "A", "reason": "whichever I saw first"})
        return CompletionResponse(text=body, parsed={"winner": "A", "reason": "first"})


async def test_a_position_biased_judge_produces_ties_not_a_fake_win_rate():
    """Uncontrolled, this judge hands every document to whoever is argued
    first. The swap-and-confirm pass turns that into visible ties."""
    payload = PairwisePayload("d0", DOC, {"vendor": "X"}, {"vendor": "Y"})
    verdict, _, flipped = await PairwiseJudge().compare_one(payload, AlwaysPrefersFirst())
    assert flipped
    assert verdict.winner == "tie"
    assert verdict.detail["position_flip"] is True


async def test_position_control_can_be_disabled_and_then_it_lies():
    payload = PairwisePayload("d0", DOC, {"vendor": "X"}, {"vendor": "Y"})
    judge = PairwiseJudge(control_position_bias=False)
    verdict, cost, flipped = await judge.compare_one(payload, AlwaysPrefersFirst())
    assert verdict.winner == "a"  # manufactured entirely by argument order
    assert not flipped
    assert cost.calls == 1  # half the cost, and worth nothing


async def test_controlling_for_position_costs_two_calls_per_document():
    payload = PairwisePayload("d0", DOC, {"vendor": "X"}, {"vendor": "Y"})
    _, cost, _ = await PairwiseJudge().compare_one(payload, MockPairwiseJudge())
    assert cost.calls == 2


async def test_an_unbiased_judge_confirms_its_verdict_in_both_orders():
    payload = PairwisePayload(
        "d0", DOC, {"vendor": "Northwind Traders"}, {"vendor": "Brightstone Manufacturing"}
    )
    verdict, _, flipped = await PairwiseJudge().compare_one(payload, MockPairwiseJudge())
    assert not flipped
    assert verdict.winner == "b"
    assert verdict.detail["confirmed_both_orders"] is True


async def test_documents_with_identical_output_are_tied_without_a_model_call():
    batch = variant_batch("Brightstone Manufacturing", n_docs=4)
    c = await compare_with_judge(batch, batch, PairwiseJudge(), MockPairwiseJudge())
    assert c.ties == 4
    assert c.cost == Cost()  # the most avoidable spend in the pipeline
    assert any("identical output" in note for note in c.notes)


async def test_only_differing_documents_reach_the_judge():
    a = variant_batch("Northwind Traders Ltd", n_docs=2)
    b = variant_batch("Brightstone Manufacturing", n_docs=2)
    c = await compare_with_judge(a, b, PairwiseJudge(), MockPairwiseJudge(), fields=["vendor"])
    assert c.n == 2
    assert c.wins_b == 2
    assert c.cost.calls == 4  # two documents, two orders each


async def test_flips_are_counted_and_reported():
    a = variant_batch("Northwind Traders Ltd", n_docs=3)
    b = variant_batch("Brightstone Manufacturing", n_docs=3)
    c = await compare_with_judge(a, b, PairwiseJudge(), AlwaysPrefersFirst())
    assert c.position_flips == 3
    assert c.ties == 3
    assert any("flipped verdict" in note for note in c.notes)


async def test_the_budget_stops_a_comparison_rather_than_overrunning_it():
    a = variant_batch("Northwind Traders Ltd", n_docs=10)
    b = variant_batch("Brightstone Manufacturing", n_docs=10)
    budget = Budget(max_calls=4)
    c = await compare_with_judge(
        a, b, PairwiseJudge(), MockPairwiseJudge(), budget=budget
    )
    assert c.n < 10
    assert any("budget exhausted" in note for note in c.notes)
