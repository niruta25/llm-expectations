"""A/B comparison between two variants.

This does not fit the expectation model, and forcing it there would have been a
mistake. A `Result` describes one document under one check; "variant B beats
variant A" is a statement about two *runs*, at a grain the framework does not
have. So comparison lives here, as functions over runs rather than as a check
inside one.

Two ways to compare, in the order you should reach for them:

1. `compare_runs` — free. Both variants already have results; whoever fails
   fewer checks on a document wins it.
2. `PairwiseJudge` — costs money, for the documents where nothing free
   separates the variants.

Both report significance, because a win rate without it is a coin flip with a
press release. On the sample sizes teams actually run — tens to low hundreds of
documents — a handful of net wins is indistinguishable from noise, and saying
so is the entire value of this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import comb
from typing import Any

from .batch import Batch
from .defaults import DEFAULT_JUDGE_TEMPERATURE
from .providers.base import CompletionRequest, ModelProvider, cost_of
from .result import Result, RunResult
from .types import Cost

TIE = "tie"


def mcnemar_exact(wins_a: int, wins_b: int) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes.

    Only the *discordant* documents carry information: the ones both variants
    got right, or both wrong, say nothing about which is better. Under the null
    hypothesis each discordant document is a coin flip, so this is an exact
    binomial test on those.

    Exact rather than the chi-square approximation because the discordant count
    is routinely under 25, which is exactly where the approximation misleads.
    """
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    k = min(wins_a, wins_b)
    tail = sum(comb(n, i) for i in range(k + 1))
    p: float = 2 * tail / 2**n
    return min(1.0, p)


@dataclass
class DocumentVerdict:
    doc_id: str
    winner: str  # "a" | "b" | "tie"
    a_failures: int = 0
    b_failures: int = 0
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "winner": self.winner,
            "a_failures": self.a_failures,
            "b_failures": self.b_failures,
            "detail": self.detail,
        }


@dataclass
class Comparison:
    a_label: str
    b_label: str
    verdicts: list[DocumentVerdict] = field(default_factory=list)
    cost: Cost = field(default_factory=Cost)
    position_flips: int = 0
    """Documents where a pairwise judge reversed its verdict when the two
    candidates swapped places. Scored as ties, and counted, because a judge
    that answers differently based on presentation order is telling you the
    difference is below its resolution."""
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.verdicts)

    @property
    def wins_a(self) -> int:
        return sum(1 for v in self.verdicts if v.winner == "a")

    @property
    def wins_b(self) -> int:
        return sum(1 for v in self.verdicts if v.winner == "b")

    @property
    def ties(self) -> int:
        return sum(1 for v in self.verdicts if v.winner == TIE)

    @property
    def win_rate_b(self) -> float:
        """B's share of *all* documents, ties included.

        The conservative number, and the one to quote. Most documents tie in a
        real A/B, so dropping them inflates the headline.
        """
        return self.wins_b / self.n if self.n else 0.0

    @property
    def win_rate_b_decided(self) -> float:
        """B's share of the documents that separated the variants."""
        decided = self.wins_a + self.wins_b
        return self.wins_b / decided if decided else 0.0

    @property
    def p_value(self) -> float:
        return mcnemar_exact(self.wins_a, self.wins_b)

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def verdict(self, alpha: float = 0.05) -> str:
        """One sentence a human can act on, including the boring answer."""
        if self.n == 0:
            return "no comparable documents"
        decided = self.wins_a + self.wins_b
        if decided == 0:
            return f"identical on all {self.n} documents"
        if not self.significant(alpha):
            return (
                f"no significant difference over {self.n} documents "
                f"({self.b_label} {self.wins_b}, {self.a_label} {self.wins_a}, "
                f"{self.ties} tied; p={self.p_value:.3f}). "
                f"{_needed_hint(self.wins_a, self.wins_b, alpha)}"
            )
        winner, w, loser, ln = (
            (self.b_label, self.wins_b, self.a_label, self.wins_a)
            if self.wins_b > self.wins_a
            else (self.a_label, self.wins_a, self.b_label, self.wins_b)
        )
        return (
            f"{winner} beats {loser} over {self.n} documents "
            f"({w} to {ln}, {self.ties} tied; p={self.p_value:.3f})"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "a": self.a_label,
            "b": self.b_label,
            "n": self.n,
            "wins_a": self.wins_a,
            "wins_b": self.wins_b,
            "ties": self.ties,
            "win_rate_b": round(self.win_rate_b, 4),
            "win_rate_b_decided": round(self.win_rate_b_decided, 4),
            "p_value": round(self.p_value, 6),
            "significant_at_05": self.significant(),
            "position_flips": self.position_flips,
            "cost": self.cost.as_dict(),
            "verdict": self.verdict(),
            "notes": self.notes,
            "verdicts": [v.as_dict() for v in self.verdicts],
        }


def _needed_hint(wins_a: int, wins_b: int, alpha: float) -> str:
    """How lopsided the discordant pairs would have to be to clear alpha."""
    n = wins_a + wins_b
    for extra in range(0, 200):
        if mcnemar_exact(min(wins_a, wins_b), max(wins_a, wins_b) + extra) < alpha:
            return (
                f"About {extra} more net wins on top of the current {n} "
                f"decided documents would be needed to call it."
            )
    return "Collect more documents before calling it."


# --------------------------------------------------------------------------
# Free comparison, from results that already exist
# --------------------------------------------------------------------------


CheckKey = tuple[str, str, str | None]
"""(expectation_id, grain, field_name) — what makes two results comparable."""


def _index(run: RunResult, on: list[str] | None) -> dict[str, dict[CheckKey, Result]]:
    by_doc: dict[str, dict[CheckKey, Result]] = {}
    for r in run.results:
        if r.success is None:
            continue  # unscored is not evidence for either side
        if on and r.expectation_id not in on:
            continue
        key = (r.expectation_id, r.grain.value, r.field_name)
        by_doc.setdefault(r.doc_id, {})[key] = r
    return by_doc


def compare_runs(
    run_a: RunResult,
    run_b: RunResult,
    a_label: str = "a",
    b_label: str = "b",
    on: list[str] | None = None,
) -> Comparison:
    """Which variant failed fewer checks, document by document.

    Compares only checks both runs actually scored on the same document: a
    check that was sampled out of one run says nothing about the other, and
    counting it would let sampling luck decide the winner.

    `on` restricts the comparison to named expectations — usually what you
    want, since a run contains checks that have nothing to do with the change
    under test.
    """
    index_a, index_b = _index(run_a, on), _index(run_b, on)
    shared_docs = [d for d in index_a if d in index_b]

    cmp = Comparison(a_label=a_label, b_label=b_label)
    if len(shared_docs) < len(index_a) or len(shared_docs) < len(index_b):
        cmp.notes.append(
            f"compared {len(shared_docs)} documents present in both runs "
            f"({len(index_a)} in {a_label}, {len(index_b)} in {b_label})"
        )

    for doc_id in shared_docs:
        a_checks, b_checks = index_a[doc_id], index_b[doc_id]
        shared_keys = [k for k in a_checks if k in b_checks]
        if not shared_keys:
            continue
        a_fail = sum(1 for k in shared_keys if a_checks[k].success is False)
        b_fail = sum(1 for k in shared_keys if b_checks[k].success is False)
        winner = "a" if a_fail < b_fail else "b" if b_fail < a_fail else TIE
        cmp.verdicts.append(
            DocumentVerdict(
                doc_id=doc_id,
                winner=winner,
                a_failures=a_fail,
                b_failures=b_fail,
                detail={"checks_compared": len(shared_keys)},
            )
        )
    return cmp


# --------------------------------------------------------------------------
# Judged comparison, for where nothing free separates the variants
# --------------------------------------------------------------------------

_PAIRWISE_SCHEMA: dict[str, Any] = {
    "name": "pairwise_verdict",
    "schema": {
        "type": "object",
        "properties": {
            "winner": {"type": "string", "enum": ["A", "B", "tie"]},
            "reason": {"type": "string"},
        },
        "required": ["winner"],
    },
}

_PAIRWISE_SYSTEM = """Two systems produced output for the same document. Decide \
which output is better, or answer "tie" if neither is clearly better.

Judge only against the document. Do not reward length, confidence or fluency — \
a shorter correct answer beats a longer one. If the document does not contain \
what is needed to tell them apart, answer "tie"; a forced choice between two \
indistinguishable outputs is noise you will later mistake for signal."""


@dataclass
class PairwisePayload:
    doc_id: str
    source_text: str
    a: dict[str, object]
    b: dict[str, object]


class PairwiseJudge:
    """Asks a model which of two outputs is better, controlling for position.

    Every document is judged twice, with the candidates swapped. A verdict
    counts only when both orders agree; when they disagree the document is a
    tie and `position_flips` is incremented.

    That doubles the cost, and it is not optional. Pairwise LLM judges have a
    well-documented preference for whichever candidate they see first, and an
    uncontrolled pairwise run will hand you a win rate manufactured by argument
    order. If the flip count is high, the honest reading is not that the judge
    is broken but that the two variants are closer together than it can resolve.
    """

    id = "pairwise_judge"

    def __init__(
        self,
        rubric: str = "",
        temperature: float = DEFAULT_JUDGE_TEMPERATURE,
        control_position_bias: bool = True,
    ) -> None:
        self.rubric = rubric
        self.temperature = temperature
        self.control_position_bias = control_position_bias

    def system_prompt(self) -> str:
        return "\n\n".join(p for p in (_PAIRWISE_SYSTEM, self.rubric.strip()) if p)

    async def _ask(
        self,
        provider: ModelProvider,
        payload: PairwisePayload,
        first: dict[str, object],
        second: dict[str, object],
        tag: str,
    ) -> tuple[str, str, Cost]:
        req = CompletionRequest(
            user=json.dumps(
                {
                    "document": payload.source_text,
                    "output_A": first,
                    "output_B": second,
                }
            ),
            system=self.system_prompt(),
            json_schema=_PAIRWISE_SCHEMA,
            temperature=self.temperature,
            tag=tag,
        )
        resp = await provider.complete(req)
        parsed = resp.parsed or {}
        raw = str(parsed.get("winner", TIE)).strip().lower()
        winner = raw if raw in ("a", "b") else TIE
        return winner, str(parsed.get("reason", "")), cost_of(provider, resp)

    async def compare_one(
        self, payload: PairwisePayload, provider: ModelProvider
    ) -> tuple[DocumentVerdict, Cost, bool]:
        forward, reason, cost = await self._ask(
            provider, payload, payload.a, payload.b, "pairwise:ab"
        )
        if not self.control_position_bias:
            return (
                DocumentVerdict(payload.doc_id, forward, detail={"reason": reason}),
                cost,
                False,
            )

        # Swapped: the judge now sees B first, so its "a" means our b.
        reverse_raw, reverse_reason, cost_b = await self._ask(
            provider, payload, payload.b, payload.a, "pairwise:ba"
        )
        reverse = {"a": "b", "b": "a", TIE: TIE}[reverse_raw]
        total = cost + cost_b

        if forward != reverse:
            return (
                DocumentVerdict(
                    payload.doc_id,
                    TIE,
                    detail={
                        "position_flip": True,
                        "forward": forward,
                        "reversed": reverse,
                        "reason": reason,
                    },
                ),
                total,
                True,
            )
        return (
            DocumentVerdict(
                payload.doc_id,
                forward,
                detail={"reason": reason, "confirmed_both_orders": True},
            ),
            total,
            False,
        )


async def compare_with_judge(
    batch_a: Batch,
    batch_b: Batch,
    judge: PairwiseJudge,
    provider: ModelProvider,
    fields: list[str] | None = None,
    a_label: str = "a",
    b_label: str = "b",
    budget: Any | None = None,
) -> Comparison:
    """Judge two variants over the documents they share.

    Only documents whose outputs actually differ are sent to the judge:
    identical outputs are a tie by construction, and paying a model to confirm
    that is the most avoidable cost in the whole pipeline.
    """
    wanted = set(fields) if fields else None

    def outputs(batch: Batch) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for doc_id, recs in batch.by_document().items():
            out[doc_id] = {
                r.field_name: r.value
                for r in recs
                if wanted is None or r.field_name in wanted
            }
        return out

    a_out, b_out = outputs(batch_a), outputs(batch_b)
    cmp = Comparison(a_label=a_label, b_label=b_label)
    identical = 0

    for doc_id in [d for d in a_out if d in b_out]:
        if a_out[doc_id] == b_out[doc_id]:
            cmp.verdicts.append(
                DocumentVerdict(doc_id, TIE, detail={"identical_output": True})
            )
            identical += 1
            continue

        if budget is not None and not await budget.reserve(2):
            cmp.notes.append(f"budget exhausted at {doc_id}; remaining documents unjudged")
            break

        payload = PairwisePayload(
            doc_id, batch_a.source(doc_id).text, a_out[doc_id], b_out[doc_id]
        )
        verdict, cost, flipped = await judge.compare_one(payload, provider)
        cmp.verdicts.append(verdict)
        cmp.cost = cmp.cost + cost
        cmp.position_flips += int(flipped)
        if budget is not None:
            await budget.record(cost)

    if identical:
        cmp.notes.append(
            f"{identical} of {len(cmp.verdicts)} documents had identical output "
            f"and were tied without a model call"
        )
    if cmp.position_flips:
        cmp.notes.append(
            f"{cmp.position_flips} documents flipped verdict when the candidates "
            f"swapped places and were scored as ties"
        )
    return cmp
