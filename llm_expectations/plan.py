"""What will run, and what it will cost — before anything is spent.

The prompts are built for real here, so the token estimate comes from the
actual bytes that would be sent rather than from a guess about them. Output is
estimated at the full budget, which makes the figure a ceiling: being told
$0.34 and paying $0.21 is fine, and the reverse is not.

Prices are a table, and a model that is not in it produces **no estimate**
rather than a plausible one. A made-up cost is the same species of error as a
made-up confidence, and this is a library about not doing that.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .config import JudgeSpec, RunConfig
from .schema import FieldKind
from .types import Item, Output

__all__ = ["Plan", "PlannedJob", "Pricing", "estimate_cost", "plan_run"]


@dataclass(frozen=True, slots=True)
class Pricing:
    """USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float


#: Anthropic list prices, as published 2026-06. Other vendors are deliberately
#: absent: this table is only as good as its last update, and a stale number
#: presented confidently is worse than no number. An unpriced model is reported
#: as unpriced, and the budget guard says it cannot enforce a cap on it.
PRICES: Mapping[str, Pricing] = {
    "claude-fable-5-1": Pricing(10.0, 50.0),
    "claude-fable-5": Pricing(10.0, 50.0),
    "claude-opus-5": Pricing(5.0, 25.0),
    "claude-opus-4-8": Pricing(5.0, 25.0),
    "claude-opus-4-7": Pricing(5.0, 25.0),
    "claude-opus-4-6": Pricing(5.0, 25.0),
    "claude-sonnet-5": Pricing(2.0, 10.0),
    "claude-sonnet-4-6": Pricing(3.0, 15.0),
    "claude-haiku-4-5": Pricing(1.0, 5.0),
}

CHARS_PER_TOKEN = 4  # a rough, deliberately generous divisor

#: What `provider: fake` reports as its model. It costs nothing because it
#: asks nothing.
SCRIPTED_MODEL = "scripted"


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """None means "this model is not in the price table", not "free".

    The one model that really is free is the scripted judge, which reaches no
    network. Reporting that as unknown would be as wrong as guessing a price.
    """
    if model == SCRIPTED_MODEL:
        return 0.0
    price = PRICES.get(model)
    if price is None:
        return None
    return (
        input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok
    ) / 1_000_000


@dataclass(frozen=True, slots=True)
class PlannedJob:
    """One job's share of the bill."""

    job: str
    judge_id: str
    model: str
    calls: int
    cached: int
    input_tokens: int
    output_tokens: int
    usd: float | None

    @property
    def new_calls(self) -> int:
        return self.calls - self.cached


@dataclass(frozen=True, slots=True)
class Plan:
    """The whole run, costed."""

    jobs: tuple[PlannedJob, ...]
    skipped: Mapping[str, int]
    unpriced_models: tuple[str, ...] = ()

    @property
    def calls(self) -> int:
        return sum(job.new_calls for job in self.jobs)

    @property
    def usd(self) -> float | None:
        if any(job.usd is None for job in self.jobs):
            return None
        return sum(job.usd or 0.0 for job in self.jobs)

    def lines(self) -> list[str]:
        width = max((len(job.job) for job in self.jobs), default=6)
        out = []
        for job in self.jobs:
            cached = f"  ({job.cached} cached)" if job.cached else ""
            out.append(f"  {job.job:<{width}} : {job.new_calls:>6,} calls{cached}")
        total = self.usd
        cost = f"est. ${total:.2f}" if total is not None else "cost not estimable"
        out.append(f"  {'':<{width}}   {'-' * 24}")
        out.append(f"  {'total':<{width}} : {self.calls:>6,} calls, {cost}")
        return out


def plan_run(
    config: RunConfig,
    items: Mapping[str, Item],
    outputs: Mapping[str, Output],
    *,
    cached_keys: frozenset[str] = frozenset(),
    free_text: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
) -> Plan:
    """Count the calls a run would make, and price them.

    Each job that spends money adds a ``PlannedJob``, so the panel and the
    claim judge appear beside triage rather than folded into it. A combined
    total would hide the thing worth seeing: the claim judge is a fraction of
    the corpus, and the triage judge is nearly all of it.
    """
    from .judges.prompts import LabelCorrectTask, label_is_judgeable

    triage = config.judges.triage
    skipped: dict[str, int] = {}
    if triage is None:
        return Plan(jobs=(), skipped={"no triage judge is wired in judges.yml": len(items)})

    spec = config.judges.judges[triage.judge]
    temperature = spec.temperature
    if temperature is None:
        temperature = config.settings.value("judge_temperature")
    max_tokens = spec.max_tokens or config.settings.value("judge_max_tokens")

    assigned = [f for f in config.schema.fields.values() if f.kind is FieldKind.ASSIGNED]
    scope = triage.scope
    item_ids = tuple(items)
    if isinstance(scope, int):
        item_ids = item_ids[:scope]
        skipped["outside the triage sample"] = len(items) - len(item_ids)

    calls = cached = input_tokens = 0
    for field in assigned:
        task = LabelCorrectTask(require_leaf=bool(config.settings.value("require_leaf", field)))
        taxonomy = config.taxonomy_for(field.name)
        for item_id in item_ids:
            output = outputs.get(item_id)
            if output is None:
                key = "no output row for the item"
                skipped[key] = skipped.get(key, 0) + 1
                continue
            if not label_is_judgeable(output, field.name):
                skipped["abstained — no label to check"] = (
                    skipped.get("abstained — no label to check", 0) + 1
                )
                continue
            request = task.build(items[item_id], output, field.name, taxonomy)
            calls += 1
            input_tokens += (len(request.system) + len(request.user)) // CHARS_PER_TOKEN
            if request.fingerprint in cached_keys:
                cached += 1

    output_tokens = (calls - cached) * max_tokens
    usd = estimate_cost(spec.model, input_tokens, output_tokens)
    job = PlannedJob(
        job="triage",
        judge_id=spec.id,
        model=spec.model,
        calls=calls,
        cached=cached,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usd=usd,
    )
    jobs = [job]
    if free_text:
        jobs.append(_claims_job(config, spec, items, outputs, free_text))
    unpriced = tuple(sorted({j.model for j in jobs if j.usd is None}))
    return Plan(
        jobs=tuple(jobs),
        skipped={k: v for k, v in skipped.items() if v},
        unpriced_models=unpriced,
    )


def _claims_job(
    config: RunConfig,
    spec: JudgeSpec,
    items: Mapping[str, Item],
    outputs: Mapping[str, Output],
    free_text: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]],
) -> PlannedJob:
    """The claim judge's share: flagged rows plus the audit sample.

    Counted separately from triage because the whole point of the free-text
    gate is that this number is a fraction of the corpus rather than all of
    it, and a combined total would hide that.
    """
    from .judges.prompts import CLAIM_TOKENS, ClaimSupportTask

    calls = input_tokens = 0
    budget = max(int(config.settings.value("judge_max_tokens")), CLAIM_TOKENS)
    for name, (suspicious, audit) in sorted(free_text.items()):
        field_spec = config.schema[name]
        task = ClaimSupportTask(
            style=field_spec.style.value if field_spec.style else "descriptive",
            max_tokens=budget,
        )
        for item_id in (*suspicious, *audit):
            output = outputs.get(item_id)
            if output is None or item_id not in items:
                continue
            request = task.build(items[item_id], output, name, None)
            calls += 1
            input_tokens += (len(request.system) + len(request.user)) // CHARS_PER_TOKEN
    output_tokens = calls * budget
    return PlannedJob(
        job="claims",
        judge_id=spec.id,
        model=spec.model,
        calls=calls,
        cached=0,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usd=estimate_cost(spec.model, input_tokens, output_tokens),
    )
