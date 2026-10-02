"""The orchestrator: collect once, analyse as often as you like.

The split in this module is the whole caching story. ``collect`` is the only
function that can cost money, and it takes items and outputs — never labels.
``analyse`` takes verdicts off disk and issues no calls at all, which is what
makes changing a threshold free.

Labels are loaded here, but they travel on their own path: they reach mode
detection and the report, and the function that calls judges has no parameter
to receive them through.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from .budget import BudgetGuard
from .cache import VerdictCache, cache_key
from .calibration import IdentityCalibrator
from .calibration.base import Calibrator
from .checks import CheckContext, run_checks
from .checks.corpus import label_shares
from .config import RunConfig, load_run
from .gates import Gates, gate_one, gate_two
from .judges.base import Judge, JudgeTask, Provider, ReplyOutcome
from .judges.panel import CHECK_ID as PANEL_CHECK
from .judges.panel import majority, sample_items
from .judges.prompts import LabelCorrectTask, label_is_judgeable
from .judges.providers import build_provider
from .judges.screening import JudgeHealth, screen, unreadable_items
from .metrics.agreement import effective_votes, fuzzy_pairs, leniency
from .plan import Plan, estimate_cost, plan_run
from .read import index_items, read_labels, read_outputs
from .report import render
from .schema import FieldKind, FieldSpec
from .taxonomy import Taxonomy
from .triage import TriageContext, build_risk_rows, resolve_strategy
from .types import Finding, Grain, Item, Label, Mode, Output, RiskRow, Status, Verdict
from .write import (
    append_index,
    finding_row,
    read_index,
    read_verdicts,
    risk_row,
    run_directory,
    verdict_row,
    write_json,
    write_jsonl,
)

__all__ = ["Dataset", "RunResult", "analyse_run", "detect_modes", "load_dataset", "run"]


@dataclass(frozen=True, slots=True)
class Dataset:
    """Everything read off disk, with labels kept to one side."""

    items: Mapping[str, Item]
    outputs: Mapping[str, Output]
    labels: tuple[Label, ...] = ()
    exclusions: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: str
    directory: Path
    verdicts: tuple[Verdict, ...]
    findings: tuple[Finding, ...]
    risk_rows: tuple[RiskRow, ...]
    health: tuple[JudgeHealth, ...]
    modes: Mapping[str, Mode]
    metrics: Mapping[str, object]
    report: str


def load_dataset(config: RunConfig) -> Dataset:
    """Read items, outputs and labels, logging every row that drops out.

    Dropping items is normal. Dropping them *because they scored badly* is how
    numbers get manufactured, and it is invisible unless every exclusion states
    a reason — so nothing is filtered silently here.
    """
    exclusions: dict[str, int] = {}
    items = {}
    for item_id, item in index_items(config.items).items():
        if not item.text.strip():
            exclusions["item text was empty"] = exclusions.get("item text was empty", 0) + 1
            continue
        items[item_id] = item

    outputs = {}
    for output in read_outputs(config.outputs):
        if output.item_id not in items:
            exclusions["output had no matching item"] = (
                exclusions.get("output had no matching item", 0) + 1
            )
            continue
        outputs[output.item_id] = output

    for item_id in items:
        if item_id not in outputs:
            exclusions["item had no output row"] = exclusions.get("item had no output row", 0) + 1

    labels = tuple(read_labels(config.labels)) if config.labels else ()
    return Dataset(items=items, outputs=outputs, labels=labels, exclusions=exclusions)


def detect_modes(config: RunConfig, labels: Sequence[Label]) -> dict[str, Mode]:
    """How much human labelling each field has, decided per field, every run.

    This reports the *shape of the data*, not whether a metric off it would
    mean anything. A field with twelve labels is in mode 1; the sample floors
    are what stop twelve rows becoming a macro F1.
    """
    annotators: dict[str, dict[str, set[str]]] = {}
    for label in labels:
        annotators.setdefault(label.field, {}).setdefault(label.item_id, set()).add(
            label.annotator
        )
    modes = {}
    for name in config.schema.fields:
        per_item = annotators.get(name, {})
        if not per_item:
            modes[name] = Mode.NO_LABELS
        elif max(len(a) for a in per_item.values()) >= 2:
            modes[name] = Mode.DOUBLE_LABELLED
        else:
            modes[name] = Mode.LABELLED
    return modes


def label_counts(config: RunConfig, labels: Sequence[Label]) -> dict[str, tuple[int, int]]:
    """Per field: how many items carry a label, and how many carry two.

    The second number is what stops ``MODE 2`` off four double-labelled items
    reading like a corpus that was labelled twice throughout.
    """
    annotators: dict[str, dict[str, set[str]]] = {}
    for label in labels:
        annotators.setdefault(label.field, {}).setdefault(label.item_id, set()).add(
            label.annotator
        )
    return {
        name: (
            len(annotators.get(name, {})),
            sum(1 for a in annotators.get(name, {}).values() if len(a) >= 2),
        )
        for name in config.schema.fields
    }


def _judge_for(config: RunConfig, judge_id: str, provider: Provider) -> Judge:
    spec = config.judges.judges[judge_id]
    temperature = spec.temperature
    if temperature is None:
        temperature = float(config.settings.value("judge_temperature"))
    return Judge(
        judge_id=spec.id,
        provider=provider,
        temperature=temperature,
        max_tokens=spec.max_tokens or int(config.settings.value("judge_max_tokens")),
    )


def _ask_cached(
    judge: Judge,
    model: str,
    task: JudgeTask,
    item: Item,
    output: Output,
    field: str,
    taxonomy: Taxonomy | None,
    *,
    cache: VerdictCache,
    guard: BudgetGuard,
) -> Verdict:
    """One question, asked at most once ever.

    The cache key is the question — judge, model, rendered prompt, item, field
    and value — and not the job that wanted the answer. A judge sitting on the
    panel *and* doing triage is asked the same thing twice by two callers and
    pays for it once.
    """
    request = judge.build(task, item, output, field, taxonomy)
    key = cache_key(
        judge_id=judge.judge_id,
        model=model,
        prompt_fingerprint=request.fingerprint,
        item_id=item.id,
        field=field,
        output_value=output.get(field),
    )
    hit = cache.get(key)
    if hit is not None:
        return hit
    if guard.exhausted:
        return _unscored(judge.judge_id, task.check_id, item.id, field, guard.cap_reason())
    _, verdict = judge.ask(task, item, output, field, taxonomy)
    stored = cache.put(key, verdict)
    guard.spend(
        estimate_cost(
            model,
            int(verdict.metadata.get("input_tokens", 0)),
            int(verdict.metadata.get("output_tokens", 0)),
        )
    )
    return stored


def _assigned_work(
    config: RunConfig,
    items: Mapping[str, Item],
    outputs: Mapping[str, Output],
    item_ids: Sequence[str],
) -> Iterator[tuple[FieldSpec, LabelCorrectTask, Taxonomy | None, Item, Output]]:
    """Every (field, item) an assigned-label judge could be asked about."""
    for field_spec in config.schema.fields.values():
        if field_spec.kind is not FieldKind.ASSIGNED:
            continue
        task = LabelCorrectTask(
            require_leaf=bool(config.settings.value("require_leaf", field_spec))
        )
        taxonomy = config.taxonomy_for(field_spec.name)
        for item_id in item_ids:
            output = outputs.get(item_id)
            if output is None or not label_is_judgeable(output, field_spec.name):
                continue
            yield field_spec, task, taxonomy, items[item_id], output


def collect(
    config: RunConfig,
    items: Mapping[str, Item],
    outputs: Mapping[str, Output],
    *,
    cache: VerdictCache,
    guard: BudgetGuard,
    providers: Mapping[str, Provider],
) -> list[Verdict]:
    """Ask the triage judge, and ask the panel.

    The signature is the label-leak guarantee: items and outputs go in, and
    there is no parameter a human label could arrive through. A judge cannot
    be graded against an answer key it was never handed.
    """
    verdicts: list[Verdict] = []
    triage, panel = config.judges.triage, config.judges.panel

    if triage is not None and triage.judge in providers:
        judge = _judge_for(config, triage.judge, providers[triage.judge])
        model = config.judges.judges[triage.judge].model
        scope = tuple(items)
        if isinstance(triage.scope, int):
            scope = scope[: triage.scope]
        for spec, task, taxonomy, item, output in _assigned_work(config, items, outputs, scope):
            verdicts.append(
                _ask_cached(
                    judge, model, task, item, output, spec.name, taxonomy,
                    cache=cache, guard=guard,
                )
            )

    if panel is not None:
        sample = sample_items(tuple(items), panel.sample)
        for member in panel.members:
            if member not in providers:
                continue
            judge = _judge_for(config, member, providers[member])
            model = config.judges.judges[member].model
            for spec, task, taxonomy, item, output in _assigned_work(
                config, items, outputs, sample
            ):
                verdict = _ask_cached(
                    judge, model, task, item, output, spec.name, taxonomy,
                    cache=cache, guard=guard,
                )
                # The triage pass may already have collected this exact
                # answer. One question, one row.
                if not any(
                    v.judge_id == member and v.item_id == item.id and v.field == spec.name
                    for v in verdicts
                ):
                    verdicts.append(verdict)
    return verdicts


def _unscored(judge_id: str, check_id: str, item_id: str, field: str, reason: str) -> Verdict:
    return Verdict(
        judge_id=judge_id,
        check_id=check_id,
        item_id=item_id,
        field=field,
        status=Status.UNSCORED,
        raw_confidence=None,
        reason=reason,
        metadata={"reply": ReplyOutcome.ERROR.value, "model": "", "cache_hit": False},
    )


def to_findings(
    run_id: str, config: RunConfig, dataset: Dataset, verdicts: Sequence[Verdict]
) -> list[Finding]:
    """Verdicts become findings — and so does everything that was not checked.

    A field nobody judged gets an unscored finding saying why. Leaving it out
    would let a run that checked one of three fields look complete.
    """
    # One finding per item and field, from the ranking judge only. A panel
    # member's verdict on the same row is the panel's business and has its
    # own check; pooling them here would make "label is correct" an average
    # over judges of different quality, with the rubber stamp pulling it up.
    triage = config.judges.triage
    ranker = triage.judge if triage is not None else None
    model = config.judges.judges[ranker].model if ranker else None
    scored = [v for v in verdicts if ranker is None or v.judge_id == ranker]
    judged = {(v.item_id, v.field) for v in scored}

    findings = [
        Finding(
            run_id=run_id,
            check=verdict.check_id,
            grain=Grain.FIELD,
            status=verdict.status,
            item_id=verdict.item_id,
            field=verdict.field,
            score=verdict.raw_confidence,
            threshold=None,
            threshold_from=None,
            evidence={
                "reason": verdict.reason,
                "reply": verdict.metadata.get("reply"),
                "raw_confidence": verdict.raw_confidence,
            },
            judge=verdict.judge_id,
            cost_usd=(
                estimate_cost(
                    str(model or verdict.metadata.get("model", "")),
                    int(verdict.metadata.get("input_tokens", 0)),
                    int(verdict.metadata.get("output_tokens", 0)),
                )
                or 0.0
            )
            if not verdict.metadata.get("cache_hit")
            else 0.0,
        )
        for verdict in scored
    ]

    for name, spec in config.schema.fields.items():
        why = (
            "the claim judge for free text arrives at M6"
            if spec.kind is FieldKind.FREE_TEXT
            else "the label was an abstention, so there is nothing to check it against"
        )
        for item_id in dataset.outputs:
            if (item_id, name) in judged:
                continue
            findings.append(
                Finding(
                    run_id=run_id,
                    check="label_correct" if spec.kind is FieldKind.ASSIGNED else "free_text",
                    grain=Grain.FIELD,
                    status=Status.UNSCORED,
                    item_id=item_id,
                    field=name,
                    evidence={"reason": why},
                )
            )
    return findings


def scoring_context(
    config: RunConfig, dataset: Dataset, verdicts: Sequence[Verdict], calibrator: Calibrator
) -> TriageContext:
    """The context Gate 2 scores every strategy on, baselines included.

    Deliberately *not* the context the queue is built from. The queue sees
    only the ranking judge, because letting panel opinions into it would make
    the panel a routing signal. But `panel_disagreement` has to be *scored* as
    a baseline, and it cannot be scored on inputs it was denied — so the gate
    gets every verdict, and every strategy competes on the same rows.

    The asymmetry is the point: disagreement is measured here and used
    nowhere. If it wins on your corpus, you select it deliberately.
    """
    by_item: dict[str, list[Verdict]] = {}
    for verdict in verdicts:
        by_item.setdefault(verdict.item_id, []).append(verdict)
    return TriageContext(
        verdicts={k: tuple(v) for k, v in by_item.items()},
        outputs=dataset.outputs,
        fields=tuple(config.schema.fields),
        calibrator=calibrator,
        item_ids=tuple(dataset.outputs),
        panel_members=tuple(config.judges.panel.members) if config.judges.panel else (),
    )


def build_triage(
    config: RunConfig,
    dataset: Dataset,
    verdicts: Sequence[Verdict],
    calibrator: Calibrator,
) -> tuple[tuple[RiskRow, ...], str]:
    """Rank on the triage judge's verdicts alone.

    Panel members' opinions are deliberately excluded. Disagreement does not
    rank errors, and letting extra judges leak into the ranking would mean the
    panel quietly became a routing signal — the first of the three things
    DESIGN.md §6 says it must not do. It ships as a scored baseline at M4.
    """
    triage = config.judges.triage
    ranker = triage.judge if triage is not None else None
    by_item: dict[str, list[Verdict]] = {}
    for verdict in verdicts:
        if ranker is not None and verdict.judge_id != ranker:
            continue
        by_item.setdefault(verdict.item_id, []).append(verdict)
    ctx = TriageContext(
        verdicts={k: tuple(v) for k, v in by_item.items()},
        outputs=dataset.outputs,
        fields=tuple(config.schema.fields),
        calibrator=calibrator,
        item_ids=tuple(dataset.outputs),
    )
    requested = config.judges.triage.strategy if config.judges.triage else "auto"
    strategy = resolve_strategy(requested, calibrated=calibrator.is_calibrated)
    return build_risk_rows(ctx, strategy.rank(ctx), strategy.id), strategy.id


def panel_analysis(
    config: RunConfig, dataset: Dataset, verdicts: Sequence[Verdict], health: Sequence[JudgeHealth]
) -> tuple[list[Finding], dict[str, Any]]:
    """Majority findings and the panel's three outputs.

    A judge outside the approval band is dropped from the aggregates and from
    the vote, and the report says which and why. Its verdicts stay on disk —
    suppressing them would hide the evidence for the exclusion.
    """
    panel = config.judges.panel
    if panel is None:
        return [], {}

    excluded = {row.judge_id for row in health if row.excluded_from_panel}
    members = tuple(m for m in panel.members if m not in excluded)
    sample = set(sample_items(tuple(dataset.items), panel.sample))
    in_panel = [
        v for v in verdicts if v.judge_id in set(panel.members) and v.item_id in sample
    ]

    by_key: dict[tuple[str, str], list[Verdict]] = {}
    for verdict in in_panel:
        by_key.setdefault((verdict.item_id, verdict.field), []).append(verdict)

    findings: list[Finding] = []
    results = []
    for (item_id, field_name), group in sorted(by_key.items()):
        result = majority(group, members=members)
        results.append(result)
        findings.append(
            Finding(
                run_id=config.run_id,
                check=PANEL_CHECK,
                grain=Grain.FIELD,
                status=result.status,
                item_id=item_id,
                field=field_name,
                evidence={
                    "votes": dict(result.votes),
                    "agreement": result.agreement,
                    "dissent": dict(result.dissent),
                    **({"instead": dict(result.instead)} if result.instead else {}),
                    **(
                        {"excluded": sorted(excluded)}
                        if excluded
                        else {}
                    ),
                },
                judge=",".join(members),
            )
        )

    scored = [r for r in results if r.status is not Status.UNSCORED]
    report = effective_votes(
        in_panel, members, warn_below=float(config.settings.value("effective_votes_warn"))
    )
    # A suggestion is only a boundary if it is a label of *that* field. A
    # judge naming something from another taxonomy is a health problem, not a
    # fuzzy pair.
    vocabularies = {
        name: frozenset(taxonomy.nodes)
        for name in config.schema.fields
        if (taxonomy := config.taxonomy_for(name)) is not None
    }
    fuzzy = fuzzy_pairs(
        in_panel,
        members,
        dataset.outputs,
        vocabularies=vocabularies,
        top=int(config.settings.value("fuzzy_pair_report")),
    )
    return findings, {
        "members": list(members),
        "excluded": sorted(excluded),
        "sampled_items": len(sample),
        "judged_rows": len(by_key),
        "majority_correct": (
            sum(r.status is Status.PASS for r in scored) / len(scored) if scored else None
        ),
        "undecided": len(results) - len(scored),
        "agreement": report.agreement,
        "mean_correlation": report.mean_correlation,
        "effective_votes": report.effective_votes,
        "votes_paid_for": report.votes_paid_for,
        "warning": report.warning,
        "leniency": [
            {
                "lenient": entry.lenient,
                "strict": entry.strict,
                "lenient_approved": entry.lenient_approved,
                "strict_approved": entry.strict_approved,
            }
            for entry in leniency(in_panel, members)
        ],
        "splits": fuzzy.splits,
        "splits_attributed": fuzzy.attributed,
        "splits_unusable": fuzzy.unusable,
        "fuzzy_pairs": [
            {"field": p.field, "left": p.left, "right": p.right, "splits": p.splits}
            for p in fuzzy.pairs
        ],
    }


def grain_rates(findings: Sequence[Finding]) -> dict[str, Any]:
    """Pass rates at both grains. The item one is the one a consumer needs.

    An item with a correct label and a made-up summary is not a usable item,
    so an item passes only if every scored finding on it passes. Unscored
    findings are counted separately and never absorbed into either rate —
    that is the whole point of having a third state.
    """
    scored = [f for f in findings if f.status is not Status.UNSCORED]
    by_item: dict[str, list[bool]] = {}
    for found in scored:
        if found.item_id is not None:
            by_item.setdefault(found.item_id, []).append(found.status is Status.PASS)
    field_rate = (
        sum(f.status is Status.PASS for f in scored) / len(scored) if scored else None
    )
    item_rate = (
        sum(all(results) for results in by_item.values()) / len(by_item) if by_item else None
    )
    return {
        "field": {
            "pass_rate": field_rate,
            "scored": len(scored),
            "unscored": len(findings) - len(scored),
        },
        "item": {
            "pass_rate": item_rate,
            "items": len(by_item),
            "failing": sum(not all(r) for r in by_item.values()),
        },
    }


def previous_run(out: Path, config: RunConfig) -> dict[str, Any]:
    """The last run's distributions, for the drift check.

    Read from the append-only index rather than from a directory listing, so
    "the previous run" means the one that actually finished last.
    """
    entries = [e for e in read_index(out) if e.get("distributions")]
    if not entries:
        return {}
    latest = dict(entries[-1])
    latest["current_prompt"] = (
        config.produced_by.prompt_version if config.produced_by else None
    )
    return latest


def what_this_run_cannot_tell_you(
    config: RunConfig,
    modes: Mapping[str, Mode],
    health: Sequence[JudgeHealth],
    gates: Gates,
) -> list[str]:
    lines = [
        "how many errors a review budget would actually find\n"
        "Error Recall@Budget needs a fitted calibration and labelled rows; M5b",
    ]
    if gates.two.skipped:
        lines.insert(0, f"whether this ranking is any better than guessing\n{gates.two.skipped}")
    withheld = gates.suppressed()
    if withheld:
        names = ", ".join(sorted(s.metric for s in withheld))
        lines.append(
            f"{names}\n"
            "withheld — a guardrail above says these cannot be computed honestly on this "
            "run, and a number that cannot be computed honestly is not reported"
        )
    free_text = [n for n, s in config.schema.fields.items() if s.kind is FieldKind.FREE_TEXT]
    if free_text:
        lines.append(
            f"whether {', '.join(free_text)} invented anything, or is filler\n"
            "cross-field agreement is the only check reading these fields today.\n"
            "specificity, copy ratio, boilerplate and the claim judge arrive at M6"
        )
    if any(mode >= Mode.LABELLED for mode in modes.values()):
        labelled = sorted(n for n, m in modes.items() if m >= Mode.LABELLED)
        lines.append(
            f"how often the judge agrees with the humans on {', '.join(labelled)}\n"
            "labels are loaded and untouched — grading the judge against them is M5"
        )
    if config.judges.panel is None:
        lines.append(
            "how good this corpus is overall, or which label pairs are fuzzy\n"
            "no panel is wired in judges.yml. A panel of two or more over a sample "
            "answers both, and locating a fuzzy boundary needs nothing but the split."
        )
    return lines


def run(
    config: RunConfig,
    *,
    out: Path,
    provider_factory: Callable[..., Provider] | None = None,
    stream: TextIO | None = None,
    ask: Callable[[str], str] | None = None,
    reuse: Path | None = None,
    now: datetime | None = None,
) -> RunResult | None:
    """Collect and analyse. Returns None if the cost prompt was declined."""
    import sys

    # Resolved here, not in the signature. A default argument binds at
    # definition time, so `provider_factory=build_provider` in the signature
    # would capture the original function and quietly ignore anyone who
    # replaced it — including a test that thought it had.
    provider_factory = provider_factory or build_provider
    stream = stream or sys.stdout
    started = time.perf_counter()
    dataset = load_dataset(config)
    earlier = previous_run(out, config)

    directory = run_directory(out, config.run_id, now=now)
    cache = VerdictCache(directory / "verdicts.jsonl")
    cache.open()
    if reuse is not None:
        # Seeded, not copied: this run's verdicts.jsonl holds what this run
        # actually asked, and the reused ones are reported as cache hits.
        cache.warm_from(reuse / "verdicts.jsonl")

    plan = plan_run(config, dataset.items, dataset.outputs)
    guard = BudgetGuard(max_usd=config.budget.max_usd, confirm=config.budget.confirm)
    if not guard.approve(plan, stream=stream, ask=ask):
        cache.close()
        stream.write("nothing was spent.\n")
        return None

    wanted = set()
    if config.judges.triage is not None:
        wanted.add(config.judges.triage.judge)
    if config.judges.panel is not None:
        wanted.update(config.judges.panel.members)
    providers = {
        judge_id: provider_factory(config.judges.judges[judge_id]) for judge_id in sorted(wanted)
    }
    verdicts = (
        collect(
            config, dataset.items, dataset.outputs,
            cache=cache, guard=guard, providers=providers,
        )
        if providers
        else []
    )
    cache.close()

    result = _analyse(
        config,
        dataset,
        verdicts,
        directory=directory,
        plan=plan,
        elapsed=time.perf_counter() - started,
        collected=[v for v in verdicts if not v.metadata.get("cache_hit")],
        run_yml=config.source,
        previous=earlier,
    )
    append_index(
        out,
        {
            "run_id": result.run_id,
            "prompt": config.produced_by.prompt_version if config.produced_by else None,
            "taxonomies": {str(t.ref): t.content_hash for t in config.taxonomies.values()},
            "items": len(dataset.items),
            "calls": result.metrics.get("calls"),
            "cost_usd": result.metrics.get("cost_usd"),
            # What the next run's drift check reads. Written here rather than
            # recomputed from that run's files, so "the previous run" means
            # the one that finished last and not the one sorted last.
            "distributions": result.metrics.get("distributions"),
        },
    )
    return result


def analyse_run(directory: Path, *, stream: TextIO | None = None) -> RunResult:
    """Re-analyse a finished run from its cache. Issues zero model calls.

    Everything downstream of the judge is recomputed: thresholds, findings,
    the ranking, the report. Nothing in this path constructs a provider.
    """
    import json

    directory = Path(directory)
    manifest = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    config = load_run(manifest["run_yml"])
    dataset = load_dataset(config)
    verdicts = read_verdicts(directory / "verdicts.jsonl")
    # The entry before this run's own, so a re-analysis compares against the
    # same baseline the original run did rather than against itself.
    entries = [e for e in read_index(directory.parent) if e.get("distributions")]
    earlier: dict[str, Any] = next(
        (e for e in reversed(entries) if e.get("run_id") != directory.name), {}
    )
    # Nothing was collected, so nothing was spent. The verdicts on disk record
    # how they were *obtained* during collection, which is a different
    # question from what this invocation cost.
    return _analyse(
        config,
        dataset,
        verdicts,
        directory=directory,
        plan=None,
        elapsed=0.0,
        collected=(),
        run_yml=Path(manifest["run_yml"]),
        previous=earlier,
    )


def _analyse(
    config: RunConfig,
    dataset: Dataset,
    verdicts: Sequence[Verdict],
    *,
    directory: Path,
    plan: Plan | None,
    elapsed: float,
    collected: Sequence[Verdict],
    run_yml: Path,
    previous: Mapping[str, Any] | None = None,
) -> RunResult:
    run_id = directory.name
    modes = detect_modes(config, dataset.labels)
    counts = label_counts(config, dataset.labels)
    health = screen(verdicts, config.settings)

    check_ctx = CheckContext(
        run_id=run_id,
        schema=config.schema,
        settings=config.settings,
        items=dataset.items,
        outputs=dataset.outputs,
        taxonomies=config.taxonomies,
        previous=previous or {},
    )
    free_findings, skipped = run_checks(check_ctx)
    panel_findings, panel = panel_analysis(config, dataset, verdicts, health)
    findings = [
        *free_findings,
        *to_findings(run_id, config, dataset, verdicts),
        *panel_findings,
    ]
    calibrator: Calibrator = IdentityCalibrator()
    risk_rows, strategy_id = build_triage(config, dataset, verdicts, calibrator)

    gate1 = gate_one(
        config, judge_health=health, items=len(dataset.items), verdicts=verdicts, panel=panel
    )
    gate2 = gate_two(
        config,
        scoring_context(config, dataset, verdicts, calibrator),
        strategy_id,
        dataset.labels,
        dataset.outputs,
    )
    gates = Gates(
        one=gate1,
        two=gate2.gate,
        table=gate2.scores,
        target_size=gate2.target_size,
        target_errors=gate2.target_errors,
    )

    # A model with no published price produces no cost, not a zero. Reporting
    # $0.00 for a run against an unpriced judge would be a made-up number.
    priced = plan is None or not plan.unpriced_models
    spent = sum(
        f.cost_usd
        for f in findings
        if (f.item_id, f.field) in {(v.item_id, v.field) for v in collected}
    )
    metrics: dict[str, object] = {
        "items": len(dataset.items),
        # `calls` and `cost_usd` are what *this invocation* spent. A
        # re-analysis spends nothing; what the original run cost is preserved
        # in the append-only index.
        "calls": len(collected),
        "cost_usd": round(spent, 6) if priced else None,
        "verdicts": len(verdicts),
        "cache_hits": len(verdicts) - len(collected),
        "elapsed_s": round(elapsed, 2),
        "strategy": strategy_id,
        "ranked_by": config.judges.triage.judge if config.judges.triage else None,
        "calibrated": calibrator.is_calibrated,
        "modes": {name: int(mode) for name, mode in modes.items()},
        "label_counts": counts,
        "judges": [
            {
                "judge": row.judge_id,
                "model": row.model,
                "calls": row.calls,
                "approval_rate": row.approval_rate,
                "cannot_decide_rate": row.cannot_decide_rate,
                "unreadable_rate": row.unreadable_rate,
                "excluded_from_panel": row.excluded_from_panel,
                "flags": [[s.value, m] for s, m in row.flags],
            }
            for row in health
        ],
        "unreadable": unreadable_items(verdicts, dataset.items),
        "exclusions": dict(dataset.exclusions),
        "ranked": sum(1 for r in risk_rows if r.triage_score is not None),
        "unranked": sum(1 for r in risk_rows if r.triage_score is None),
        "grains": grain_rates(findings),
        "distributions": {
            name: label_shares(check_ctx, name)
            for name, spec in config.schema.fields.items()
            if spec.kind is FieldKind.ASSIGNED
        },
        "checks_not_run": [
            {"check": s.check, "field": s.field, "reason": s.reason} for s in skipped
        ],
        "panel": panel,
        "gates": {
            "measurement_sound": {
                "status": gates.one.status,
                "results": [
                    {"id": r.id, "severity": r.severity.value, "message": r.message}
                    for r in gates.one.results
                ],
            },
            "beats_baselines": {
                "status": gates.two.status,
                "skipped": gates.two.skipped,
                "results": [
                    {"id": r.id, "severity": r.severity.value, "message": r.message}
                    for r in gates.two.results
                ],
                "target_items": gates.target_size,
                "target_errors": gates.target_errors,
                "table": [
                    {
                        "strategy": s.strategy,
                        "baseline": s.is_baseline,
                        "auc": s.estimate.value,
                        "ci_low": s.estimate.low,
                        "ci_high": s.estimate.high,
                        "n": s.estimate.n,
                        "under_floor": s.estimate.under_floor,
                        "unranked": s.unranked,
                        "unranked_errors": s.unranked_errors,
                    }
                    for s in gates.table
                ],
            },
        },
        "suppressed": [
            {"metric": s.metric, "reason": s.reason} for s in gates.suppressed()
        ],
    }

    cannot = what_this_run_cannot_tell_you(config, modes, health, gates)
    report = render(
        run_id=run_id,
        schema=config.schema,
        taxonomies=config.taxonomies,
        modes=modes,
        label_counts=counts,
        health=health,
        findings=findings,
        skipped=skipped,
        panel=panel,
        gates=gates,
        risk_rows=risk_rows,
        verdicts=verdicts,
        items=dataset.items,
        metrics=metrics,
        cannot_tell=cannot,
        exclusions=dataset.exclusions,
    )

    write_jsonl(directory / "findings.jsonl", (finding_row(f) for f in findings))
    write_jsonl(directory / "risk.jsonl", (risk_row(r) for r in risk_rows))
    write_json(directory / "metrics.json", metrics)
    write_json(
        directory / "run.json",
        {
            "run_id": run_id,
            "run_yml": str(run_yml.resolve()),
            "produced_by": {
                "model": config.produced_by.model if config.produced_by else None,
                "prompt_version": config.produced_by.prompt_version if config.produced_by else None,
            },
            "taxonomies": {
                str(t.ref): t.content_hash for t in config.taxonomies.values()
            },
            "modes": {name: int(mode) for name, mode in modes.items()},
            "settings_used": {
                "judge_temperature": config.settings.resolve("judge_temperature").value,
                "judge_max_tokens": config.settings.resolve("judge_max_tokens").value,
                "judge_approval_warn": config.settings.resolve("judge_approval_warn").value,
                "parse_failure_warn": config.settings.resolve("parse_failure_warn").value,
            },
            "cannot_tell": cannot,
            "metrics": metrics,
        },
    )
    (directory / "report.md").write_text(report, encoding="utf-8")

    # Verdicts are already on disk from the cache, but a re-analysed run that
    # was warmed from elsewhere should still carry its own copy.
    if not (directory / "verdicts.jsonl").exists():
        write_jsonl(directory / "verdicts.jsonl", (verdict_row(v) for v in verdicts))

    return RunResult(
        run_id=run_id,
        directory=directory,
        verdicts=tuple(verdicts),
        findings=tuple(findings),
        risk_rows=risk_rows,
        health=health,
        modes=modes,
        metrics=metrics,
        report=report,
    )
