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

import dataclasses
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from .budget import BudgetGuard
from .cache import VerdictCache, cache_key
from .calibration import IdentityCalibrator
from .calibration.base import Calibrator, StaleCalibration
from .calibration.base import fingerprint as calibration_fingerprint
from .calibration.platt import FieldCalibrator, PlattCalibrator
from .calibration.quality import reliability
from .checks import CheckContext, run_checks
from .checks.corpus import label_shares
from .config import JudgeSpec, RunConfig, load_run
from .gates import Gates, gate_one, gate_two
from .judges.base import Judge, JudgeTask, Provider, ReplyOutcome
from .judges.panel import CHECK_ID as PANEL_CHECK
from .judges.panel import majority, sample_items
from .judges.prompts import (
    CLAIM_TOKENS,
    ClaimSupportTask,
    LabelCorrectTask,
    free_text_is_judgeable,
    label_is_judgeable,
)
from .judges.providers import build_provider
from .judges.screening import JudgeHealth, screen, unreadable_items
from .metrics.agreement import annotator_agreement, effective_votes, fuzzy_pairs, leniency
from .metrics.classification import (
    TreeBucket,
    bucket,
    classify,
    confusion_direction,
    judge_direction,
    primary_labels,
)
from .plan import Plan, estimate_cost, plan_run
from .read import index_items, read_labels, read_outputs
from .report import render
from .schema import FieldKind, FieldSpec
from .stability import (
    StabilityReport,
    measure_stability,
    regenerate_sample,
    write_regenerated,
)
from .taxonomy import Taxonomy
from .triage import TriageContext, build_risk_rows, resolve_strategy
from .triage.evaluate import compare
from .triage.strategies import BASELINES, STRATEGIES
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


def check_context(
    config: RunConfig, dataset: Dataset, run_id: str, previous: Mapping[str, Any] | None = None
) -> CheckContext:
    return CheckContext(
        run_id=run_id,
        schema=config.schema,
        settings=config.settings,
        items=dataset.items,
        outputs=dataset.outputs,
        taxonomies=config.taxonomies,
        previous=previous or {},
    )


def free_text_plan(
    config: RunConfig,
    items: Mapping[str, Item],
    outputs: Mapping[str, Output],
    flagged: Mapping[str, set[str]],
) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Which free-text rows to judge: the suspicious ones, plus an audit.

    This is the part that makes free text cheaper than assigned, and it is
    the opposite of what you would guess. The assigned free checks catch
    *format* problems, so a perfectly formed wrong label sails through and
    the triage judge has to see everything. The free-text checks catch
    *content* problems — no specific detail, contradicts the label — and
    those genuinely predict invention, so they gate the expensive call.

    The audit sample is why the gate is measured rather than trusted. Judging
    a slice of the rows nothing flagged is the only way to find out how much
    the free checks are missing, and without it "the gate is predictive"
    would be a claim this library makes about itself without evidence.
    """
    rate = float(config.settings.value("free_text_audit_rate"))
    plan: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for name, spec in config.schema.fields.items():
        if spec.kind is not FieldKind.FREE_TEXT:
            continue
        eligible = [
            i
            for i in items
            if (row := outputs.get(i)) is not None and free_text_is_judgeable(row, name)
        ]
        suspicious = tuple(sorted(i for i in eligible if name in flagged.get(i, set())))
        rest = [i for i in eligible if i not in set(suspicious)]
        # At least one, whenever an audit was asked for and there is anything
        # to audit. A rate that rounds to zero on a small corpus silently
        # turns the measurement off, and the gate goes back to being trusted
        # rather than checked — which is the thing the audit exists to stop.
        wanted = round(rate * len(rest))
        if rate > 0 and rest:
            wanted = max(1, wanted)
        audit = sample_items(tuple(rest), wanted, seed=f"audit:{name}")
        plan[name] = (suspicious, tuple(sorted(audit)))
    return plan


def flagged_by_free_checks(findings: Sequence[Finding]) -> dict[str, set[str]]:
    """Which (item, field) pairs a free check already found fault with."""
    gates = {"specificity", "copy_ratio", "length_in_bounds", "cross_field_agreement"}
    out: dict[str, set[str]] = {}
    for found in findings:
        if found.check in gates and found.status is Status.FAIL and found.item_id:
            out.setdefault(found.item_id, set()).add(str(found.field))
    return out


def collect(
    config: RunConfig,
    items: Mapping[str, Item],
    outputs: Mapping[str, Output],
    *,
    cache: VerdictCache,
    guard: BudgetGuard,
    providers: Mapping[str, Provider],
    free_text: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
) -> list[Verdict]:
    """Ask the triage judge, the panel, and the claim judge.

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

    if triage is not None and free_text and triage.judge in providers:
        judge = _judge_for(config, triage.judge, providers[triage.judge])
        model = config.judges.judges[triage.judge].model
        for name, (suspicious, audit) in sorted(free_text.items()):
            text_spec = config.schema[name]
            claims = ClaimSupportTask(
                style=text_spec.style.value if text_spec.style else "descriptive",
                max_tokens=max(int(config.settings.value("judge_max_tokens")), CLAIM_TOKENS),
            )
            for item_id in (*suspicious, *audit):
                written = outputs.get(item_id)
                if written is None:
                    continue
                verdict = _ask_cached(
                    judge, model, claims, items[item_id], written, name, None,
                    cache=cache, guard=guard,
                )
                verdicts.append(
                    dataclasses.replace(
                        verdict,
                        metadata={
                            **verdict.metadata,
                            "selected_by": "free check" if item_id in set(suspicious) else "audit",
                        },
                    )
                )
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
    # Label verdicts only. Claim verdicts have their own builder, because
    # the pass/fail there is a threshold applied to a support rate rather
    # than the judge's own verdict — and emitting both would count every
    # judged free-text row twice, in the findings and in the grain rates.
    scored = [
        v
        for v in verdicts
        if v.check_id == "label_correct" and (ranker is None or v.judge_id == ranker)
    ]
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
            "nothing here was flagged by a free check and it was not in the audit "
            "sample, so no claim judge was paid for it"
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

    if len(members) < 2:
        # Config refuses a configured panel of one because every number a
        # panel exists for is undefined there. Exclusions can produce the
        # same state at runtime, and it is the same non-answer: one judge's
        # verdict relabelled as a consensus would be the worst of both.
        return [], {
            "members": list(members),
            "excluded": sorted(excluded),
            "sampled_items": len(set(sample_items(tuple(dataset.items), panel.sample))),
            "collapsed": (
                f"only {len(members)} of {len(panel.members)} judges survived screening. "
                "A panel needs two to have an opinion about — no agreement, no dissent "
                "and no effective vote count can be computed, so none are reported."
            ),
        }
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


def human_findings(run_id: str, config: RunConfig, dataset: Dataset) -> list[Finding]:
    """One row per labelled item: was it right, and if not, what kind of wrong.

    The aggregates say how often; this says which, and puts the bucket on the
    row so a reviewer opening the item knows whether they are looking at a
    sibling, a hedge or a misread before they start.
    """
    advice = {
        TreeBucket.RIGHT_PARENT: "a sibling — the boundary between two definitions",
        TreeBucket.TOO_SHALLOW: "stopped at a parent — hedging, not misreading",
        TreeBucket.WRONG: "a different branch — the model is not reading the item",
        TreeBucket.UNKNOWN: "not a label in this taxonomy at all",
        TreeBucket.ABSTAINED: "declined to pick a label",
    }
    findings: list[Finding] = []
    for name, spec in config.schema.fields.items():
        if spec.kind is not FieldKind.ASSIGNED:
            continue
        taxonomy = config.taxonomy_for(name)
        if taxonomy is None:
            continue
        for item_id, answer in primary_labels(dataset.labels, name).items():
            output = dataset.outputs.get(item_id)
            if output is None:
                continue
            placed = bucket(output.get(name), answer, taxonomy)
            findings.append(
                Finding(
                    run_id=run_id,
                    check="label_tree_bucket",
                    grain=Grain.FIELD,
                    # An abstention is not a wrong answer. It has its own
                    # check, and failing it here would count a taxonomy gap
                    # as a model error.
                    status=(
                        Status.PASS
                        if placed is TreeBucket.EXACT
                        else Status.UNSCORED
                        if placed is TreeBucket.ABSTAINED
                        else Status.FAIL
                    ),
                    item_id=item_id,
                    field=name,
                    evidence={
                        "bucket": placed.value,
                        "human": answer,
                        "assigned": output.get(name),
                        "why": advice.get(placed, ""),
                    },
                )
            )
    return findings


def against_humans(
    config: RunConfig, dataset: Dataset, verdicts: Sequence[Verdict], modes: Mapping[str, Mode]
) -> dict[str, Any]:
    """Everything the human answers unlock, per field — mode 1 and mode 2.

    Labels reach this function and no earlier. Downstream of the judge they
    are used for exactly four things, and three of them are here: scoring the
    model, scoring the judge, and asking whether the taxonomy is crisp. The
    fourth, fitting a calibration, arrives at M5b.
    """
    out: dict[str, Any] = {}
    floor = int(config.settings.value("min_n_per_label"))
    top_pairs = int(config.settings.value("fuzzy_pair_report"))

    for name, spec in config.schema.fields.items():
        if spec.kind is not FieldKind.ASSIGNED or modes.get(name, Mode.NO_LABELS) < Mode.LABELLED:
            continue
        taxonomy = config.taxonomy_for(name)
        if taxonomy is None:
            continue
        scored = classify(
            name, dataset.outputs, dataset.labels, taxonomy,
            label_floor=floor, top_confusions=max(top_pairs * 2, 10),
        )
        entry: dict[str, Any] = {
            "n": scored.n,
            "macro_f1": scored.macro_f1,
            "accuracy": scored.accuracy,
            "majority_baseline": scored.majority_baseline,
            "beats_majority": scored.beats_majority,
            "buckets": {k.value: v for k, v in scored.buckets.items()},
            "labels": [
                {
                    "label": row.label,
                    "support": row.support,
                    "precision": row.precision,
                    "recall": row.recall,
                    "f1": row.f1,
                    "under_floor": row.under_floor,
                    "weak": row.weak,
                }
                for row in scored.scores
            ],
            "confusion": [
                {"truth": c.truth, "predicted": c.predicted, "n": c.n} for c in scored.confusion
            ],
            "confusion_direction": confusion_direction(scored.confusion, top=top_pairs),
            "weak_labels": [row.label for row in scored.weak_labels],
        }

        if modes.get(name) is Mode.DOUBLE_LABELLED:
            humans = annotator_agreement(dataset.labels, name, top=top_pairs)
            entry["annotators"] = {
                "compared": humans.compared,
                "agreement": humans.agreement,
                "disagreements": humans.disagreements,
                "concentration": humans.concentration,
                "who": list(humans.annotators),
                "pairs": [
                    {"left": a, "right": b, "n": n} for a, b, n in humans.pairs
                ],
            }
        out[name] = entry

    fields = tuple(
        name
        for name, spec in config.schema.fields.items()
        if spec.kind is FieldKind.ASSIGNED and modes.get(name, Mode.NO_LABELS) >= Mode.LABELLED
    )
    directions = judge_direction(verdicts, dataset.outputs, dataset.labels, fields)
    judges: list[dict[str, Any]] = []
    for row in directions:
        stated, happened = _confidence_pairs(verdicts, row.judge_id, dataset, fields)
        curve = reliability(stated, happened)
        judges.append(
            {
                "judge": row.judge_id,
                "scored": row.scored,
                "accuracy": row.accuracy,
                "always_approve_accuracy": row.always_approve_accuracy,
                "beats_always_approve": (
                    None
                    if row.accuracy is None or row.always_approve_accuracy is None
                    else row.accuracy > row.always_approve_accuracy
                ),
                "approves_wrong": row.approves_wrong_rate,
                "rejects_right": row.rejects_right_rate,
                "leaning": row.leaning,
                "confidence": {
                    "n": curve.n,
                    "ece": curve.ece,
                    "brier": curve.brier,
                    "brier_base_rate": curve.brier_base_rate,
                    "beats_base_rate": curve.beats_base_rate,
                    "broken": curve.broken,
                    "direction": curve.direction,
                    "bins": [
                        {
                            "low": b.low,
                            "high": b.high,
                            "n": b.n,
                            "stated": b.stated,
                            "observed": b.observed,
                        }
                        for b in curve.bins
                        if b.reportable
                    ],
                },
            }
        )
    return {"fields": out, "judges": judges}


def _confidence_pairs(
    verdicts: Sequence[Verdict],
    judge_id: str,
    dataset: Dataset,
    fields: Sequence[str],
) -> tuple[list[float], list[bool]]:
    """What the judge claimed, and whether it turned out to be so.

    "Confidence" is read as confidence *in the judge's own verdict*, so a
    confident rejection that was right counts as a hit. Reading it as
    confidence that the label is correct would score every honest rejection
    as a miss.
    """
    from .metrics.classification import primary_labels

    truth: dict[tuple[str, str], str] = {}
    for name in fields:
        for item_id, answer in primary_labels(dataset.labels, name).items():
            truth[(item_id, name)] = answer

    stated: list[float] = []
    happened: list[bool] = []
    for verdict in verdicts:
        if verdict.judge_id != judge_id or verdict.status is Status.UNSCORED:
            continue
        if verdict.raw_confidence is None:
            continue
        human = truth.get((verdict.item_id, verdict.field))
        output = dataset.outputs.get(verdict.item_id)
        if human is None or output is None:
            continue
        really_right = str(output.get(verdict.field)) == human
        stated.append(verdict.raw_confidence)
        happened.append((verdict.status is Status.PASS) == really_right)
    return stated, happened


def run_fingerprint(config: RunConfig) -> str:
    """What a calibration fitted on this run would only ever be valid for.

    Judge, model, prompt and taxonomy version. A calibration fitted on
    prompt p7 tells you nothing about p8 — the judge is answering a
    different question — and using it anyway is how a number keeps looking
    trustworthy after the thing under it moved.
    """
    triage = config.judges.triage
    judge_id = triage.judge if triage else ""
    model = config.judges.judges[judge_id].model if judge_id else ""
    taxonomies = ",".join(sorted(str(t.ref) for t in config.taxonomies.values()))
    prompt = config.produced_by.prompt_version if config.produced_by else ""
    return calibration_fingerprint(
        judge_id=judge_id,
        model=model,
        prompt_hash=prompt,
        check_id="label_correct",
        taxonomy_ref=taxonomies,
    )


def check_calibration_age(stored: str | None, current: str, *, where: str) -> None:
    """Refuse a calibration fitted against something that has since moved."""
    if stored is None or stored == current:
        return
    raise StaleCalibration(
        f"the calibration in {where} was fitted against a different judge, model, "
        f"prompt version or taxonomy ({stored} vs {current} now).\n"
        "  Refit it, or run uncalibrated and say so. A curve fitted on one question "
        "does not describe the answers to another."
    )


def fit_calibration(
    config: RunConfig, dataset: Dataset, verdicts: Sequence[Verdict], modes: Mapping[str, Mode]
) -> tuple[Calibrator, dict[str, Any]]:
    """Fit one curve per labelled assigned field, plus a shared fallback.

    Returns the calibrator the run will rank with and a report of what was
    fitted. When nothing could be fitted the identity passthrough comes
    back, and every number downstream of it stays stamped uncalibrated.
    """
    from .metrics.classification import primary_labels

    triage = config.judges.triage
    if triage is None:
        return IdentityCalibrator(), {}

    identity = IdentityCalibrator()
    rows: dict[str, list[tuple[float, bool]]] = {}
    for verdict in verdicts:
        if verdict.judge_id != triage.judge:
            continue
        score = identity.error_probability(verdict)
        if score is None:
            continue
        truth = primary_labels(dataset.labels, verdict.field).get(verdict.item_id)
        output = dataset.outputs.get(verdict.item_id)
        if truth is None or output is None:
            continue
        rows.setdefault(verdict.field, []).append(
            (score, str(output.get(verdict.field)) != truth)
        )

    stamp = run_fingerprint(config)
    report: dict[str, Any] = {"fingerprint": stamp, "fields": {}, "fitted": False}
    calibrator = FieldCalibrator(fingerprint=stamp)

    pooled: list[tuple[float, bool]] = []
    for name, pairs in sorted(rows.items()):
        pooled.extend(pairs)
        if modes.get(name, Mode.NO_LABELS) < Mode.LABELLED:
            continue
        fitted = PlattCalibrator()
        fit = fitted.fit_from_rows(
            [s for s, _ in pairs], [e for _, e in pairs], fingerprint=stamp
        )
        calibrator.per_field[name] = fitted
        report["fields"][name] = fit.to_dict()

    if pooled and not any(c.is_calibrated for c in calibrator.per_field.values()):
        fallback = PlattCalibrator()
        fit = fallback.fit_from_rows(
            [s for s, _ in pooled], [e for _, e in pooled], fingerprint=stamp
        )
        calibrator.fallback = fallback
        report["fields"]["(all fields pooled)"] = fit.to_dict()

    if not calibrator.is_calibrated:
        report["why"] = (
            "nothing was fitted, so the ranking is by raw judge confidence and is "
            "stamped uncalibrated everywhere it appears."
        )
        return IdentityCalibrator(), report

    report["fitted"] = True
    report["reorders"] = calibrator.reorders
    report["model"] = calibrator.to_dict()
    if not calibrator.reorders:
        report["why"] = (
            "one curve covers everything, and a Platt curve is monotone — the review "
            "order is identical to the uncalibrated one. What the calibration buys is "
            "probabilities that mean what they say, not a better ordering. Two or more "
            "fields each with enough labelled rows would let it reorder."
        )
    usable: Calibrator = calibrator
    return usable, report


def triage_evaluation(
    config: RunConfig,
    ctx: TriageContext,
    strategy_id: str,
    labels: Sequence[Label],
    outputs: Mapping[str, Output],
) -> dict[str, Any]:
    """Error Recall@Budget for every strategy, on one target.

    The table that settles whether a judge is worth paying for at the budget
    you actually review at — and whether panel disagreement beats it on your
    corpus rather than on somebody else's.
    """
    from .gates import build_target
    from .metrics.ranking import degenerate_target

    triage = config.judges.triage
    if triage is None or not labels:
        return {}
    target = build_target(labels, outputs, tuple(config.schema.fields))
    if not target.truth or not target.positives:
        return {
            "skipped": (
                "no labelled errors to find, so there is no recall to measure"
                if target.truth
                else "no human labels, so true errors cannot be counted"
            )
        }

    # The same floor that governs Gate 2. A target too thin to support an
    # AUC is too thin to support a recall curve off the same rows, and a
    # table that printed anyway would be the more persuasive of the two.
    reason = degenerate_target(
        target,
        min_items=int(config.settings.value("minority_class_min_items")),
        min_share=float(config.settings.value("minority_class_min_share")),
    )
    if reason is not None:
        return {"skipped": reason}

    wanted = [strategy_id, *(b for b in BASELINES if b != strategy_id)]
    ranked: dict[str, Mapping[str, float | None]] = {}
    for name in wanted:
        strategy = STRATEGIES.get(name)
        if strategy is None:
            continue
        if "panel" in strategy.requires and not ctx.panel_members:
            continue
        if "calibration" in strategy.requires and not ctx.calibrator.is_calibrated:
            continue
        ranked[name] = strategy.rank(ctx)

    results = compare(
        ranked,
        target.truth,
        triage.budgets,
        baselines=BASELINES,
        resamples=int(config.settings.value("bootstrap_resamples")),
        seed=ctx.seed,
        min_n=int(config.settings.value("min_n_ranking")),
    )
    return {
        "target_items": len(target.truth),
        "target_errors": target.positives,
        "budgets": list(triage.budgets),
        "strategies": [
            {
                "strategy": r.strategy,
                "baseline": r.is_baseline,
                "ranked": r.ranked,
                "unranked": r.unranked,
                "unranked_errors": r.unranked_errors,
                "points": [
                    {
                        "budget": p.budget,
                        "n_reviewed": p.n_reviewed,
                        "errors_found": round(p.errors_found, 2),
                        "error_recall": p.error_recall,
                        "precision": p.precision,
                        "wasted": p.wasted,
                        "ci_low": p.ci_low,
                        "ci_high": p.ci_high,
                    }
                    for p in r.points
                ],
            }
            for r in results
        ],
    }


def claim_findings(run_id: str, config: RunConfig, verdicts: Sequence[Verdict]) -> list[Finding]:
    """One finding per judged free-text row, with the invented sentence on it.

    The threshold lives here rather than in the judge: the judge reports which
    claims the item supports, and ``min_claim_support`` decides whether that
    rate is good enough. A judge that applied the threshold itself would make
    the number unchangeable without paying again.
    """
    findings: list[Finding] = []
    for verdict in verdicts:
        if verdict.check_id != "claims_supported":
            continue
        spec = config.schema.fields.get(verdict.field)
        if spec is None:
            continue
        threshold = config.settings.resolve("min_claim_support", spec)
        rate = verdict.detail.get("support_rate")
        if verdict.status is Status.UNSCORED or not isinstance(rate, (int, float)):
            findings.append(
                Finding(
                    run_id=run_id,
                    check="claims_supported",
                    grain=Grain.FIELD,
                    status=Status.UNSCORED,
                    item_id=verdict.item_id,
                    field=verdict.field,
                    judge=verdict.judge_id,
                    evidence={"reason": verdict.reason},
                )
            )
            continue
        findings.append(
            Finding(
                run_id=run_id,
                check="claims_supported",
                grain=Grain.FIELD,
                status=Status.PASS if rate >= threshold.value else Status.FAIL,
                item_id=verdict.item_id,
                field=verdict.field,
                score=float(rate),
                threshold=threshold.value,
                threshold_from=threshold.source,
                judge=verdict.judge_id,
                evidence={
                    "unsupported": list(verdict.detail.get("unsupported") or []),
                    "claims": verdict.detail.get("claims"),
                    "selected_by": verdict.metadata.get("selected_by"),
                    **(
                        {"missing": verdict.detail["missing"]}
                        if verdict.detail.get("missing")
                        else {}
                    ),
                },
            )
        )
    return findings


def free_gate_quality(
    plan: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]],
    findings: Sequence[Finding],
) -> dict[str, Any]:
    """Did the free checks actually predict invention, or is that folklore?

    DESIGN.md §8 claims the free-text gate is predictive, which is why it is
    allowed to decide what gets paid for. The audit sample is how that claim
    gets checked rather than repeated: of the rows nothing flagged, how many
    did the judge find invention in anyway?
    """
    by_selection: dict[str, dict[str, int]] = {}
    for found in findings:
        if found.check != "claims_supported" or found.status is Status.UNSCORED:
            continue
        bucket = str(found.evidence.get("selected_by") or "unknown")
        tally = by_selection.setdefault(bucket, {"judged": 0, "failed": 0})
        tally["judged"] += 1
        tally["failed"] += int(found.status is Status.FAIL)

    audited = by_selection.get("audit", {"judged": 0, "failed": 0})
    flagged = by_selection.get("free check", {"judged": 0, "failed": 0})
    out: dict[str, Any] = {
        "flagged": flagged,
        "audited": audited,
        "fields": {
            name: {"flagged": len(suspicious), "audited": len(audit)}
            for name, (suspicious, audit) in plan.items()
        },
    }
    if flagged["judged"]:
        out["precision"] = flagged["failed"] / flagged["judged"]
        if not flagged["failed"]:
            out["gate_note"] = (
                f"the free checks flagged {flagged['judged']} row(s) and the judge found "
                "nothing ungrounded in any of them — the gate is spending on rows that "
                "were fine"
            )
    if audited["judged"]:
        out["miss_rate"] = audited["failed"] / audited["judged"]
        out["why"] = (
            f"{audited['failed']} of {audited['judged']} audited rows that no free check "
            "flagged turned out to have an unsupported claim. That is what the free gate "
            "is missing, measured rather than assumed."
            if audited["failed"]
            else f"none of the {audited['judged']} audited rows that no free check flagged "
            "had an unsupported claim — the gate held on this sample."
        )
    return out


def _stability_metrics(report: StabilityReport | None) -> dict[str, Any]:
    if report is None:
        return {}
    return {
        "kind": report.kind,
        "temperature": report.temperature,
        "runs": report.runs,
        "sampled": report.sampled,
        "unreadable": report.unreadable,
        "note": report.note,
        "reads_as_broken": report.reads_as_broken,
        "fields": [
            {
                "field": f.field,
                "compared": f.compared,
                "agreed": f.agreed,
                "stability": f.stability,
                "flipped": [{"item_id": i, "answers": list(a)} for i, a in f.flipped],
            }
            for f in report.fields
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
    lines: list[str] = []
    if not gates.table:
        lines.append(
            "how many errors a review budget would actually find\n"
            "Error Recall@Budget counts true errors against a target big enough to "
            "rank on, and this run has neither"
        )
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
    unlabelled = sorted(n for n, m in modes.items() if m is Mode.NO_LABELS)
    if unlabelled:
        lines.append(
            f"how accurate {', '.join(unlabelled)} actually is\n"
            "no human answers for this field, so there is nothing to score it against. "
            "~100 labelled rows would unlock accuracy, macro F1 and the confusion matrix."
        )
    single = sorted(n for n, m in modes.items() if m is Mode.LABELLED)
    if single:
        lines.append(
            f"whether the taxonomy itself is crisp for {', '.join(single)}\n"
            "one annotator cannot tell you whether two people can separate two labels. "
            "A second opinion on ~100 items would."
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

    # The free checks run before anything is spent, because their results
    # decide which free-text rows are worth paying a judge for.
    gate_findings, _ = run_checks(check_context(config, dataset, config.run_id, earlier))
    free_text = free_text_plan(
        config, dataset.items, dataset.outputs, flagged_by_free_checks(gate_findings)
    )
    plan = plan_run(config, dataset.items, dataset.outputs, free_text=free_text)
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
    stability: StabilityReport | None = None
    regenerated: list[dict[str, Any]] = []
    unreadable = 0
    hook = config.produced_by.regenerate if config.produced_by else None
    if hook is not None:
        spec = JudgeSpec(
            id="regenerate",
            provider=hook.provider,
            model=hook.model,
            api_key_env=hook.api_key_env,
            endpoint=hook.endpoint,
            temperature=hook.temperature,
        )
        regenerated, unreadable = regenerate_sample(
            config,
            dataset.items,
            provider_factory(spec),
            fields=tuple(config.schema.fields),
        )
        stability = measure_stability(
            config, dataset.outputs, regenerated, unreadable=unreadable
        )
        write_regenerated(directory, regenerated)

    verdicts = (
        collect(
            config, dataset.items, dataset.outputs,
            cache=cache, guard=guard, providers=providers, free_text=free_text,
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
        free_text=free_text,
        stability=stability,
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
    stored = directory / "calibration.json"
    if stored.exists():
        # The project can move between a run and a re-analysis. A curve
        # fitted against the old taxonomy or prompt does not describe this
        # one, and silently reusing it is how a stale number survives.
        check_calibration_age(
            json.loads(stored.read_text(encoding="utf-8")).get("fingerprint"),
            run_fingerprint(config),
            where=str(stored),
        )

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
        free_text=free_text_plan(
            config,
            dataset.items,
            dataset.outputs,
            flagged_by_free_checks(run_checks(check_context(config, dataset, directory.name))[0]),
        ),
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
    free_text: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
    stability: StabilityReport | None = None,
) -> RunResult:
    run_id = directory.name
    modes = detect_modes(config, dataset.labels)
    counts = label_counts(config, dataset.labels)
    health = screen(verdicts, config.settings)

    check_ctx = check_context(config, dataset, run_id, previous)
    free_findings, skipped = run_checks(check_ctx)
    panel_findings, panel = panel_analysis(config, dataset, verdicts, health)
    humans = against_humans(config, dataset, verdicts, modes)
    label_findings = human_findings(run_id, config, dataset)
    claims = claim_findings(run_id, config, verdicts)
    gate_quality = free_gate_quality(free_text or {}, claims)
    stability_metrics = _stability_metrics(stability)
    findings = [
        *free_findings,
        *to_findings(run_id, config, dataset, verdicts),
        *panel_findings,
        *label_findings,
        *claims,
    ]
    calibrator, calibration = fit_calibration(config, dataset, verdicts, modes)
    risk_rows, strategy_id = build_triage(config, dataset, verdicts, calibrator)
    scoring = scoring_context(config, dataset, verdicts, calibrator)
    triage_eval = triage_evaluation(config, scoring, strategy_id, dataset.labels, dataset.outputs)

    gate1 = gate_one(
        config, judge_health=health, items=len(dataset.items), verdicts=verdicts, panel=panel
    )
    gate2 = gate_two(config, scoring, strategy_id, dataset.labels, dataset.outputs)
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
        "vs_humans": humans,
        "free_text_gate": gate_quality,
        "stability": stability_metrics,
        "calibration": calibration,
        "triage_eval": triage_eval,
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
        humans=humans,
        free_text_gate=gate_quality,
        stability=stability_metrics,
        calibration=calibration,
        triage_eval=triage_eval,
        gates=gates,
        risk_rows=risk_rows,
        verdicts=verdicts,
        items=dataset.items,
        metrics=metrics,
        cannot_tell=cannot,
        exclusions=dataset.exclusions,
    )

    if calibration:
        write_json(directory / "calibration.json", calibration)
    if triage_eval:
        write_json(directory / "triage_eval.json", triage_eval)
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
