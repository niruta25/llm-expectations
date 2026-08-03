"""JTBD session classification, end to end and offline.

    python demo_jtbd.py

Five sections, in the order you would actually do the work:

  1. the free baseline    what a gold set tells you for nothing
  2. calibration          fitting a judge against those human labels
  3. the guarded run      both tiers, both grains, with cost
  4. grading the grader   kappa between judge and human
  5. production           the same suite with no gold, and what changes

The whole point of the ordering is that the judge has to earn its place. By
section 3 you already know the accuracy, the confusion matrix and the label
distribution, and all of it was free.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "examples" / "jtbd_session_classification"))

from data import FIELDS, load_sessions  # noqa: E402
from taxonomy import BASELINE_DISTRIBUTION, LABELS, RUBRIC  # noqa: E402

from llmex import (  # noqa: E402
    ConsoleSink,
    Grain,
    LabelledDecision,
    PlanError,
    Planner,
    Runner,
    Suite,
    calibrate_judge,
    normalise_label,
    report_from_batch,
)
from llmex.strategies import ScorePayload  # noqa: E402

SUITE = {
    "suite": "jtbd_session_classification",
    "version": "4",
    "providers": {"cheap_judge": {"plugin": "mock_judge"}},
    "strategies": {
        "jtbd_judge": {"plugin": "label_judge", "label_set": LABELS, "rubric": RUBRIC}
    },
    "budget": {"max_usd": 5.0},
    "expectations": [
        # Tier 1 — free, and it answers the question outright where gold exists.
        {"type": "expect_field_matches_gold", "fields": list(FIELDS), "severity": "warn"},
        # Tier 2 — free, and catches what accuracy cannot.
        {
            "type": "expect_label_distribution_stable",
            "fields": ["jtbd_label"],
            "max_share": 0.5,
            "baseline": BASELINE_DISTRIBUTION,
            "max_shift": 0.25,
            "severity": "warn",
        },
        # Tier 3 — costs money, so it runs last and only where it earns its keep.
        {
            "type": "expect_extraction_label_trustworthy",
            "fields": ["jtbd_label"],
            "provider": "cheap_judge",
            "strategy": "jtbd_judge",
            "calibration": "jtbd_session_v1",
            "audit_rate": 1.0,
            "severity": "error",
        },
        # Tier 4 — free, and grades the grader.
        {
            "type": "expect_judge_agrees_with_gold",
            "fields": ["jtbd_label"],
            "min_kappa": 0.4,
            "severity": "warn",
        },
    ],
}


def section(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def show_baseline(batch) -> None:
    section("1. the free baseline — what a gold set tells you for nothing")
    rep = report_from_batch(batch, "jtbd_label")
    print(f"  sessions          {rep.n}")
    print(f"  accuracy          {rep.accuracy:.1%}")
    print(f"  macro F1          {rep.macro_f1:.3f}   <- the honest one")
    print("\n  weakest labels:")
    for m in rep.worst_labels(3):
        print(
            f"    {m.label:<22} f1={m.f1:.2f}  precision={m.precision:.2f} "
            f"recall={m.recall:.2f}  n={m.support}"
        )
    print("\n  confusions:")
    for (gold, pred), n in sorted(rep.matrix.items()):
        if gold != pred:
            print(f"    {gold:<22} labelled as {pred:<22} x{n}")
    print("\n  Not one model call was made to learn any of this.")


async def build_calibration(batch, suite) -> object:
    section("2. calibration — fitting the judge against those same human labels")
    provider = suite.providers["cheap_judge"]
    strategy = suite.strategies["jtbd_judge"]

    decisions: list[LabelledDecision] = []
    for doc_id, recs in batch.by_document().items():
        rec = next(r for r in recs if r.field_name == "jtbd_label")
        scored = await strategy.score(
            ScorePayload(doc_id, batch.source(doc_id).text, {"jtbd_label": rec.value}),
            provider,
        )
        decisions.append(
            LabelledDecision(
                doc_id=doc_id,
                field_name="jtbd_label",
                score=scored.field_scores["jtbd_label"],
                predicted_label=normalise_label(rec.value),
                gold_label=normalise_label(rec.meta["gold_label"]),
            )
        )

    def fit(target: float):
        return calibrate_judge(
            id="jtbd_session_v1",
            decisions=decisions,
            provider_id=provider.id,
            model_version=provider.model_version,
            strategy_id=strategy.id,
            target_precision=target,
        )

    cal = fit(0.90)
    print(f"  labelled decisions {cal.n_labels}")
    print(f"  auroc              {cal.metrics['auroc']:.3f}")
    print(f"  precision@errors   {cal.metrics['precision_at_num_errors']:.3f}")

    if cal.metrics["recall_at_target_precision"] == 0.0:
        print("\n  at target precision 0.90:")
        print("    no threshold qualifies. Every cut that catches a real error on")
        print("    this corpus also catches a correct label, so 90% precision is")
        print("    not available at any recall above zero.")
        print("\n  This is the framework declining to hand you a number you could")
        print("  not defend. The options are a better judge, more labels, or an")
        print("  honest lower target — not a threshold typed in anyway.")
        cal = fit(0.80)
        print("\n  refitted at target precision 0.80:")

    m = cal.metrics
    print(f"    threshold        {cal.thresholds['__default__']:.3f}")
    print(f"    recall           {m['recall_at_target_precision']:.1%}  <- what it cost")
    print(f"    kappa vs human   {m['cohens_kappa']:.3f}")
    print(f"    raw agreement    {m['judge_human_agreement']:.1%}")
    print(f"    ref              {cal.ref}")
    return cal


def show_guard(batch) -> None:
    section("3a. the guard — an uncalibrated judge cannot block a run")
    cfg = {**SUITE, "expectations": [
        {**SUITE["expectations"][2], "calibration": None, "severity": "error"}
    ]}
    cfg["expectations"][0].pop("calibration")
    try:
        Planner().plan(Suite.from_dict(cfg), batch)
    except PlanError as exc:
        print(f"  {exc}")


async def main() -> None:
    batch = load_sessions()
    show_baseline(batch)

    suite = Suite.from_dict(SUITE)
    cal = await build_calibration(batch, suite)

    show_guard(batch)

    section("3b. the full run — four tiers, both grains, cost")
    suite = Suite.from_dict(SUITE, calibrations={"jtbd_session_v1": cal})
    plan = Planner().plan(suite, batch)
    print(f"  planned {len(plan.steps)} steps, ~{plan.estimated_calls} model calls, "
          f"~${plan.estimated_usd:.4f}")
    for w in plan.warnings:
        print(f"  ! {w}")
    run = await Runner(sinks=[ConsoleSink(limit=6)]).run(suite, batch, plan)

    section("4. grading the grader")
    agreement = next(
        (r for r in run.results if r.expectation_id == "expect_judge_agrees_with_gold"), None
    )
    if agreement is not None:
        d = agreement.evidence.detail
        print(f"  kappa vs human    {d['cohens_kappa']:.3f}  (threshold {d['min_kappa']})")
        print(f"  raw agreement     {d['raw_agreement']:.1%}  over {d['n']} sessions")
        print(f"  verdict           {'PASS' if agreement.success else 'FAIL'}")
    print("\n  Raw agreement always flatters. On a skewed taxonomy a judge that")
    print("  approves everything agrees with humans most of the time; kappa is")
    print("  what says whether it measured anything.")

    section("5. the grain gap")
    fields = [r for r in run.results if r.grain is Grain.FIELD and r.success is not None]
    docs = [r for r in run.results if r.grain is Grain.DOCUMENT and r.success is not None]
    fp = sum(1 for r in fields if r.success) / len(fields)
    dp = sum(1 for r in docs if r.success) / len(docs)
    print(f"  field-grain pass rate     {fp:.1%}  ({len(fields)} checks)")
    print(f"  document-grain pass rate  {dp:.1%}  ({len(docs)} checks)")
    print(f"  gap                       {fp - dp:.1%}")

    section("6. the same suite in production, where nobody has labelled anything")
    unlabelled = load_sessions(with_gold=False)
    prod = await Runner().run(
        Suite.from_dict(SUITE, calibrations={"jtbd_session_v1": cal}), unlabelled
    )
    by_exp: dict[str, int] = {}
    for r in prod.unscored():
        by_exp[r.expectation_id] = by_exp.get(r.expectation_id, 0) + 1
    print("  unscored results, by check:")
    for exp_id, n in sorted(by_exp.items()):
        print(f"    {exp_id:<42} {n}")
    print("\n  Nothing silently passed. The checks that needed gold said so,")
    print("  and the judge — which never needed it — carried on.")

    print("\n" + "=" * 74)
    print("Every number above came from a fixture judge that scores on lexical")
    print("overlap. It is a plumbing check, not a result. Point this at a real")
    print("provider and a real gold set before believing any of it.")
    print("=" * 74)


if __name__ == "__main__":
    asyncio.run(main())
