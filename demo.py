"""End-to-end walkthrough. Runs offline against the mock provider, in under a second.

    python demo.py

Four sections: the plan-time guards that refuse to spend money, a calibration
fitted from labelled scores, a full run over a seeded corpus, and the grain gap
that is the whole reason this framework exists.
"""

from __future__ import annotations

import asyncio

from llmex import (
    Batch,
    Capability,
    ConsoleSink,
    Evidence,
    ExtractionRecord,
    LabelledScore,
    PlanError,
    Planner,
    Runner,
    SourceDoc,
    Suite,
    calibrate,
    field_check,
)
from llmex.registry import PROVIDERS
from llmex.strategies import ScorePayload

# --------------------------------------------------------------------------
# A tiny corpus. Two documents contain values the model invented.
# --------------------------------------------------------------------------

DOCS = {
    "inv-001": "Payment of $1,530.00 USD was issued to Brightstone Manufacturing for the invoice dated February 12, 2024.",
    "inv-002": "Invoice 88231 from Redwood Logistics, total EUR 4,215.60, dated 03 March 2024.",
    "inv-003": "Northgate Supply billed $890.25 USD on January 8, 2024 under PO 4471.",
    "inv-004": "Statement issued to Calder & Sons for GBP 2,100.00 covering services in April 2024.",
    "inv-005": "Vendor: Ashfield Components. Amount due $7,340.10 USD. Invoice date May 30, 2024.",
    "inv-006": "Total of CAD 512.00 payable to Lakeview Print Co, dated June 14, 2024.",
}

EXTRACTIONS = [
    # doc,     vendor,                      date,                total,       currency, ok?
    ("inv-001", "Brightstone Manufacturing", "February 12, 2024", "1,530.00", "USD", True),
    ("inv-002", "Redwood Logistics",         "03 March 2024",     "4,215.60", "EUR", True),
    ("inv-003", "Northgate Supply",          "January 8, 2024",   "890.25",   "USD", True),
    ("inv-004", "Calder and Sons Ltd",       "April 2024",        "2,100.00", "GBP", False),  # vendor invented
    ("inv-005", "Ashfield Components",       "May 30, 2024",      "7,430.10", "USD", False),  # digits transposed
    ("inv-006", "Lakeview Print Co",         "June 14, 2024",     "512.00",   "CAD", True),
]

BASE_CFG = {
    "suite": "invoice_extraction",
    "version": "3",
    "providers": {"cheap_verifier": {"plugin": "mock"}},
    "budget": {"max_usd": 5.0},
    "aggregate": {
        "field_to_document": "weighted_harmonic",
        "field_weights": {"total_amount": 3.0, "currency": 0.5},
    },
    "expectations": [
        {"type": "expect_field_grounded_in_source",
         "fields": ["vendor", "total_amount"], "min_ratio": 0.92},
        {"type": "expect_field_type",
         "types": {"vendor": "string", "total_amount": "string"}, "severity": "warn"},
        {"type": "expect_field_null_rate_between", "max_rate": 0.2, "severity": "warn"},
        {"type": "expect_field_trustworthy",
         "provider": "cheap_verifier", "strategy": "diverse_ensemble",
         "audit_rate": 1.0, "calibration": "invoices_v1", "severity": "error"},
    ],
}


def build_batch() -> Batch:
    records = []
    for doc_id, vendor, date, total, currency, _ in EXTRACTIONS:
        for name, value in [
            ("vendor", vendor),
            ("invoice_date", date),
            ("total_amount", total),
            ("currency", currency),
        ]:
            records.append(
                ExtractionRecord(
                    doc_id=doc_id,
                    field_name=name,
                    value=value,
                    prompt_version="invoice-extract@v3",
                    generator_model="mock-1.0",
                )
            )
    return Batch(
        records,
        source_resolver=lambda d: SourceDoc(d, DOCS[d]),
        schema={"vendor": "string", "invoice_date": "string",
                "total_amount": "string", "currency": "string"},
    )


@field_check(id="expect_currency_iso4217", fields=["currency"])
def currency_is_iso(rec, batch, ctx):
    """A whole expectation in one function — the sync shim in action."""
    known = {"USD", "EUR", "GBP", "CAD", "JPY", "AUD"}
    return rec.value in known, Evidence("enum", {"allowed": sorted(known), "got": rec.value})


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def fmt(d: dict) -> str:
    return "{" + ", ".join(f"{k}: {v:.2f}" for k, v in sorted(d.items())) + "}"


# --------------------------------------------------------------------------


def show_plan_guards(batch: Batch) -> None:
    section("1. plan-time guards — fail before spending money")

    uncalibrated = {**BASE_CFG, "expectations": [
        {"type": "expect_field_trustworthy", "provider": "cheap_verifier",
         "strategy": "diverse_ensemble", "severity": "error"}]}
    try:
        Planner().plan(Suite.from_dict(uncalibrated), batch)
    except PlanError as exc:
        print(f"  calibration guard   -> {exc}")

    class NoLogprobs(PROVIDERS.get("mock")):
        id = "no_logprobs"
        model_version = "nolp-1"
        capabilities = frozenset({Capability.STRUCTURED_OUTPUT})

    PROVIDERS.register("no_logprobs", NoLogprobs)
    gap = {**BASE_CFG,
           "providers": {"v": {"plugin": "no_logprobs"}},
           "expectations": [{"type": "expect_field_trustworthy", "provider": "v",
                             "strategy": "logprob", "severity": "warn"}]}
    try:
        Planner().plan(Suite.from_dict(gap), batch)
    except PlanError as exc:
        print(f"  capability guard    -> {exc}")


async def build_calibration(batch: Batch):
    section("2. calibration — thresholds derived, not typed")

    suite = Suite.from_dict(BASE_CFG)
    provider = suite.providers["cheap_verifier"]
    strategy = suite.strategies["diverse_ensemble"]
    truth = {d: ok for d, *_, ok in EXTRACTIONS}

    labels: list[LabelledScore] = []
    for doc_id, recs in batch.by_document().items():
        values = {r.field_name: r.value for r in recs}
        scored = await strategy.score(
            ScorePayload(doc_id, batch.source(doc_id).text, values, batch.schema), provider
        )
        for name, score in scored.field_scores.items():
            grounded = str(values[name]) in batch.source(doc_id).text
            labels.append(LabelledScore(doc_id, name, score, grounded))
        labels.append(LabelledScore(doc_id, None, scored.doc_score, truth[doc_id]))

    cal = calibrate(
        id="invoices_v1",
        labels=labels,
        provider_id=provider.id,
        model_version=provider.model_version,
        strategy_id=strategy.id,
        target_precision=0.9,
    )
    print(f"  labels           {cal.n_labels}")
    print(f"  auroc            {cal.metrics['auroc']:.3f}")
    print(f"  precision@errors {cal.metrics['precision_at_num_errors']:.3f}")
    print(f"  confidence gap   {cal.metrics['confidence_gap']:.3f}")
    print(f"  recall @ p=0.90  {cal.metrics['recall_at_target_precision']:.3f}")
    print(f"  thresholds       {fmt(cal.thresholds)}")
    print(f"  ref              {cal.ref}")
    print("\n  NB: AUROC reads 1.00 because MockProvider scores by substring")
    print("      containment, which is exactly what these labels encode. Real")
    print("      verifiers land far lower. It is a plumbing check, not a claim.")
    return cal


async def main() -> None:
    batch = build_batch()
    show_plan_guards(batch)
    cal = await build_calibration(batch)

    section("3. full run — both grains, evidence, cost")
    cfg = {**BASE_CFG, "expectations": [
        *BASE_CFG["expectations"],
        {"type": "expect_currency_iso4217", "severity": "warn"},
    ]}
    suite = Suite.from_dict(cfg, calibrations={"invoices_v1": cal})
    plan = Planner().plan(suite, batch)
    print(f"  planned {len(plan.steps)} steps, "
          f"~{plan.estimated_calls} model calls, ~${plan.estimated_usd:.4f}")
    for w in plan.warnings:
        print(f"  ! {w}")

    run = await Runner(sinks=[ConsoleSink()]).run(suite, batch, plan)

    section("4. the grain gap — where field-grain flatters you")
    field_rows = [r for r in run.results if r.grain.value == "field" and r.success is not None]
    doc_rows = [r for r in run.results if r.grain.value == "document" and r.success is not None]
    fp = sum(1 for r in field_rows if r.success) / len(field_rows)
    dp = sum(1 for r in doc_rows if r.success) / len(doc_rows)
    print(f"  field-grain pass rate     {fp:.1%}  ({len(field_rows)} checks)")
    print(f"  document-grain pass rate  {dp:.1%}  ({len(doc_rows)} checks)")
    print(f"  gap                       {fp - dp:.1%}")
    print("\n  A document is wrong if any field is wrong. Report the second number.")


if __name__ == "__main__":
    asyncio.run(main())
