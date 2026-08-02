"""Shared fixtures. Every test runs offline against the mock provider."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llmex import Batch, ExtractionRecord, SourceDoc

GOLDEN = Path(__file__).parent / "golden" / "invoices"

DOC = "Paid $1,530.00 to Brightstone Manufacturing on February 12, 2024."

# Which seeded error types the free deterministic tier can actually catch. The
# rest put a real source value (or no value) in the output; they need the model
# tier or a cross-field rule. See tests/golden/invoices/README.md.
GROUNDING_DETECTABLE = {"fabrication", "transposition", "format_drift"}

GROUNDED_FIELDS = ["vendor", "invoice_date", "total_amount", "currency"]


def _jsonl(name: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (GOLDEN / name).read_text().splitlines() if line]


@pytest.fixture(scope="session")
def golden_docs() -> dict[str, str]:
    return {d["doc_id"]: d["text"] for d in _jsonl("docs.jsonl")}


@pytest.fixture(scope="session")
def golden_answers() -> dict[tuple[str, str], dict[str, Any]]:
    return {(a["doc_id"], a["field_name"]): a for a in _jsonl("answers.jsonl")}


@pytest.fixture
def golden_batch(golden_docs: dict[str, str]) -> Batch:
    records = [
        ExtractionRecord(
            doc_id=r["doc_id"],
            field_name=r["field_name"],
            value=r["value"],
            span=tuple(r["span"]) if r.get("span") else None,
            prompt_version="invoice-extract@v3",
            generator_model="gen-2",
        )
        for r in _jsonl("extractions.jsonl")
    ]
    return Batch(
        records,
        source_resolver=lambda d: SourceDoc(d, golden_docs[d]),
        schema={f: "string" for f in GROUNDED_FIELDS},
    )


@pytest.fixture
def tiny_batch() -> Batch:
    """Two fields, one document, both grounded in DOC."""
    return make_tiny_batch()


def make_tiny_batch(
    vendor: str = "Brightstone Manufacturing", total: str = "1,530.00"
) -> Batch:
    recs = [
        ExtractionRecord("d1", "vendor", vendor, generator_model="gen-1"),
        ExtractionRecord("d1", "total_amount", total, generator_model="gen-1"),
    ]
    return Batch(recs, lambda d: SourceDoc(d, DOC))


def cfg(**over: Any) -> dict[str, Any]:
    """Minimal suite config: one mock provider, one grounding check."""
    base: dict[str, Any] = {
        "suite": "t",
        "providers": {"v": {"plugin": "mock"}},
        "expectations": [{"type": "expect_field_grounded_in_source"}],
    }
    base.update(over)
    return base
