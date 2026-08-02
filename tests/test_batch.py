"""Batch: lazy source resolution and deterministic ordering."""

from __future__ import annotations

from llmex import Batch, ExtractionRecord, SourceDoc


def _recs():
    return [
        ExtractionRecord("d2", "vendor", "A"),
        ExtractionRecord("d1", "vendor", "B"),
        ExtractionRecord("d1", "total", "1.00"),
    ]


def test_source_resolver_is_called_once_per_doc():
    calls: list[str] = []

    def resolver(doc_id: str) -> SourceDoc:
        calls.append(doc_id)
        return SourceDoc(doc_id, "text")

    b = Batch(_recs(), resolver)
    for _ in range(3):
        b.source("d1")
        b.source("d2")
    assert calls == ["d1", "d2"]


def test_doc_ids_and_field_names_preserve_insertion_order():
    b = Batch(_recs(), lambda d: SourceDoc(d, ""))
    assert b.doc_ids == ["d2", "d1"]
    assert b.field_names == ["vendor", "total"]


def test_for_fields_star_returns_everything():
    b = Batch(_recs(), lambda d: SourceDoc(d, ""))
    assert b.for_fields(["*"]) is b.records
    assert [r.field_name for r in b.for_fields(["total"])] == ["total"]
    assert b.for_fields(["nope"]) == []


def test_by_document_groups_and_len_counts_records():
    b = Batch(_recs(), lambda d: SourceDoc(d, ""))
    grouped = b.by_document()
    assert set(grouped) == {"d1", "d2"}
    assert len(grouped["d1"]) == 2
    assert len(b) == 3


def test_record_key():
    assert ExtractionRecord("d1", "vendor", "A").key == ("d1", "vendor")
