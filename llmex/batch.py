"""The unit of work.

Observe-only by design: the framework never calls the extractor. It receives
records that already exist and a way to fetch the source text they came from.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceDoc:
    doc_id: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractionRecord:
    """One field extracted from one document by one run of one extractor.

    `span` is optional but valuable: when present, groundedness is an O(1)
    slice comparison instead of a window scan, and it verifies the model
    extracted from the location it claimed.
    """

    doc_id: str
    field_name: str
    value: object
    span: tuple[int, int] | None = None
    prompt_version: str | None = None
    generator_model: str | None = None
    extraction_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        return (self.doc_id, self.field_name)


class Batch:
    """Records plus lazy source access.

    The resolver is a callable rather than a dict so a batch can span a corpus
    that does not fit in memory. Resolved docs are cached for the batch's life.
    """

    def __init__(
        self,
        records: Iterable[ExtractionRecord],
        source_resolver: Callable[[str], SourceDoc],
        schema: dict[str, Any] | None = None,
    ) -> None:
        self.records: list[ExtractionRecord] = list(records)
        self._resolver = source_resolver
        self.schema: dict[str, Any] = schema or {}
        self._cache: dict[str, SourceDoc] = {}

    def source(self, doc_id: str) -> SourceDoc:
        if doc_id not in self._cache:
            self._cache[doc_id] = self._resolver(doc_id)
        return self._cache[doc_id]

    def by_document(self) -> dict[str, list[ExtractionRecord]]:
        grouped: dict[str, list[ExtractionRecord]] = defaultdict(list)
        for rec in self.records:
            grouped[rec.doc_id].append(rec)
        return dict(grouped)

    def for_fields(self, names: list[str]) -> list[ExtractionRecord]:
        if names == ["*"]:
            return self.records
        wanted = set(names)
        return [r for r in self.records if r.field_name in wanted]

    @property
    def doc_ids(self) -> list[str]:
        """Insertion-ordered, so output is deterministic across runs."""
        seen: dict[str, None] = {}
        for rec in self.records:
            seen.setdefault(rec.doc_id, None)
        return list(seen)

    @property
    def field_names(self) -> list[str]:
        seen: dict[str, None] = {}
        for rec in self.records:
            seen.setdefault(rec.field_name, None)
        return list(seen)

    def __len__(self) -> int:
        return len(self.records)


@dataclass
class Context:
    """Everything an expectation may need that is not the data itself.

    `prior` is the mechanism behind cheap-tiers-gate-expensive-tiers:
    deterministic steps write into it, the model tier reads it to decide which
    documents to escalate.
    """

    providers: dict[str, Any] = field(default_factory=dict)
    strategies: dict[str, Any] = field(default_factory=dict)
    calibrations: dict[str, Any] = field(default_factory=dict)
    budget: Any | None = None
    prior: dict[tuple[str, str | None], bool] = field(default_factory=dict)
    prior_results: list[Any] = field(default_factory=list)
    """Every result produced by earlier steps, in tier order.

    `prior` answers "is this field clean?" and is what the model tier routes
    on. This is the full record, for DERIVED checks that need to read a
    specific earlier expectation's verdicts rather than their conjunction.
    """
    run_id: str = ""
