"""Expectation contract.

The core is async because providers are IO-bound and concurrency is the whole
game. But most expectations are three lines of pure Python, so two shims exist:

  SyncExpectation   subclass, implement `check()` as a normal method
  @field_check      decorate a single function operating on one record

Both are wrapped into the async contract by the base class, so the runner only
ever awaits. Set `blocking=True` to get offloaded to a worker thread.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from .batch import Batch, Context, ExtractionRecord
from .defaults import DEFAULT_STRATEGY_ALIAS
from .registry import EXPECTATIONS
from .result import Result
from .types import Capability, Evidence, Grain, Kind, Provenance, Severity

FieldVerdict = bool | tuple[bool, Evidence]
FieldCheckFn = Callable[[ExtractionRecord, Batch, Context], FieldVerdict]


class Expectation(ABC):
    id: str = "expectation"
    version: str = "1"  # bump on semantic change; the manifest records it
    kind: Kind = Kind.DETERMINISTIC
    grain: Grain = Grain.FIELD
    required_capabilities: frozenset[Capability] = frozenset()

    default_strategy: str = DEFAULT_STRATEGY_ALIAS
    """Strategy used when the config names none.

    Declared on the base rather than read at the call site so the planner and
    the expectation resolve the *same* strategy. A variant that defaults to a
    different one would otherwise have its capabilities and calibration
    validated against a strategy it never runs.
    """

    def __init__(self, severity: Severity | str = Severity.ERROR, **config: Any) -> None:
        self.severity = Severity(severity)
        self.config: dict[str, Any] = config

    @property
    def provenance(self) -> Provenance:
        return Provenance(expectation_id=self.id, expectation_version=self.version)

    def estimate_calls(self, batch: Batch) -> int:
        """Model calls this expectation will make. Deterministic checks: zero."""
        return 0

    @abstractmethod
    async def validate(self, batch: Batch, ctx: Context) -> list[Result]:
        """Always returns a list: one field-grain check over a batch produces
        many results, and a uniform return type keeps the runner simple."""

    def __repr__(self) -> str:
        return f"<{self.id}@{self.version} {self.kind.value}/{self.grain.value}>"


class SyncExpectation(Expectation):
    """Write `check()` as a plain method. The base handles the async contract."""

    blocking: bool = False

    @abstractmethod
    def check(self, batch: Batch, ctx: Context) -> list[Result]:
        ...

    async def validate(self, batch: Batch, ctx: Context) -> list[Result]:
        if self.blocking:
            return await asyncio.to_thread(self.check, batch, ctx)
        return self.check(batch, ctx)


def field_check(
    id: str,  # noqa: A002 - mirrors Expectation.id, which is the public vocabulary
    version: str = "1",
    fields: list[str] | None = None,
    blocking: bool = False,
) -> Callable[[FieldCheckFn], FieldCheckFn]:
    """Turn one function into a registered field-grain expectation.

        @field_check(id="expect_field_not_empty")
        def not_empty(rec, batch, ctx):
            return bool(rec.value), Evidence("emptiness", {"value": rec.value})

    The function returns `bool` or `(bool, Evidence)`.
    """

    def wrap(fn: FieldCheckFn) -> FieldCheckFn:
        def check(self: SyncExpectation, batch: Batch, ctx: Context) -> list[Result]:
            targets = self.config.get("fields") or fields or ["*"]
            out: list[Result] = []
            for rec in batch.for_fields(targets):
                verdict = fn(rec, batch, ctx)
                if isinstance(verdict, tuple):
                    ok, evidence = verdict
                else:
                    ok, evidence = verdict, Evidence()
                out.append(
                    Result(
                        expectation_id=self.id,
                        grain=Grain.FIELD,
                        doc_id=rec.doc_id,
                        field_name=rec.field_name,
                        success=bool(ok),
                        severity=self.severity,
                        observed=rec.value,
                        evidence=evidence,
                        provenance=self.provenance,
                    )
                )
            return out

        # The generated class must carry `check` in its namespace at creation.
        # Assigning `cls.check = fn` afterwards leaves __abstractmethods__
        # populated and instantiation fails with "Can't instantiate abstract class".
        cls = type(
            f"{id}_expectation",
            (SyncExpectation,),
            {
                "id": id,
                "version": version,
                "blocking": blocking,
                "__doc__": fn.__doc__,
                "check": check,
            },
        )
        EXPECTATIONS.register(id, cls)
        fn.expectation = cls  # type: ignore[attr-defined]
        return fn

    return wrap
