"""The check seam: one signature, three grains, and a registry keyed by kind.

Every check has the same shape whatever kind of field it belongs to, so the
runner walks a list and collects findings. **It never branches on field kind.**
A lookup keyed by kind is not a branch — adding a kind means registering
against it, not editing the runner (DESIGN.md §4).

Checks differ by *what they look at*, not by what kind of field they are for:

    FIELD    one item, one field    "is this label in the taxonomy?"
    ITEM     one item, several      "does the summary agree with the label?"
    CORPUS   the whole batch        "did one label swallow 60% of everything?"

Cross-field consistency is therefore an item check and belongs to neither
kind — it sits above both and reads whatever fields the schema names.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Protocol, runtime_checkable

from ..config import Resolved, Settings
from ..schema import FieldKind, FieldSpec, Schema
from ..taxonomy import Taxonomy
from ..types import Finding, Grain, Item, Output, Status

__all__ = [
    "CheckContext",
    "CheckSpec",
    "Skipped",
    "checks_for",
    "finding",
    "register",
    "run_checks",
]


@dataclass(frozen=True, slots=True)
class CheckContext:
    """Everything a check may read.

    No verdicts and no labels. These are the free checks: they run before
    anything is spent and they cannot see the answer key. A check that needed
    either would be a different kind of check.
    """

    run_id: str
    schema: Schema
    settings: Settings
    items: Mapping[str, Item]
    outputs: Mapping[str, Output]
    taxonomies: Mapping[str, Taxonomy]
    previous: Mapping[str, Any] = dataclass_field(default_factory=dict)

    def taxonomy_for(self, field_name: str) -> Taxonomy | None:
        spec = self.schema[field_name]
        return None if spec.taxonomy is None else self.taxonomies.get(spec.taxonomy.id)

    def resolve(self, key: str, spec: FieldSpec | None = None) -> Resolved[Any]:
        return self.settings.resolve(key, spec)


@runtime_checkable
class Check(Protocol):
    """``check(field, ctx) -> findings``. That is the whole interface."""

    def __call__(self, spec: FieldSpec, ctx: CheckContext) -> Sequence[Finding]: ...


@dataclass(frozen=True, slots=True)
class Skipped:
    """A check that did not run, and why.

    Not a finding: a check switched off by configuration produced no result at
    all, which is different from a result of "we did not check this". Reported
    so the field's section still accounts for every check that exists.
    """

    check: str
    field: str
    reason: str


@dataclass(frozen=True, slots=True)
class CheckSpec:
    """One registered check: what it is called, what it looks at, when it runs."""

    name: str
    grain: Grain
    kinds: frozenset[FieldKind]
    run: Check
    applies: Callable[[FieldSpec, CheckContext], str | None] | None = None

    def skip_reason(self, spec: FieldSpec, ctx: CheckContext) -> str | None:
        return self.applies(spec, ctx) if self.applies else None


_REGISTRY: list[CheckSpec] = []


def register(
    name: str,
    grain: Grain,
    kinds: Sequence[FieldKind],
    *,
    applies: Callable[[FieldSpec, CheckContext], str | None] | None = None,
) -> Callable[[Check], Check]:
    """Register a check against the kinds it reads. Order of registration is
    the order it appears in the report, so the ladder reads top to bottom."""

    def decorate(run: Check) -> Check:
        _REGISTRY.append(
            CheckSpec(name=name, grain=grain, kinds=frozenset(kinds), run=run, applies=applies)
        )
        return run

    return decorate


def checks_for(kind: FieldKind) -> tuple[CheckSpec, ...]:
    return tuple(spec for spec in _REGISTRY if kind in spec.kinds)


def run_checks(ctx: CheckContext) -> tuple[list[Finding], list[Skipped]]:
    """Walk every registered check for every field. No branching on kind."""
    findings: list[Finding] = []
    skipped: list[Skipped] = []
    for spec in ctx.schema.fields.values():
        for check in checks_for(spec.kind):
            reason = check.skip_reason(spec, ctx)
            if reason is not None:
                skipped.append(Skipped(check=check.name, field=spec.name, reason=reason))
                continue
            findings.extend(check.run(spec, ctx))
    return findings, skipped


def finding(
    ctx: CheckContext,
    *,
    check: str,
    grain: Grain,
    status: Status,
    item_id: str | None = None,
    field: str | None = None,
    score: float | None = None,
    threshold: Resolved[Any] | Any = None,
    evidence: Mapping[str, Any] | None = None,
) -> Finding:
    """Build a finding, carrying the threshold's provenance when it has one.

    A number printed without the bar it cleared is not a result, and a bar
    printed without where it came from is not arguable.
    """
    value = threshold.value if isinstance(threshold, Resolved) else threshold
    source = threshold.source if isinstance(threshold, Resolved) else None
    return Finding(
        run_id=ctx.run_id,
        check=check,
        grain=grain,
        status=status,
        item_id=item_id,
        field=field,
        score=score,
        threshold=value,
        threshold_from=source,
        evidence=dict(evidence or {}),
    )
