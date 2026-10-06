"""Two runs, head to head — and the one thing it refuses to tell you.

Reads two finished run directories off disk and issues no model calls. What it
shows: the metrics side by side, which labels moved, and which individual
items got better or worse.

**What it will not do is say which run is better.** DESIGN.md §14 defers the
significance test to v1 with a specific reason: most items tie in a real A/B,
so three net wins over two hundred items is a coin flip, and quoting a win
rate without a test is how underpowered changes get shipped. A comparison that
printed "B is better" off a net delta would be the exact failure this library
exists to prevent, so the paired counts are shown — improved, regressed, and
the usually-much-larger unchanged — and the conclusion is withheld.

Two runs on different taxonomy versions refuse to compare without a migration
mapping. Lining labels up by name across an edit would read a rename as a
distribution shift and blame the model for a change to the taxonomy.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .taxonomy import Migration, TaxonomyError, parse_ref
from .types import Status

__all__ = [
    "ComparisonError",
    "Delta",
    "LoadedRun",
    "RunComparison",
    "compare_runs",
    "load_run_directory",
]

#: Metrics worth lining up, and how to read each one.
HEADLINE: tuple[tuple[str, str, str], ...] = (
    ("macro_f1", "macro F1", "ratio"),
    ("accuracy", "accuracy", "share"),
    ("majority_baseline", "majority-label baseline", "share"),
)


class ComparisonError(ValueError):
    """The two runs cannot be compared, and the reason is worth reading."""


@dataclass(frozen=True, slots=True)
class LoadedRun:
    """One finished run, as it sits on disk."""

    directory: Path
    run_id: str
    manifest: Mapping[str, Any]
    metrics: Mapping[str, Any]
    findings: tuple[Mapping[str, Any], ...]

    @property
    def prompt(self) -> str | None:
        produced = self.manifest.get("produced_by") or {}
        return produced.get("prompt_version")

    @property
    def model(self) -> str | None:
        produced = self.manifest.get("produced_by") or {}
        return produced.get("model")

    @property
    def taxonomies(self) -> Mapping[str, str]:
        return self.manifest.get("taxonomies") or {}

    @property
    def items(self) -> int:
        value = self.metrics.get("items")
        return int(value) if isinstance(value, int) else 0

    def field_scores(self, name: str) -> Mapping[str, Any]:
        return ((self.metrics.get("vs_humans") or {}).get("fields") or {}).get(name) or {}

    def distribution(self, name: str) -> Mapping[str, float]:
        return (self.metrics.get("distributions") or {}).get(name) or {}

    def outcome_by_item(self, name: str) -> dict[str, bool]:
        """Per item: did every scored check on this field pass?

        Read from the findings rather than from an aggregate, because the
        interesting question in a comparison is which *items* moved, and an
        aggregate cannot answer it.
        """
        per_item: dict[str, list[bool]] = {}
        for row in self.findings:
            if row.get("field") != name or not row.get("item_id"):
                continue
            if row.get("status") == Status.UNSCORED.value:
                continue
            per_item.setdefault(str(row["item_id"]), []).append(
                row.get("status") == Status.PASS.value
            )
        return {item: all(results) for item, results in per_item.items()}


def load_run_directory(directory: str | Path) -> LoadedRun:
    """Read one run's outputs. Nothing here calls a model."""
    directory = Path(directory)
    manifest_path = directory / "run.json"
    if not manifest_path.exists():
        raise ComparisonError(
            f"{directory} does not look like a run directory — no run.json in it"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics_path = directory / "metrics.json"
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    )
    findings_path = directory / "findings.jsonl"
    findings: list[Mapping[str, Any]] = []
    if findings_path.exists():
        with findings_path.open(encoding="utf-8") as handle:
            findings = [json.loads(line) for line in handle if line.strip()]
    return LoadedRun(
        directory=directory,
        run_id=str(manifest.get("run_id") or directory.name),
        manifest=manifest,
        metrics=metrics,
        findings=tuple(findings),
    )


@dataclass(frozen=True, slots=True)
class Delta:
    """One metric in both runs, and the gap between them."""

    name: str
    kind: str
    before: float | None
    after: float | None

    @property
    def change(self) -> float | None:
        if self.before is None or self.after is None:
            return None
        return self.after - self.before


@dataclass(frozen=True, slots=True)
class FieldComparison:
    """One field across two runs."""

    field: str
    deltas: tuple[Delta, ...]
    moved_labels: tuple[tuple[str, float], ...]
    improved: tuple[str, ...]
    regressed: tuple[str, ...]
    unchanged: int
    only_in_a: int
    only_in_b: int
    untranslatable: int = 0

    @property
    def net(self) -> int:
        return len(self.improved) - len(self.regressed)

    @property
    def compared(self) -> int:
        return len(self.improved) + len(self.regressed) + self.unchanged


@dataclass(frozen=True, slots=True)
class RunComparison:
    """Both runs, every shared field, and what cannot be concluded."""

    a: LoadedRun
    b: LoadedRun
    fields: tuple[FieldComparison, ...]
    notes: tuple[str, ...] = ()
    cannot_tell: tuple[str, ...] = field(default_factory=tuple)


def _version_check(
    a: LoadedRun, b: LoadedRun, migration: Migration | None
) -> tuple[list[str], dict[str, Migration]]:
    """Refuse a comparison across a taxonomy edit without a mapping."""
    notes: list[str] = []
    usable: dict[str, Migration] = {}
    for ref_a in sorted(a.taxonomies):
        name = parse_ref(ref_a).id
        matching = [r for r in b.taxonomies if parse_ref(r).id == name]
        if not matching:
            notes.append(f"{ref_a} is not used by {b.run_id} at all")
            continue
        ref_b = matching[0]
        if ref_a == ref_b:
            if a.taxonomies[ref_a] != b.taxonomies[ref_b]:
                raise ComparisonError(
                    f"both runs claim {ref_a} but its content differs between them "
                    f"({a.taxonomies[ref_a][:8]}... vs {b.taxonomies[ref_b][:8]}...).\n"
                    "  One of them ran against an edited file without the version being "
                    "bumped. Nothing below would be comparable."
                )
            continue
        if migration is None:
            raise ComparisonError(
                f"{a.run_id} used {ref_a} and {b.run_id} used {ref_b}.\n"
                "  Without a migration mapping these cannot be compared: lining labels up "
                "by name across a taxonomy edit reads a rename as a distribution shift and "
                "blames the model for a change you made.\n"
                "  Pass --migration pointing at a file that maps one version onto the other."
            )
        if (str(migration.source), str(migration.target)) != (ref_a, ref_b):
            raise ComparisonError(
                f"the migration maps {migration.source} onto {migration.target}, but these "
                f"runs used {ref_a} and {ref_b}. A mapping applied between the wrong pair "
                "of versions produces a comparison that looks fine and means nothing."
            )
        usable[name] = migration
        notes.append(
            f"{ref_a} → {ref_b} through {migration.path.name if migration.path else 'a mapping'}"
        )
        if migration.dropped:
            notes.append(
                f"{len(migration.dropped)} label(s) have no equivalent in {ref_b} "
                f"({', '.join(migration.dropped[:3])}) — items carrying them are excluded "
                "and counted"
            )
        if migration.merged:
            merged = "; ".join(
                f"{' + '.join(sources)} → {target}"
                for target, sources in list(migration.merged.items())[:2]
            )
            notes.append(f"labels merged: {merged}")
    return notes, usable


def _distribution_shift(
    before: Mapping[str, float], after: Mapping[str, float], migration: Migration | None
) -> tuple[tuple[tuple[str, float], ...], int]:
    """How far each label's share moved, translated when versions differ."""
    translated: dict[str, float] = {}
    untranslatable = 0
    for label, share in before.items():
        if migration is None:
            translated[label] = translated.get(label, 0.0) + share
            continue
        target = migration.translate(label)
        if target is None:
            untranslatable += 1
            continue
        translated[target] = translated.get(target, 0.0) + share

    moves = {
        label: round(after.get(label, 0.0) - translated.get(label, 0.0), 4)
        for label in sorted(set(translated) | set(after))
    }
    ranked = tuple(sorted(moves.items(), key=lambda kv: -abs(kv[1])))
    return ranked, untranslatable


def compare_runs(
    a: LoadedRun, b: LoadedRun, *, migration: Migration | None = None, top_labels: int = 5
) -> RunComparison:
    """Line two runs up, and say plainly what the comparison cannot settle."""
    notes, migrations = _version_check(a, b, migration)

    if a.prompt and b.prompt and a.prompt != b.prompt:
        notes.append(
            f"the prompt changed, {a.prompt} → {b.prompt}. That is what a comparison is "
            "for; it also means every difference below has more than one possible cause."
        )
    if a.model and b.model and a.model != b.model:
        notes.append(
            f"the producing model changed, {a.model} → {b.model}. A prompt difference and "
            "a model difference cannot be told apart from one comparison."
        )

    in_b = b.metrics.get("modes") or {}
    shared = [name for name in (a.metrics.get("modes") or {}) if name in in_b]
    comparisons: list[FieldComparison] = []
    for name in shared:
        scores_a, scores_b = a.field_scores(name), b.field_scores(name)
        deltas = tuple(
            Delta(
                name=label,
                kind=kind,
                before=_number(scores_a.get(key)),
                after=_number(scores_b.get(key)),
            )
            for key, label, kind in HEADLINE
            if scores_a.get(key) is not None or scores_b.get(key) is not None
        )

        field_migration = migrations.get(name) or next(iter(migrations.values()), None)
        moved, untranslatable = _distribution_shift(
            a.distribution(name), b.distribution(name), field_migration
        )

        before, after = a.outcome_by_item(name), b.outcome_by_item(name)
        both = sorted(set(before) & set(after))
        improved = tuple(i for i in both if not before[i] and after[i])
        regressed = tuple(i for i in both if before[i] and not after[i])
        comparisons.append(
            FieldComparison(
                field=name,
                deltas=deltas,
                moved_labels=moved[:top_labels],
                improved=improved,
                regressed=regressed,
                unchanged=len(both) - len(improved) - len(regressed),
                only_in_a=len(set(before) - set(after)),
                only_in_b=len(set(after) - set(before)),
                untranslatable=untranslatable,
            )
        )

    return RunComparison(
        a=a,
        b=b,
        fields=tuple(comparisons),
        notes=tuple(notes),
        cannot_tell=_cannot_tell(comparisons),
    )


def _cannot_tell(fields: Sequence[FieldComparison]) -> tuple[str, ...]:
    """The conclusion this command deliberately will not draw."""
    lines = []
    for comparison in fields:
        if not comparison.compared:
            continue
        lines.append(
            f"whether {comparison.field} actually got better or worse\n"
            f"{len(comparison.improved)} improved and {len(comparison.regressed)} regressed "
            f"out of {comparison.compared} shared items, with {comparison.unchanged} "
            f"unchanged. A net {comparison.net:+d} is not a result without a test over the "
            "items that actually differ — most items tie in a real A/B, and quoting a win "
            "rate without one is how underpowered changes get shipped. That test is v1."
        )
    return tuple(lines)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def load_and_compare(
    first: str | Path,
    second: str | Path,
    *,
    migration_path: str | Path | None = None,
) -> RunComparison:
    """Load two directories and compare them."""
    migration = None
    if migration_path is not None:
        from .taxonomy import load_migration

        try:
            migration = load_migration(migration_path)
        except TaxonomyError as exc:
            raise ComparisonError(str(exc)) from None
    return compare_runs(
        load_run_directory(first), load_run_directory(second), migration=migration
    )
