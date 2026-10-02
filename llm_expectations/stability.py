"""Does the model give the same answer twice?

A diagnostic, on a sample, and it never touches your outputs. Three rules
from DESIGN.md §5 are enforced rather than advised:

**Sample only.** Regenerating a corpus to measure its stability costs what
producing it cost. The config requires a sample size and this module never
looks past it.

**Never overwrites your outputs.** Regenerated answers go to their own file in
the run directory. The outputs under test are the one thing that must survive
the test.

**Two numbers, labelled differently, never averaged.** They answer different
questions and only the temperature says which you are getting:

    temperature 0     SERVING stability. Expect ~100%. Anything less means
                      your serving stack is nondeterministic, which is worth
                      knowing and is not a model problem.

    temperature > 0   DECISION stability. How fragile is this label when the
                      model rolls the dice? A field that flips a quarter of
                      the time is telling you the boundary is unclear, not
                      that the stack is broken.

Reporting one number for both would hide whichever you actually have.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Regenerate, RunConfig
from .judges.base import JudgeError, JudgeRequest, Provider
from .judges.panel import sample_items
from .schema import FieldKind
from .types import Item, Output

__all__ = [
    "FieldStability",
    "StabilityReport",
    "measure_stability",
    "regenerate_sample",
    "write_regenerated",
]

_JSON = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True, slots=True)
class FieldStability:
    """One field's agreement across repeated generations."""

    field: str
    compared: int
    agreed: int
    flipped: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @property
    def stability(self) -> float | None:
        return self.agreed / self.compared if self.compared else None


@dataclass(frozen=True, slots=True)
class StabilityReport:
    """What repeated generation showed, and which question it answers."""

    kind: str
    temperature: float
    runs: int
    sampled: int
    fields: tuple[FieldStability, ...]
    unreadable: int = 0
    note: str | None = None

    @property
    def reads_as_broken(self) -> str | None:
        """What a low number here would mean, which depends on the kind.

        The same 85% is a serving bug in one reading and a fuzzy boundary in
        the other, and a report that did not say which would send someone to
        the wrong place.
        """
        scored = [(f, f.stability) for f in self.fields if f.stability is not None]
        if not scored:
            return None
        # `x or default` is wrong on a float: a stability of 0.0 is the
        # worst possible result and would read as missing.
        lowest, value = min(scored, key=lambda pair: pair[1])
        if value >= 0.99:
            return None
        if self.kind == "serving":
            return (
                f"{lowest.field} answered differently {1 - value:.0%} of the time at "
                "temperature 0. That is not the model being uncertain — a deterministic "
                "request returned two answers, so something in your serving path is "
                "nondeterministic."
            )
        return (
            f"{lowest.field} flips {1 - value:.0%} of the time when the model rolls the "
            "dice. That is how fragile the label is, not whether it is right — a fragile "
            "label usually means the boundary is unclear."
        )


def _prompt(regenerate: Regenerate) -> str:
    try:
        return regenerate.prompt_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise JudgeError(
            f"no prompt file at {regenerate.prompt_file}. The stability check has to send "
            "the same prompt that produced the outputs; it cannot reconstruct it."
        ) from None


def regenerate_sample(
    config: RunConfig,
    items: Mapping[str, Item],
    provider: Provider,
    *,
    fields: Sequence[str],
) -> tuple[list[dict[str, Any]], int]:
    """Ask the producing model again, over a sample, N times.

    Returns the regenerated rows and how many replies could not be read.
    Unreadable replies are counted and excluded — never defaulted, for the
    same reason a judge's unreadable reply is never defaulted.
    """
    regenerate = config.produced_by.regenerate if config.produced_by else None
    if regenerate is None:
        return [], 0

    template = _prompt(regenerate)
    chosen = sample_items(tuple(items), regenerate.sample, seed="stability")
    rows: list[dict[str, Any]] = []
    unreadable = 0
    for attempt in range(regenerate.runs):
        for item_id in chosen:
            request = JudgeRequest(
                system=template,
                user=items[item_id].text,
                max_tokens=int(config.settings.value("judge_max_tokens")) * 4,
                temperature=regenerate.temperature,
            )
            try:
                reply = provider.complete(request)
            except JudgeError:
                unreadable += 1
                continue
            parsed = _read(reply.text, fields)
            if parsed is None:
                unreadable += 1
                continue
            rows.append({"item_id": item_id, "run": attempt, **parsed})
    return rows, unreadable


def _read(text: str, fields: Sequence[str]) -> dict[str, Any] | None:
    match = _JSON.search(text or "")
    if match is None:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return {name: payload.get(name) for name in fields if name in payload} or None


def measure_stability(
    config: RunConfig,
    original: Mapping[str, Output],
    regenerated: Sequence[Mapping[str, Any]],
    *,
    unreadable: int = 0,
    top_flips: int = 5,
) -> StabilityReport | None:
    """Compare the regenerated answers against each other and the originals.

    The original counts as one generation. A model that agrees with its two
    re-runs but not with the output on disk has told you the output on disk
    came from something else.
    """
    regenerate = config.produced_by.regenerate if config.produced_by else None
    if regenerate is None:
        return None

    names = [
        name
        for name, spec in config.schema.fields.items()
        if spec.kind is FieldKind.ASSIGNED
    ]
    by_item: dict[str, dict[str, list[Any]]] = {}
    for row in regenerated:
        item_id = str(row.get("item_id"))
        for name in names:
            if name in row:
                by_item.setdefault(item_id, {}).setdefault(name, []).append(row[name])

    fields: list[FieldStability] = []
    for name in names:
        compared = agreed = 0
        flips: list[tuple[str, tuple[str, ...]]] = []
        for item_id, answers in sorted(by_item.items()):
            values = answers.get(name)
            if not values:
                continue
            produced = original.get(item_id)
            seen = [*values, produced.get(name)] if produced is not None else list(values)
            compared += 1
            distinct = Counter(str(v) for v in seen)
            if len(distinct) == 1:
                agreed += 1
            elif len(flips) < top_flips:
                flips.append((item_id, tuple(sorted(distinct))))
        fields.append(
            FieldStability(field=name, compared=compared, agreed=agreed, flipped=tuple(flips))
        )

    note = None
    if unreadable:
        note = (
            f"{unreadable} regenerated reply(s) could not be read and are excluded. They "
            "are not counted as agreement — a default would invent stability."
        )
    return StabilityReport(
        kind=regenerate.measures,
        temperature=regenerate.temperature,
        runs=regenerate.runs,
        sampled=len(by_item),
        fields=tuple(fields),
        unreadable=unreadable,
        note=note,
    )


def write_regenerated(directory: Path, rows: Sequence[Mapping[str, Any]]) -> Path | None:
    """Write the regenerated answers to their own file, never to yours."""
    if not rows:
        return None
    path = directory / "regenerated.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
