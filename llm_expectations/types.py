"""The value types every other module speaks in.

This module imports nothing from the rest of the package, and nothing else in
the package may make it. If something here needs to import upward, the layering
is wrong (DESIGN.md §12).

Two absences are deliberate and are pinned by tests:

``Verdict`` has no error probability and no triage score. A judge does not
produce either; a calibrator computes one later from a verdict plus a fitted
model (DESIGN.md §7). A field here would invite writing it at judge time, which
is the bug that section exists to prevent. It lives on ``RiskRow`` instead.

``Output`` has no label. Human answers travel as ``Label`` on a separate path
that judge-calling code does not accept, so a judge cannot be graded against an
answer key it was shown (DESIGN.md §2).
"""

from __future__ import annotations

# ``field`` is the domain's own noun here, so dataclasses keeps its qualifier.
import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any

__all__ = [
    "ABSTAIN",
    "Exclusion",
    "Finding",
    "Grain",
    "Item",
    "Label",
    "Mode",
    "Output",
    "RiskRow",
    "Severity",
    "Status",
    "Verdict",
]

#: The reserved output value meaning "the model declined to pick a label".
#: A JSON ``null`` in an output file means the same thing. A taxonomy may not
#: define a label with this name — ``taxonomy.py`` rejects one that does.
ABSTAIN = "abstain"


class Status(Enum):
    """The three states a check or a verdict can land in.

    ``UNSCORED`` is the one that earns its keep: without a third state, a run
    that quietly stopped checking reports green. It never becomes a pass —
    :attr:`is_pass` is the only way to ask, and it answers ``False`` here.
    """

    PASS = "pass"  # noqa: S105 — a check result, not a credential
    FAIL = "fail"
    UNSCORED = "unscored"

    @property
    def is_pass(self) -> bool:
        return self is Status.PASS

    @property
    def is_scored(self) -> bool:
        return self is not Status.UNSCORED


class Grain(Enum):
    """What a check looks at — not what kind of field it belongs to."""

    FIELD = "field"  # one item, one field
    ITEM = "item"  # one item, several fields
    CORPUS = "corpus"  # the whole batch


class Severity(Enum):
    """How hard a guardrail bites. None of them fail the run.

    ``STOP`` removes the numbers it invalidates and reports why; it does not
    stop the run (DESIGN.md §10).
    """

    STOP = "stop"
    WARN = "warn"
    NOTE = "note"


class Mode(IntEnum):
    """How much human labelling a field has, decided per field, every run.

    Ordered so ``mode >= Mode.LABELLED`` is the natural way to ask whether a
    metric that needs an answer key can be computed at all.
    """

    NO_LABELS = 0  # screening only
    LABELLED = 1  # one label per item
    DOUBLE_LABELLED = 2  # two independent labels, so the taxonomy can be judged


@dataclass(frozen=True, slots=True)
class Item:
    """The unit being processed — a session, a ticket, a document."""

    id: str
    text: str
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Output:
    """What the model produced for one item, all fields side by side.

    ``values`` holds the row as it arrived, minus ``item_id``. The schema — not
    this type — decides which keys are fields under test; anything else rides
    along (a producer confidence, a prompt id) and is available to checks.
    """

    item_id: str
    values: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def get(self, field_name: str) -> Any:
        return self.values.get(field_name)

    def is_abstain(self, field_name: str) -> bool:
        value = self.values.get(field_name)
        return value is None or value == ABSTAIN


@dataclass(frozen=True, slots=True)
class Label:
    """A human answer for one item and field. Optional, and kept apart.

    ``annotator`` is what makes two-annotator checks possible: the same
    ``(item_id, field)`` appearing twice under different annotators is a
    disagreement, not a duplicate.
    """

    item_id: str
    field: str
    label: str
    annotator: str


@dataclass(frozen=True, slots=True)
class Verdict:
    """One judge's opinion on one output.

    Both field kinds return this same shape, which is what lets approval rates,
    parse rates, panel voting, leniency, caching and the health table be
    written once (DESIGN.md §4).
    """

    judge_id: str
    check_id: str
    item_id: str
    field: str
    status: Status
    raw_confidence: float | None
    reason: str
    detail: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Finding:
    """One check's result for one row. Carries *why*, not just a verdict.

    ``threshold_from`` records which layer supplied the threshold — a built-in
    default, ``settings.yml``, or the field's own block — so the report can
    print the bar next to the number that cleared or missed it.
    """

    run_id: str
    check: str
    grain: Grain
    status: Status
    item_id: str | None = None
    field: str | None = None
    score: float | None = None
    threshold: Any = None
    threshold_from: str | None = None
    evidence: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    judge: str | None = None
    cost_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class RiskRow:
    """One item's place in the review queue, and where that place came from.

    ``calibrated`` is not decoration. An uncalibrated score is a ranking by a
    number the model emitted, which is not a probability of error, and every
    number derived from it is stamped as such.

    ``triage_score`` is ``None`` when the judge could not decide or its reply
    could not be read. Such an item still gets a row — leaving it out is how
    the unjudged items get forgotten — but it has no place in the order, and
    inventing one for it would be the third-state failure in miniature.
    """

    item_id: str
    triage_score: float | None
    strategy: str
    calibrated: bool
    raw_confidence_mean: float | None = None
    calibrated_error_probability: float | None = None
    calibration_id: str | None = None


@dataclass(frozen=True, slots=True)
class Exclusion:
    """One reason rows left the run, and how many.

    Dropping items is normal. Dropping them *because they scored badly* is how
    numbers get manufactured, and it is invisible unless every exclusion states
    a reason — so exclusions are data, not a log line.
    """

    kind: str  # "filtered" | "dropped"
    reason: str
    n: int
    item_ids: tuple[str, ...] = ()
