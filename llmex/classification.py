"""Classification metrics for label-valued extractions.

When the extracted value is a category rather than a span, the question
"is this right?" has an answer a human can supply cheaply and exactly. That
makes classification the one place where a gold set gives you truth directly,
and where a judge can therefore be measured against something.

Everything here is pure Python, no numpy, matching the calibration module.
`report()` deliberately reports macro F1 alongside accuracy: accuracy on an
unbalanced corpus is the same flattering number that field-grain pass rates
are, for the same reason.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .batch import Batch
from .defaults import GOLD_LABEL_KEY

Pair = tuple[str, str]
"""(gold, predicted)."""


def normalise_label(value: object) -> str:
    """Casefold and treat `-`, `_` and spaces as the same separator.

    `Password Reset`, `password_reset` and `password-reset` are one label
    written three ways, and a taxonomy that survives contact with several
    annotators will contain all three.
    """
    text = str(value).strip().casefold()
    for sep in ("-", "_"):
        text = text.replace(sep, " ")
    return " ".join(text.split())


def confusion_matrix(pairs: Iterable[Pair]) -> dict[Pair, int]:
    """Counts keyed by (gold, predicted). Off-diagonal entries are the errors."""
    counts: dict[Pair, int] = {}
    for gold, pred in pairs:
        counts[(gold, pred)] = counts.get((gold, pred), 0) + 1
    return counts


@dataclass
class LabelMetrics:
    label: str
    support: int
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "support": self.support,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class ClassificationReport:
    n: int
    accuracy: float
    macro_f1: float
    per_label: dict[str, LabelMetrics] = field(default_factory=dict)
    matrix: dict[Pair, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "per_label": {k: v.as_dict() for k, v in sorted(self.per_label.items())},
            # Tuple keys are not JSON-serialisable; "gold|predicted" is.
            "matrix": {f"{g}|{p}": n for (g, p), n in sorted(self.matrix.items())},
        }

    def worst_labels(self, n: int = 3) -> list[LabelMetrics]:
        """Where the taxonomy is failing, which is usually where it is ambiguous."""
        ranked = sorted(self.per_label.values(), key=lambda m: (m.f1, -m.support))
        return ranked[:n]


def report(pairs: Sequence[Pair]) -> ClassificationReport:
    if not pairs:
        return ClassificationReport(n=0, accuracy=1.0, macro_f1=1.0)

    matrix = confusion_matrix(pairs)
    labels = sorted({label for pair in pairs for label in pair})
    correct = sum(n for (g, p), n in matrix.items() if g == p)

    per_label: dict[str, LabelMetrics] = {}
    for label in labels:
        tp = matrix.get((label, label), 0)
        predicted = sum(n for (g, p), n in matrix.items() if p == label)
        actual = sum(n for (g, p), n in matrix.items() if g == label)
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        denom = precision + recall
        f1 = 2 * precision * recall / denom if denom else 0.0
        per_label[label] = LabelMetrics(label, actual, precision, recall, f1)

    # Macro, not micro: every label counts the same regardless of how rare it
    # is, so a taxonomy that fails only on its tail cannot hide behind volume.
    scored = [m for m in per_label.values() if m.support]
    macro_f1 = sum(m.f1 for m in scored) / len(scored) if scored else 0.0

    return ClassificationReport(
        n=len(pairs),
        accuracy=correct / len(pairs),
        macro_f1=macro_f1,
        per_label=per_label,
        matrix=matrix,
    )


def pairs_from_batch(
    batch: Batch,
    field_name: str,
    gold_key: str = GOLD_LABEL_KEY,
    normalise: bool = True,
) -> list[Pair]:
    """(gold, predicted) for every record of `field_name` that carries a label.

    Records without gold are omitted rather than counted as anything: an
    unlabelled record is not evidence either way.
    """
    out: list[Pair] = []
    for rec in batch.records:
        if rec.field_name != field_name or gold_key not in rec.meta:
            continue
        gold, pred = rec.meta[gold_key], rec.value
        if normalise:
            out.append((normalise_label(gold), normalise_label(pred)))
        else:
            out.append((str(gold), str(pred)))
    return out


def report_from_batch(
    batch: Batch,
    field_name: str,
    gold_key: str = GOLD_LABEL_KEY,
    normalise: bool = True,
) -> ClassificationReport:
    return report(pairs_from_batch(batch, field_name, gold_key, normalise))
