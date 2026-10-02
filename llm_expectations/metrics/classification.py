"""Grading assigned labels against human answers.

Four things here, and the order matters because the first is a trap.

**Accuracy is reported and never alone.** On a skewed taxonomy, always
guessing the biggest label scores well and knows nothing, so accuracy always
appears beside the majority-label baseline it has to beat. Macro F1 is the
headline because it weights a rare label the same as a common one.

**The confusion matrix is the actionable output.** Not the score — the score
tells you there is a problem, the matrix tells you where.

**Tree buckets say which kind of wrong.** A sibling, a parent and a different
branch are three different failures with three different fixes, and a single
"wrong" throws that away.

**Direction separates a taxonomy bug from a prompt bug.** Confusion that runs
both ways between two labels means nobody can tell them apart; confusion that
runs one way means the model is biased and the definitions are fine.

Nothing here calls a model. It reads outputs and labels, which is what keeps
``analyse`` free.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from ..taxonomy import Taxonomy
from ..types import ABSTAIN, Label, Output, Status, Verdict

__all__ = [
    "Classification",
    "ConfusionPair",
    "JudgeDirection",
    "LabelScore",
    "TreeBucket",
    "bucket",
    "classify",
    "confusion_direction",
    "judge_direction",
    "primary_labels",
]


class TreeBucket(str, Enum):
    """Which kind of wrong, which is a different question from how often.

    Each bucket points at a different fix: a sibling means two definitions
    need sharpening, a parent means the model is hedging, a different branch
    means it is not reading the item.
    """

    EXACT = "exact"
    RIGHT_PARENT = "right_parent"  # a sibling — the boundary is the problem
    TOO_SHALLOW = "too_shallow"  # stopped at the parent — hedging
    WRONG = "wrong"  # a different branch entirely
    ABSTAINED = "abstained"  # declined to answer; not a wrong answer
    UNKNOWN = "unknown"  # not a label in this taxonomy at all


def bucket(predicted: object, truth: str, taxonomy: Taxonomy) -> TreeBucket:
    """Place one prediction against one human answer."""
    if predicted is None or predicted == ABSTAIN:
        return TreeBucket.ABSTAINED
    if not isinstance(predicted, str) or predicted not in taxonomy:
        return TreeBucket.UNKNOWN
    if truth not in taxonomy:
        return TreeBucket.UNKNOWN
    if predicted == truth:
        return TreeBucket.EXACT
    if predicted in taxonomy.ancestors(truth):
        return TreeBucket.TOO_SHALLOW
    parent = taxonomy.parent_of(truth)
    # A shared parent is only partial credit when there is a parent to
    # share. In a flat vocabulary every label is a root, so "right parent"
    # would be true of every confusion and would advise sharpening two
    # definitions on a tree that has no branches to confuse.
    if parent is not None and taxonomy.parent_of(predicted) == parent:
        return TreeBucket.RIGHT_PARENT
    return TreeBucket.WRONG


@dataclass(frozen=True, slots=True)
class LabelScore:
    """One label's precision, recall and F1, with the support behind them."""

    label: str
    support: int
    predicted: int
    correct: int
    floor: int = 30

    @property
    def precision(self) -> float | None:
        return self.correct / self.predicted if self.predicted else None

    @property
    def recall(self) -> float | None:
        return self.correct / self.support if self.support else None

    @property
    def f1(self) -> float | None:
        precision, recall = self.precision, self.recall
        if not precision or not recall:
            return 0.0 if (self.support or self.predicted) else None
        return 2 * precision * recall / (precision + recall)

    @property
    def under_floor(self) -> bool:
        """Too few examples of this label for its numbers to mean much."""
        return self.support < self.floor

    @property
    def weak(self) -> bool:
        """Recall below 0.5 — the threshold the design says to flag."""
        return self.recall is not None and self.recall < 0.5


@dataclass(frozen=True, slots=True)
class ConfusionPair:
    """How often one label was predicted when another was the answer."""

    truth: str
    predicted: str
    n: int


@dataclass(frozen=True, slots=True)
class Classification:
    """Everything the human answers say about one assigned field."""

    field: str
    n: int
    scores: tuple[LabelScore, ...]
    buckets: Mapping[TreeBucket, int]
    confusion: tuple[ConfusionPair, ...]
    majority_label: str | None
    label_floor: int = 30

    @property
    def accuracy(self) -> float | None:
        return self.buckets.get(TreeBucket.EXACT, 0) / self.n if self.n else None

    @property
    def majority_baseline(self) -> float | None:
        """What "always guess the biggest label" would score.

        Accuracy is never printed without this. A taxonomy where one label
        holds 60% of the corpus hands a useless model 60% accuracy, and the
        number only means something next to what it had to beat.
        """
        if self.majority_label is None or not self.n:
            return None
        support = next(
            (s.support for s in self.scores if s.label == self.majority_label), 0
        )
        return support / self.n

    @property
    def macro_f1(self) -> float | None:
        """The headline: every label counts the same, however rare.

        Averaged over labels that *appear in the human answers*, not over
        every label the taxonomy defines. A tree with forty labels of which
        six are used would otherwise score near zero no matter how good the
        model is, which measures the taxonomy's size rather than the model.
        """
        present = [s for s in self.scores if s.support]
        values = [s.f1 for s in present if s.f1 is not None]
        return sum(values) / len(values) if values else None

    @property
    def weak_labels(self) -> tuple[LabelScore, ...]:
        return tuple(s for s in self.scores if s.weak and not s.under_floor)

    @property
    def beats_majority(self) -> bool | None:
        accuracy, baseline = self.accuracy, self.majority_baseline
        if accuracy is None or baseline is None:
            return None
        return accuracy > baseline


def primary_labels(labels: Sequence[Label], field: str) -> dict[str, str]:
    """One human answer per item, from the first annotator alphabetically.

    Mode 2's second opinions are a different measurement — whether the
    *taxonomy* is crisp — and pooling them here would make the answer key
    depend on which annotator was read last.
    """
    primary: dict[str, str] = {}
    for label in sorted(labels, key=lambda label: label.annotator):
        if label.field == field:
            primary.setdefault(label.item_id, label.label)
    return primary


def classify(
    field: str,
    outputs: Mapping[str, Output],
    labels: Sequence[Label],
    taxonomy: Taxonomy,
    *,
    label_floor: int = 30,
    top_confusions: int = 10,
) -> Classification:
    """Score one assigned field against the human answers."""
    truth_by_item = primary_labels(labels, field)
    pairs: list[tuple[str, object]] = [
        (truth, outputs[item_id].get(field))
        for item_id, truth in truth_by_item.items()
        if item_id in outputs
    ]

    buckets: Counter[TreeBucket] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    support: Counter[str] = Counter()
    predicted: Counter[str] = Counter()
    correct: Counter[str] = Counter()

    for truth, prediction in pairs:
        placed = bucket(prediction, truth, taxonomy)
        buckets[placed] += 1
        support[truth] += 1
        if placed is TreeBucket.ABSTAINED:
            continue
        name = str(prediction)
        predicted[name] += 1
        if placed is TreeBucket.EXACT:
            correct[name] += 1
        else:
            confusion[(truth, name)] += 1

    scores = tuple(
        LabelScore(
            label=name,
            support=support.get(name, 0),
            predicted=predicted.get(name, 0),
            correct=correct.get(name, 0),
            floor=label_floor,
        )
        for name in sorted(set(support) | set(predicted))
    )
    biggest = support.most_common(1)
    return Classification(
        field=field,
        n=len(pairs),
        scores=scores,
        buckets=dict(buckets),
        confusion=tuple(
            ConfusionPair(truth, name, n)
            for (truth, name), n in confusion.most_common(top_confusions)
        ),
        majority_label=biggest[0][0] if biggest else None,
        label_floor=label_floor,
    )


#: Below this many confusions across both directions, a lopsided split is
#: just as likely to be chance as bias. One-nil is not evidence of anything.
DIRECTION_MIN_TOTAL = 10


def confusion_direction(
    confusion: Sequence[ConfusionPair], *, top: int = 5, min_total: int = DIRECTION_MIN_TOTAL
) -> list[dict[str, object]]:
    """Is each confusable pair symmetric or one-way?

    The distinction is the whole value of having labels here:

        SYMMETRIC   payment_failed -> card_declined  23
                    card_declined -> payment_failed  19
                    nobody can separate these     -> fix the TAXONOMY

        ASYMMETRIC  payment_failed -> refund         31
                    refund -> payment_failed          2
                    the model leans one way       -> fix the PROMPT

    A pair is called one-way when the heavier direction carries at least
    four fifths of the traffic. Below that the evidence does not separate
    the two explanations, and it says so rather than picking one.

    And below ``min_total`` confusions in all, neither word is used: a pair
    seen once, in one direction, is not evidence of a one-way bias — it is a
    pair seen once. Those are still listed, marked ``too few to tell``, so a
    boundary that is starting to show is visible without being overclaimed.
    """
    counts = {(pair.truth, pair.predicted): pair.n for pair in confusion}
    seen: set[frozenset[str]] = set()
    out = []
    for (truth, predicted), n in sorted(counts.items(), key=lambda kv: -kv[1]):
        key = frozenset({truth, predicted})
        if key in seen:
            continue
        seen.add(key)
        back = counts.get((predicted, truth), 0)
        total = n + back
        share = max(n, back) / total
        if total < min_total:
            shape = "too few to tell"
            verdict = f"only {total} confusion(s) — not enough to call a direction"
        elif share >= 0.8:
            shape = "asymmetric"
            verdict = "the model leans one way — fix the prompt"
        else:
            shape = "symmetric"
            verdict = "neither direction dominates — fix the taxonomy"
        heavy = (truth, predicted) if n >= back else (predicted, truth)
        out.append(
            {
                "pair": sorted(key),
                "shape": shape,
                "forward": {"from": heavy[0], "to": heavy[1], "n": max(n, back)},
                "backward": {"from": heavy[1], "to": heavy[0], "n": min(n, back)},
                "total": total,
                "verdict": verdict,
            }
        )
    return out[:top]


@dataclass(frozen=True, slots=True)
class JudgeDirection:
    """Which way a judge fails, which is not the same as how often.

    A judge that only ever waves wrong labels through adds nothing to a panel
    that already fails that way. Two lenient judges are nearly one judge; what
    helps is one that fails the other way, and this is the table that tells
    you which you have.
    """

    judge_id: str
    approved_wrong: int
    wrong_total: int
    rejected_right: int
    right_total: int
    agreed: int
    scored: int

    @property
    def approves_wrong_rate(self) -> float | None:
        return self.approved_wrong / self.wrong_total if self.wrong_total else None

    @property
    def rejects_right_rate(self) -> float | None:
        return self.rejected_right / self.right_total if self.right_total else None

    @property
    def accuracy(self) -> float | None:
        return self.agreed / self.scored if self.scored else None

    @property
    def always_approve_accuracy(self) -> float | None:
        """What a judge that never flags anything would score.

        The baseline a judge's accuracy has to clear. Most outputs are
        correct, so "approve everything" is a strong-looking number and a
        judge that cannot beat it is costing money for nothing.
        """
        return self.right_total / self.scored if self.scored else None

    @property
    def leaning(self) -> str:
        approves, rejects = self.approves_wrong_rate, self.rejects_right_rate
        if approves is None or rejects is None:
            return "unknown"
        if approves > rejects * 1.5:
            return "lenient"
        if rejects > approves * 1.5:
            return "strict"
        return "balanced"


def judge_direction(
    verdicts: Sequence[Verdict],
    outputs: Mapping[str, Output],
    labels: Sequence[Label],
    fields: Sequence[str],
) -> list[JudgeDirection]:
    """Grade each judge against the human answers, one row per judge.

    This is downstream of every judge call. The judge was never shown a
    label; the label arrives here, after the verdict is on disk, and is used
    only to score it.
    """
    truth: dict[tuple[str, str], str] = {}
    for name in fields:
        for item_id, answer in primary_labels(labels, name).items():
            truth[(item_id, name)] = answer

    tallies: dict[str, dict[str, int]] = {}
    for verdict in verdicts:
        if verdict.status is Status.UNSCORED:
            continue
        human = truth.get((verdict.item_id, verdict.field))
        output = outputs.get(verdict.item_id)
        if human is None or output is None:
            continue
        really_right = str(output.get(verdict.field)) == human
        said_right = verdict.status is Status.PASS
        tally = tallies.setdefault(
            verdict.judge_id,
            {
                "approved_wrong": 0,
                "wrong_total": 0,
                "rejected_right": 0,
                "right_total": 0,
                "agreed": 0,
                "scored": 0,
            },
        )
        tally["scored"] += 1
        tally["agreed"] += int(said_right == really_right)
        if really_right:
            tally["right_total"] += 1
            tally["rejected_right"] += int(not said_right)
        else:
            tally["wrong_total"] += 1
            tally["approved_wrong"] += int(said_right)

    return [
        JudgeDirection(judge_id=judge_id, **tally) for judge_id, tally in sorted(tallies.items())
    ]
