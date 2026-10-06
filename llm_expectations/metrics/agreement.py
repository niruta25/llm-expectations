"""What a panel's votes are worth, and where its disagreement concentrates.

Nothing here calls a model. It reads verdicts off disk, which is what makes
``analyse`` free.

The headline number is **effective votes**: three judges who agree about
everything are one judge you paid for three times, and the raw agreement rate
does not say that out loud.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

from ..types import Label, Status, Verdict

__all__ = [
    "AgreementReport",
    "AnnotatorAgreement",
    "FuzzyPair",
    "FuzzyReport",
    "Leniency",
    "annotator_agreement",
    "effective_votes",
    "fuzzy_pairs",
    "leniency",
    "pairwise_agreement",
]


@dataclass(frozen=True, slots=True)
class AgreementReport:
    """How much the panel agreed, and how many opinions that is worth."""

    members: tuple[str, ...]
    compared: int
    agreement: float | None
    mean_correlation: float | None
    effective_votes: float | None
    warning: str | None = None

    @property
    def votes_paid_for(self) -> int:
        return len(self.members)


def _decisions(
    verdicts: Sequence[Verdict], members: Sequence[str]
) -> dict[tuple[str, str], dict[str, int]]:
    """Per (item, field), each member's decision as 1 approve / 0 reject.

    Only decisions. A ``cannot_decide`` is not half a vote, and an unreadable
    reply is not a vote at all — defaulting either would be inventing
    agreement out of a hole in the data.
    """
    rows: dict[tuple[str, str], dict[str, int]] = {}
    allowed = set(members)
    for verdict in verdicts:
        if verdict.judge_id not in allowed or verdict.status is Status.UNSCORED:
            continue
        key = (verdict.item_id, verdict.field)
        rows.setdefault(key, {})[verdict.judge_id] = int(verdict.status is Status.PASS)
    return rows


def pairwise_agreement(
    verdicts: Sequence[Verdict], members: Sequence[str]
) -> tuple[float | None, int]:
    """Share of comparable rows where two members landed the same way."""
    same = total = 0
    for decisions in _decisions(verdicts, members).values():
        for left, right in combinations(sorted(decisions), 2):
            total += 1
            same += decisions[left] == decisions[right]
    return (same / total if total else None), total


def _correlation(pairs: Sequence[tuple[int, int]]) -> float | None:
    """Phi between two binary raters: the correlation the design effect wants.

    ``None`` when either rater is constant across the sample. A judge that
    said the same thing every time has no variance to correlate, and the
    conventional substitutes — zero, or one — would each be a claim the data
    does not make.
    """
    n = len(pairs)
    if n < 2:
        return None
    left = [a for a, _ in pairs]
    right = [b for _, b in pairs]
    mean_left, mean_right = sum(left) / n, sum(right) / n
    cov = sum((a - mean_left) * (b - mean_right) for a, b in pairs)
    var_left = sum((a - mean_left) ** 2 for a in left)
    var_right = sum((b - mean_right) ** 2 for b in right)
    if var_left == 0 or var_right == 0:
        return None
    return float(cov / (var_left * var_right) ** 0.5)


def effective_votes(
    verdicts: Sequence[Verdict], members: Sequence[str], *, warn_below: float = 0.5
) -> AgreementReport:
    """How many independent opinions a panel of m is actually worth.

    The design-effect formula for m correlated raters::

        n_eff = m / (1 + (m - 1) * rho)

    with ``rho`` the mean pairwise correlation between members. Perfectly
    correlated judges give one effective vote however many you bought;
    uncorrelated ones give you all m. That is the quantity the warning in
    DESIGN.md §1 is about — "three judges giving 1.1 opinions means you pay 3×
    for nothing" — and a raw agreement rate cannot express it, because high
    agreement on a lopsided target is partly just the base rate.
    """
    members = tuple(members)
    m = len(members)
    agreement, compared = pairwise_agreement(verdicts, members)
    rows = _decisions(verdicts, members)

    if m < 2:
        return AgreementReport(
            members, compared, agreement, None, None,
            "a panel of one has no agreement to measure",
        )

    correlations = []
    constant = []
    for left, right in combinations(members, 2):
        pairs = [
            (d[left], d[right]) for d in rows.values() if left in d and right in d
        ]
        value = _correlation(pairs)
        if value is None:
            constant.append(f"{left}/{right}")
        else:
            correlations.append(value)

    if not correlations:
        return AgreementReport(
            members, compared, agreement, None, None,
            "no pair of judges varied enough to correlate — at least one of them "
            "answered the same way on every item, so there is no independence to measure",
        )

    rho = sum(correlations) / len(correlations)
    # A negative mean correlation would push n_eff above m, which is not a
    # thing you can buy. Judges that disagree more than chance are still worth
    # at most m opinions.
    n_eff = min(float(m), m / (1 + (m - 1) * rho)) if (1 + (m - 1) * rho) > 0 else float(m)

    warning = None
    if n_eff / m < warn_below:
        warning = (
            f"you are paying for {m} opinions and receiving about {n_eff:.1f}. "
            "Judges that fail the same way are nearly one judge; what helps is one "
            "that fails the other way."
        )
    if constant:
        note = f"{len(constant)} pair(s) could not be correlated: {', '.join(constant)}"
        warning = f"{warning} {note}" if warning else note

    return AgreementReport(members, compared, agreement, rho, n_eff, warning)


@dataclass(frozen=True, slots=True)
class Leniency:
    """Who waved it through when two judges disagreed."""

    lenient: str
    strict: str
    lenient_approved: int
    strict_approved: int

    @property
    def ratio(self) -> float | None:
        return self.lenient_approved / self.strict_approved if self.strict_approved else None


def leniency(verdicts: Sequence[Verdict], members: Sequence[str]) -> list[Leniency]:
    """Rank judges against each other by who approves when they split.

    Free, and it needs no human labels: it says nothing about who is *right*,
    only who is softer. The absolute version — how often a judge waves through
    a wrong label versus rejects a right one — needs labels, and is
    ``judge_direction`` in ``classification.py``.
    """
    counts: Counter[tuple[str, str]] = Counter()
    for decisions in _decisions(verdicts, members).values():
        for left, right in combinations(sorted(decisions), 2):
            if decisions[left] == decisions[right]:
                continue
            approver, rejecter = (left, right) if decisions[left] else (right, left)
            counts[(approver, rejecter)] += 1

    seen: set[frozenset[str]] = set()
    out = []
    for left, right in combinations(sorted(set(members)), 2):
        key = frozenset({left, right})
        if key in seen:
            continue
        seen.add(key)
        left_approved, right_approved = counts[(left, right)], counts[(right, left)]
        if not (left_approved or right_approved):
            continue
        if left_approved >= right_approved:
            out.append(Leniency(left, right, left_approved, right_approved))
        else:
            out.append(Leniency(right, left, right_approved, left_approved))
    return sorted(out, key=lambda entry: -(entry.lenient_approved - entry.strict_approved))


@dataclass(frozen=True, slots=True)
class FuzzyPair:
    """One boundary of one field that the judges keep splitting on.

    Undirected: ``left`` and ``right`` are sorted, not assigned-then-suggested.
    Which way the confusion runs needs human labels to answer.
    """

    field: str
    left: str
    right: str
    splits: int

    def __str__(self) -> str:
        return f"{self.left} ↔ {self.right}"


@dataclass(frozen=True, slots=True)
class FuzzyReport:
    """Where panel disagreement concentrates, and how much of it is placeable."""

    pairs: tuple[FuzzyPair, ...]
    splits: int
    attributed: int
    unusable: int = 0

    @property
    def unattributed(self) -> int:
        return self.splits - self.attributed


def fuzzy_pairs(
    verdicts: Sequence[Verdict],
    members: Sequence[str],
    outputs: Mapping[str, object],
    *,
    vocabularies: Mapping[str, frozenset[str]] | None = None,
    top: int = 5,
) -> FuzzyReport:
    """Which label boundaries the panel keeps splitting on.

    The second half of each pair is the rejecting judge's own suggestion,
    never the taxonomy's shape. A sibling of the assigned label is a plausible
    guess, and a guess is not a finding — a split whose dissenter named no
    alternative is counted as unattributed rather than filled in.

    ``vocabularies`` maps a field to the labels its taxonomy actually defines.
    A suggestion outside that set is a judge naming something that does not
    exist for this field — usually a label from a different taxonomy — and it
    is discarded rather than reported. A hallucinated label is a judge-health
    problem, and promoting it to a fuzzy *boundary* would send someone to
    rewrite two definitions over a pair that was never real.

    Pairs are **undirected**. A boundary two judges keep falling either side
    of is one boundary, and counting ``a ↔ b`` apart from ``b ↔ a`` would
    split the evidence for it in half and bury it. Direction is a separate
    question — whether the confusion is symmetric (fix the taxonomy) or
    one-way (fix the prompt) — and answering it needs human labels, so it
    comes from the confusion matrix, which needs them.
    """
    rows = _decisions(verdicts, members)
    by_key: dict[tuple[str, str], list[Verdict]] = {}
    for verdict in verdicts:
        by_key.setdefault((verdict.item_id, verdict.field), []).append(verdict)

    counts: Counter[tuple[str, str, str]] = Counter()
    split = attributed = unusable = 0
    for (item_id, field), decisions in rows.items():
        if len(set(decisions.values())) < 2:
            continue
        split += 1
        output = outputs.get(item_id)
        assigned = output.get(field) if output is not None else None  # type: ignore[attr-defined]
        known = vocabularies.get(field) if vocabularies else None
        suggestions = {
            str(v.detail["instead"])
            for v in by_key.get((item_id, field), ())
            if v.status is Status.FAIL and isinstance(v.detail.get("instead"), str)
        }
        if not isinstance(assigned, str) or not suggestions:
            continue
        real = {s for s in suggestions if s != assigned and (known is None or s in known)}
        if not real:
            unusable += bool(suggestions - {assigned})
            continue
        attributed += 1
        for suggestion in real:
            left, right = sorted((assigned, suggestion))
            counts[(field, left, right)] += 1

    pairs = tuple(
        FuzzyPair(field, left, right, n)
        for (field, left, right), n in counts.most_common(top)
    )
    return FuzzyReport(pairs=pairs, splits=split, attributed=attributed, unusable=unusable)


@dataclass(frozen=True, slots=True)
class AnnotatorAgreement:
    """Whether two people can separate the labels — mode 2's question.

    The strongest evidence the fuzzy-pair detector takes. A model confusing
    two labels might be a model problem; two trained humans confusing the
    same two labels is not. The taxonomy is wrong, and no amount of prompt
    work will fix it.
    """

    field: str
    compared: int
    agreed: int
    pairs: tuple[tuple[str, str, int], ...] = ()
    annotators: tuple[str, ...] = ()

    @property
    def agreement(self) -> float | None:
        return self.agreed / self.compared if self.compared else None

    @property
    def disagreements(self) -> int:
        return self.compared - self.agreed

    @property
    def concentration(self) -> float | None:
        """Share of all disagreement sitting on the single worst pair.

        Scattered disagreement is annotators being human. Concentrated
        disagreement is one boundary nobody can see, and that is a finding.
        """
        if not self.disagreements or not self.pairs:
            return None
        return self.pairs[0][2] / self.disagreements


def annotator_agreement(
    labels: Sequence[Label], field: str, *, top: int = 5
) -> AnnotatorAgreement:
    """Compare every pair of annotators who labelled the same item.

    Items with one annotator contribute nothing — they are not evidence
    either way, and counting them as agreement would dilute the rate towards
    a reassuring number as a corpus grows.
    """
    by_item: dict[str, dict[str, str]] = {}
    for label in labels:
        if label.field == field:
            by_item.setdefault(label.item_id, {})[label.annotator] = label.label

    compared = agreed = 0
    pairs: Counter[tuple[str, str]] = Counter()
    people: set[str] = set()
    for answers in by_item.values():
        if len(answers) < 2:
            continue
        people.update(answers)
        for left, right in combinations(sorted(answers), 2):
            compared += 1
            if answers[left] == answers[right]:
                agreed += 1
            else:
                first, second = sorted((answers[left], answers[right]))
                pairs[(first, second)] += 1

    return AnnotatorAgreement(
        field=field,
        compared=compared,
        agreed=agreed,
        pairs=tuple((a, b, n) for (a, b), n in pairs.most_common(top)),
        annotators=tuple(sorted(people)),
    )
