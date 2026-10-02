"""Aggregates computed from findings and verdicts on disk.

Nothing in this package calls a model. That is the rule that makes ``analyse``
free, and it is what lets you change a threshold, add a metric or fix a
reporting bug and re-run for nothing.
"""

from __future__ import annotations

from .agreement import (
    AgreementReport,
    AnnotatorAgreement,
    FuzzyPair,
    FuzzyReport,
    Leniency,
    annotator_agreement,
    effective_votes,
    fuzzy_pairs,
    leniency,
    pairwise_agreement,
)
from .classification import (
    Classification,
    JudgeDirection,
    LabelScore,
    TreeBucket,
    classify,
    confusion_direction,
    judge_direction,
)

__all__ = [
    "AgreementReport",
    "AnnotatorAgreement",
    "FuzzyPair",
    "Classification",
    "FuzzyReport",
    "JudgeDirection",
    "LabelScore",
    "TreeBucket",
    "Leniency",
    "annotator_agreement",
    "effective_votes",
    "classify",
    "confusion_direction",
    "fuzzy_pairs",
    "judge_direction",
    "leniency",
    "pairwise_agreement",
]
