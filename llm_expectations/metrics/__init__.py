"""Aggregates computed from findings and verdicts on disk.

Nothing in this package calls a model. That is the rule that makes ``analyse``
free, and it is what lets you change a threshold, add a metric or fix a
reporting bug and re-run for nothing.
"""

from __future__ import annotations

from .agreement import (
    AgreementReport,
    FuzzyPair,
    FuzzyReport,
    Leniency,
    effective_votes,
    fuzzy_pairs,
    leniency,
    pairwise_agreement,
)

__all__ = [
    "AgreementReport",
    "FuzzyPair",
    "FuzzyReport",
    "Leniency",
    "effective_votes",
    "fuzzy_pairs",
    "leniency",
    "pairwise_agreement",
]
