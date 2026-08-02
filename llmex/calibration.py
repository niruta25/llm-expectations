"""Calibration.

A model-based expectation is a classifier: it predicts "this extraction is
wrong." An uncalibrated classifier is an opinion. This module turns labelled
examples into a threshold you can defend in a design review, and gives the
planner something to refuse when it is missing.

Thresholds are *derived* from a target precision, never typed by hand.

All metrics are implemented without numpy or sklearn so the framework installs
cleanly with only pyyaml.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .defaults import (
    DEFAULT_TARGET_PRECISION,
    DEFAULT_THRESHOLD_KEY,
    FINGERPRINT_LENGTH,
    GOLD_SET_HASH_LENGTH,
    MIN_LABELS_PER_FIELD,
)


def auroc(scores: list[float], correct: list[bool]) -> float:
    """Rank-based AUROC (Mann-Whitney U).

    Measures how well low scores rank the wrong extractions above the right
    ones. Ties are handled by average rank. Returns nan when one class is
    absent, because AUROC is undefined there.
    """
    pos = [s for s, c in zip(scores, correct, strict=True) if not c]  # errors: should score LOW
    neg = [s for s, c in zip(scores, correct, strict=True) if c]
    if not pos or not neg:
        return float("nan")
    ranked = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and scores[ranked[j + 1]] == scores[ranked[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[ranked[k]] = avg
        i = j + 1
    rank_sum_err = sum(r for r, c in zip(ranks, correct, strict=True) if not c)
    n_err, n_ok = len(pos), len(neg)
    u = rank_sum_err - n_err * (n_err + 1) / 2
    return 1.0 - (u / (n_err * n_ok))


def precision_at_k(scores: list[float], correct: list[bool], k: int | None = None) -> float:
    """If you review the k lowest-scoring items, what share are actually wrong?

    k defaults to the true number of errors. More actionable than AUROC for
    capacity planning: it answers the question a reviewer rota actually asks.
    """
    n_err = sum(1 for c in correct if not c)
    k = k or n_err
    if k == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])[:k]
    return sum(1 for i in order if not correct[i]) / k


def confidence_gap(scores: list[float], correct: list[bool]) -> float:
    """Mean score of correct minus mean score of incorrect.

    Unlike the rank metrics this is scale-sensitive, so a human can interpret
    raw score magnitudes rather than only relative ordering.
    """
    ok = [s for s, c in zip(scores, correct, strict=True) if c]
    bad = [s for s, c in zip(scores, correct, strict=True) if not c]
    if not ok or not bad:
        return float("nan")
    return sum(ok) / len(ok) - sum(bad) / len(bad)


def threshold_for_precision(
    scores: list[float], correct: list[bool], target_precision: float
) -> tuple[float, float]:
    """Highest-recall threshold at which flagged items are `target_precision` accurate.

    Returns (threshold, achieved_recall) so you can see what you are giving up.
    Flag anything scoring below the threshold.
    """
    candidates = sorted(set(scores))
    best = (0.0, 0.0)
    n_err = sum(1 for c in correct if not c) or 1
    for t in candidates:
        flagged = [(s, c) for s, c in zip(scores, correct, strict=True) if s < t]
        if not flagged:
            continue
        tp = sum(1 for _, c in flagged if not c)
        precision = tp / len(flagged)
        recall = tp / n_err
        if precision >= target_precision and recall > best[1]:
            best = (t, recall)
    return best


@dataclass
class Calibration:
    id: str
    gold_set_hash: str
    provider_id: str
    model_version: str
    strategy_id: str
    metrics: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    n_labels: int = 0
    created_at: str = ""

    def fingerprint(self) -> str:
        raw = f"{self.gold_set_hash}|{self.provider_id}|{self.model_version}|{self.strategy_id}"
        return hashlib.sha256(raw.encode()).hexdigest()[:FINGERPRINT_LENGTH]

    def valid_for(self, provider_id: str, model_version: str, strategy_id: str) -> bool:
        return (
            self.provider_id == provider_id
            and self.model_version == model_version
            and self.strategy_id == strategy_id
        )

    def threshold_for(self, field_name: str | None) -> float | None:
        if field_name and field_name in self.thresholds:
            return self.thresholds[field_name]
        return self.thresholds.get(DEFAULT_THRESHOLD_KEY)

    @property
    def ref(self) -> str:
        """Lands in `Result.threshold_source`, so every result carries a
        pointer to the evidence for its own threshold."""
        return f"calibration:{self.id}@{self.fingerprint()}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "gold_set_hash": self.gold_set_hash,
            "provider_id": self.provider_id,
            "model_version": self.model_version,
            "strategy_id": self.strategy_id,
            "metrics": self.metrics,
            "thresholds": self.thresholds,
            "n_labels": self.n_labels,
            "created_at": self.created_at,
            "fingerprint": self.fingerprint(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Calibration:
        return cls(
            id=d["id"],
            gold_set_hash=d["gold_set_hash"],
            provider_id=d["provider_id"],
            model_version=d["model_version"],
            strategy_id=d["strategy_id"],
            metrics=d.get("metrics", {}),
            thresholds=d.get("thresholds", {}),
            n_labels=d.get("n_labels", 0),
            created_at=d.get("created_at", ""),
        )


@dataclass
class LabelledScore:
    doc_id: str
    field_name: str | None
    score: float
    is_correct: bool


def calibrate(
    id: str,  # noqa: A002 - mirrors Calibration.id, the public vocabulary
    labels: list[LabelledScore],
    provider_id: str,
    model_version: str,
    strategy_id: str,
    target_precision: float = DEFAULT_TARGET_PRECISION,
    gold_set_hash: str = "",
) -> Calibration:
    scores = [s.score for s in labels]
    correct = [s.is_correct for s in labels]

    thresholds: dict[str, float] = {}
    default_t, default_r = threshold_for_precision(scores, correct, target_precision)
    thresholds[DEFAULT_THRESHOLD_KEY] = default_t

    per_field: dict[str, list[LabelledScore]] = {}
    for s in labels:
        if s.field_name:
            per_field.setdefault(s.field_name, []).append(s)

    per_field_auroc: dict[str, float] = {}
    for name, group in per_field.items():
        if len(group) < MIN_LABELS_PER_FIELD:
            continue  # too few labels to derive a defensible number
        gs = [g.score for g in group]
        gc = [g.is_correct for g in group]
        t, _ = threshold_for_precision(gs, gc, target_precision)
        if t > 0:
            thresholds[name] = t
        per_field_auroc[name] = auroc(gs, gc)

    return Calibration(
        id=id,
        gold_set_hash=gold_set_hash
        or hashlib.sha256(
            json.dumps([(s.doc_id, s.field_name) for s in labels], sort_keys=True).encode()
        ).hexdigest()[:GOLD_SET_HASH_LENGTH],
        provider_id=provider_id,
        model_version=model_version,
        strategy_id=strategy_id,
        metrics={
            "auroc": auroc(scores, correct),
            "precision_at_num_errors": precision_at_k(scores, correct),
            "confidence_gap": confidence_gap(scores, correct),
            "recall_at_target_precision": default_r,
            "target_precision": target_precision,
            "per_field_auroc": per_field_auroc,
        },
        thresholds=thresholds,
        n_labels=len(labels),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
