"""Platt scaling: turning a judge's feeling into a probability, or failing to.

A logistic curve fitted to map an uncalibrated score onto observed error
rates. One feature, two parameters, which is deliberate — with a hundred
labelled rows anything richer memorises the sample. Isotonic regression is
the obvious third calibrator and is deferred for the same reason: it needs
more data than most teams have at the start and overfits badly below a few
hundred labels.

**What the probability is of.** DESIGN.md §7 words it as "P(this verdict is
wrong)", but a confident rejection would then score near zero and sort last,
putting the judge's clearest catches at the bottom of the review queue. The
number that makes the ranking work — and the one Error Recall@Budget counts
against — is P(**the output** is wrong). That is what is fitted here, with
the verdict's direction folded into the input, exactly as the identity
calibrator already reads it.

**The evaluation is cross-fitted, and that is not optional.** Fitting on a
hundred rows and then reporting how well the fit scores those same hundred
rows is in-sample performance wearing a lab coat — the single most common way
a calibration gets trusted that should not be. ``fit`` therefore does two
things: it fits one calibrator on everything, which is the artifact you would
deploy, and it refits k times on k-1 folds to produce an out-of-fold
prediction for every row, which is what the quality numbers are computed on.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..metrics.stats import FloatArray
from ..types import Label, Status, Verdict
from .base import FitReport
from .identity import IdentityCalibrator
from .quality import reliability

__all__ = ["MIN_ROWS", "FieldCalibrator", "PlattCalibrator", "fit_sigmoid", "raw_scores"]

#: Below this many labelled rows a two-parameter fit is memorising noise.
MIN_ROWS = 100

#: Folds for the out-of-fold evaluation. Five is the usual compromise
#: between a fold large enough to fit on and enough folds to average over.
FOLDS = 5


def fit_sigmoid(
    scores: FloatArray, outcomes: FloatArray, *, iterations: int = 100, tol: float = 1e-7
) -> tuple[float, float]:
    """Fit ``P(outcome) = 1 / (1 + exp(a * score + b))`` by Newton steps.

    Targets are smoothed away from 0 and 1 the way Platt's original method
    prescribes. With a hundred rows, a fold that happens to be all-positive
    would otherwise drive a coefficient to infinity and produce a calibrator
    that reports certainty it has not earned.
    """
    positives = float(outcomes.sum())
    negatives = float(len(outcomes) - positives)
    high = (positives + 1.0) / (positives + 2.0)
    low = 1.0 / (negatives + 2.0)
    target = np.where(outcomes, high, low)

    a, b = 0.0, float(np.log((negatives + 1.0) / (positives + 1.0)))
    for _ in range(iterations):
        z = a * scores + b
        # exp is evaluated on the stable side of zero in both branches.
        p = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
        q = 1.0 - p
        # The fitted curve is P = sigmoid(-(a*s + b)); the gradient below
        # follows that sign convention throughout.
        diff = target - q
        grad_a = float(np.sum(scores * diff))
        grad_b = float(np.sum(diff))
        if abs(grad_a) < tol and abs(grad_b) < tol:
            break
        w = p * q
        h_aa = float(np.sum(w * scores * scores)) + 1e-10
        h_ab = float(np.sum(w * scores))
        h_bb = float(np.sum(w)) + 1e-10
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        a -= (h_bb * grad_a - h_ab * grad_b) / det
        b -= (h_aa * grad_b - h_ab * grad_a) / det
    return a, b


def _apply(a: float, b: float, scores: FloatArray) -> FloatArray:
    z = -(a * scores + b)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


@dataclass
class PlattCalibrator:
    """A fitted logistic map from raw confidence to probability of error."""

    id: str = "platt"
    a: float = 0.0
    b: float = 0.0
    fitted_on: int = 0
    fingerprint: str | None = None
    _fitted: bool = field(default=False, repr=False)
    _identity: IdentityCalibrator = field(default_factory=IdentityCalibrator, repr=False)

    @property
    def is_calibrated(self) -> bool:
        """False until a fit has actually succeeded.

        An unfitted Platt is not a calibration, and every number downstream
        of it must be stamped the same way the identity passthrough is.
        """
        return self._fitted

    def fit(
        self, verdicts: Sequence[Verdict], labels: Sequence[Label]
    ) -> FitReport:  # pragma: no cover — the run uses fit_from_rows
        raise NotImplementedError(
            "PlattCalibrator.fit needs outputs to know whether a verdict was right. "
            "Use fit_from_rows, which takes the scores and outcomes already paired."
        )

    def fit_from_rows(
        self,
        scores: list[float],
        errors: list[bool],
        *,
        fingerprint: str | None = None,
        folds: int = FOLDS,
        seed: int = 0,
    ) -> FitReport:
        """Fit on everything, then score honestly on out-of-fold predictions.

        ``scores`` are uncalibrated risk in [0, 1] — higher means more likely
        to be an error. ``errors`` is what the human answers say actually
        happened.
        """
        x = np.asarray(scores, dtype=float)
        y = np.asarray(errors, dtype=bool)
        n = len(x)
        self.fingerprint = fingerprint

        if n < MIN_ROWS:
            return FitReport(
                calibration_id=self.id,
                fitted=False,
                n=n,
                notes=(
                    f"{n} labelled rows, below the {MIN_ROWS} a two-parameter fit needs. "
                    "Nothing was fitted; the ranking stays uncalibrated and says so.",
                ),
            )
        if y.all() or not y.any():
            return FitReport(
                calibration_id=self.id,
                fitted=False,
                n=n,
                notes=(
                    "every labelled row fell on the same side — all errors or none. "
                    "There is no boundary to fit, and a curve through one class would "
                    "return the base rate dressed as a probability.",
                ),
            )

        self.a, self.b = fit_sigmoid(x, y)
        self.fitted_on = n
        self._fitted = True

        out_of_fold = self._cross_fit(x, y, folds=folds, seed=seed)
        curve = reliability(out_of_fold.tolist(), y.tolist())
        notes: list[str] = [
            f"quality measured out of fold over {folds} folds — fitting and scoring on "
            "the same rows reports how well a curve remembers, not how well it predicts",
        ]
        if curve.broken:
            notes.append(
                f"expected calibration error {curve.ece:.3f} is above 0.10 — the stated "
                "probabilities do not match observed rates"
            )
        if curve.beats_base_rate is False:
            notes.append(
                "the Brier score is no better than always predicting the base rate. "
                "The calibration adds nothing; fitting one is not the same as it helping."
            )
        if curve.direction:
            notes.append(curve.direction)

        return FitReport(
            calibration_id=self.id,
            fitted=True,
            n=n,
            ece=curve.ece,
            brier=curve.brier,
            brier_base_rate=curve.brier_base_rate,
            reliability=tuple(
                {
                    "low": b.low,
                    "high": b.high,
                    "n": b.n,
                    "stated": b.stated,
                    "observed": b.observed,
                    "reportable": b.reportable,
                }
                for b in curve.bins
            ),
            notes=tuple(notes),
        )

    def _cross_fit(
        self, x: FloatArray, y: FloatArray, *, folds: int, seed: int
    ) -> FloatArray:
        """An out-of-sample prediction for every row, by refitting k times."""
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(x))
        out = np.zeros(len(x), dtype=float)
        for k in range(folds):
            held = order[k::folds]
            kept = np.setdiff1d(order, held, assume_unique=False)
            if not len(kept) or not len(held):
                continue
            if y[kept].all() or not y[kept].any():
                # This fold's training half has one class. Predicting its
                # base rate is the honest fallback; a fitted curve would be
                # a constant pretending to be a model.
                out[held] = float(y[kept].mean())
                continue
            a, b = fit_sigmoid(x[kept], y[kept])
            out[held] = _apply(a, b, x[held])
        return out

    def score(self, raw: float) -> float:
        """Map one uncalibrated score through the fitted curve."""
        if not self._fitted:
            return raw
        return float(_apply(self.a, self.b, np.asarray([raw], dtype=float))[0])

    def error_probability(self, verdict: Verdict) -> float | None:
        """P(the output is wrong), or None when there is nothing to read."""
        raw = self._identity.error_probability(verdict)
        if raw is None:
            return None
        return self.score(raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "a": self.a,
            "b": self.b,
            "fitted": self._fitted,
            "fitted_on": self.fitted_on,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PlattCalibrator:
        """Rebuild from `calibration.json`. JSON is `Any` and honestly so."""
        calibrator = cls(
            a=float(payload.get("a", 0.0)),
            b=float(payload.get("b", 0.0)),
            fitted_on=int(payload.get("fitted_on", 0) or 0),
            fingerprint=payload.get("fingerprint"),
        )
        calibrator._fitted = bool(payload.get("fitted"))
        return calibrator


def raw_scores(verdicts: Sequence[Verdict]) -> dict[tuple[str, str], float]:
    """The uncalibrated risk each verdict implies, keyed by item and field."""
    identity = IdentityCalibrator()
    out: dict[tuple[str, str], float] = {}
    for verdict in verdicts:
        if verdict.status is Status.UNSCORED:
            continue
        value = identity.error_probability(verdict)
        if value is not None:
            out[(verdict.item_id, verdict.field)] = value
    return out


@dataclass
class FieldCalibrator:
    """One fitted curve per field, with a shared one behind them.

    Why per field, and not one curve for the corpus: **Platt scaling is
    monotone.** A single global curve maps a higher raw score to a higher
    probability, always, so it cannot reorder anything — ``calibrated_risk``
    and ``raw_confidence`` would post byte-identical recall at every budget
    and the comparison table would be theatre.

    What a global calibration buys is real but different: probabilities that
    mean what they say, so "the top 500 are each about 18% likely to be
    wrong" becomes a sentence you can act on, and a threshold can be set at a
    target precision. It does not buy a better ordering.

    Per field it *can* reorder, because "is this jtbd label right" and "is
    this outcome right" are different tasks with different base rates, and a
    raw 0.7 on one is not a raw 0.7 on the other. That is the only way an
    ordering improves here, and when there is too little data per field to
    fit separately the report says the ranking is unchanged rather than
    implying the calibration earned something.
    """

    id: str = "platt"
    per_field: dict[str, PlattCalibrator] = field(default_factory=dict)
    fallback: PlattCalibrator | None = None
    fingerprint: str | None = None
    _identity: IdentityCalibrator = field(default_factory=IdentityCalibrator, repr=False)

    @property
    def is_calibrated(self) -> bool:
        return any(c.is_calibrated for c in self.per_field.values()) or bool(
            self.fallback and self.fallback.is_calibrated
        )

    @property
    def reorders(self) -> bool:
        """Can this calibration change the review order at all?

        Only when two or more fields carry their own curve. One curve over
        everything is a monotone map and leaves the order exactly as it was.
        """
        return len([c for c in self.per_field.values() if c.is_calibrated]) >= 2

    def _for(self, field_name: str) -> PlattCalibrator | None:
        chosen = self.per_field.get(field_name)
        if chosen is not None and chosen.is_calibrated:
            return chosen
        if self.fallback is not None and self.fallback.is_calibrated:
            return self.fallback
        return None

    def fit(
        self, verdicts: Sequence[Verdict], labels: Sequence[Label]
    ) -> FitReport:  # pragma: no cover — the run uses the per-field fitter
        raise NotImplementedError("fit each field through PlattCalibrator.fit_from_rows")

    def error_probability(self, verdict: Verdict) -> float | None:
        raw = self._identity.error_probability(verdict)
        if raw is None:
            return None
        chosen = self._for(verdict.field)
        return raw if chosen is None else chosen.score(raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "fingerprint": self.fingerprint,
            "reorders": self.reorders,
            "per_field": {k: v.to_dict() for k, v in self.per_field.items()},
            "fallback": self.fallback.to_dict() if self.fallback else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FieldCalibrator:
        raw_fields: Mapping[str, Any] = payload.get("per_field") or {}
        fallback = payload.get("fallback")
        return cls(
            per_field={k: PlattCalibrator.from_dict(v) for k, v in raw_fields.items()},
            fallback=PlattCalibrator.from_dict(fallback) if fallback else None,
            fingerprint=payload.get("fingerprint"),
        )
