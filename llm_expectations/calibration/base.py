"""The calibration interface, and the guard that keeps one from going stale."""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..types import Label, Verdict

__all__ = ["Calibrator", "FitReport", "StaleCalibration", "fingerprint"]


class StaleCalibration(RuntimeError):
    """A fitted calibration is being used against something it was not fitted on."""


@dataclass(frozen=True, slots=True)
class FitReport:
    """Whether the calibration worked — not whether one was produced.

    Fitting a calibrator is not the same as it helping. These are the numbers
    that say which happened, and each has a way of reading as broken:
    an expected calibration error above ~0.10 means the stated probabilities do
    not match observed rates; a Brier score no better than the base rate means
    the calibration added nothing.
    """

    calibration_id: str
    fitted: bool
    n: int
    ece: float | None = None
    brier: float | None = None
    brier_base_rate: float | None = None
    reliability: tuple[Mapping[str, Any], ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["reliability"] = [dict(bin_) for bin_ in self.reliability]
        payload["notes"] = list(self.notes)
        return payload


@runtime_checkable
class Calibrator(Protocol):
    """Raw confidence in, probability of error out.

    ``fit`` is the only place in the library where labels and verdicts meet,
    and it is downstream of every judge call. A judge cannot be graded against
    an answer key it was shown because it was never shown one.
    """

    @property
    def id(self) -> str: ...

    @property
    def is_calibrated(self) -> bool: ...

    def fit(self, verdicts: Sequence[Verdict], labels: Sequence[Label]) -> FitReport: ...

    def error_probability(self, verdict: Verdict) -> float | None: ...


def fingerprint(
    *,
    judge_id: str,
    model: str,
    prompt_hash: str,
    check_id: str,
    taxonomy_ref: str | None,
) -> str:
    """What a fitted calibration is only valid for.

    The same guard as the taxonomy hash, for the same reason: a calibration
    fitted on prompt p7 tells you nothing about p8, and using it anyway is how
    a number keeps looking trustworthy after the thing under it moved.
    """
    payload = "\x00".join([judge_id, model, prompt_hash, check_id, taxonomy_ref or ""])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
