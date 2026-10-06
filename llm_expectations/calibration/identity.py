"""The passthrough. It is not a calibration, and it says so on every number."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..types import Label, Status, Verdict
from .base import FitReport

__all__ = ["IdentityCalibrator"]

UNCALIBRATED_NOTICE = (
    "triage ranking is UNCALIBRATED\n"
    "    ranked by raw judge confidence, which is not a probability of error\n"
    "    ~100 labelled rows would let it be fitted and measured"
)


@dataclass(frozen=True, slots=True)
class IdentityCalibrator:
    """Hands back the judge's own confidence, read as a chance of error.

    It exists so the pipeline has one shape in both modes. Without it, mode 0
    would take a different code path from mode 1 and the uncalibrated case
    would be the one nobody tested.

    DESIGN.md §7 gives the rule as "returns 1 − raw_confidence". That is the
    answer when the judge said the label was *correct*. A judge that said
    **incorrect** with confidence 0.9 is not reporting a 10% chance of error —
    it is reporting a 90% one, and ranking it near the bottom would put the
    judge's clearest rejections last. So the confidence is read against the
    verdict's own direction. In the PASS case this is the design's formula
    unchanged.
    """

    id: str = "identity"
    is_calibrated: bool = False

    def fit(self, verdicts: Sequence[Verdict], labels: Sequence[Label]) -> FitReport:
        """Nothing is fitted, and the report says that rather than returning zeros."""
        return FitReport(
            calibration_id=self.id,
            fitted=False,
            n=len(labels),
            notes=(
                "the identity calibrator fits nothing — it passes raw confidence through, "
                "and raw confidence is not a probability of error",
                "a fitted calibrator (platt) needs ~100 labelled rows for the field",
            ),
        )

    def error_probability(self, verdict: Verdict) -> float | None:
        """None when there is nothing to read — never a stand-in number."""
        confidence = verdict.raw_confidence
        if confidence is None or verdict.status is Status.UNSCORED:
            return None
        return confidence if verdict.status is Status.FAIL else 1.0 - confidence
