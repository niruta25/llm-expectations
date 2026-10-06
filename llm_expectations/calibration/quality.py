"""Does 0.9 mean 90%?

A judge reporting "confidence 0.9" is stating a feeling. Whether that feeling
tracks anything is a measurable question, and this is where it is measured —
against human answers, downstream of every judge call.

The same functions answer the question twice over: for a judge's raw stated
confidence, and for the ``FitReport`` of a fitted calibrator. "Do these stated
probabilities match observed rates" is one question whether the numbers came
out of a model's mouth or out of a logistic fit, and writing it twice would
let the two answers drift.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["Bin", "Reliability", "brier", "expected_calibration_error", "reliability"]

#: Above this, stated probabilities do not match observed rates.
ECE_BROKEN = 0.10


@dataclass(frozen=True, slots=True)
class Bin:
    """One confidence band, and what actually happened inside it."""

    low: float
    high: float
    n: int
    stated: float
    observed: float

    @property
    def gap(self) -> float:
        return self.observed - self.stated

    @property
    def reportable(self) -> bool:
        """No bin under twenty rows gets its own number.

        A bin holding three items produces a gap that is mostly noise, and
        printing it invites somebody to act on it.
        """
        return self.n >= 20


@dataclass(frozen=True, slots=True)
class Reliability:
    """Whether a set of stated probabilities can be believed."""

    bins: tuple[Bin, ...]
    n: int
    ece: float | None
    brier: float | None
    base_rate: float | None
    brier_base_rate: float | None

    @property
    def broken(self) -> bool:
        return self.ece is not None and self.ece > ECE_BROKEN

    @property
    def beats_base_rate(self) -> bool | None:
        """Does predicting per-item beat always predicting the base rate?

        A Brier score no better than the base rate's means the confidences
        add nothing — the judge might as well have returned one number for
        every item.
        """
        if self.brier is None or self.brier_base_rate is None:
            return None
        return self.brier < self.brier_base_rate

    @property
    def direction(self) -> str | None:
        """Which way the miscalibration runs, when it runs one way.

        Bins that all sit on one side of the diagonal are a systematic
        over- or under-statement, which is fixable. Bins scattered either
        side are noise, which is not the same problem.
        """
        usable = [b for b in self.bins if b.reportable]
        if len(usable) < 2:
            return None
        if all(b.gap < 0 for b in usable):
            return "overconfident — stated probabilities run above observed rates"
        if all(b.gap > 0 for b in usable):
            return "underconfident — stated probabilities run below observed rates"
        return None


def reliability(
    stated: Sequence[float], happened: Sequence[bool], *, bins: int = 10
) -> Reliability:
    """Bin stated probabilities and compare each bin to what occurred."""
    paired = [(float(s), bool(h)) for s, h in zip(stated, happened, strict=True) if s is not None]
    n = len(paired)
    if not n:
        return Reliability((), 0, None, None, None, None)

    edges = [i / bins for i in range(bins + 1)]
    out: list[Bin] = []
    for low, high in zip(edges, edges[1:], strict=False):
        inside = [
            (s, h) for s, h in paired if (low <= s < high or (high == 1.0 and s == 1.0))
        ]
        if not inside:
            continue
        out.append(
            Bin(
                low=low,
                high=high,
                n=len(inside),
                stated=sum(s for s, _ in inside) / len(inside),
                observed=sum(h for _, h in inside) / len(inside),
            )
        )

    base_rate = sum(h for _, h in paired) / n
    return Reliability(
        bins=tuple(out),
        n=n,
        ece=expected_calibration_error(out, n),
        brier=brier([s for s, _ in paired], [h for _, h in paired]),
        base_rate=base_rate,
        brier_base_rate=brier([base_rate] * n, [h for _, h in paired]),
    )


def expected_calibration_error(bins: Sequence[Bin], n: int) -> float | None:
    """Mean gap between stated and observed, weighted by bin size."""
    if not n or not bins:
        return None
    return sum(b.n * abs(b.gap) for b in bins) / n


def brier(stated: Sequence[float], happened: Sequence[bool]) -> float | None:
    """Mean squared error of the stated probabilities. Lower is better."""
    pairs = list(zip(stated, happened, strict=True))
    if not pairs:
        return None
    return sum((s - float(h)) ** 2 for s, h in pairs) / len(pairs)
