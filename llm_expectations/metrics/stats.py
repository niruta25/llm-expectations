"""Intervals, and the resampling unit that makes them honest.

A number without an interval is not a result. The interval here is a bootstrap
percentile interval, and the thing it resamples is the **item** — never the
``(item, field)`` pair.

That is not a detail. Several fields of one item are produced by one model
call from one document: when the model misreads the document, every field on
that item moves together. Resampling fields independently treats those as
separate draws, which fakes independence and shrinks every interval it
touches. The function below cannot be called the wrong way: it takes values
grouped by item, so there is no signature that resamples a field on its own.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["Estimate", "FloatArray", "bootstrap_ci", "proportion"]

#: Spelled out rather than bare ``np.ndarray`` because numpy's own stubs
#: require the parameters under strict typing, and the two numpy majors this
#: package supports disagree about the default.
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Estimate:
    """A point estimate, its interval, and whether it can support a claim."""

    value: float | None
    low: float | None = None
    high: float | None = None
    n: int = 0
    resamples: int = 0
    floor: int | None = None
    note: str | None = None

    @property
    def under_floor(self) -> bool:
        """Too few rows for this number to mean anything.

        It is still reported — with the interval and a line saying so. A wide
        interval is not a weak result, it is no result, and hiding it would
        lose the only information the sample actually carries.
        """
        return self.floor is not None and self.n < self.floor

    @property
    def width(self) -> float | None:
        if self.low is None or self.high is None:
            return None
        return self.high - self.low

    def beats(self, other: Estimate) -> bool:
        """Does this estimate's interval clear the other's point estimate?

        The question Gate 2 asks. A judge whose interval overlaps a baseline's
        number has not been shown to beat it, however much higher its point
        estimate sits.
        """
        if self.low is None or other.value is None:
            return False
        return self.low > other.value

    def format(self, places: int = 2) -> str:
        if self.value is None:
            return "—"
        if self.low is None:
            return f"{self.value:.{places}f}"
        return f"{self.value:.{places}f} [{self.low:.{places}f}, {self.high:.{places}f}]"


def proportion(values: Sequence[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def bootstrap_ci(
    by_item: Mapping[str, Sequence[float]] | Sequence[Sequence[float]],
    statistic: Callable[[FloatArray], float | None],
    *,
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    floor: int | None = None,
) -> Estimate:
    """Resample whole items, recompute, and take the percentile interval.

    ``by_item`` maps an item to the values it contributed. An item either
    appears in a resample with all of its values or not at all, which is the
    whole point: the correlation between one item's fields is preserved
    instead of being averaged away.

    ``statistic`` receives the pooled values of one resample and returns the
    number. Returning ``None`` (a resample with nothing to compute on) drops
    that resample rather than contributing a zero.
    """
    raw = list(by_item.values()) if isinstance(by_item, Mapping) else list(by_item)
    groups: list[FloatArray] = [np.asarray(g, dtype=float) for g in raw if len(g)]
    if not groups:
        return Estimate(value=None, n=0, resamples=0, floor=floor)

    pooled = np.concatenate(groups)
    point = statistic(pooled)
    n_items = len(groups)

    if n_items < 2 or resamples < 1:
        return Estimate(
            value=point,
            n=n_items,
            resamples=0,
            floor=floor,
            note="too few items to resample — no interval",
        )

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        picked = rng.integers(0, n_items, size=n_items)
        sample = np.concatenate([groups[i] for i in picked])
        value = statistic(sample)
        if value is not None:
            draws.append(value)

    if not draws:
        return Estimate(
            value=point,
            n=n_items,
            resamples=resamples,
            floor=floor,
            note="no resample could be computed",
        )

    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(draws, [tail, 100 - tail])
    note = None
    if floor is not None and n_items < floor:
        note = (
            f"n = {n_items}, below the floor of {floor}. This interval cannot support a "
            "conclusion — a wide interval is not a weak result, it is no result."
        )
    return Estimate(
        value=point,
        low=float(low),
        high=float(high),
        n=n_items,
        resamples=len(draws),
        floor=floor,
        note=note,
    )
