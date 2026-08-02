"""Field -> document rollup.

Default is harmonic mean, which behaves as a soft minimum. That is the correct
semantic when a document is wrong if *any* field is wrong: the arithmetic mean
of nineteen 0.99s and one 0.02 is 0.94, which reads as healthy. The harmonic
mean of the same set is 0.29, which reads as broken. It is broken.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

from .registry import AGGREGATORS

_EPS = 1e-6

Scores = Mapping[str, float | None] | Sequence[float | None]
Aggregator = Callable[..., float]


def _clean(scores: Scores) -> list[float]:
    """Accepts a mapping of field -> score or a bare sequence.

    The runner passes a mapping so weighted aggregators can see field names;
    the unweighted ones must not choke on it. Naive `for s in scores` over a
    dict iterates keys, which is a real bug this guard exists to prevent.
    """
    values: Iterable[float | None]
    values = scores.values() if isinstance(scores, Mapping) else scores
    return [max(_EPS, min(1.0, float(s))) for s in values if s is not None]


@AGGREGATORS.plugin("harmonic")
def harmonic(scores: Scores, weights: Mapping[str, float] | None = None) -> float:
    vals = _clean(scores)
    if not vals:
        return 1.0
    return len(vals) / sum(1.0 / v for v in vals)


@AGGREGATORS.plugin("arithmetic")
def arithmetic(scores: Scores, weights: Mapping[str, float] | None = None) -> float:
    """Included mainly for comparison, and to make the point. Almost never right."""
    vals = _clean(scores)
    return sum(vals) / len(vals) if vals else 1.0


@AGGREGATORS.plugin("minimum")
def minimum(scores: Scores, weights: Mapping[str, float] | None = None) -> float:
    """Hard minimum. Zero tolerance, and very noisy."""
    vals = _clean(scores)
    return min(vals) if vals else 1.0


@AGGREGATORS.plugin("weighted_harmonic")
def weighted_harmonic(scores: Scores, weights: Mapping[str, float] | None = None) -> float:
    """Field criticality as weight. A wrong `total_amount` should hurt more
    than a wrong `notes`, and this is where that judgement gets encoded."""
    if isinstance(scores, Mapping):
        items: list[tuple[str, float | None]] = list(scores.items())
    else:
        items = [(str(i), s) for i, s in enumerate(scores)]
    weights = weights or {}
    num, den = 0.0, 0.0
    for name, raw in items:
        if raw is None:
            continue
        w = float(weights.get(name, 1.0))
        v = max(_EPS, min(1.0, float(raw)))
        num += w
        den += w / v
    return num / den if den else 1.0


def get(name: str) -> Aggregator:
    agg: Aggregator = AGGREGATORS.get(name)
    return agg
