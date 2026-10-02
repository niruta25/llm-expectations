"""Turning verdicts into an order to review in.

Ranking is a separate step with its own machinery, because the obvious
shortcut — rank by ``1 − confidence`` and call it risk — quietly asserts that a
number the model emitted is a probability of error. It is not, and that
assertion is the failure this library exists to catch (DESIGN.md §7).

Strategies are a plug point rather than a formula. Six ship by the end, and
four of them exist to be beaten. M1 ships the one that works without labels;
the baselines arrive with Gate 2 at M4 and the comparison table at M5b.
"""

from __future__ import annotations

from .base import TriageContext, TriageStrategy, assert_no_labels, build_risk_rows
from .strategies import STRATEGIES, RawConfidenceStrategy, resolve_strategy

__all__ = [
    "STRATEGIES",
    "RawConfidenceStrategy",
    "TriageContext",
    "TriageStrategy",
    "assert_no_labels",
    "build_risk_rows",
    "resolve_strategy",
]
