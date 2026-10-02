"""The cost guard: ask before spending, and stop when the cap is reached.

Over budget, items past the cap are marked **unscored** — never passed. A run
that ran out of money and reported green would be the exact failure the third
result state exists to prevent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TextIO

from .plan import Plan

__all__ = ["BudgetExceeded", "BudgetGuard"]


class BudgetExceeded(RuntimeError):
    """The cap was reached. Remaining work is unscored, not skipped silently."""


@dataclass
class BudgetGuard:
    """Tracks spend against a cap, when the spend is knowable at all."""

    max_usd: float | None = None
    confirm: bool = True
    spent_usd: float = 0.0
    priced: bool = True
    _stopped: bool = field(default=False, init=False)

    @property
    def exhausted(self) -> bool:
        return self._stopped

    def spend(self, usd: float | None) -> None:
        """Record what a call cost. An unpriced call spends an unknown amount."""
        if usd is None:
            self.priced = False
            return
        self.spent_usd += usd
        if self.max_usd is not None and self.spent_usd >= self.max_usd:
            self._stopped = True

    def cap_reason(self) -> str:
        return (
            f"the ${self.max_usd:.2f} budget was reached after ${self.spent_usd:.2f}. "
            "Items past the cap are unscored, not passed."
        )

    def approve(
        self,
        plan: Plan,
        *,
        stream: TextIO,
        ask: Callable[[str], str] | None = None,
    ) -> bool:
        """Show the plan and get a yes. Returns False when the user declines."""
        stream.write("PLAN\n")
        for line in plan.lines():
            stream.write(line + "\n")
        for reason, count in sorted(plan.skipped.items()):
            stream.write(f"  skipped {count:,}: {reason}\n")

        estimate = plan.usd
        if estimate is None:
            stream.write(
                "\n  ⚠ cost cannot be estimated: no published price for "
                f"{', '.join(plan.unpriced_models)}.\n"
            )
            if self.max_usd is not None:
                # Say this plainly rather than let a cap look enforced.
                stream.write(
                    f"    The ${self.max_usd:.2f} cap cannot be enforced against an "
                    "unpriced model.\n"
                )
        elif self.max_usd is not None and estimate > self.max_usd:
            stream.write(
                f"\n  ⚠ the estimate (${estimate:.2f}) is over the "
                f"${self.max_usd:.2f} budget. Work past the cap will be unscored.\n"
            )

        if not self.confirm or plan.calls == 0:
            return True
        answer = (ask or input)("\nContinue? [y/N] ").strip().lower()
        return answer in {"y", "yes"}
