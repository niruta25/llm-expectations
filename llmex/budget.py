"""Budget guard.

When money runs out the correct behaviour is to mark results unscored, never
to pass them. A run that quietly stops checking and reports green is worse
than one that fails loudly.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .defaults import UNLIMITED_CALLS
from .types import Cost


@dataclass
class Budget:
    max_usd: float = float("inf")
    max_calls: int = UNLIMITED_CALLS
    spent: Cost = field(default_factory=Cost)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def reserve(self, est_calls: int, est_usd: float = 0.0) -> bool:
        """Check-and-hold before making calls. False means do not proceed.

        Known gap (M6): this is not a true two-phase reservation. Concurrent
        documents can each pass `reserve()` and collectively overshoot. The fix
        is a reservation token that `record()` settles.
        """
        async with self._lock:
            if self.spent.calls + est_calls > self.max_calls:
                return False
            if self.spent.usd + est_usd > self.max_usd:
                return False
            return True

    async def record(self, cost: Cost) -> None:
        async with self._lock:
            self.spent = self.spent + cost

    @property
    def exhausted(self) -> bool:
        return self.spent.usd >= self.max_usd or self.spent.calls >= self.max_calls

    def remaining_usd(self) -> float:
        return max(0.0, self.max_usd - self.spent.usd)
