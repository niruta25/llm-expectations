"""The result record.

Three states, not two:
  success=True   passed
  success=False  failed
  success=None   deliberately unscored (sampled out, budget capped, provider error)

The third state is load-bearing. Without it, dropped coverage looks like a pass
and the quality signal silently rots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .defaults import THRESHOLD_SOURCE_MANUAL
from .types import Cost, Evidence, Grain, Provenance, Severity, SkipReason


@dataclass
class Result:
    expectation_id: str
    grain: Grain
    doc_id: str
    field_name: str | None = None

    success: bool | None = None
    score: float | None = None
    threshold: float | None = None
    # "manual" | "calibration:{id}@{fingerprint}" — what makes an audit possible.
    threshold_source: str = THRESHOLD_SOURCE_MANUAL

    severity: Severity = Severity.ERROR
    observed: object = None
    evidence: Evidence = field(default_factory=Evidence)
    cost: Cost = field(default_factory=Cost)
    provenance: Provenance | None = None
    skip_reason: SkipReason | None = None

    @property
    def key(self) -> tuple[str, str | None]:
        return (self.doc_id, self.field_name)

    @property
    def blocking_failure(self) -> bool:
        return self.success is False and self.severity is Severity.ERROR

    def as_dict(self) -> dict[str, Any]:
        return {
            "expectation_id": self.expectation_id,
            "grain": self.grain.value,
            "doc_id": self.doc_id,
            "field_name": self.field_name,
            "success": self.success,
            "score": None if self.score is None else round(self.score, 4),
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "severity": self.severity.value,
            "observed": self.observed,
            "evidence": self.evidence.as_dict(),
            "cost": self.cost.as_dict(),
            "provenance": self.provenance.as_dict() if self.provenance else {},
            "skip_reason": self.skip_reason.value if self.skip_reason else None,
        }


@dataclass
class RunResult:
    run_id: str
    results: list[Result]
    cost: Cost
    manifest: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def failures(self) -> list[Result]:
        return [r for r in self.results if r.success is False]

    def unscored(self) -> list[Result]:
        return [r for r in self.results if r.success is None]

    def summary(self) -> dict[str, Any]:
        by_grain: dict[str, dict[str, int]] = {}
        for r in self.results:
            bucket = by_grain.setdefault(r.grain.value, {"pass": 0, "fail": 0, "unscored": 0})
            if r.success is True:
                bucket["pass"] += 1
            elif r.success is False:
                bucket["fail"] += 1
            else:
                bucket["unscored"] += 1
        return {
            "run_id": self.run_id,
            "by_grain": by_grain,
            "blocking_failures": sum(1 for r in self.results if r.blocking_failure),
            "cost": self.cost.as_dict(),
            "warnings": self.warnings,
        }
