"""The on-disk shapes, both directions.

One module owns every row format, because ``analyse`` reading a file that
``run`` wrote is the property the whole caching story rests on. A writer and a
reader that drift apart would make re-analysis silently wrong rather than
loudly broken.

Timestamped run folders, so several runs a day never collide, and one line per
run in ``index.jsonl`` — which is what "show me every run this week", the drift
check and the taxonomy hash guard all read.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import Finding, Grain, RiskRow, Status, Verdict

__all__ = [
    "append_index",
    "finding_row",
    "read_verdicts",
    "risk_row",
    "run_directory",
    "verdict_from_row",
    "verdict_row",
    "write_jsonl",
    "write_json",
]


def run_directory(out: Path, run_id: str, *, now: datetime | None = None) -> Path:
    """``out/2026-09-29_1432_jtbd-p8``, and never an existing one.

    Two runs in the same minute must not share a directory: the second would
    append its verdicts to the first's cache file and overwrite its report,
    which silently merges two runs into one. A suffix is added only when that
    would otherwise happen, so the documented shape is what you normally see.
    """
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d_%H%M")
    directory = out / f"{stamp}_{run_id}"
    attempt = 2
    while directory.exists():
        directory = out / f"{stamp}-{attempt}_{run_id}"
        attempt += 1
    directory.mkdir(parents=True)
    return directory


def verdict_row(verdict: Verdict) -> dict[str, Any]:
    """One judge call, as written. Nothing is averaged away at write time."""
    return {
        "judge": verdict.judge_id,
        "check": verdict.check_id,
        "item_id": verdict.item_id,
        "field": verdict.field,
        "status": verdict.status.value,
        "raw_confidence": verdict.raw_confidence,
        "reason": verdict.reason,
        "detail": dict(verdict.detail),
        "metadata": dict(verdict.metadata),
    }


def verdict_from_row(row: Mapping[str, Any]) -> Verdict:
    return Verdict(
        judge_id=str(row["judge"]),
        check_id=str(row["check"]),
        item_id=str(row["item_id"]),
        field=str(row["field"]),
        status=Status(row["status"]),
        raw_confidence=row.get("raw_confidence"),
        reason=str(row.get("reason", "")),
        detail=dict(row.get("detail") or {}),
        metadata=dict(row.get("metadata") or {}),
    )


def read_verdicts(path: Path) -> list[Verdict]:
    """Read a verdict file back. This is what makes ``analyse`` free."""
    if not path.exists():
        return []
    verdicts = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                verdicts.append(verdict_from_row(json.loads(line)))
            except (ValueError, KeyError) as exc:
                raise ValueError(f"{path}:{number}: not a verdict row — {exc}") from None
    return verdicts


def finding_row(finding: Finding) -> dict[str, Any]:
    """One check result. Carries *why*, not just a verdict."""
    return {
        "run_id": finding.run_id,
        "check": finding.check,
        "grain": finding.grain.value,
        "item_id": finding.item_id,
        "field": finding.field,
        "status": finding.status.value,
        "score": finding.score,
        "threshold": _plain(finding.threshold),
        "threshold_from": finding.threshold_from,
        "evidence": dict(finding.evidence),
        "judge": finding.judge,
        "cost_usd": round(finding.cost_usd, 6),
    }


def finding_from_row(row: Mapping[str, Any]) -> Finding:
    return Finding(
        run_id=str(row["run_id"]),
        check=str(row["check"]),
        grain=Grain(row["grain"]),
        status=Status(row["status"]),
        item_id=row.get("item_id"),
        field=row.get("field"),
        score=row.get("score"),
        threshold=row.get("threshold"),
        threshold_from=row.get("threshold_from"),
        evidence=dict(row.get("evidence") or {}),
        judge=row.get("judge"),
        cost_usd=float(row.get("cost_usd") or 0.0),
    )


def risk_row(row: RiskRow) -> dict[str, Any]:
    """One item's place in the review queue, and where that place came from."""
    return {
        "item_id": row.item_id,
        "triage_score": row.triage_score,
        "calibrated_error_probability": row.calibrated_error_probability,
        "raw_confidence_mean": row.raw_confidence_mean,
        "strategy": row.strategy,
        "calibrated": row.calibrated,
        "calibration_id": row.calibration_id,
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_plain) + "\n")
            written += 1
    return written


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_plain) + "\n",
        encoding="utf-8",
    )


def append_index(out: Path, entry: Mapping[str, Any]) -> None:
    """One line per run. Append-only; nothing rewrites history here."""
    out.mkdir(parents=True, exist_ok=True)
    with (out / "index.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, default=_plain) + "\n")


def read_index(out: Path) -> list[dict[str, Any]]:
    path = out / "index.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _plain(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value") and hasattr(value, "name"):  # an Enum
        return value.value
    return str(value)
