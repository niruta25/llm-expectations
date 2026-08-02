"""Result sinks.

Async so a warehouse or HTTP sink is a drop-in. Ship JSONL and console;
everything else is a plugin.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from .registry import SINKS
from .result import RunResult


@runtime_checkable
class Sink(Protocol):
    async def emit(self, run: RunResult) -> None:
        ...


@SINKS.plugin("jsonl")
class JsonlSink:
    """One JSON object per result, plus a sidecar manifest pinning the run."""

    def __init__(self, path: str, manifest_path: str | None = None) -> None:
        self.path = Path(path)
        self.manifest_path = (
            Path(manifest_path) if manifest_path else self.path.with_suffix(".manifest.json")
        )

    async def emit(self, run: RunResult) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as fh:
            for r in run.results:
                fh.write(json.dumps({"run_id": run.run_id, **r.as_dict()}, default=str) + "\n")
        self.manifest_path.write_text(json.dumps(run.manifest, indent=2, default=str))


DEFAULT_FAILURE_LIMIT = 20
EVIDENCE_PREVIEW_CHARS = 110
"""Console layout only — the full evidence is always in the JSONL row."""


@SINKS.plugin("console")
class ConsoleSink:
    def __init__(
        self,
        show: str = "failures",
        limit: int = DEFAULT_FAILURE_LIMIT,
        evidence_chars: int = EVIDENCE_PREVIEW_CHARS,
    ) -> None:
        self.show = show
        self.limit = limit
        self.evidence_chars = evidence_chars

    async def emit(self, run: RunResult) -> None:
        s = run.summary()
        print(f"\nrun {run.run_id}")
        for grain, counts in sorted(s["by_grain"].items()):
            print(
                f"  {grain:<9} pass={counts['pass']:<4} fail={counts['fail']:<4} "
                f"unscored={counts['unscored']}"
            )
        print(f"  blocking failures: {s['blocking_failures']}")
        print(f"  cost: ${s['cost']['usd']:.4f} over {s['cost']['calls']} calls")
        for w in run.warnings:
            print(f"  ! {w}")

        if self.show == "failures":
            rows = [r for r in run.results if r.success is False]
            if rows:
                print("\n  failures:")
            for r in rows[: self.limit]:
                where = f"{r.doc_id}/{r.field_name}" if r.field_name else r.doc_id
                score = f" score={r.score:.2f}" if r.score is not None else ""
                thr = f" thr={r.threshold:.2f}" if r.threshold is not None else ""
                print(f"    [{r.grain.value:<8}] {r.expectation_id:<34} {where}{score}{thr}")
                if r.evidence.detail:
                    detail = json.dumps(r.evidence.detail, default=str)[: self.evidence_chars]
                    print(f"      {r.evidence.kind}: {detail}")
