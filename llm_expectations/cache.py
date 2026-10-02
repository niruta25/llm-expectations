"""Judge calls are the only expensive thing here, so they are written down.

Every verdict lands on disk the moment it comes back, and every piece of
analysis reads that file rather than the model. Change a threshold, add a
check, fix a reporting bug, try a different target — re-run instantly, pay
nothing. Cheap reporting code that only runs after an expensive collection is
its own failure mode, and this removes it (DESIGN.md §3).

The key is ``(judge, model, rendered prompt, item, field, output value, tag)``.
The *rendered* prompt, because it carries the taxonomy definitions with it:
tighten a definition and every verdict reached under the old wording stops
being a hit, which is right — it was a different question.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from types import TracebackType
from typing import Any

from .types import Verdict
from .write import read_verdicts, verdict_row

__all__ = ["VerdictCache", "cache_key"]


def cache_key(
    *,
    judge_id: str,
    model: str,
    prompt_fingerprint: str,
    item_id: str,
    field: str,
    output_value: Any,
    tag: str | None = None,
) -> str:
    payload = json.dumps(
        [judge_id, model, prompt_fingerprint, item_id, field, output_value, tag],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class VerdictCache:
    """Append-only verdict store with an in-memory index.

    ``tag`` is the escape hatch for when you deliberately want two runs with
    identical prompts kept apart — a re-measurement, or an A/B of the same
    prompt against itself.
    """

    def __init__(self, path: Path, *, tag: str | None = None) -> None:
        self.path = Path(path)
        self.tag = tag
        self._index: dict[str, Verdict] = {}
        self._handle: Any = None
        self.hits = 0
        self.misses = 0

    def __enter__(self) -> VerdictCache:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load(self.path)
        self._handle = self.path.open("a", encoding="utf-8")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def warm_from(self, other: Path | str) -> int:
        """Seed from a previous run's verdicts without copying them forward.

        Reused verdicts are hits: they are not re-paid for and not re-written
        into this run's file, so this run's ``verdicts.jsonl`` holds what this
        run actually asked. The count is reported rather than absorbed.
        """
        return self._load(Path(other))

    def _load(self, path: Path) -> int:
        loaded = 0
        for verdict in read_verdicts(path):
            key = verdict.metadata.get("cache_key")
            if isinstance(key, str):
                self._index.setdefault(key, verdict)
                loaded += 1
        return loaded

    def get(self, key: str) -> Verdict | None:
        verdict = self._index.get(key)
        if verdict is None:
            self.misses += 1
            return None
        self.hits += 1
        return Verdict(
            judge_id=verdict.judge_id,
            check_id=verdict.check_id,
            item_id=verdict.item_id,
            field=verdict.field,
            status=verdict.status,
            raw_confidence=verdict.raw_confidence,
            reason=verdict.reason,
            detail=verdict.detail,
            metadata={**verdict.metadata, "cache_hit": True},
        )

    def put(self, key: str, verdict: Verdict) -> Verdict:
        """Index and write, in that order, immediately.

        Not buffered. A run that dies halfway through has still paid for every
        call it made, and the point of this file is that you never pay twice.
        """
        stamped = Verdict(
            judge_id=verdict.judge_id,
            check_id=verdict.check_id,
            item_id=verdict.item_id,
            field=verdict.field,
            status=verdict.status,
            raw_confidence=verdict.raw_confidence,
            reason=verdict.reason,
            detail=verdict.detail,
            metadata={**verdict.metadata, "cache_key": key},
        )
        self._index[key] = stamped
        if self._handle is not None:
            self._handle.write(json.dumps(verdict_row(stamped), ensure_ascii=False) + "\n")
            self._handle.flush()
        return stamped

    def __len__(self) -> int:
        return len(self._index)

    def verdicts(self) -> Iterable[Verdict]:
        return self._index.values()
