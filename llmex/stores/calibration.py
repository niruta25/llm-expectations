"""Calibration storage.

Suites reference calibrations by id; the store is injected. A warehouse-backed
store belongs in a plugin distribution — core ships the file one so a team can
version calibrations in the repo alongside the suite that uses them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..calibration import Calibration


@runtime_checkable
class CalibrationStore(Protocol):
    def get(self, id: str) -> Calibration | None:  # noqa: A002
        ...

    def put(self, cal: Calibration) -> None:
        ...

    def list(self) -> list[str]:
        ...


class FileCalibrationStore:
    """JSON under `calibrations/{id}.json`."""

    def __init__(self, root: str | Path = "calibrations") -> None:
        self.root = Path(root)

    def _path(self, id: str) -> Path:  # noqa: A002
        return self.root / f"{id}.json"

    def get(self, id: str) -> Calibration | None:  # noqa: A002
        path = self._path(id)
        if not path.exists():
            return None
        return Calibration.from_dict(json.loads(path.read_text()))

    def put(self, cal: Calibration) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._path(cal.id).write_text(json.dumps(cal.as_dict(), indent=2))

    def list(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.stem for p in self.root.glob("*.json"))

    def load_all(self) -> dict[str, Calibration]:
        """Everything on disk, keyed by id — the shape `Suite.from_dict` wants."""
        out: dict[str, Calibration] = {}
        for name in self.list():
            cal = self.get(name)
            if cal is not None:
                out[name] = cal
        return out
