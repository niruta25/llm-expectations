"""Declarative suite definition.

YAML is the interface most people will touch. Everything it can express is
also constructible in Python, so notebooks and tests never need a temp file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .budget import Budget
from .defaults import (
    DEFAULT_AGGREGATOR,
    DEFAULT_SUITE_NAME,
    DEFAULT_SUITE_VERSION,
    UNLIMITED_CALLS,
)
from .expectation import Expectation
from .registry import EXPECTATIONS, PROVIDERS, STRATEGIES
from .types import Severity


@dataclass
class Suite:
    name: str
    version: str = "1"
    expectations: list[Expectation] = field(default_factory=list)
    providers: dict[str, Any] = field(default_factory=dict)
    strategies: dict[str, Any] = field(default_factory=dict)
    budget: Budget = field(default_factory=Budget)
    aggregator: str = DEFAULT_AGGREGATOR
    field_weights: dict[str, float] = field(default_factory=dict)
    calibrations: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls, cfg: dict[str, Any], calibrations: dict[str, Any] | None = None
    ) -> Suite:
        providers: dict[str, Any] = {}
        for alias, raw in (cfg.get("providers") or {}).items():
            spec = dict(raw)
            plugin = spec.pop("plugin")
            providers[alias] = PROVIDERS.get(plugin)(**spec)

        strategies: dict[str, Any] = {}
        for alias, raw in (cfg.get("strategies") or {}).items():
            spec = dict(raw)
            plugin = spec.pop("plugin")
            strategies[alias] = STRATEGIES.get(plugin)(**spec)
        # Every registered built-in is auto-aliased by its id, so
        # `strategy: diverse_ensemble` works with no `strategies:` block at all.
        for builtin in STRATEGIES.names():
            strategies.setdefault(builtin, STRATEGIES.get(builtin)())

        expectations: list[Expectation] = []
        for raw in cfg.get("expectations") or []:
            spec = dict(raw)
            etype = spec.pop("type")
            severity = Severity(spec.pop("severity", "error"))
            expectations.append(EXPECTATIONS.get(etype)(severity=severity, **spec))

        b = cfg.get("budget") or {}
        budget = Budget(
            max_usd=float(b.get("max_usd", float("inf"))),
            max_calls=int(b.get("max_calls", UNLIMITED_CALLS)),
        )

        agg = cfg.get("aggregate") or {}
        return cls(
            name=cfg.get("suite", DEFAULT_SUITE_NAME),
            version=str(cfg.get("version", DEFAULT_SUITE_VERSION)),
            expectations=expectations,
            providers=providers,
            strategies=strategies,
            budget=budget,
            aggregator=agg.get("field_to_document", DEFAULT_AGGREGATOR),
            field_weights=agg.get("field_weights", {}),
            calibrations=calibrations or {},
        )

    @classmethod
    def from_yaml(
        cls, path: str | Path, calibrations: dict[str, Any] | None = None
    ) -> Suite:
        cfg: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(cfg, calibrations)

    def fingerprint(self) -> dict[str, Any]:
        """Goes into the run manifest. Two runs whose fingerprints differ were
        produced by different functions and are not comparable."""
        return {
            "suite": self.name,
            "version": self.version,
            "expectations": [
                {
                    "id": e.id,
                    "version": e.version,
                    "kind": e.kind.value,
                    "grain": e.grain.value,
                    "severity": e.severity.value,
                    "config": e.config,
                }
                for e in self.expectations
            ],
            "providers": {
                k: {"id": v.id, "model_version": v.model_version}
                for k, v in self.providers.items()
            },
            "calibrations": {
                k: getattr(c, "ref", None) for k, c in self.calibrations.items()
            },
            "aggregator": self.aggregator,
        }
