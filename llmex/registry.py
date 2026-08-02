"""Plugin registries.

Two ways in:
  1. `pip install llmex-anthropic` -> discovered via importlib entry points
  2. `PROVIDERS.register("mine", MyProvider)` -> in-process, for tests and notebooks

Both resolve through the same lookup so core code never branches on origin.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self, group: str) -> None:
        self.group = group
        self._local: dict[str, Any] = {}
        self._entry_cache: dict[str, Any] | None = None

    def register(self, name: str, obj: T) -> T:
        self._local[name] = obj
        return obj

    def plugin(self, name: str) -> Callable[[T], T]:
        """Decorator form: @PROVIDERS.plugin("openai")"""

        def wrap(obj: T) -> T:
            return self.register(name, obj)

        return wrap

    def _entries(self) -> dict[str, Any]:
        # A broken third-party plugin must not stop the framework importing.
        # It surfaces as a missing name with an `available: [...]` message.
        if self._entry_cache is None:
            found: dict[str, Any] = {}
            try:
                for ep in entry_points(group=self.group):
                    found[ep.name] = ep.load()
            except Exception:  # noqa: S110 - deliberately swallowed, see above
                pass
            self._entry_cache = found
        return self._entry_cache

    def get(self, name: str) -> Any:
        if name in self._local:
            return self._local[name]
        entries = self._entries()
        if name in entries:
            return entries[name]
        raise KeyError(
            f"no plugin '{name}' in group '{self.group}'. available: {sorted(self.names())}"
        )

    def names(self) -> list[str]:
        return sorted(set(self._local) | set(self._entries()))

    def __repr__(self) -> str:
        return f"<Registry {self.group} n={len(self.names())}>"


PROVIDERS = Registry("llmex.providers")
STRATEGIES = Registry("llmex.strategies")
EXPECTATIONS = Registry("llmex.expectations")
AGGREGATORS = Registry("llmex.aggregators")
SINKS = Registry("llmex.sinks")
ENGINES = Registry("llmex.engines")  # M8 — declared now so the name is settled
