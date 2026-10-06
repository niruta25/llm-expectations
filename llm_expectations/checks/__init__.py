"""The free checks. They run before anything is spent and cannot see labels.

Importing this package registers every check. Nothing outside it needs to know
which checks exist — the schema says what kind each field is, and the registry
answers which checks read that kind.
"""

from __future__ import annotations

from . import (  # noqa: F401 — imported for registration
    assigned,
    copied,
    corpus,
    free_text,
    item,
)
from .base import CheckContext, CheckSpec, Skipped, checks_for, run_checks
from .corpus import label_shares

__all__ = [
    "CheckContext",
    "CheckSpec",
    "Skipped",
    "checks_for",
    "label_shares",
    "run_checks",
]
