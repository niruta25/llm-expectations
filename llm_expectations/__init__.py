"""llm-expectations — quality checks for LLM outputs that are judgements *about*
a document, not values copied *out of* one.

Pre-alpha. The skeleton (M0) is in place: the value types, the schema, the
taxonomy with its version hash, the readers, and the config merge. Nothing
calls a model yet and no checks run — that starts at M1. See DESIGN.md §12.

  - assigned fields   a label chosen from a versioned taxonomy
  - free text fields  a sentence written about the item

Two gates run before any quality number is reported: can the measurement be
trusted, and is the judge better than guessing. Results carry three states —
pass, fail, and *not checked* — and the third never silently becomes a pass.

See https://github.com/niruta25/llm-expectations
"""

from __future__ import annotations

from .config import ConfigError, RunConfig, Settings, load_run
from .read import ReadError, index_items, read_items, read_labels, read_outputs
from .schema import FieldKind, FieldSpec, Schema, SchemaError, TextStyle
from .taxonomy import Taxonomy, TaxonomyError, check_recorded_hash, load_taxonomy
from .types import (
    ABSTAIN,
    Exclusion,
    Finding,
    Grain,
    Item,
    Label,
    Mode,
    Output,
    RiskRow,
    Severity,
    Status,
    Verdict,
)

__version__ = "0.0.0"

__all__ = [
    "ABSTAIN",
    "ConfigError",
    "Exclusion",
    "FieldKind",
    "FieldSpec",
    "Finding",
    "Grain",
    "Item",
    "Label",
    "Mode",
    "Output",
    "ReadError",
    "RiskRow",
    "RunConfig",
    "Schema",
    "SchemaError",
    "Settings",
    "Severity",
    "Status",
    "Taxonomy",
    "TaxonomyError",
    "TextStyle",
    "Verdict",
    "__version__",
    "check_recorded_hash",
    "index_items",
    "load_run",
    "load_taxonomy",
    "read_items",
    "read_labels",
    "read_outputs",
]
