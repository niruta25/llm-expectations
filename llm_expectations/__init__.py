"""llm-expectations — quality checks for LLM outputs that are judgements *about*
a document, not values copied *out of* one.

Pre-alpha, and usable for one thing: pointing a judge at a corpus and getting
back the order a human should open it in. That is M1, the first shippable
milestone (DESIGN.md §12). The free checks arrive at M2, the panel at M3, the
baselines that say whether the ranking beats guessing at M4, and the
operating-point table at M5b.

  - assigned fields   a label chosen from a versioned taxonomy
  - free text fields  a sentence written about the item

Two gates run before any quality number is reported: can the measurement be
trusted, and is the judge better than guessing. Results carry three states —
pass, fail, and *not checked* — and the third never silently becomes a pass.

Running a project is ``from llm_expectations.run import run, analyse_run``.
Those two are deliberately not re-exported here: a package attribute named
``run`` would shadow the ``llm_expectations.run`` module, and an import that
silently returns a function instead of a module is a bad afternoon.

See https://github.com/niruta25/llm-expectations
"""

from __future__ import annotations

from .calibration import Calibrator, IdentityCalibrator
from .config import ConfigError, RunConfig, Settings, load_run
from .judges import Judge, JudgeError, LabelCorrectTask
from .judges.fake import FakeProvider
from .plan import plan_run
from .read import ReadError, index_items, read_items, read_labels, read_outputs
from .schema import FieldKind, FieldSpec, Schema, SchemaError, TextStyle
from .taxonomy import Taxonomy, TaxonomyError, check_recorded_hash, load_taxonomy
from .triage import TriageContext, TriageStrategy
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
    "Calibrator",
    "ConfigError",
    "Exclusion",
    "FakeProvider",
    "FieldKind",
    "FieldSpec",
    "Finding",
    "Grain",
    "IdentityCalibrator",
    "Item",
    "Judge",
    "JudgeError",
    "Label",
    "LabelCorrectTask",
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
    "TriageContext",
    "TriageStrategy",
    "Verdict",
    "__version__",
    "check_recorded_hash",
    "index_items",
    "load_run",
    "load_taxonomy",
    "plan_run",
    "read_items",
    "read_labels",
    "read_outputs",
]
