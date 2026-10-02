"""llm-expectations — quality checks for LLM outputs that are judgements *about*
a document, not values copied *out of* one.

Pre-alpha, and both gates are now in place. Nothing is reported until the
measurement has been shown sound, and no ranking is called usable until it
has beaten random, output length, majority label and panel disagreement on
the same rows, with an interval that clears the best of them.

Underneath: the free checks, a triage judge that ranks what a human should
open first, and a panel that measures over a sample and locates the label
boundaries it keeps splitting on. A guardrail that fires withholds the
numbers it invalidates and names them, rather than printing them with a
caveat nobody reads.

That is M0 through M4 (DESIGN.md §12). Classification metrics arrive at M5
and the operating-point table at M5b.

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
from .checks import CheckContext, run_checks
from .config import ConfigError, RunConfig, Settings, load_run
from .gates import Gate, Gates, Suppression, gate_one, gate_two
from .judges import Judge, JudgeError, LabelCorrectTask
from .judges.fake import FakeProvider
from .metrics import AgreementReport, FuzzyPair, effective_votes
from .metrics.ranking import auc
from .metrics.stats import Estimate, bootstrap_ci
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
    "AgreementReport",
    "Calibrator",
    "CheckContext",
    "ConfigError",
    "Exclusion",
    "FakeProvider",
    "FieldKind",
    "FieldSpec",
    "Finding",
    "Estimate",
    "FuzzyPair",
    "Gate",
    "Gates",
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
    "Suppression",
    "TextStyle",
    "TriageContext",
    "TriageStrategy",
    "Verdict",
    "__version__",
    "auc",
    "bootstrap_ci",
    "check_recorded_hash",
    "effective_votes",
    "gate_one",
    "gate_two",
    "index_items",
    "load_run",
    "load_taxonomy",
    "plan_run",
    "read_items",
    "read_labels",
    "read_outputs",
    "run_checks",
]
