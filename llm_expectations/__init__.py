"""llm-expectations — quality checks for LLM outputs that are judgements *about*
a document, not values copied *out of* one.

Pre-alpha. Both gates are in place, and human answers now grade both the
model and the judges that score it.

With labels: macro F1 beside the baseline it has to beat, a confusion matrix,
and four tree buckets that say *which kind* of wrong — a sibling, a hedge or
a misread each point at a different fix. With a second annotator: whether two
people can separate two labels at all, which is the strongest evidence a
taxonomy is the problem. And each judge is graded on which way it fails, and
on whether its stated confidences mean anything.

Underneath: the free checks, a triage judge that ranks what a human should
open first, and a panel that measures over a sample. A guardrail that fires
withholds the numbers it invalidates and names them.

And the triage score is finally allowed to be called calibrated. A Platt
curve per field maps raw confidence onto observed error rates, its quality is
measured out of fold, and Error Recall@Budget answers the question a review
budget actually poses — how many of the errors does opening the first 1%
actually find — for the judge and every baseline at once.

Free text is covered too, and it is cheaper than assigned — the opposite of
what you would guess. Length, specificity, copy ratio and boilerplate cost
nothing, and because they catch *content* problems rather than format ones
they genuinely gate the one expensive check: a judge that splits the text into
claims and says which the item does not support. An audit sample measures what
that gate misses instead of assuming it misses nothing.

And runs can be set against each other: ``compare`` lines two up, names the
labels that moved and the items that improved or regressed — and refuses to
say which run is better, because most items tie in a real A/B and a net delta
without a test is how underpowered changes get shipped. Two runs on different
taxonomy versions refuse to compare at all without a migration mapping.

That is M0 through M7 (DESIGN.md §12) — every build milestone. What remains is
polish, and the v1 list in §13.

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

from .calibration import (
    Calibrator,
    FieldCalibrator,
    IdentityCalibrator,
    PlattCalibrator,
)
from .checks import CheckContext, run_checks
from .compare import ComparisonError, compare_runs, load_run_directory
from .config import ConfigError, RunConfig, Settings, load_run
from .gates import Gate, Gates, Suppression, gate_one, gate_two
from .judges import ClaimSupportTask, Judge, JudgeError, LabelCorrectTask
from .judges.fake import FakeProvider
from .metrics import AgreementReport, FuzzyPair, effective_votes
from .metrics.classification import Classification, TreeBucket, classify
from .metrics.ranking import auc
from .metrics.stats import Estimate, bootstrap_ci
from .plan import plan_run
from .read import ReadError, index_items, read_items, read_labels, read_outputs
from .schema import FieldKind, FieldSpec, Schema, SchemaError, TextStyle
from .taxonomy import Taxonomy, TaxonomyError, check_recorded_hash, load_taxonomy
from .triage import TriageContext, TriageStrategy
from .triage.evaluate import error_recall_at_budget
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
    "ComparisonError",
    "ClaimSupportTask",
    "Classification",
    "ConfigError",
    "Exclusion",
    "FakeProvider",
    "FieldCalibrator",
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
    "PlattCalibrator",
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
    "TreeBucket",
    "TriageContext",
    "TriageStrategy",
    "Verdict",
    "__version__",
    "auc",
    "bootstrap_ci",
    "check_recorded_hash",
    "compare_runs",
    "classify",
    "effective_votes",
    "error_recall_at_budget",
    "gate_one",
    "gate_two",
    "index_items",
    "load_run",
    "load_run_directory",
    "load_taxonomy",
    "plan_run",
    "read_items",
    "read_labels",
    "read_outputs",
    "run_checks",
]
