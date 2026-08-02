"""llm-expectations — a data quality framework for LLM extractions.

    97% field accuracy is 26% document accuracy. Measure both.

Great Expectations' vocabulary, rebuilt for checks that are stochastic, cost
money, and cannot be trusted until they are calibrated.

Quick start::

    import llmex

    batch = llmex.Batch(records, source_resolver=lambda d: llmex.SourceDoc(d, text))
    suite = llmex.Suite.from_yaml("suite.yml")
    run = llmex.Runner(sinks=[llmex.ConsoleSink()]).run_sync(suite, batch)

The distribution is `llm-expectations`; the import is `llmex`.
"""

from __future__ import annotations

# Importing these for their side effect: the decorators inside register the
# built-ins, so `Suite.from_dict({"type": "expect_field_type"})` resolves
# without the caller importing anything.
from . import expectations as expectations  # noqa: E402,F401  (isort: keep last)
from . import providers as providers  # noqa: E402,F401
from . import strategies as strategies  # noqa: E402,F401
from .aggregate import arithmetic, harmonic, minimum, weighted_harmonic
from .batch import Batch, Context, ExtractionRecord, SourceDoc
from .budget import Budget
from .calibration import (
    Calibration,
    LabelledScore,
    auroc,
    calibrate,
    confidence_gap,
    precision_at_k,
    threshold_for_precision,
)
from .expectation import Expectation, SyncExpectation, field_check
from .planner import Plan, Planner, Step
from .registry import (
    AGGREGATORS,
    ENGINES,
    EXPECTATIONS,
    PROVIDERS,
    SINKS,
    STRATEGIES,
    Registry,
)
from .result import Result, RunResult
from .runner import Runner
from .sinks import ConsoleSink, JsonlSink, Sink
from .stores import CalibrationStore, FileCalibrationStore
from .suite import Suite
from .types import (
    Capability,
    Cost,
    Evidence,
    Grain,
    Kind,
    PlanError,
    Provenance,
    Severity,
    SkipReason,
)

__version__ = "0.1.0"

__all__ = [
    # domain
    "Batch",
    "Context",
    "ExtractionRecord",
    "SourceDoc",
    "Result",
    "RunResult",
    "Cost",
    "Evidence",
    "Provenance",
    # enums and errors
    "Capability",
    "Grain",
    "Kind",
    "PlanError",
    "Severity",
    "SkipReason",
    # contracts
    "Expectation",
    "SyncExpectation",
    "field_check",
    "Sink",
    # orchestration
    "Budget",
    "Plan",
    "Planner",
    "Runner",
    "Step",
    "Suite",
    # calibration
    "Calibration",
    "CalibrationStore",
    "FileCalibrationStore",
    "LabelledScore",
    "auroc",
    "calibrate",
    "confidence_gap",
    "precision_at_k",
    "threshold_for_precision",
    # aggregation
    "arithmetic",
    "harmonic",
    "minimum",
    "weighted_harmonic",
    # sinks and registries
    "ConsoleSink",
    "JsonlSink",
    "Registry",
    "AGGREGATORS",
    "ENGINES",
    "EXPECTATIONS",
    "PROVIDERS",
    "SINKS",
    "STRATEGIES",
    "__version__",
]
