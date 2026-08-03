"""Every tunable constant in the framework, in one place.

Nothing here imports anything: this is the bottom of the dependency graph.

The rule this module enforces is that a number which changes what a run
*concludes* must be named, documented and overridable — never a literal buried
at a call site. A threshold typed inline is exactly the failure the calibration
subsystem exists to prevent, and the same logic applies to the sampling rate,
the cost estimate and the escalation floor.

Constants that only affect presentation (column widths, truncation) live with
the code that prints them.
"""

from __future__ import annotations

# -- sentinels --------------------------------------------------------------

CORPUS_DOC_ID = "__corpus__"
"""`doc_id` for corpus-grain results, which describe the batch, not a document."""

DEFAULT_THRESHOLD_KEY = "__default__"
"""Key under which a calibration stores its fall-back threshold."""

THRESHOLD_SOURCE_MANUAL = "manual"
"""`Result.threshold_source` when no calibration defended the number.

Alert on `threshold_source = 'manual' AND severity = 'error'`: that combination
is a guard that slipped through.
"""

GOLD_LABEL_KEY = "gold_label"
"""Where a human label rides on an extraction: `ExtractionRecord.meta`.

Gold belongs to the record, not to a parallel structure that can fall out of
sync with it. A judge must never see this in a production run — only a
calibration or evaluation run compares against it.
"""

# -- default aliases --------------------------------------------------------

DEFAULT_PROVIDER_ALIAS = "default"
DEFAULT_STRATEGY_ALIAS = "diverse_ensemble"
DEFAULT_AGGREGATOR = "harmonic"
DEFAULT_SUITE_NAME = "unnamed"
DEFAULT_SUITE_VERSION = "1"

# -- grounding --------------------------------------------------------------

DEFAULT_MIN_RATIO = 0.92
"""Similarity below which a value counts as absent from the source.

Published hallucination rates span 0.23%-20.23% across model/dataset
configurations, so this is a per-field, per-model property you should measure
rather than inherit. The default is a starting point, not a finding.
"""

GROUNDING_WINDOW_DIVISOR = 3
"""The fuzzy scan steps by `len(needle) // this`.

Smaller means finer alignment and more comparisons. M6 replaces the scan with
an n-gram prefilter, at which point this goes away.
"""

# -- sampling and routing ---------------------------------------------------

DEFAULT_AUDIT_RATE = 0.05
"""Share of clean documents escalated to the model tier anyway.

The audit stratum is the only thing that tells you whether the cheap gates
work. Set it from the calibration's precision/recall curve, not from intuition.
"""

DEFAULT_SAMPLING_SEED = 7
"""Seeded so two runs over one batch audit the same documents and stay comparable."""

MIN_ESTIMATED_ESCALATION_RATE = 0.1
"""Floor on the escalation share used when *estimating* cost.

Suspect documents are escalated on top of the audit stratum, so the true rate
exceeds `audit_rate` by an amount that depends on data the planner has not seen
yet. Estimating at the configured rate alone would under-quote every run.
"""

# -- verification thresholds ------------------------------------------------

UNCALIBRATED_THRESHOLD = 0.5
"""Threshold for a model-based check running without a calibration.

Only reachable at severity `warn` or `info`: the planner refuses to let a
blocking check run on an undefended number. It is deliberately the least
informative value available, and every result carrying it is stamped
`threshold_source = "manual"` so it is greppable.
"""

# -- cost estimation --------------------------------------------------------

ESTIMATED_TOKENS_IN = 2000
ESTIMATED_TOKENS_OUT = 200
"""Shape of a typical verifier call, for plan-time cost estimation only.

A rough guess by construction — the planner has not read the documents yet.
Override per run with `Planner(estimated_tokens_in=..., estimated_tokens_out=...)`
once you have measured your own corpus; the estimate appears in the manifest so
you can compare it against `actual`.
"""

# -- classification ---------------------------------------------------------

DEFAULT_MAX_LABEL_SHARE = 0.9
"""Share of the corpus the commonest label may hold before it reads as collapse.

A classifier that has quietly learned to answer the majority label for
everything still scores well on accuracy in an unbalanced corpus. The
distribution is what gives it away.
"""

DEFAULT_MIN_DISTINCT_LABELS = 2
DEFAULT_MAX_DISTRIBUTION_SHIFT = 0.25
"""Total variation distance from a baseline distribution before it is drift."""

# -- judges -----------------------------------------------------------------

DEFAULT_JUDGE_TEMPERATURE = 0.0
"""A judge is a measuring instrument. Sampling noise in an instrument is a
defect, not creativity."""

# -- ensembles --------------------------------------------------------------

DEFAULT_ENSEMBLE_CALLS = 5
"""Members beyond about five yield under 1% improvement. Do not add more
without measuring."""

DEFAULT_ENSEMBLE_TIMEOUT_S = 20.0

# -- budget -----------------------------------------------------------------

UNLIMITED_CALLS = 2**31
"""Stand-in for "no call cap". Not `inf`, because the field is an int."""

# -- calibration ------------------------------------------------------------

MIN_LABELS_PER_FIELD = 10
"""Below this, a per-field threshold is worse than no per-field threshold."""

DEFAULT_TARGET_PRECISION = 0.9

FINGERPRINT_LENGTH = 12
GOLD_SET_HASH_LENGTH = 16

# -- identifiers ------------------------------------------------------------

RUN_ID_LENGTH = 12
