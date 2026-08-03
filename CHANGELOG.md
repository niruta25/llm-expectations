# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — judge-backed classification (J1–J6)

- **J1, free tier.** `expect_field_matches_gold` compares a label against a
  human one carried on `ExtractionRecord.meta["gold_label"]`; a record nobody
  labelled reports unscored, never passed. `expect_label_distribution_stable`
  catches collapse onto a majority label and drift from a baseline, both of
  which survive a perfect accuracy score. `llmex.classification` adds a
  confusion matrix and a report carrying macro F1 beside accuracy.
- **J2, the judge.** `label_judge` strategy — rubric and label set are
  configuration, so one strategy grades any taxonomy. It is the first built-in
  to populate `ScoreSet.explanations`, which `trust_score` evidence has always
  read as `""`. `MockLabelJudge` scores on label-vocabulary overlap with the
  document: a real, weak signal chosen so calibrations land somewhere honest.
- **J3, calibration.** `cohens_kappa` and `agreement_rate`, the multi-class
  `LabelledDecision`, and `calibrate_judge()` adding kappa, raw agreement and
  per-label thresholds gated at the same ten-label minimum.
- **J4, wiring.** `expect_extraction_label_trustworthy` — a genuinely thin
  subclass of `ExpectFieldTrustworthy` overriding three class attributes and no
  methods. `expect_judge_agrees_with_gold` grades the grader at corpus grain.
- **`Kind.DERIVED`** and `Context.prior_results`: a fourth tier, free and last,
  for checks computed from results other checks produced. A check that grades
  another check cannot run in the statistical tier, because what it reads does
  not exist yet.
- **J5, example.** `examples/jtbd_session_classification/` with a synthetic
  corpus, a one-line taxonomy swap, and a loader seam that takes a private path
  for real data. `demo_jtbd.py` runs the whole thing offline.
- **J6, docs.** README section, DESIGN.md §6.1 subsection on the judge
  contract, cookbook recipe §15.4, and `suite.jtbd.example.yml`.

### Fixed — found by the JTBD work

- **Verification cost was double-counted.** Every field row and the document
  row carried the same `ScoreSet.cost`, and the runner sums cost across rows,
  so one call per document was reported as three. Cost now lands once, on the
  document row.
- **`expect_field_trustworthy` ignored its `fields` config**, sending every
  field on the document to the verifier. A suite naming two fields on a
  twelve-field schema silently paid to grade the other ten.
- **A subclass's `default_strategy` was honoured at run time but not at plan
  time**, so capability negotiation and the staleness guard validated against a
  strategy that would never run. `default_strategy` is now declared on the
  `Expectation` base and read by both.
- `MockLabelJudge` tokenised `None` as the word "none" instead of treating it
  as an absent label.

### Added

- **M0** — packaging (`llm-expectations` distribution, `llmex` import name),
  entry-point groups for all six registries, CI over Python 3.10–3.13 running
  ruff, mypy strict and pytest.
- **M1** — domain core: `Kind`/`Grain`/`Severity`/`Capability`/`SkipReason`
  enums, `Cost` (max-latency `__add__`), `Evidence`, `Provenance`, `PlanError`,
  `SourceDoc`, `ExtractionRecord`, `Batch` with a memoising source resolver,
  `Context`, `Result`, `RunResult`, `Registry` and the six singletons, the
  `Expectation` ABC with `SyncExpectation` and `@field_check` shims.
- **M2** — four aggregators (`harmonic`, `weighted_harmonic`, `arithmetic`,
  `minimum`), four deterministic/statistical expectations, `Runner` with tier
  ordering, `ctx.prior` merging and field→document rollup with dedupe,
  `ConsoleSink` and `JsonlSink`, YAML/dict suite loading with a fingerprint.
- **M3** — provider contract with `CostModel`/`Limits`, `MockProvider`,
  `HTTPChatProvider` skeleton, strategy contract, four strategies
  (`single_judge`, `per_field_judge`, `diverse_ensemble`, `logprob`) and
  `expect_field_trustworthy` emitting both grains from one pass.
- **M4** — `Planner` with alias, capability, calibration, staleness,
  correlated-verifier and budget guards; `Plan.ordered()` tier ranking.
- **M5** — calibration metrics without numpy (`auroc` with average-rank tie
  handling, `precision_at_k`, `confidence_gap`, `threshold_for_precision`),
  the `Calibration` object with fingerprinting, `calibrate()` gated at ten
  labels per field, and `FileCalibrationStore`.

### Fixed

- **No invented scores.** Three places substituted a stand-in number for a
  measurement that was never taken, and in each the stand-in compared as a pass
  against the uncalibrated `0.5` threshold:
  - a field absent from `ScoreSet.field_scores` defaulted to `0.5`. It is now
    reported `success=None` with `NOT_APPLICABLE` — absent is not passing.
  - `per_field_judge` and `diverse_ensemble` filled unscored fields with `0.5`.
    They now omit the field, letting the expectation mark it unscored.
  - `logprob` returned a flat `0.7` for every field regardless of input. It now
    scores from the generator's own probabilities, carried on
    `ExtractionRecord.meta["logprob"]` and exposed as `ScorePayload.logprobs`,
    and raises `StrategyError` when they are absent rather than inventing one.
- `expect_field_trustworthy` reported `"aggregation": "harmonic"` in document
  evidence no matter what the strategy actually did. Strategies now declare it
  via `ScoreSet.aggregation`.
- `diverse_ensemble` returned a fabricated `0.5` when *every* member failed or
  timed out. That score cleared a middling threshold and read as a pass, so a
  dead provider produced a green run — the exact failure the unscored state
  exists to prevent. It now raises `StrategyError`, which
  `expect_field_trustworthy` converts into a `PROVIDER_ERROR` skip. Partial
  losses are unchanged: they still return a score and report `n_calls_dropped`.

### Changed

- Every constant that changes what a run concludes now lives in `llmex.defaults`
  with a stated rationale, instead of as a literal at a call site: grounding
  ratio, audit rate, sampling seed, escalation floor used for estimation,
  uncalibrated threshold, ensemble size and timeout, per-field label minimum,
  and the `__corpus__` / `__default__` / `manual` sentinels. Defaults are
  unchanged; the demo output is byte-for-byte identical.
- `Planner(estimated_tokens_in=…, estimated_tokens_out=…)` replaces the
  hardcoded 2000/200-token guess behind every cost estimate.

### Known limitations

Tracked in DESIGN.md §16.2. The load-bearing ones: no response cache (M6),
budget reservation is not two-phase (M6), `expect_fields_to_satisfy` still uses
`eval` and is unsafe for untrusted config (M7), and `HTTPChatProvider._post`
is unimplemented (no real provider ships in core).
