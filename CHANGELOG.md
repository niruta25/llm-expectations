# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
