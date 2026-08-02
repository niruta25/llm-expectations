# llm-expectations — Design and Implementation Plan

**A pluggable data quality framework for LLM extractions.**

| | |
|---|---|
| Distribution | `llm-expectations` on PyPI · import name `llmex` |
| Status | Design approved, reference prototype exists (see Appendix A) |
| Target | Python 3.10+, no warehouse dependency |
| Scope | Quality checks over already-extracted structured data |
| Non-scope | Extraction itself, prompt management, model serving |

---

## Table of contents

1. [Problem statement](#1-problem-statement)
2. [Design principles](#2-design-principles)
3. [Architecture](#3-architecture)
4. [Domain model](#4-domain-model)
5. [Plugin contracts](#5-plugin-contracts)
6. [Built-in catalog](#6-built-in-catalog)
7. [Execution model](#7-execution-model)
8. [Calibration subsystem](#8-calibration-subsystem)
9. [Configuration reference](#9-configuration-reference)
10. [Persistence and result schema](#10-persistence-and-result-schema)
11. [Repository layout](#11-repository-layout)
12. [Implementation plan](#12-implementation-plan)
13. [Testing strategy](#13-testing-strategy)
14. [Operations](#14-operations)
15. [Extension cookbook](#15-extension-cookbook)
16. [Non-goals and known limitations](#16-non-goals-and-known-limitations)
17. [Open questions](#17-open-questions)
- [Appendix A — prototype status](#appendix-a--prototype-status)
- [Appendix B — glossary](#appendix-b--glossary)

---

## 1. Problem statement

### 1.1 What breaks when you point warehouse DQ at LLM output

Great Expectations, dbt tests, Soda and every other data quality framework
rest on three assumptions. All three fail for LLM extractions.

**Assumption 1: there is a reconcilable source of truth.**
In ETL you can count rows against the upstream system, checksum a file, or
join to a dimension. For an extraction, the "correct" value exists only inside
prose that a human would have to read. There is nothing to reconcile against.

**Assumption 2: the transformation is deterministic and versioned by code.**
An extractor is a stochastic function whose behaviour changes when the model
changes, when the prompt changes, when the temperature changes, and sometimes
for no visible reason at all. A prompt edit is a silent schema change that no
diff tool will flag.

**Assumption 3: malformation is the failure mode.**
Structured-output APIs guarantee syntactically valid JSON conforming to the
declared schema. Every structural test passes while the values are wrong. The
failure mode is *plausibility*, not malformation — a fabricated vendor name is
a perfectly well-typed string.

### 1.2 The grain problem

This is the single most consequential design input, and it is empirically
established rather than a matter of taste.

Published 2026 benchmarks on frontier models doing structured extraction
report these paired numbers on identical data:

| Task | Field accuracy | Document accuracy |
|---|---|---|
| PII extraction | 0.966 – 0.979 | 0.260 – 0.460 |
| Financial entity extraction | 0.887 – 0.949 | 0.422 – 0.700 |
| Insurance claim extraction | 0.750 – 0.775 | 0.300 – 0.400 |

Document accuracy counts a document as wrong if *any* field is wrong. A system
at 97% field accuracy is at 26% document accuracy. If downstream consumers need
whole records — and they almost always do — reporting field pass-rates is
actively misleading.

Two consequences flow directly into the design:

- The framework reports at **both grains**, always, for every check.
- Field-to-document rollup defaults to a **soft minimum** (harmonic mean),
  not an average. Nineteen fields at 0.99 and one at 0.02 averages to 0.94
  (reads healthy) and harmonises to 0.29 (reads broken — which it is).

### 1.3 The verification economics problem

A model-based check costs roughly what the extraction cost. You cannot run
one on every record and call it quality assurance; you have built a second
pipeline with the same reliability profile as the first.

The framework therefore treats **cost as a first-class dimension**: every check
declares its kind, the planner estimates spend before executing, and a budget
guard degrades gracefully. "Degrades gracefully" specifically means marking
results *unscored*, never passing them.

### 1.4 The trust-laundering problem

An LLM judging an LLM's output is an opinion, not a measurement, until you
have shown that its scores rank wrong extractions above right ones. The
framework makes this structural rather than advisory: **a model-based
expectation with blocking severity and no calibration is a configuration
error that fails at plan time.**

### 1.5 What good looks like

A run produces, for every `(document, field)` and every `(document)`:

- a verdict (pass / fail / deliberately unscored)
- a score where one is meaningful, with the threshold and its provenance
- evidence explaining the verdict (a span, a ratio, a judge explanation)
- the cost incurred
- the exact function that produced it (model, prompt version, strategy)

and a manifest pinning the entire run so two runs can be compared.

---

## 2. Design principles

Each principle states the rule, why it exists, and what it forces.

### P1 — Async core, sync shims

`Expectation.validate` is `async def`. Providers are IO-bound and concurrency
is the entire performance story.

**Forces:** two ergonomic escape hatches so simple checks stay simple —
`SyncExpectation.check()` for a normal method, and `@field_check` for a single
function. Both are wrapped into the async contract by the base class. The
runner only ever awaits. A `blocking=True` flag routes CPU-heavy sync checks
through `asyncio.to_thread`.

### P2 — Observe-only

The framework never invokes the extractor. A `Batch` carries records that
already exist plus a resolver for their source documents.

**Forces:** composes with any extraction stack. Costs us
`expect_extraction_consistent_across_samples`, which needs re-generation; that
is scoped as an opt-in hook, not smuggled into the core.

### P3 — Python-native, warehouse-optional

No database required to run. SQL pushdown is an execution-engine plugin, not
the foundation.

**Forces:** the deterministic tier must be implementable in pure Python first.
Pushdown becomes an optimisation with an equivalence test against the Python
path.

### P4 — Both grains, always

Every check reports at field grain and document grain.

**Forces:** the `Result` type carries `grain` plus a nullable `field_name`.
Expectations that compute their own document score (model-based ones do, via
the strategy) own it; everything else gets a rollup synthesised by the runner.
The runner must dedupe on `(expectation_id, doc_id)` or it double-reports.

### P5 — Three result states, not two

`success` is `True`, `False`, or `None`. `None` means *deliberately unscored*:
sampled out, budget capped, provider errored, not applicable.

**Forces:** without it, dropped coverage is indistinguishable from a pass and
the quality signal silently rots. Budget exhaustion must never convert a run
to green.

### P6 — Provider and strategy are orthogonal

A **provider** knows transport, auth, tokens and money. A **strategy** knows
how to interrogate a model. Neither knows about the other's concerns.

**Forces:** you can A/B a cheap verifier against an expensive one without
touching any expectation, and reuse the five-call ensemble on a local model.
If an expectation hardcodes a model, both become impossible.

### P7 — Fail at plan time, not run time

Capability gaps, missing calibrations, stale calibrations and misconfigured
aliases are detected before a single token is spent.

**Forces:** providers must declare capabilities declaratively; strategies must
declare requirements; the planner performs set-difference negotiation.

### P8 — Thresholds are derived, never typed

A threshold comes from a calibration fitted at a target precision against
labelled examples.

**Forces:** `Calibration` is a first-class persisted object with a fingerprint
over `(gold_set_hash, provider, model_version, strategy)`. Any drift in those
invalidates it.

### P9 — Evidence over booleans

Every result carries structured evidence: the matched span, the fuzzy ratio,
the closest source text, the judge's explanation, the aggregation used, the
weakest field.

**Forces:** an `Evidence` type on every result path, including skips. A red
dashboard nobody can action is worse than no dashboard.

### P10 — Cheap tiers gate expensive tiers

Deterministic checks run first; their verdicts feed the model tier's routing
decision.

**Forces:** the runner threads prior results into `Context.prior`, and the
model-based expectation escalates suspect documents plus a random audit
stratum of clean ones. The audit stratum is the only thing that tells you
whether the cheap gates work.

---

## 3. Architecture

### 3.1 Layers

```
┌──────────────────────────────────────────────────────────────┐
│  Interface        YAML suite  ·  Python API  ·  CLI          │
├──────────────────────────────────────────────────────────────┤
│  Orchestration    Planner ──> Runner ──> Sinks               │
│                   (guards, cost)  (tiers, budget, cache)     │
├──────────────────────────────────────────────────────────────┤
│  Contracts        Expectation · Provider · Strategy ·        │
│                   Aggregator · Sink · ExecutionEngine        │
├──────────────────────────────────────────────────────────────┤
│  Domain           Batch · ExtractionRecord · SourceDoc ·     │
│                   Result · Cost · Evidence · Provenance      │
└──────────────────────────────────────────────────────────────┘
```

Dependencies point downward only. Domain imports nothing from the framework.
Contracts import domain. Orchestration imports contracts. Interface imports
orchestration. A circular import means a layering mistake.

### 3.2 Request lifecycle

```
Batch (records + lazy source resolver)
   │
   ├─> Planner
   │      capability negotiation      → PlanError on gap
   │      calibration guard           → PlanError if blocking + uncalibrated
   │      staleness guard             → PlanError if calibration fingerprint drifted
   │      correlated-verifier check   → warning
   │      cost estimation             → warning if over budget
   │      tier ordering               → deterministic, statistical, model_based
   │
   ├─> Runner
   │      for each step in tier order:
   │        deterministic/statistical → run, merge verdicts into ctx.prior
   │        model_based               → route by ctx.prior + audit_rate
   │                                     reserve budget → strategy.score()
   │                                     → provider.complete() × N in parallel
   │      rollup field → document for checks that did not self-score
   │
   └─> RunResult(results, cost, manifest, warnings) ──> Sinks
```

### 3.3 The six plugin seams

| Seam | Entry-point group | Swaps | Ships with core |
|---|---|---|---|
| Expectation | `llmex.expectations` | Check types | 5 built-ins |
| Provider | `llmex.providers` | Model transport + capabilities | `mock`, `HTTPChatProvider` base |
| Strategy | `llmex.strategies` | How you interrogate a model | 4 built-ins |
| Aggregator | `llmex.aggregators` | Field→document semantics | 4 built-ins |
| Sink | `llmex.sinks` | Result destinations | `console`, `jsonl` |
| Execution engine | `llmex.engines` | Where deterministic checks run | `python` (M2), `sql` (M8) |

Registration is via `importlib.metadata` entry points, the same mechanism dbt
uses for adapters. `pip install llmex-anthropic` makes a provider available with
zero core changes. An in-process `register()` path exists for tests and
notebooks; both resolve through the same lookup so core code never branches
on origin.

### 3.4 Why provider ≠ strategy, concretely

```python
# WRONG — model baked into the check
class ExpectTrustworthy:
    async def validate(self, batch, ctx):
        resp = await openai.chat(model="gpt-5", messages=[...])

# RIGHT — three independent axes
expectation = ExpectFieldTrustworthy(provider="cheap", strategy="ensemble",
                                     calibration="invoices_v1")
```

With the second form, swapping `gpt-4.1-mini` for a local Qwen is a one-line
YAML change, and comparing `single_judge` against `diverse_ensemble` on the
same provider is another. With the first, both require code edits and the
calibration cannot be keyed to what actually produced the score.

---

## 4. Domain model

All domain types are dataclasses with no framework dependencies. They must be
JSON-serialisable via `.as_dict()` where they cross a boundary.

### 4.1 Enumerations

```python
class Kind(str, Enum):
    DETERMINISTIC = "deterministic"   # pure function of the data, free
    STATISTICAL   = "statistical"     # aggregate over the batch, cheap
    MODEL_BASED   = "model_based"     # costs money, non-deterministic

class Grain(str, Enum):
    FIELD    = "field"        # one (doc_id, field_name)
    DOCUMENT = "document"     # one doc_id, all its fields
    CORPUS   = "corpus"       # the whole batch

class Severity(str, Enum):
    ERROR = "error"   # blocking
    WARN  = "warn"    # reported, non-blocking
    INFO  = "info"    # informational only

class Capability(str, Enum):
    STRUCTURED_OUTPUT = "structured_output"
    LOGPROBS          = "logprobs"
    SYSTEM_PROMPT     = "system_prompt"
    BATCH_API         = "batch_api"
    EMBEDDINGS        = "embeddings"

class SkipReason(str, Enum):
    BUDGET_EXHAUSTED = "budget_exhausted"
    SAMPLED_OUT      = "sampled_out"
    UPSTREAM_FAILED  = "upstream_failed"
    PROVIDER_ERROR   = "provider_error"
    NOT_APPLICABLE   = "not_applicable"
```

`Kind` drives tier ordering and budget treatment. `Grain` drives reporting.
`Capability` drives plan-time negotiation. `SkipReason` makes the `None`
verdict state auditable.

### 4.2 Value types

```python
@dataclass
class Cost:
    usd: float = 0.0
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0    # NB: __add__ takes max(), not sum() — parallel
    cache_hits: int = 0
```

`Cost.__add__` sums everything except latency, which takes the maximum,
because verifier calls run in parallel. Summing latency across an ensemble
would report 5× the wall-clock truth.

```python
@dataclass
class Evidence:
    kind: str = "none"          # "exact_match" | "fuzzy_match" | "trust_score" | ...
    detail: dict = {}           # kind-specific payload

@dataclass
class Provenance:
    expectation_id: str
    expectation_version: str
    provider_id: str | None = None
    model_version: str | None = None
    strategy_id: str | None = None
    prompt_version: str | None = None
```

`prompt_version` is the field teams omit and then cannot debug. It must be
carried from the extraction record through to every result. Treat
`(model, prompt, schema)` as one versioned artifact: a change to any of them
means historical rows were produced by a different function and are not
comparable. This is an SCD problem wearing a new hat.

### 4.3 Batch and records

```python
@dataclass(frozen=True)
class SourceDoc:
    doc_id: str
    text: str
    meta: dict = {}

@dataclass(frozen=True)
class ExtractionRecord:
    doc_id: str
    field_name: str
    value: object
    span: tuple[int, int] | None = None
    prompt_version: str | None = None
    generator_model: str | None = None
    extraction_id: str | None = None
    meta: dict = {}
```

`span` is optional but enormously valuable: when present, groundedness becomes
an O(1) slice comparison instead of an O(n) window scan, and it verifies the
model extracted from the location it claimed.

```python
class Batch:
    def __init__(self, records, source_resolver: Callable[[str], SourceDoc],
                 schema: dict | None = None)
    def source(self, doc_id) -> SourceDoc          # memoised
    def by_document(self) -> dict[str, list[ExtractionRecord]]
    def for_fields(self, names: list[str]) -> list[ExtractionRecord]   # ["*"] = all
    @property doc_ids -> list[str]                 # insertion-ordered
    @property field_names -> list[str]
```

The resolver is a **callable, not a dict**, so a batch can span a corpus that
does not fit in memory. Resolved docs are cached for the batch's lifetime.
`doc_ids` and `field_names` preserve insertion order for deterministic output.

### 4.4 Context

```python
@dataclass
class Context:
    providers: dict[str, ModelProvider]
    strategies: dict[str, ScoringStrategy]
    calibrations: dict[str, Calibration]
    budget: Budget | None
    prior: dict[tuple[str, str], bool]   # (doc_id, field) -> passed all cheap checks
    run_id: str
```

`prior` is the mechanism behind P10. Deterministic steps write into it; the
model tier reads it to decide escalation.

### 4.5 Result

```python
@dataclass
class Result:
    expectation_id: str
    grain: Grain
    doc_id: str
    field_name: str | None = None      # None for DOCUMENT/CORPUS grain

    success: bool | None = None
    score: float | None = None
    threshold: float | None = None
    threshold_source: str = "manual"   # "manual" | "calibration:{id}@{fingerprint}"

    severity: Severity = Severity.ERROR
    observed: object = None
    evidence: Evidence = Evidence()
    cost: Cost = Cost()
    provenance: Provenance | None = None
    skip_reason: SkipReason | None = None

    @property blocking_failure -> bool   # success is False and severity is ERROR
```

`threshold_source` is what makes an audit possible. A reviewer can see at a
glance whether a number was defended by a calibration or typed by someone.

```python
@dataclass
class RunResult:
    run_id: str
    results: list[Result]
    cost: Cost
    manifest: dict
    warnings: list[str]

    def failures(self) -> list[Result]
    def unscored(self) -> list[Result]
    def summary(self) -> dict     # counts by grain, blocking failures, cost
```

---

## 5. Plugin contracts

### 5.1 Registry

```python
class Registry:
    def __init__(self, group: str)                 # entry-point group name
    def register(self, name: str, obj) -> obj      # in-process
    def plugin(self, name: str)                    # decorator form
    def get(self, name: str)                       # local first, then entry points
    def names(self) -> list[str]                   # union, sorted

PROVIDERS    = Registry("llmex.providers")
STRATEGIES   = Registry("llmex.strategies")
EXPECTATIONS = Registry("llmex.expectations")
AGGREGATORS  = Registry("llmex.aggregators")
SINKS        = Registry("llmex.sinks")
ENGINES      = Registry("llmex.engines")        # M8
```

Entry-point discovery is wrapped in a bare `except` and cached. A broken
third-party plugin must not prevent the framework from importing; it should
surface as a missing name with a helpful `available: [...]` message.

### 5.2 Expectation

```python
class Expectation(ABC):
    id: str                                       # "expect_field_grounded_in_source"
    version: str                                  # bump on semantic change
    kind: Kind
    grain: Grain
    required_capabilities: frozenset[Capability] = frozenset()

    def __init__(self, severity: Severity = Severity.ERROR, **config)
    def estimate_calls(self, batch: Batch) -> int          # 0 for deterministic
    @abstractmethod
    async def validate(self, batch: Batch, ctx: Context) -> list[Result]
```

Returns a **list**, always — a field-grain check over a batch produces many
results, and a uniform return type keeps the runner simple.

`version` is not decoration. When you change what a check means, bump it; the
manifest records it, and two runs with different expectation versions are not
comparable even if the suite name matches.

#### Sync shim (P1)

```python
class SyncExpectation(Expectation):
    blocking: bool = False           # True → asyncio.to_thread

    @abstractmethod
    def check(self, batch: Batch, ctx: Context) -> list[Result]

    async def validate(self, batch, ctx):
        if self.blocking:
            return await asyncio.to_thread(self.check, batch, ctx)
        return self.check(batch, ctx)
```

#### One-function shim

```python
@field_check(id="expect_currency_iso4217", fields=["currency"])
def currency_is_iso(rec, batch, ctx):
    ok = rec.value in {"USD", "EUR", "GBP", "CAD"}
    return ok, Evidence("enum", {"got": rec.value})
```

The function returns `bool` or `(bool, Evidence)`. The decorator generates a
`SyncExpectation` subclass and registers it.

**Implementation note (learned the hard way):** the generated class must be
built with `type(name, bases, namespace)` including `check` in the namespace.
Assigning `cls.check = fn` *after* class creation leaves `__abstractmethods__`
populated and instantiation fails with `Can't instantiate abstract class`.

### 5.3 Model provider

```python
@dataclass(frozen=True)
class CostModel:
    usd_per_1k_in: float = 0.0
    usd_per_1k_out: float = 0.0
    def price(self, tokens_in: int, tokens_out: int) -> float

@dataclass(frozen=True)
class Limits:
    max_concurrency: int = 8
    requests_per_minute: int | None = None
    max_context_tokens: int = 128_000

@dataclass
class CompletionRequest:
    user: str
    system: str | None = None
    json_schema: dict | None = None
    temperature: float = 0.0
    max_tokens: int = 1024
    want_logprobs: bool = False
    tag: str = ""            # which strategy template issued this — for debugging

@dataclass
class CompletionResponse:
    text: str = ""
    parsed: dict | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    logprobs: list[float] | None = None
    meta: dict = {}

@runtime_checkable
class ModelProvider(Protocol):
    id: str
    model_version: str
    capabilities: frozenset[Capability]
    cost: CostModel
    limits: Limits
    async def complete(self, req: CompletionRequest) -> CompletionResponse: ...
```

A provider knows **nothing** about expectations, scores or thresholds. That is
the whole point. `tag` exists so that when an ensemble disagrees you can tell
which framing produced which score.

### 5.4 Scoring strategy

```python
@dataclass
class ScorePayload:
    doc_id: str
    source_text: str
    extraction: dict[str, object]
    schema: dict = {}
    instructions: str = ""

@dataclass
class ScoreSet:
    doc_score: float
    field_scores: dict[str, float]
    explanations: dict[str, str] = {}
    cost: Cost = Cost()
    n_calls_used: int = 0
    n_calls_dropped: int = 0

@runtime_checkable
class ScoringStrategy(Protocol):
    id: str
    required_capabilities: frozenset[Capability]
    def estimate_calls(self, n_fields: int) -> int
    async def score(self, payload: ScorePayload, provider) -> ScoreSet: ...
```

`n_calls_dropped` surfaces timeout-discarded ensemble members. If it is
consistently non-zero, either the timeout is too tight or the provider is
degraded — both worth knowing and invisible otherwise.

### 5.5 Aggregator

```python
Aggregator = Callable[[Mapping[str, float] | Sequence[float], dict | None], float]
```

Must accept **either** a mapping of field→score or a bare sequence. The runner
passes a mapping so weighted variants can see field names; unweighted variants
must not choke on it. (This is a real bug the prototype's tests caught: naive
`for s in scores` over a dict iterates keys.)

### 5.6 Sink

```python
class Sink(Protocol):
    async def emit(self, run: RunResult) -> None: ...
```

Async so that warehouse and HTTP sinks are drop-in.

### 5.7 Execution engine (M8)

```python
class ExecutionEngine(Protocol):
    id: str
    def can_pushdown(self, expectation: Expectation) -> bool
    async def execute(self, expectation, batch, ctx) -> list[Result]
```

Deferred to M8. The contract is stated now so the deterministic tier is written
without assuming in-process Python execution.

---

## 6. Built-in catalog

Naming follows the Great Expectations convention so the vocabulary is
learnable. Semantics differ where they must.

### 6.1 Expectations

| id | kind | grain | Cost | Catches |
|---|---|---|---|---|
| `expect_field_type` | deterministic | field | free | Type/nullability violations |
| `expect_field_grounded_in_source` | deterministic | field | free | Fabricated values |
| `expect_fields_to_satisfy` | deterministic | document | free | Cross-field rule violations |
| `expect_field_null_rate_between` | statistical | corpus | free | Silent drift, extraction collapse |
| `expect_field_trustworthy` | model_based | field + document | $$ | Plausible-but-wrong values |

#### `expect_field_grounded_in_source`

**The highest-value check in the framework, and it costs nothing.** If a value
cannot be located in the source text, the model invented it.

Config: `fields` (list or `["*"]`), `min_ratio` (default 0.92),
`allow_null` (default true).

Resolution order:
1. Null/empty → pass if `allow_null`, evidence `null_value`
2. Exact substring match → pass, evidence `exact_match`, score 1.0
3. `span` present → compare `source[start:end]` against the value with
   `SequenceMatcher`, evidence `span_match` including the quoted source
4. Otherwise → sliding-window fuzzy scan, evidence `fuzzy_match` including
   `closest_source_text`

The window scan steps by `len(needle)//3` and short-circuits on a perfect
match. This is O(n·m) in the worst case; for large documents M6 should add an
n-gram index prefilter.

**Design note.** There is academic work (anchor-constrained extraction) that
pushes this further upstream: build an inventory of all meaningful spans
*before* extraction so the model can only select from a closed set. That is an
extraction-side intervention and out of scope under P2, but it is the correct
long-term fix and worth flagging to whoever owns the extractor. Published
hallucination rates span 0.23%–20.23% across model/dataset configurations —
a two-order-of-magnitude spread, meaning this is a per-field, per-model
property you must measure rather than assume.

#### `expect_field_type`

Config: `types` (`{field: json_type}`), `nullable` (list).

Cheap, and it almost always passes when the extractor uses structured-output
mode — which is exactly why it is necessary but nowhere near sufficient. Keep
it for the case where someone swaps to a non-constrained decode path.

#### `expect_fields_to_satisfy`

Config: `expression` (Python expression over the document's fields),
`label` (human name).

Document-grain cross-field rule. Sums, date ordering, referential rules,
currency enums.

**Security note.** The prototype uses `eval` with `{"__builtins__": {}}`. That
is adequate for a trusted config file and inadequate for anything user-supplied.
M7 must replace it with a restricted AST evaluator (walk the tree, allow only
`Compare`, `BoolOp`, `BinOp`, `Name`, `Constant`, `Subscript`; reject
everything else). Track as a hard blocker for any multi-tenant deployment.

**Design guidance.** Fields requiring computation are not extraction fields.
If a value is derived — a total, a duration, an inferred end date — pull it out
of the model's job and compute it in this layer from extracted primitives. You
get determinism, testability, and a smaller surface for the expensive tier.

#### `expect_field_null_rate_between`

Config: `min_rate` (0.0), `max_rate` (1.0).

Corpus-grain silent-degradation detector. Group by `prompt_version` in the sink
and a regression stops being "quality dropped" and becomes "quality dropped
when we shipped prompt v7."

M5 should add siblings: `expect_field_cardinality_stable`,
`expect_value_distribution_similar_to_baseline`.

#### `expect_field_trustworthy`

Config: `provider`, `strategy`, `calibration`, `audit_rate` (0.05),
`threshold` (fallback, manual), `seed`.

Routes documents to a verifier and **emits results at both grains from a
single pass** — field scores from `ScoreSet.field_scores`, document score from
`ScoreSet.doc_score`.

Sampling: escalate a document if any of its fields already failed a cheap
check (`ctx.prior`), OR with probability `audit_rate`. The audit stratum is
what tells you whether the cheap gates work at all.

Skip paths, all producing `success=None` with a `SkipReason`:
`SAMPLED_OUT`, `BUDGET_EXHAUSTED`, `PROVIDER_ERROR`.

### 6.2 Scoring strategies

| id | Calls/doc | Requires | Notes |
|---|---|---|---|
| `logprob` | 0 | `LOGPROBS` | Free but unavailable on many APIs and poorly calibrated |
| `single_judge` | 1 | `STRUCTURED_OUTPUT` | Weakest — a holistic judge systematically overlooks individual fields |
| `per_field_judge` | n fields | `STRUCTURED_OUTPUT` | Best per-field recall; dies on 50+ field schemas via token rate limits |
| `diverse_ensemble` | ~5 | `STRUCTURED_OUTPUT` | Best measured accuracy/cost point |

#### `diverse_ensemble` in detail

Five parallel calls with deliberately **different** framings. Diversity is the
mechanism — identical prompts reproduce the same blind spot five times.

| tag | framing |
|---|---|
| `holistic` | Argue why the extraction might be wrong, then rate each field |
| `strict` | Treat unverifiable or partially correct values as wrong |
| `grounding` | Decide whether each value appears in the source |
| `omission` | Look for information in the source missing from the extraction |
| `format` | Check type, format and normalisation against the schema |

Three properties that matter:

1. **No sequential dependencies** — all calls issue concurrently via
   `asyncio.wait`, so wall-clock is one call, not five.
2. **Timeout discards laggards** rather than blocking the batch. Pending tasks
   are cancelled and counted in `n_calls_dropped`.
3. **Aggregation is two-stage** — arithmetic mean *across templates* for each
   field (they are noisy estimates of the same quantity), then harmonic mean
   *across fields* for the document score (failure is the union of component
   failures).

Empirically, additional ensemble members beyond ~5 yield under 1% improvement.
Do not add more without measuring.

### 6.3 Aggregators

| id | Semantics | When |
|---|---|---|
| `harmonic` | Soft minimum | Default. Any bad field poisons the document |
| `weighted_harmonic` | Soft min with field criticality | When `total_amount` matters more than `notes` |
| `arithmetic` | Mean | Almost never. Included for comparison and to make the point |
| `minimum` | Hard minimum | Zero tolerance, very noisy |

All clamp inputs to `[EPS, 1.0]` so a zero score does not divide by zero.

---

## 7. Execution model

### 7.1 Planning

```python
class Planner:
    def plan(self, suite: Suite, batch: Batch) -> Plan
```

For every model-based expectation, in order:

| Guard | Condition | Outcome |
|---|---|---|
| Alias resolution | provider/strategy alias not in suite | `PlanError` listing defined aliases |
| Capability | `strategy.required_capabilities - provider.capabilities` non-empty | `PlanError` naming the missing capabilities |
| Calibration | `severity is ERROR` and no calibration | `PlanError` — "an uncalibrated model score is an opinion, not a threshold" |
| Staleness | calibration fingerprint ≠ current provider/model/strategy | `PlanError` — "recalibrate" |
| Correlation | `provider.model_version` in `{r.generator_model}` | **Warning** — errors will be correlated |
| Budget | estimated spend > `budget.max_usd` | **Warning** — checks will be marked unscored |

The correlation check is a warning rather than an error because there are
legitimate reasons to self-verify (no second model available, cost) — but it
must be loud, because a self-graded score reads systematically high.

`Plan.ordered()` sorts steps by `{DETERMINISTIC: 0, STATISTICAL: 1,
MODEL_BASED: 2}`. This ordering is load-bearing, not cosmetic (P10).

### 7.2 Running

```python
class Runner:
    async def run(self, suite, batch, plan=None) -> RunResult
    def run_sync(self, suite, batch, plan=None) -> RunResult   # asyncio.run
```

Steps:

1. Build `Context` from the suite.
2. For each step in tier order, `await expectation.validate(batch, ctx)`.
3. After each deterministic/statistical step, merge field verdicts into
   `ctx.prior` with AND semantics: a field is "clean" only if every cheap check
   passed it.
4. Roll up field→document for any `(expectation_id, doc_id)` pair that does not
   already have a document result. **This dedupe is mandatory** — model-based
   expectations self-score, and without the guard they get two contradictory
   document rows.
5. Assemble the manifest, emit to all sinks.

**M6 upgrade:** deterministic steps are mutually independent and should fan out
with `asyncio.gather`, with `ctx.prior` merged after the gather completes.
The prototype runs them sequentially; correctness is identical, throughput is not.

### 7.3 Sampling and routing

```
for each document:
    suspect = any field failed a cheap check
    audit   = rng.random() < audit_rate       # seeded for reproducibility
    escalate = suspect or audit
```

The RNG is seeded from config so the same batch produces the same audit
sample across runs — otherwise you cannot compare two runs.

Typical operating point: 100% of suspect documents plus a 5% audit stratum,
landing near the 1–5% human-review budget that production document-processing
deployments report.

### 7.4 Budget

```python
@dataclass
class Budget:
    max_usd: float = inf
    max_calls: int = 2**31
    async def reserve(self, est_calls, est_usd=0.0) -> bool   # check-and-hold
    async def record(self, cost: Cost) -> None
```

Guarded by an `asyncio.Lock`. `reserve()` is called before issuing calls;
`False` means produce unscored results with `BUDGET_EXHAUSTED`.

**Known gap (M6):** reserve/record is not a true two-phase reservation —
concurrent documents can each pass `reserve()` and collectively overshoot.
Fix by holding a reservation token that `record()` settles.

### 7.5 Caching (M6)

Cache key:

```
sha256(expectation_id, expectation_version, config_hash,
       input_hash, provider_id, model_version, strategy_id, prompt_version)
```

`input_hash` covers the source text and the extraction values. Any drift in
any component is a miss. Cache hits increment `Cost.cache_hits` and contribute
zero USD.

This is the single largest cost win available: without it, every retry
re-bills every verifier call.

### 7.6 Concurrency and rate limits (M6)

The prototype relies on a per-provider `asyncio.Semaphore`. That does not
survive a real TPM ceiling. M6 needs:

- A token-bucket limiter per provider keyed on `Limits.requests_per_minute`
  and a token-per-minute estimate
- Exponential backoff with jitter on 429, honouring `Retry-After`
- A circuit breaker that trips a provider after N consecutive failures and
  produces `PROVIDER_ERROR` skips rather than hanging the run

---

## 8. Calibration subsystem

### 8.1 Why this exists

A model-based expectation is a binary classifier: it predicts "this extraction
is wrong." An uncalibrated classifier is an opinion. This subsystem turns
labelled examples into a threshold you can defend in a design review, and gives
the planner something to refuse when it is missing (P8).

### 8.2 Metrics

All implemented without numpy or sklearn — the framework must install cleanly
with only `pyyaml`.

**AUROC** (rank-based, Mann-Whitney U). Measures how well low scores rank the
wrong extractions above the right ones. Handles ties by average rank.

```
pos = scores of INCORRECT extractions  (should be low)
neg = scores of CORRECT extractions    (should be high)
U   = rank_sum(incorrect) - n_err(n_err+1)/2
AUROC = 1 - U/(n_err · n_ok)
```

**Precision @ num-errors.** The operational question: *if we review the K
lowest-scoring items, what share are actually wrong?* K defaults to the true
error count. More actionable than AUROC for capacity planning.

**Confidence gap.** Mean score of correct minus mean score of incorrect. Unlike
the rank metrics this is scale-sensitive, so a human can interpret raw score
magnitudes rather than only relative ordering.

**Threshold for target precision.** Sweep candidate thresholds, keep the one
maximising recall subject to `precision >= target`. Returns
`(threshold, achieved_recall)` so you can see what you are giving up.

### 8.3 The Calibration object

```python
@dataclass
class Calibration:
    id: str
    gold_set_hash: str
    provider_id: str
    model_version: str
    strategy_id: str
    metrics: dict          # auroc, precision_at_num_errors, confidence_gap,
                           # recall_at_target_precision, per_field_auroc
    thresholds: dict       # field_name -> float, plus "__default__"
    n_labels: int
    created_at: str

    def fingerprint(self) -> str     # sha256 of gold_set|provider|model|strategy
    def valid_for(self, provider_id, model_version, strategy_id) -> bool
    def threshold_for(self, field_name) -> float | None
    @property ref -> str             # "calibration:{id}@{fingerprint}"
```

Per-field thresholds are only emitted when a field has **at least 10 labels**.
Below that there is no defensible number and the default is used. This
guardrail matters: per-field thresholds fitted on three examples are worse than
no per-field thresholds.

`ref` lands in `Result.threshold_source`, so every result carries a pointer to
the evidence for its own threshold.

### 8.4 Workflow

```
1.  Sample documents (stratified, not random — oversample rare field patterns)
2.  Run the strategy over them, capture per-field and per-document scores
3.  Have humans label correctness, double-annotated with adjudication
4.  calibrate(id, labels, provider_id, model_version, strategy_id,
              target_precision=0.9)
5.  Persist. Reference by id from the suite.
6.  Re-run whenever provider, model, strategy, prompt or gold set changes.
```

### 8.5 The gold set is not trustworthy either

When researchers benchmarked frontier models against existing public
structured-extraction datasets, they found that many apparent model "errors"
were mistakes in the benchmark's own ground truth, and concluded that every
public dataset they reviewed was too noisy to support reliable accuracy
measurement. Documented failure patterns included inconsistent annotation of
conceptually identical cases, ambiguous category boundaries, and labels that
captured a descriptive clause instead of the value.

Your internal gold set will have the same disease. Therefore:

- **Double-annotate** and adjudicate disagreements.
- **Treat inter-annotator disagreement as a signal that the field definition
  is ambiguous**, not that an annotator was careless. Fix the definition.
- **Version and content-hash** the gold set; it is an input to the calibration
  fingerprint.
- Budget for label maintenance as an ongoing cost, not a one-off.

### 8.6 Storage (M5)

```python
class CalibrationStore(Protocol):
    def get(self, id: str) -> Calibration | None
    def put(self, cal: Calibration) -> None
    def list(self) -> list[str]
```

Ship a `FileCalibrationStore` writing JSON under `calibrations/{id}.json`, and
a `WarehouseCalibrationStore` for teams that want them versioned alongside
results. Suites reference calibrations by id; the store is injected.

---

## 9. Configuration reference

Everything expressible in YAML is also constructible in Python, so notebooks
and tests never need a temp file.

### 9.1 Full schema

```yaml
suite: invoice_extraction          # required, string
version: 3                         # required, coerced to string

providers:                         # alias -> provider spec
  cheap_verifier:
    plugin: openai                 # entry-point name; required
    model: gpt-4.1-mini            # passed to the provider constructor
    max_concurrency: 20
  local:
    plugin: vllm
    endpoint: http://gpu-01:8000
    model: qwen3-32b

strategies:                        # alias -> strategy spec; optional
  ensemble:                        # all built-ins are auto-aliased by their id
    plugin: diverse_ensemble
    n_calls: 5
    timeout_s: 20

budget:
  max_usd: 40
  max_calls: 20000

aggregate:
  field_to_document: weighted_harmonic
  field_weights:
    total_amount: 3.0
    currency: 0.5

expectations:                      # list; each needs `type`
  - type: expect_field_grounded_in_source
    fields: ["vendor", "invoice_date", "total_amount"]
    min_ratio: 0.92
    severity: error                # error | warn | info; default error

  - type: expect_field_trustworthy
    provider: cheap_verifier       # alias from `providers`
    strategy: ensemble             # alias from `strategies`
    audit_rate: 0.05
    calibration: invoices_v1       # id resolved via the CalibrationStore
    severity: error                # requires the calibration, per P8
```

### 9.2 Loading

```python
Suite.from_dict(cfg, calibrations={"invoices_v1": cal}) -> Suite
Suite.from_yaml(path, calibrations=...) -> Suite
suite.fingerprint() -> dict        # goes into the run manifest
```

All registered built-in strategies are auto-aliased by their id, so
`strategy: diverse_ensemble` works with no `strategies:` block at all.

### 9.3 Planned additions (M7)

- `!from_calibration {calibration: invoices_v1, target_precision: 0.9}` as an
  inline threshold tag, so target precision lives with the check rather than
  with the calibration run.
- `sample:` block with richer routing rules than a flat `audit_rate`
  (`{rule: all_below, on: deterministic_pass, else_rate: 0.05}`).
- `include:` for suite composition across teams.
- JSON Schema for the config file, validated before construction, so typos
  produce a line number rather than a `KeyError`.

---

## 10. Persistence and result schema

### 10.1 Sinks

| id | Output |
|---|---|
| `console` | Human summary: counts by grain, blocking failures, cost, failure detail with evidence |
| `jsonl` | One JSON object per result plus a sidecar `.manifest.json` |
| `warehouse` (M7) | Two tables, below |

### 10.2 Warehouse landing tables

```sql
CREATE TABLE dq_result (
    run_id            VARCHAR   NOT NULL,
    expectation_id    VARCHAR   NOT NULL,
    expectation_ver   VARCHAR   NOT NULL,
    grain             VARCHAR   NOT NULL,   -- field | document | corpus
    doc_id            VARCHAR   NOT NULL,
    field_name        VARCHAR,              -- NULL for document/corpus grain
    success           BOOLEAN,              -- NULL = deliberately unscored
    skip_reason       VARCHAR,
    score             DOUBLE,
    threshold         DOUBLE,
    threshold_source  VARCHAR   NOT NULL,   -- "manual" | "calibration:id@fp"
    severity          VARCHAR   NOT NULL,
    observed          VARCHAR,
    evidence_kind     VARCHAR,
    evidence          JSON,
    provider_id       VARCHAR,
    model_version     VARCHAR,
    strategy_id       VARCHAR,
    prompt_version    VARCHAR,
    cost_usd          DOUBLE,
    cost_calls        INTEGER,
    created_at        TIMESTAMP NOT NULL
);

CREATE TABLE dq_run (
    run_id            VARCHAR   PRIMARY KEY,
    suite_name        VARCHAR   NOT NULL,
    suite_version     VARCHAR   NOT NULL,
    suite_fingerprint JSON      NOT NULL,
    estimated_calls   INTEGER,
    estimated_usd     DOUBLE,
    actual_usd        DOUBLE,
    actual_calls      INTEGER,
    n_records         INTEGER,
    n_documents       INTEGER,
    warnings          JSON,
    started_at        TIMESTAMP NOT NULL
);
```

Partition `dq_result` by `created_at` date. Cluster on `(expectation_id,
doc_id)`. The columns you will actually group by in anger are
`prompt_version`, `model_version` and `threshold_source`.

### 10.3 Derived views worth shipping

```sql
-- the grain gap, which is the number that matters
SELECT prompt_version,
       AVG(CASE WHEN grain='field'    THEN success::INT END) AS field_pass_rate,
       AVG(CASE WHEN grain='document' THEN success::INT END) AS doc_pass_rate
FROM dq_result WHERE success IS NOT NULL GROUP BY 1;

-- coverage rot: how much are we quietly not checking?
SELECT expectation_id, skip_reason, COUNT(*)
FROM dq_result WHERE success IS NULL GROUP BY 1, 2;

-- undefended thresholds
SELECT expectation_id, COUNT(*)
FROM dq_result WHERE threshold_source = 'manual' AND severity = 'error'
GROUP BY 1;
```

---

## 11. Repository layout

### 11.1 Naming and packaging

| Surface | Value | Why |
|---|---|---|
| PyPI distribution | `llm-expectations` | Discoverability. People search "great expectations for LLM" |
| Import name | `llmex` | Five characters at the top of every file |
| Entry-point groups | `llmex.providers`, `llmex.strategies`, ... | Must match the import name |
| Plugin distributions | `llmex-openai`, `llmex-anthropic` | Short, obviously in-family |
| GitHub | `llm-expectations/llm-expectations` | Matches the distribution |

Distribution name and import name deliberately differ, following
`beautifulsoup4` → `bs4` and `scikit-learn` → `sklearn`. The long name wins
search traffic; the short one wins ergonomics.

**Entry-point group names are public API.** Every third-party plugin declares
against `llmex.providers` and friends. Renaming after plugins exist is a
breaking change for the whole ecosystem, not a find-and-replace. This is
settled and must not move after M0.

**Disambiguation.** The `expect_*` convention is borrowed because it is
learnable; nobody owns the word "expect". This project is not a Great
Expectations port or plugin — suites are not interchangeable, the `Result`
type is different, and there is no GX analogue for the calibration layer. The
README states this explicitly to head off the support burden.

### 11.2 Tree

```
llm-expectations/
├── pyproject.toml               # entry points for all six registries
├── README.md
├── DESIGN.md                    # this document
├── CHANGELOG.md
├── suite.example.yml
├── demo.py                      # offline end-to-end walkthrough
│
├── llmex/
│   ├── __init__.py              # public API + built-in module imports
│   ├── types.py                 # enums, Cost, Evidence, Provenance, PlanError
│   ├── batch.py                 # SourceDoc, ExtractionRecord, Batch, Context
│   ├── result.py                # Result, RunResult
│   ├── registry.py              # Registry + the six singletons
│   ├── expectation.py           # Expectation, SyncExpectation, @field_check
│   ├── aggregate.py             # harmonic, weighted_harmonic, arithmetic, minimum
│   ├── budget.py                # Budget
│   ├── calibration.py           # metrics, Calibration, calibrate()
│   ├── cache.py                 # M6
│   ├── planner.py               # Planner, Plan, Step, guards
│   ├── runner.py                # Runner, tier ordering, rollup
│   ├── suite.py                 # Suite, YAML loading, fingerprint
│   ├── sinks.py                 # ConsoleSink, JsonlSink
│   ├── cli.py                   # M7
│   │
│   ├── providers/
│   │   ├── base.py              # protocol, CostModel, Limits, request/response
│   │   └── mock.py              # MockProvider, HTTPChatProvider skeleton
│   ├── strategies/
│   │   ├── base.py              # protocol, ScorePayload, ScoreSet
│   │   └── builtin.py           # single_judge, per_field_judge, ensemble, logprob
│   ├── expectations/
│   │   ├── builtin.py           # the five built-ins
│   │   └── ast_eval.py          # M7, replaces eval()
│   └── stores/
│       └── calibration.py       # M5, File + Warehouse stores
│
├── plugins/                     # separately published distributions
│   ├── llmex-openai/
│   ├── llmex-anthropic/
│   ├── llmex-bedrock/
│   └── llmex-warehouse/
│
├── tests/
│   ├── conftest.py              # shared fixtures
│   ├── test_types.py
│   ├── test_batch.py
│   ├── test_aggregate.py
│   ├── test_calibration.py
│   ├── test_expectations.py
│   ├── test_planner.py          # all guards
│   ├── test_runner.py           # tiers, rollup, dedupe, prior merging
│   ├── test_strategies.py
│   ├── test_budget.py
│   ├── test_suite.py
│   ├── test_registry.py
│   └── golden/                  # fixture corpora with known-seeded errors
│
└── docs/
    ├── quickstart.md
    ├── writing-expectations.md
    ├── writing-providers.md
    ├── calibration-guide.md
    └── cost-management.md
```

---

## 12. Implementation plan

Nine milestones. M0–M5 produce a framework that is genuinely usable; M6–M8 make
it production-grade. Each milestone lists its goal, files, ordered tasks, and
acceptance criteria. **A milestone is not done until its acceptance criteria
pass in CI.**

Effort estimates assume one engineer working with the reference prototype
available (Appendix A).

---

### M0 — Skeleton and CI · ~0.5 day

**Goal.** A repository that installs, lints, type-checks and runs an empty test
suite in CI.

**Files.** `pyproject.toml`, `.github/workflows/ci.yml`, `.gitignore`,
`README.md`, `llmex/__init__.py`, `tests/conftest.py`

**Tasks.**
1. `pyproject.toml` with `name = "llm-expectations"`, `requires-python = ">=3.10"`,
   dependency on `pyyaml` only, `[tool.setuptools.packages.find] include = ["llmex*"]`,
   and declared entry-point groups for all six registries (empty is fine).
   **Register the PyPI name with a 0.0.0 placeholder before writing code** — PyPI
   names are permanent and this one is not yet claimed.
2. Dev extras: `pytest`, `pytest-asyncio`, `ruff`, `mypy`.
3. `ruff` config: line length 100, select `E,F,I,UP,B,S`. `S` (bandit) matters
   because of the `eval` in `expect_fields_to_satisfy`.
4. `mypy` config: `strict = true` for `llmex/`, relaxed for `tests/`.
5. CI matrix over Python 3.10–3.13 running lint, typecheck, tests.
6. `pytest-asyncio` in `auto` mode so `async def test_` works without decorators.

**Acceptance.** `pip install -e ".[dev]"` succeeds; `ruff check`, `mypy llmex`,
`pytest` all pass on an empty suite; CI green on all Python versions.

---

### M1 — Domain core · ~1 day

**Goal.** Every type in section 4, plus the registry, plus the expectation base
and both sync shims. No checks yet.

**Files.** `types.py`, `batch.py`, `result.py`, `registry.py`, `expectation.py`

**Tasks.**
1. `types.py` — all five enums, `Cost` (with the max-latency `__add__`),
   `Evidence`, `Provenance`, `PlanError`. Every type gets `.as_dict()`.
2. `batch.py` — `SourceDoc`, `ExtractionRecord` (both frozen), `Batch` with the
   memoising callable resolver, `Context`.
3. `result.py` — `Result`, `RunResult` with `failures()`, `unscored()`,
   `summary()`.
4. `registry.py` — `Registry` with local-then-entry-point resolution, the
   `plugin()` decorator, and the six singletons. Entry-point discovery must be
   exception-safe and cached.
5. `expectation.py` — `Expectation` ABC, `SyncExpectation`, `@field_check`.
   **Build the generated class with `type(name, bases, namespace)` including
   `check`.** Assigning it post-creation leaves `__abstractmethods__` populated.

**Acceptance.**
- `Cost() + Cost()` sums tokens and takes max latency — asserted in a test.
- `Batch.source()` calls the resolver exactly once per doc_id (assert with a
  counting fake).
- `Batch.doc_ids` preserves insertion order.
- A `@field_check`-decorated function is instantiable and appears in
  `EXPECTATIONS.names()`.
- `Registry.get("nope")` raises `KeyError` whose message lists available names.
- `mypy --strict` clean.

---

### M2 — Deterministic tier and runner v1 · ~2 days

**Goal.** The framework does useful work with no model, no money, no network.

**Files.** `aggregate.py`, `expectations/builtin.py` (four deterministic and
statistical checks), `runner.py`, `sinks.py`, `suite.py`

**Tasks.**
1. `aggregate.py` — four aggregators. **`_clean()` must accept a Mapping or a
   Sequence**; the runner passes a mapping and naive iteration yields keys.
   Clamp to `[EPS, 1.0]`.
2. `expect_field_type` — the simplest check, use it to shake out the shape.
3. `expect_field_grounded_in_source` — the four-branch resolution order from
   §6.1, with distinct `Evidence.kind` per branch.
4. `expect_fields_to_satisfy` — document grain. Ship with `eval` + empty
   builtins and an explicit `# TODO(M7): AST evaluator` plus a `noqa: S307`.
5. `expect_field_null_rate_between` — corpus grain, exercises the third grain.
6. `runner.py` — tier ordering, `_merge_prior` with AND semantics, `_rollup`
   with the **`already_scored` dedupe set**, manifest assembly, `run_sync`.
7. `sinks.py` — `ConsoleSink` and `JsonlSink`.
8. `suite.py` — `from_dict`, `from_yaml`, `fingerprint`. Auto-alias every
   registered built-in strategy.

**Acceptance.**
- A seeded fixture corpus with known hallucinations: grounding flags exactly
  the seeded ones, no more, no fewer.
- Both grains present in `RunResult.results`.
- No duplicate `(expectation_id, doc_id)` document rows.
- `harmonic([0.99]*19 + [0.02]) < 0.35` while `arithmetic(...) > 0.9`.
- `harmonic({"a": 1.0, "b": 1.0})` does not raise on the mapping form.
- Manifest round-trips through JSON.

---

### M3 — Provider and strategy layer · ~2 days

**Goal.** Model-based checking works, against a mock and at least one real API.

**Files.** `providers/base.py`, `providers/mock.py`, `strategies/base.py`,
`strategies/builtin.py`, `expectations/builtin.py` (add
`expect_field_trustworthy`), `plugins/llmex-openai/`

**Tasks.**
1. `providers/base.py` — protocol, `CostModel`, `Limits`, request/response,
   `cost_of()` helper.
2. `providers/mock.py` — deterministic `MockProvider` that scores on substring
   containment with per-tag jitter so ensemble members legitimately disagree.
   **This is the backbone of the whole test suite**; get it right.
3. `HTTPChatProvider` skeleton with an unimplemented `_post`.
4. `strategies/base.py` — protocol, `ScorePayload`, `ScoreSet`.
5. `single_judge`, then `per_field_judge` (fan out with `gather`), then
   `diverse_ensemble` (five templates, `asyncio.wait` with timeout, cancel
   pending, count dropped, two-stage aggregation), then `logprob` (mostly to
   have something that requires an unusual capability).
6. `expect_field_trustworthy` — dual-grain emission, escalation logic, all
   three skip paths.
7. First real plugin distribution: `plugins/llmex-openai` with its own
   `pyproject.toml` declaring the entry point.

**Acceptance.**
- `diverse_ensemble` completes in roughly one call's latency, not five
  (assert wall-clock against a mock with 50 ms latency).
- A strategy timeout produces `n_calls_dropped > 0` and still returns a score.
- Provider exception → `PROVIDER_ERROR` skip, run completes.
- `expect_field_trustworthy` emits both a field row per field and exactly one
  document row per escalated document.
- `pip install -e plugins/llmex-openai` makes `plugin: openai` resolvable with
  no core change.

---

### M4 — Planner and guards · ~1 day

**Goal.** Every misconfiguration fails before a token is spent.

**Files.** `planner.py`, `runner.py` (accept a pre-built plan)

**Tasks.**
1. `Plan`, `Step`, `Planner`.
2. Alias resolution with a helpful error listing defined aliases.
3. Capability negotiation via set difference; error names the missing
   capabilities and the offending provider and model version.
4. Calibration guard: `severity is ERROR` and no calibration → `PlanError`.
5. Staleness guard: `calibration.valid_for(provider, model, strategy)`.
6. Correlated-verifier warning.
7. Cost estimation and over-budget warning.
8. `Plan.ordered()`.

**Acceptance.** One test per guard asserting the exception type and a
substring of the message. Plus a positive test: the same config at
`severity: warn` plans cleanly.

---

### M5 — Calibration subsystem · ~1.5 days

**Goal.** Thresholds are derived and auditable.

**Files.** `calibration.py`, `stores/calibration.py`, `cli.py` (calibrate
subcommand)

**Tasks.**
1. `auroc` with average-rank tie handling. Test against a hand-computed case
   including ties.
2. `precision_at_k`, `confidence_gap`, `threshold_for_precision`.
3. `Calibration` with `fingerprint`, `valid_for`, `threshold_for`, `ref`.
4. `calibrate()` — global threshold plus per-field thresholds, gated at a
   **minimum of 10 labels per field**.
5. `FileCalibrationStore` (JSON under `calibrations/`).
6. `llmex calibrate --suite s.yml --gold gold.jsonl --target-precision 0.9`.
7. Additional statistical expectations: `expect_field_cardinality_stable`,
   `expect_value_distribution_similar_to_baseline`.

**Acceptance.**
- AUROC returns exactly 1.0 for perfectly separated scores, 0.5 for random,
  and handles all-ties without dividing by zero.
- A field with 9 labels gets no per-field threshold; with 10 it does.
- `Calibration.ref` appears in `Result.threshold_source` end to end.
- Changing `model_version` invalidates the calibration at plan time.

---

### M6 — Execution hardening · ~2.5 days

**Goal.** Survives contact with a real API at real volume.

**Files.** `cache.py`, `providers/base.py`, `budget.py`, `runner.py`

**Tasks.**
1. **Response cache** — the §7.5 key. `MemoryCache` and `DiskCache`
   (content-addressed files). Wire through `ScoringStrategy` so every provider
   call checks it. Cache hits set `Cost.cache_hits` and zero USD.
   *This is the highest-value item in the entire milestone.*
2. **Two-phase budget** — `reserve()` returns a token that `record()` settles
   and that a context manager releases on failure. Closes the concurrent
   overshoot gap.
3. **Rate limiting** — token bucket per provider from
   `Limits.requests_per_minute` plus a tokens-per-minute estimate.
4. **Retries** — exponential backoff with jitter on 429/5xx, honour
   `Retry-After`, cap attempts.
5. **Circuit breaker** — trip after N consecutive failures, produce
   `PROVIDER_ERROR` skips rather than hanging.
6. **Parallel deterministic tier** — `asyncio.gather` across independent
   steps, merge `ctx.prior` after.
7. **Grounding prefilter** — n-gram index over the source so the fuzzy window
   scan stops being O(n·m) on long documents.

**Acceptance.**
- Second identical run reports `cache_hits > 0` and `usd == 0`.
- A fake provider returning 429 twice then succeeding produces one successful
  result and exactly three attempts.
- 100 concurrent documents against `Budget(max_usd=0.01)` never exceed it.
- Deterministic tier wall-clock scales sub-linearly in step count.

---

### M7 — Config, CLI, safety, docs · ~2 days

**Files.** `cli.py`, `expectations/ast_eval.py`, `suite.py`, `sinks.py`,
`docs/`

**Tasks.**
1. **Replace `eval`** with a restricted AST evaluator. Walk the tree; allow
   `Compare`, `BoolOp`, `UnaryOp`, `BinOp`, `Name`, `Constant`, `Subscript`,
   `Tuple`, `List`; reject everything else with a clear error. Fuzz it.
2. JSON Schema for the config, validated before construction.
3. `!from_calibration` YAML tag.
4. Richer `sample:` block.
5. `include:` for suite composition.
6. CLI: `llmex validate`, `llmex plan` (dry run with cost estimate), `llmex run`,
   `llmex calibrate`, `llmex explain <doc_id>`.
7. `WarehouseSink` writing the §10.2 tables.
8. The five docs pages.

**Acceptance.** `llmex plan` prints estimated cost and every warning without
issuing a call. AST evaluator rejects `__import__`, attribute access, calls,
and comprehensions. Fuzz corpus produces no unhandled exception.

---

### M8 — SQL pushdown engine · ~3 days · optional

**Goal.** Deterministic checks execute in the warehouse for corpus-scale batches.

**Tasks.**
1. `ExecutionEngine` protocol and `ENGINES` registry.
2. `PythonEngine` wrapping current behaviour.
3. `SqlEngine` with `can_pushdown()` per expectation; compile grounding,
   type and null-rate checks to SQL.
4. **Equivalence test harness**: the same fixture batch through both engines
   must produce identical results modulo ordering. Non-negotiable.

**Acceptance.** Equivalence suite green over every pushdown-capable
expectation across at least two dialects.

---

### M9 — Ecosystem · ongoing

Additional provider distributions (`llmex-anthropic`, `llmex-bedrock`,
`llmex-vllm`), an `nli_entailment` strategy using a small local model for
groundedness at near-zero marginal cost, a dbt-artifact sink, and an
OpenLineage emitter.

### Dependency graph

```
M0 ──> M1 ──> M2 ──> M3 ──> M4 ──> M5 ──> M6 ──> M7 ──> M8
                │              └──────────┘        │
                └──> (M2 alone is shippable)       └──> M9
```

M2 is the first shippable increment: deterministic groundedness checking with
dual-grain reporting and zero cost is already worth deploying.

---

## 13. Testing strategy

### 13.1 The prime directive

**Every test runs offline.** No API key, no network, no fixture recording.
`MockProvider` is the backbone: deterministic, free, and scriptable. If a test
needs a real provider, it belongs in a separate opt-in integration suite gated
behind an environment variable, never in CI's default path.

### 13.2 Layers

| Layer | What to test | Style |
|---|---|---|
| Value types | `Cost.__add__` latency semantics, serialisation round-trips | Unit |
| Aggregators | Soft-min property, mapping/sequence duality, zero handling | Property-based where possible |
| Calibration math | AUROC against hand-computed cases including ties; threshold monotonicity | Unit with fixtures |
| Expectations | Seeded corpora with known errors; exact flag sets | Golden fixtures |
| Planner | One test per guard, exception type plus message substring | Unit |
| Runner | Tier ordering, `prior` merging, rollup dedupe, skip propagation | Integration with mock |
| Strategies | Concurrency, timeout dropping, exception isolation | Async unit |
| Budget | Concurrent overshoot, unscored-not-passed | Async stress |
| Suite | YAML round-trip, alias resolution, fingerprint stability | Unit |

### 13.3 Golden fixtures

`tests/golden/` holds small corpora with **deliberately seeded errors** and an
answer key:

```
tests/golden/invoices/
    docs.jsonl          # doc_id, text
    extractions.jsonl   # doc_id, field, value, span, prompt_version
    answers.jsonl       # doc_id, field, is_correct, error_type
```

Seed the error types that actually occur in production, which are documented
and specific:

| Error type | Example |
|---|---|
| `fabrication` | Vendor name not present in source |
| `transposition` | 7,430.10 for 7,340.10 |
| `wrong_instance` | Picked the wrong date when several appear |
| `misattribution` | Associated a value with an unrelated concept |
| `span_drift` | Right value, wrong offsets |
| `omission` | Null where the source has a value |
| `format_drift` | `2024-02-31` — well-formed and impossible |
| `derived_error` | Computed field wrong though its inputs are right |

The last one is the argument for keeping derived fields out of the model's job
entirely; the fixture exists to demonstrate that the framework catches it via
`expect_fields_to_satisfy` rather than via an expensive judge.

### 13.4 Tests that must exist

Non-negotiable, because each pins a bug the prototype actually hit or a
principle that is easy to erode:

1. `harmonic` on a mapping does not raise (the dict-iteration bug).
2. A `@field_check` function is instantiable (the `__abstractmethods__` bug).
3. No duplicate document rows for a self-scoring expectation (the rollup bug).
4. Budget exhaustion yields `success is None`, never `True` (P5).
5. Uncalibrated + `severity: error` raises at plan time (P8).
6. Capability gap raises at plan time (P7).
7. Stale calibration raises at plan time (P8).
8. Field pass-rate exceeds document pass-rate on the seeded corpus (P4 — if
   this ever inverts, the rollup is broken).

### 13.5 Coverage targets

90% line coverage on `llmex/`, with `planner.py`, `runner.py`, `aggregate.py`
and `calibration.py` at 100%. Those four are where a silent bug produces
confidently wrong quality numbers, which is worse than a crash.

---

## 14. Operations

### 14.1 Where each tier runs

| Tier | Cadence | Environment |
|---|---|---|
| Deterministic | Every batch, 100% coverage | Inline with the extraction pipeline |
| Statistical | Per batch, aggregate | Same |
| Model-based | Per batch, sampled | Async worker pool, off the critical path |
| Calibration | On model/prompt/gold-set change | Manual or scheduled job |

Never put model-based checks in a per-PR CI path. The cost is unbounded and
the signal is noisy at small sample sizes. CI runs deterministic checks over a
frozen fixture corpus.

### 14.2 Alerting

Alert on the derived views from §10.3, not on raw failure counts:

- **Grain gap widening** — field pass-rate holding while document pass-rate
  falls means errors are spreading across fields.
- **Coverage rot** — `success IS NULL` share rising means you are quietly
  checking less than you think.
- **Undefended thresholds** — any `threshold_source = 'manual'` with
  `severity = 'error'` is a P8 violation that slipped through.
- **Ensemble drop rate** — `n_calls_dropped` trending up means timeouts are
  too tight or the provider is degraded.

### 14.3 Cost control

1. Cache first (M6) — the largest single win.
2. Tier ruthlessly. Anything expressible deterministically must not reach the
   model tier.
3. Use a cheap verifier. Verification is an easier task than generation;
   a small fast model is usually sufficient and is what production
   deployments actually run.
4. Set `audit_rate` from the calibration's precision/recall curve, not from
   intuition. The 1–5% human-review figure is an *output* of that curve, not
   an input.

### 14.4 Versioning discipline

Treat `(model, prompt, schema, expectation_version, calibration)` as one
versioned artifact. A change to any component means historical results were
produced by a different function and must not be compared naively. The
manifest exists to make this checkable; the alert exists because nobody checks
it voluntarily.

---

## 15. Extension cookbook

### 15.1 A new provider

```python
class MyProvider:
    id = "my-provider"
    model_version = "v1.2"
    capabilities = frozenset({Capability.STRUCTURED_OUTPUT, Capability.SYSTEM_PROMPT})
    cost = CostModel(usd_per_1k_in=0.15, usd_per_1k_out=0.60)
    limits = Limits(max_concurrency=20, requests_per_minute=500)

    def __init__(self, model: str, api_key: str | None = None, **kw): ...

    async def complete(self, req: CompletionRequest) -> CompletionResponse: ...
```

```toml
[project.entry-points."llmex.providers"]
my-provider = "llmex_myprovider:MyProvider"
```

**Declare capabilities honestly.** Over-declaring `LOGPROBS` moves a failure
from plan time to run time, which is exactly the failure mode P7 exists to
prevent.

### 15.2 A new expectation, one function

```python
@field_check(id="expect_currency_iso4217", fields=["currency"])
def currency_is_iso(rec, batch, ctx):
    ok = rec.value in {"USD", "EUR", "GBP", "CAD"}
    return ok, Evidence("enum", {"got": rec.value})
```

### 15.3 A new expectation, full control

```python
@EXPECTATIONS.plugin("expect_field_matches_dimension")
class ExpectFieldMatchesDimension(SyncExpectation):
    id = "expect_field_matches_dimension"
    version = "1"
    kind = Kind.DETERMINISTIC
    grain = Grain.FIELD

    def check(self, batch, ctx) -> list[Result]:
        allowed = set(self.config["values"])
        return [
            Result(expectation_id=self.id, grain=Grain.FIELD,
                   doc_id=r.doc_id, field_name=r.field_name,
                   success=r.value in allowed, severity=self.severity,
                   observed=r.value,
                   evidence=Evidence("dimension", {"n_allowed": len(allowed)}),
                   provenance=self.provenance)
            for r in batch.for_fields(self.config.get("fields", ["*"]))
        ]
```

Set `blocking = True` if `check()` does heavy CPU work; it will be offloaded.

### 15.4 A new strategy

```python
@STRATEGIES.plugin("nli_entailment")
class NliEntailment:
    id = "nli_entailment"
    required_capabilities = frozenset()      # local model, no provider needed

    def estimate_calls(self, n_fields): return 0

    async def score(self, payload, provider) -> ScoreSet:
        # run a small local NLI model; entailment probability = field score
        ...
```

The most interesting unbuilt strategy. Groundedness via a small local
entailment model costs approximately nothing per document and catches the
dominant error class.

### 15.5 A new aggregator

```python
@AGGREGATORS.plugin("p10")
def tenth_percentile(scores, weights=None) -> float:
    vals = sorted(_clean(scores))
    return vals[max(0, int(len(vals) * 0.10) - 1)] if vals else 1.0
```

Must accept a mapping or a sequence.

---

## 16. Non-goals and known limitations

### 16.1 Explicit non-goals

| Not doing | Why |
|---|---|
| Extraction / generation | P2. Composes with any extractor instead |
| Prompt management | Separate concern with mature tools |
| Model serving | Providers are thin adapters over someone else's endpoint |
| A UI | Sinks feed existing BI. Building a dashboard is a different product |
| Streaming / per-record | Batch-oriented by design; a streaming adapter can wrap it |

### 16.2 Known limitations

| Limitation | Impact | Milestone |
|---|---|---|
| No response cache | Retries re-bill every call | M6 |
| Budget reservation not two-phase | Concurrent overshoot possible | M6 |
| `eval` in `expect_fields_to_satisfy` | Unsafe for untrusted config | M7 |
| Semaphore-only rate limiting | Will not survive a real TPM ceiling | M6 |
| O(n·m) fuzzy grounding scan | Slow on long documents | M6 |
| Deterministic steps run sequentially | Throughput, not correctness | M6 |
| No self-consistency strategy | Needs generator access, forbidden by P2 | Opt-in hook, undecided |
| `HTTPChatProvider._post` unimplemented | No real provider in core | M3 |

### 16.3 Things that look like bugs and are not

- **`Cost.__add__` takes max latency.** Verifier calls run in parallel; summing
  would report 5× the wall-clock truth.
- **`success=None` is not a pass.** It is the point of P5.
- **The demo reports AUROC 1.00.** `MockProvider` scores by substring
  containment, which is exactly what the demo's labels encode. Real verifiers
  land far lower. It is a plumbing check, not a capability claim.
- **Document pass-rate is much worse than field pass-rate.** That is the
  finding, not a defect.

---

## 17. Open questions

1. **Self-consistency under observe-only.** Measuring generator agreement
   across k samples catches a failure class nothing else does, but needs
   re-generation. Options: an opt-in `regenerate: Callable` on `Batch`; a
   separate `llmex-generate` companion; or accept the gap. *Leaning toward the
   opt-in callable, since it keeps the default posture intact.*
2. **Corpus-grain baselines.** `expect_value_distribution_similar_to_baseline`
   needs a stored baseline. Reuse `CalibrationStore`, or a separate
   `BaselineStore` with its own retention?
3. **Multi-tenancy.** If suites come from untrusted users, the AST evaluator is
   necessary but not sufficient — provider credentials and budget isolation
   also need a story.
4. **Nested extractions.** The current model is flat `(doc, field)`. Real
   schemas nest (`line_items[].amount`). Options: flatten with a path syntax
   (`line_items.3.amount`) at ingest, or make `field_name` a JSONPath.
   *Leaning toward flattening at the `ExtractionRecord` boundary so the core
   stays flat.*
5. **Cross-document checks.** Deduplication and entity resolution across a
   corpus do not fit the current grains. New `Grain.ENTITY`, or out of scope?

---

## Appendix A — prototype status

A working reference implementation exists: 22 files, ~2,400 lines, 16 tests
passing, demo runs offline in under a second.

| Component | Status |
|---|---|
| `types`, `batch`, `result`, `registry` | Complete (M1) |
| Four deterministic/statistical expectations | Complete (M2) |
| Four aggregators | Complete (M2) |
| Runner with tiers, prior merging, rollup dedupe | Complete (M2) |
| `MockProvider`, `HTTPChatProvider` skeleton | Complete (M3), `_post` unimplemented |
| Four strategies incl. `diverse_ensemble` | Complete (M3) |
| `expect_field_trustworthy`, dual grain | Complete (M3) |
| Planner with all four guards | Complete (M4) |
| Calibration metrics and `calibrate()` | Complete (M5) |
| `CalibrationStore` | Not started (M5) |
| Cache, rate limiting, retries, circuit breaker | Not started (M6) |
| CLI, AST evaluator, warehouse sink | Not started (M7) |
| SQL pushdown | Not started (M8) |

Demo output on a six-document corpus with two seeded hallucinations:

```
1. plan-time guards
   calibration guard   -> severity=error requires a calibration.
   capability guard    -> strategy 'logprob' requires ['logprobs'] but
                          provider 'no_logprobs' (nolp-1) does not expose them.
   correlated verifier -> verifier model 'mock-1.0' also generated these
                          extractions.

2. calibration    auroc 1.000  thresholds {__default__: 0.95}
                  ref calibration:invoices_v1@cf7d7d6c5f78

3. full run       corpus pass=4 · document pass=8 fail=4 · field pass=50 fail=4
                  $0.0024 over 150 calls

4. grain gap      field-grain    92.6%
                  document-grain 66.7%
                  gap            25.9%
```

Three bugs the prototype surfaced, all now pinned by tests and called out in
their respective milestones: the `__abstractmethods__` decorator bug (M1), the
aggregator mapping-iteration bug (M2), and the rollup duplication bug (M2).

---

## Appendix B — glossary

| Term | Meaning |
|---|---|
| **Batch** | A set of extraction records plus a resolver for their sources |
| **Calibration** | A fitted threshold with metrics, fingerprinted to a specific provider/model/strategy/gold set |
| **Capability** | A declared provider feature negotiated at plan time |
| **Document accuracy** | Share of documents where *every* field is correct |
| **Evidence** | Structured explanation attached to every result |
| **Field accuracy** | Share of individual fields correct — the flattering metric |
| **Grain** | The unit a result describes: field, document, or corpus |
| **Grain gap** | Field pass-rate minus document pass-rate |
| **Groundedness** | Whether an extracted value appears in the source text |
| **Provenance** | The exact function that produced a result |
| **Provider** | Adapter over a model endpoint: transport, auth, cost, capabilities |
| **Soft minimum** | Harmonic mean; punished by its worst input |
| **Strategy** | How a model is interrogated, independent of which model |
| **Trust laundering** | Presenting an uncalibrated model score as a quality measurement |
| **Unscored** | `success=None`; deliberately not checked, never a pass |

---

*End of document.*
