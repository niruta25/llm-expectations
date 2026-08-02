# llm-expectations

**97% field accuracy is 26% document accuracy. Measure both.**

That gap is not a rounding error — it is what published benchmarks report for
frontier models doing structured extraction, on the same data. A document is
wrong if *any* field is wrong, and almost every team reports the flattering
number.

`llm-expectations` is a data quality framework for LLM extractions. Great
Expectations' vocabulary, rebuilt for checks that are stochastic, cost money,
and cannot be trusted until they are calibrated.

```bash
pip install llm-expectations
```

```python
import llmex
```

The distribution is `llm-expectations`; the import is `llmex`. Same pattern as
`beautifulsoup4` → `bs4`.

```bash
python demo.py          # full walkthrough, offline, no API key
pytest -q               # 130 tests, all offline
```

## The position

**We make it a configuration error to ship an uncalibrated LLM judge.**

A model-based check is a classifier. An uncalibrated classifier is an opinion.
So a check with blocking severity and no calibration fails at plan time, before
a single token is spent:

```
PlanError: expect_field_trustworthy: severity=error requires a calibration.
An uncalibrated model score is an opinion, not a threshold.
Either attach one or drop to severity=warn.
```

Three more guards fire in the same pass: a capability gap between strategy and
provider, a calibration fitted on a different model, and a verifier that also
generated the extractions it is grading.

## Decisions baked in

| Decision | Consequence |
|---|---|
| Async core, sync shims | `Expectation.validate` is `async`. `SyncExpectation.check` and `@field_check` cover the 90% case in one function |
| Observe-only | The framework never calls your extractor. `Batch` takes records that already exist plus a lazy source resolver |
| Python-native | No warehouse required. SQL pushdown is a future execution-engine plugin, not the foundation |
| Both grains | Every check reports at field *and* document grain. Rollup defaults to harmonic |

## Why harmonic rollup

Nineteen fields at 0.99 and one at 0.02: the arithmetic mean is 0.94, which
reads healthy. The harmonic mean is 0.29, which reads broken. It is broken.

Use `weighted_harmonic` when some fields matter more than others:

```yaml
aggregate:
  field_to_document: weighted_harmonic
  field_weights: {total_amount: 3.0, currency: 0.5}
```

## Five plugin seams

| Registry | Entry-point group | Swaps |
|---|---|---|
| `PROVIDERS` | `llmex.providers` | Transport, auth, token accounting, capabilities |
| `STRATEGIES` | `llmex.strategies` | How you interrogate a model |
| `EXPECTATIONS` | `llmex.expectations` | Check types |
| `AGGREGATORS` | `llmex.aggregators` | Field → document semantics |
| `SINKS` | `llmex.sinks` | Where results land |

Provider and strategy are deliberately orthogonal. Bake a model into an
expectation and you can never A/B a cheaper verifier or reuse the ensemble on
a local box.

### Adding a provider

```python
class MyProvider:
    id = "my-provider"
    model_version = "v1"
    capabilities = frozenset({Capability.STRUCTURED_OUTPUT})
    cost = CostModel(usd_per_1k_in=0.15, usd_per_1k_out=0.60)
    limits = Limits(max_concurrency=20)

    async def complete(self, req: CompletionRequest) -> CompletionResponse: ...
```

Ship it as its own distribution:

```toml
[project.entry-points."llmex.providers"]
my-provider = "llmex_myprovider:MyProvider"
```

`pip install llmex-myprovider` makes it available. No core changes.

### Adding an expectation in one function

```python
@field_check(id="expect_currency_iso4217", fields=["currency"])
def currency_is_iso(rec, batch, ctx):
    ok = rec.value in {"USD", "EUR", "GBP", "CAD"}
    return ok, Evidence("enum", {"got": rec.value})
```

## Three result states, not two

`success=None` means *deliberately unscored* — sampled out, budget capped, or
the provider errored. Without it, dropped coverage looks identical to a pass
and your quality signal silently rots. Budget exhaustion never converts to a
green run.

## Calibration

```python
cal = calibrate(
    id="invoices_v1", labels=labelled_scores,
    provider_id=p.id, model_version=p.model_version, strategy_id=s.id,
    target_precision=0.9,
)
# -> auroc, precision@num_errors, confidence_gap, per-field thresholds
```

Thresholds are derived from a target precision, never typed by hand. AUROC is
rank-based (Mann-Whitney U), implemented without numpy or sklearn.

Note: the demo reports AUROC 1.00 because `MockProvider` scores by substring
containment, which is exactly what the labels encode. Real verifiers land far
lower. Treat the number as a plumbing check, not a capability claim.

## Sampling economics

Deterministic tiers run first and their verdicts feed `ctx.prior`. The model
tier then escalates every document that already failed a cheap check, plus a
random audit stratum of the ones that passed. The audit stratum is the only
thing that tells you whether the cheap gates work.

```yaml
- type: expect_field_trustworthy
  audit_rate: 0.05      # 5% of clean docs, 100% of suspect ones
```

## Not affiliated with Great Expectations

We borrow the `expect_*` naming convention because it is learnable. This is not
a Great Expectations port or plugin: suites are not interchangeable, the result
type is different, and the execution model is different.

## Not built yet

- SQL pushdown execution engine (deterministic checks currently run in Python)
- Response cache keyed on `(expectation, config, input_hash, model, prompt_version)`
- Rate-limit-aware scheduling beyond a per-provider semaphore
- Real provider adapters — `HTTPChatProvider` is a skeleton, `_post` is unimplemented
- `expect_extraction_consistent_across_samples` (needs generator access, which
  observe-only forbids by design — would require an opt-in re-generation hook)

See [DESIGN.md](DESIGN.md) for the full architecture and the nine-milestone
implementation plan.
