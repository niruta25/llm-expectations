# llm-expectations

**Quality checks for LLM outputs that are judgements *about* a document, not
values copied *out of* one.**

```bash
pip install llm-expectations     # not published yet
```

```python
import llm_expectations
```

A model reads a document and produces structured output. Some of it is copied
from the document — an amount, a date, a name — and you can check those by
searching the text. But most interesting LLM output is *about* the document:

```
   jtbd      billing.payment_failed        a label chosen from a taxonomy
   summary   "Maya's card was declined."    a sentence written about the item
   outcome   resolved                       a judgement on what happened
```

None of that appears in the source text, so there is nothing to reconcile
against. This tool covers those two kinds of field — **assigned** labels and
**free text** — without pretending to certainty it has not earned.

## Status

**Pre-alpha.** `0.0.0` reserves the name and has no functionality — it installs
with zero dependencies and prints its own status. The design is settled and
written up; the build has not started.

## Reading order

| | For | |
|---|---|---|
| [`docs/pitch.html`][pitch] | anyone | Why this exists, no jargon |
| [`docs/overview.html`][overview] | engineers | Architecture at a glance |
| [**DESIGN.md**][design] | implementers | Every decision, threshold and default |

[pitch]: https://github.com/niruta25/llm-expectations/blob/main/docs/pitch.html
[overview]: https://github.com/niruta25/llm-expectations/blob/main/docs/overview.html
[design]: https://github.com/niruta25/llm-expectations/blob/main/DESIGN.md

## What it does, in short

- **Two gates first.** Can the measurement be trusted? Is the judge better than
  guessing? Nothing else is reported until both pass.
- **One judge ranks, a panel measures.** A single judge's score decides which
  items a human should open. A small panel estimates quality and finds which
  parts of your taxonomy are fuzzy.
- **A confidence is not a probability.** Raw judge confidence, calibrated error
  probability and the triage score are three separate things, never silently
  interconverted. With no labels you still get a ranking — stamped uncalibrated.
- **Error Recall@Budget is the headline.** "Review 1% and you find 18% of the
  errors" beats an AUC. Every ranking strategy, including panel disagreement, is
  scored against the trivial baselines rather than assumed to work.
- **Three answers, not two.** Pass, fail, and *"we did not check this, here is
  why"* — which never silently becomes a pass.
- **Collect once, analyse many times.** Judge calls are cached to disk; every
  metric re-runs for free.
- **Says what it cannot tell you.** Every report names the numbers it could not
  compute and what would unlock them.

## Build order

| | | |
|---|---|---|
| M0 | skeleton | |
| M1 | one judge, end to end | **first shippable** |
| M2 | free checks + full report | |
| M3 | panel | |
| M4 | guardrails + stats | shippable |
| M5 | labels (Mode 1) | |
| M6 | free text | shippable |
| M7 | across runs | |
| M8 | polish | |

## License

Apache-2.0
