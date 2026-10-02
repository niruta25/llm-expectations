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

**Pre-alpha, and feature-complete for assigned labels.** Both gates, human
answers grading the model and its judges, a fitted calibration, and the
operating-point table a review budget actually needs.

```bash
llm-expectations check examples/jtbd/taxonomy.yml   # static health, no data
llm-expectations plan  examples/jtbd/run.yml        # what it will cost, no calls
llm-expectations run   examples/jtbd/run.yml        # collect and analyse
llm-expectations analyse out/<run>/                 # re-analyse from cache, free
llm-expectations triage-eval out/<run>/             # the table alone, free
```

```
  ┌ GATES ─────────────────────────────────────────────────────────────┐
  │  ✓ measurement is sound       PASS                                  │
  │  ✓ judge beats baselines      PASS                                  │
  └─────────────────────────────────────────────────────────────────────┘

  CALIBRATION
    jtbd         n=1200   ECE 0.002 ✓   Brier 0.057 vs 0.145 base rate
      quality measured out of fold over 5 folds

  OPERATING POINT   Error Recall@Budget
    312 known errors in 1200 labelled items

    strategy             budget  reviewed   found   recall   wasted
    calibrated_risk       1.0%        12       8     2.6%    31.6%  ← the judge
    calibrated_risk      20.0%       240     164    52.6%    31.6%
    panel_disagreement    1.0%        12       7     2.3%    39.2%
    random                1.0%        12       3     1.0%    75.0%
```

Every number carries what it has to beat, and nothing claims more than it
earned. Calibration quality is measured **out of fold** — fitting on a
hundred rows and reporting how well the fit scores those same hundred rows is
in-sample performance wearing a lab coat. A calibration fitted against one
prompt version refuses to be reused against another. And because a Platt
curve is monotone, the report says plainly when a calibration bought you
meaningful probabilities but *not* a better ordering.

A worked project with deliberately planted defects lives in
[`examples/jtbd/`][example]; at thirteen items it correctly *fails* Gate 2 and
fits no calibration, which is what a fixture that size should do.

[example]: https://github.com/niruta25/llm-expectations/tree/main/examples/jtbd

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
| M0 | skeleton | done |
| M1 | one judge, end to end | done — first shippable |
| M2 | free checks + full report | done |
| M3 | panel | done |
| M4 | guardrails + stats | done — shippable |
| M5 | labels (Mode 1) | done |
| M5b | calibration + triage evaluation | **done** — shippable |
| M6 | free text | shippable |
| M7 | across runs | |
| M8 | polish | |

## License

Apache-2.0
