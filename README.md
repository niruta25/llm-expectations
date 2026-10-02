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

**Pre-alpha.** Both gates work, and human answers now grade the model *and*
the judges scoring it.

```bash
llm-expectations check examples/jtbd/taxonomy.yml   # static health, no data
llm-expectations plan  examples/jtbd/run.yml        # what it will cost, no calls
llm-expectations run   examples/jtbd/run.yml        # collect and analyse
llm-expectations analyse out/<run>/                 # re-analyse from cache, free
```

```
  ┌ GATES ─────────────────────────────────────────────────────────────┐
  │  ✓ measurement is sound       PASS                                  │
  │  ✓ judge beats baselines      PASS                                  │
  └─────────────────────────────────────────────────────────────────────┘

    strategy                 AUC  95% interval           n
    raw_confidence          0.99  [0.99, 1.00]         400  ← the judge
    random                  0.49  [0.41, 0.56]         400  ← baseline
    output_length           0.50  [0.50, 0.50]         400  ← baseline

  vs humans   400 labelled items
    macro F1                      0.91   the headline
    accuracy                     89.2%   ✓   majority-label baseline 18.2%
    exact / right parent / too shallow / wrong
    89% / 4% / 4% / 2%
    confusable pairs
      billing.card_declined → billing.payment_failed
        5 this way, 5 back — symmetric
        neither direction dominates — fix the taxonomy
    two annotators             95.0% agree over 120 double-labelled items
      billing.card_declined ↔ billing.payment_failed
        6 of 6 disagreements (100%)
      your annotators cannot separate these two either.
```

Every number carries what it has to beat. Accuracy never appears without the
majority-label baseline; a ranking never appears without random and output
length; a judge's accuracy never appears without "approve everything". Four
tree buckets say *which kind* of wrong, because a sibling, a hedge and a
misread need three different fixes. Confusion that runs both ways is a
taxonomy bug; confusion that runs one way is a prompt bug.

A worked project with deliberately planted defects lives in
[`examples/jtbd/`][example]; at thirteen items it correctly *fails* Gate 2,
which is what a fixture that size should do.

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
| M5 | labels (Mode 1) | **done** |
| M5b | calibration + triage evaluation | shippable |
| M6 | free text | shippable |
| M7 | across runs | |
| M8 | polish | |

## License

Apache-2.0
