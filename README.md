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

**Pre-alpha, and every build milestone is in.** Both gates, free checks on
both kinds of field, a panel, human answers grading the model and its judges,
a fitted calibration, the operating-point table a review budget needs, and
runs set against each other.

```bash
llm-expectations check examples/jtbd/taxonomy.yml   # static health, no data
llm-expectations plan  examples/jtbd/run.yml        # what it will cost, no calls
llm-expectations run   examples/jtbd/run.yml        # collect and analyse
llm-expectations analyse out/<run>/                 # re-analyse from cache, free
llm-expectations triage-eval out/<run>/             # the table alone, free
llm-expectations compare out/a out/b                # two runs, head to head
```

```
  A  2026-10-02_0747_jtbd-p7              400 items   prompt p7
  B  2026-10-02_0747_jtbd-p8              400 items   prompt p8

  ⚠ the prompt changed, p7 → p8. That is what a comparison is for; it also
    means every difference below has more than one possible cause.

                                      A        B     change
    macro F1                       0.71     0.89      +0.18
    accuracy                      71.0%    89.2%    +18.2pp

    items that moved          400 shared
      improved                   107
      regressed                   34
      unchanged                  259

  ┌ WHAT THIS COMPARISON CANNOT TELL YOU ──────────────────────────────────┐
  │  ✗ whether jtbd actually got better or worse
  │      A net +73 is not a result without a test over the items that
  │      actually differ — most items tie in a real A/B, and quoting a win
  │      rate without one is how underpowered changes get shipped.
  └────────────────────────────────────────────────────────────────────────┘
```

**`compare` will not tell you which run is better**, and that refusal is the
point. Two runs on different taxonomy versions refuse to compare at all
without a migration mapping — lining labels up by name across a rename reads
an edit you made as a distribution the model shifted.

Every number carries what it has to beat. Calibration quality is measured out
of fold. A calibration fitted against one prompt version refuses to be reused
against another. A wide interval is reported as no result rather than hidden.

A worked project with eight deliberately planted defects lives in
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
| M5b | calibration + triage evaluation | done — shippable |
| M6 | free text | done — shippable |
| M7 | across runs | **done** |
| M8 | polish, docs, worked example | |

## License

Apache-2.0
