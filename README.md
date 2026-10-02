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

**Pre-alpha, and covering both kinds of field.** Both gates, human answers
grading the model and its judges, a fitted calibration, the operating-point
table a review budget needs, and free text.

```bash
llm-expectations check examples/jtbd/taxonomy.yml   # static health, no data
llm-expectations plan  examples/jtbd/run.yml        # what it will cost, no calls
llm-expectations run   examples/jtbd/run.yml        # collect and analyse
llm-expectations analyse out/<run>/                 # re-analyse from cache, free
llm-expectations triage-eval out/<run>/             # the table alone, free
```

```
  ── summary ────────────────────────────────── free_text · descriptive ──

  free checks
    length in bounds              100.0%   ✓
    specific, not filler           92.3%   ✗  1 of 13 failed
      s-11     nothing here is specific to this item
    copy ratio                     92.3%   ✗  1 of 13 failed
      s-13     28 of 28 words are one lifted run
    boilerplate                    0 of 13   ✓  near-identical
    agrees with other fields       81.8%   ✗  2 of 11 failed

  judge
    claims the item supports       92.3%   ✗  1 of 13 failed
      s-10     1 invented: 'we issued a refund of $49'

  FREE-TEXT GATE
    summary                3 flagged by a free check, 10 audited
    of the flagged rows       33% had an unsupported claim
    1 of 10 audited rows that no free check flagged turned out to have an
    unsupported claim. That is what the free gate is missing, measured
    rather than assumed.
```

Free text is **cheaper** than assigned, which is the opposite of what you
would guess. The assigned free checks catch format problems, so a
well-formed wrong label sails through and the judge has to see everything.
The free-text checks catch *content* problems, so they genuinely gate the
expensive call — and the audit sample measures what the gate misses instead
of assuming it misses nothing.

Every number carries what it has to beat, and nothing claims more than it
earned. Calibration quality is measured out of fold. A calibration fitted
against one prompt version refuses to be reused against another. A wide
interval is reported as no result rather than hidden.

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
| M6 | free text | **done** — shippable |
| M7 | across runs | |
| M8 | polish | |

## License

Apache-2.0
