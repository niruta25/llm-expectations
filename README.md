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
against. This tool covers all three kinds of field — **assigned** labels,
**free text**, and values **copied** out of the document — without pretending
to certainty it has not earned.

## Status

**Alpha — every milestone in the design is built, and all three kinds of
field with it.** Both gates, a panel, human answers grading the model and its
judges, a fitted calibration, the operating-point table a review budget
needs, and runs set against each other.

### Try it without an API key

```bash
pip install -e .
llm-expectations run examples/ticket-routing/run-offline.yml
```

`provider: fake` answers from a hash of the prompt — no network, no key, no
cost. Everything upstream of the judge is real on real data, and Gate 1 stops
the run to say the rest is not.

### The commands

```bash
llm-expectations check examples/ticket-routing/queues.yml  # static health, no data
llm-expectations plan  examples/ticket-routing/run.yml     # what it will cost, no calls
llm-expectations run   examples/ticket-routing/run.yml     # collect and analyse
llm-expectations analyse out/<run>/                        # re-analyse from cache, free
llm-expectations triage-eval out/<run>/                    # the table alone, free
llm-expectations compare out/a out/b                       # two runs, head to head
```

### What it refuses to do

Most of the work here is in the absences.

- A guardrail that fires **withholds the numbers it invalidates** and names
  them, rather than printing them with a caveat nobody reads.
- **Unscored never becomes a pass.** An unreadable judge reply is counted,
  never defaulted — a coin-flip default is uncorrelated by construction and
  would quietly move every agreement number in the run.
- **No ranking is called validated** without labels and a target big enough to
  rank on. A metric under its sample floor is printed with its interval and a
  line saying it cannot support a conclusion: a wide interval is not a weak
  result, it is no result.
- **`compare` will not tell you which run is better.** It shows improved,
  regressed and the usually much larger unchanged, and withholds the verdict —
  most items tie in a real A/B, and a net delta without a test is how
  underpowered changes get shipped.
- **A calibration refuses to outlive its prompt version.** Two runs on
  different taxonomy versions refuse to compare without a migration mapping.
- **A judge never sees a human label.** The context a triage strategy receives
  has no field for one, and an assertion confirms it at runtime.

Calibration quality is measured **out of fold**. Intervals resample whole
items, never `(item, field)` pairs. Every number carries what it has to beat:
accuracy next to the majority-label baseline, a ranking next to random and
output length, a judge's accuracy next to approving everything.

## Worked projects, one per kind of field

Each ships its data, its config, and the deterministic script that built the
corpus. All of them run offline with no API key.

| | kind | the question it answers |
|---|---|---|
| [**ticket-routing**][routing] | `assigned` | 320 support tickets into nine queues. Can we auto-route, and which tickets still need a person? |
| [**release-notes**][notes] | `free_text` | 260 merged pull requests into one customer-facing line each. Did any of them invent a feature? |
| [**invoice-extraction**][invoices] | `copied` | 300 supplier invoices. How often is the amount wrong, and would we know? |

Three findings they are built to show, one each:

- **The taxonomy, not the model.** Two annotators disagree with each other on
  one pair of queues 86% of the time. No prompt fixes that.
- **Free text is cheaper than assigned.** Its free checks catch *content*
  problems, so they can decide what gets paid for — 70% off the judge bill,
  with an audit sample measuring what the gate misses.
- **Present is not correct.** 96% of invoice totals are in the document and
  89.5% are the right one. Two thirds of the wrong ones pass every free
  check, because the carriage charge is in the document too.

[`examples/jtbd/`][example] is the fixture rather than a project: thirteen
items, eight planted defects, and a manifest naming the check that must catch
each. It correctly *fails* Gate 2 and fits no calibration, which is what a
fixture that size should do. [All four][examples].

[examples]: https://github.com/niruta25/llm-expectations/tree/main/examples
[routing]: https://github.com/niruta25/llm-expectations/tree/main/examples/ticket-routing
[notes]: https://github.com/niruta25/llm-expectations/tree/main/examples/release-notes
[invoices]: https://github.com/niruta25/llm-expectations/tree/main/examples/invoice-extraction
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
| M7 | across runs | done |
| M8 | polish, docs, worked example | **done** |

What is deliberately *not* built is in [DESIGN.md §14][design] — derived
thresholds, the A/B significance test, multi-label, reasoning judges — each
with a reason to wait and a seam already in place.

## License

Apache-2.0
