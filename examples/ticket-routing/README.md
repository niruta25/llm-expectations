# examples/ticket-routing — an **assigned** project

> A support desk wants to stop routing tickets by hand. A model reads each
> ticket and picks a queue and a severity. Before anyone turns that on, two
> questions have to be answered: **is it good enough**, and **which tickets
> still need a person**?

This is the worked project for the **assigned** kind of field — an answer
chosen from a versioned list of labels. It is the kind with the weakest free
checks and the strongest need for a judge, because a well-formed wrong label
passes everything that costs nothing.

```
320 tickets   2 assigned fields   280 labelled by a person   70 twice
```

```
run.yml        ties it together
schema.yml     two assigned fields, and the third layer of the settings merge
queues.yml     queues@v2 — nine queues under four teams
severity.yml   severity@v1 — three values, flat on purpose
judges.yml     three judges, two jobs
settings.yml   the project layer: one threshold that is right for one field
items.jsonl    the tickets
outputs.jsonl  what the router answered
labels.jsonl   what people answered, including 70 second opinions
generate.py    rebuilds all three, deterministically
```

The tickets are synthetic and `generate.py` is right there, so nobody has to
take the numbers on trust. What is not synthetic is the **shape** of the
errors: siblings confused, a label hedged up to its parent, a queue invented
outright, in roughly the proportions a real router gets them wrong. The
generator plants them and tells the tool nothing.

## The real-life workflow

### 1. Check the taxonomy before you have any data

```bash
llm-expectations check examples/ticket-routing/queues.yml
```

```
queues@v2   13 labels, 9 leaves
content hash e1b4008e758c

no issues.
```

This costs nothing and needs no outputs. It is looking for the taxonomy
problems that turn into model problems later — two definitions so alike they
are one label, a label with no examples, a branch nothing can fall into. The
content hash is what makes a later run refuse to compare against this one if
the file moves without its version.

### 2. Find out what it will cost

```bash
llm-expectations plan examples/ticket-routing/run.yml
```

```
PLAN
  triage        :    631 calls
  panel:judge-b :    297 calls
  panel:judge-c :    297 calls
                  ------------------------
  total         :  1,225 calls, est. $2.18
  skipped 9: abstained — no label to check
```

Every job that spends money is on its own line. The panel is two thirds of
the bill here, which is the thing a combined total would hide — and the thing
you would want to know before deciding whether three judges are worth it.

No calls are made. If any judge's model is missing from the price table, the
total reads *cost not estimable* rather than guessing.

### 3. Run it

```bash
llm-expectations run examples/ticket-routing/run.yml
```

Verdicts are cached to disk as they arrive, so every later analysis is free.

### 4. Read the gates before you read anything else

Nothing below the gates means anything until both pass. Gate 1 asks whether
the measurement is sound; Gate 2 asks whether the judge beats guessing:

```
    ranked against 89 known errors in 280 labelled items
    strategy                 AUC  95% interval           n
    calibrated_risk         0.48  [0.41, 0.55]         280  ← the judge
    random                  0.44  [0.37, 0.52]         280  ← baseline
    output_length           0.50  [0.50, 0.50]         280  ← baseline
    majority_label          0.51  [0.45, 0.57]         280  ← baseline
    panel_disagreement      0.49  [0.42, 0.57]         275  ← baseline
    ✗ calibrated_risk scores 0.48 [0.41, 0.55] and majority_label scores
      0.51 [0.45, 0.57] on the same rows. The judge's interval does not
      clear the baseline, so a judge is not buying you anything here — and
      saying that is worth more than a dashboard.
```

Those are the real numbers from `run-offline.yml`, and they are at chance
because offline *nothing was asked* — a scripted judge ranks no better than
a shuffle, and the table says so instead of flattering it. Wire real judges
in and the judge's row is the one that has to move; the four baselines stay
where they are, which is the point of printing them.

What matters is the comparison, not the number. A judge at 0.72 against a
`majority_label` baseline at 0.71 has earned nothing, and a judge whose
interval overlaps a baseline's has not cleared it.

**This corpus is sized to produce an answer**, which is the difference
between it and [`examples/jtbd`](../jtbd). 89 errors in 280 labelled tickets
clears both floors Gate 2 enforces (30 errors, and 5% of the labelled set),
so the gate computes the number instead of refusing to. The thirteen-item
example is sized to prove the refusal works; this one is sized to get past it.

### 5. Add labels, and get the three things only labels unlock

Labelling 280 of 320 tickets is more than a real project would do up front.
Start with a hundred and re-analyse — it costs nothing, because the verdicts
are already on disk:

```bash
llm-expectations analyse out/<run>/
```

With labels in place you get the model's accuracy next to the majority-label
baseline, the judge graded against the same answers, and a fitted
calibration — which is what turns a judge's confidence into a probability
you can set a review threshold against.

### 6. Change the prompt, and compare

```bash
llm-expectations compare out/routing-v2 out/routing-v3
```

`compare` will show improved, regressed, and the usually much larger
unchanged — and **refuses to say which run is better**. Most items tie in a
real A/B, and a net delta without a test is how underpowered changes ship.

## What this project is for

### Finding the boundary your taxonomy cannot hold

The most valuable line in the report is not about the model:

```
    two annotators             68.6% agree over 70 double-labelled items
      billing.invoicing ↔ billing.payment_failure
        19 of 22 disagreements (86%)
      your annotators cannot separate these two either. That is not a
      model problem — merge them or rewrite both definitions.
```

Nearly nine in ten disagreements between two people land on one pair of
queues. No amount of prompt engineering fixes that. `queues@v2` draws the
line at *was a charge attempted*, and the tickets that break it are the ones
where a customer says "the money is wrong" without saying which.

Three independent sources of evidence point at that pair, and the report
keeps them apart rather than averaging them:

- **the panel**, when judges split and the dissenter names the queue it
  would assign instead;
- **the confusion matrix**, which adds direction — symmetric means fix the
  taxonomy, one-way means fix the prompt;
- **two annotators**, the strongest of the three, above.

### Knowing which numbers you have not earned

Nine queues over 280 labelled tickets is about thirty each, which is exactly
the per-label floor. Two queues fall under it and the report says so rather
than printing a per-queue F1 off a dozen rows:

```
    macro F1                      0.83   the headline
      built from 9 per-label scores, 2 of which sit below the per-label
      floor. The average cannot support a conclusion that its parts
      cannot.
```

The fix is more labels on those two queues, not a lower floor. If you want
per-queue numbers for all nine, budget for thirty labelled tickets per queue
before you start.

### Where a threshold comes from

`settings.yml` sets `max_label_share: 0.25` — reasonable for nine queues, and
wrong for a three-value severity, where a third of the board in one value is
normal. The severity block in `schema.yml` puts it back, and the report shows
which layer won on each line:

```
    largest label share             11.3%   ✓  max 25% (settings.yml)
    largest label share             36.2%   ✓  max 50% (schema.yml:severity)
```

Field block beats project file beats built-in default, and every number is
printed next to the threshold it was judged against and where that threshold
came from.

### The two fields are deliberately different shapes

`queues@v2` is a tree, so a wrong answer can be *partly* right: a sibling
means two definitions need sharpening, a parent means the model hedged, a
different branch means it is not reading the ticket. Those are three
different fixes and the report separates them.

`severity@v1` is flat. With no parents there is no branch to have partly got
right, so every miss lands in `wrong`. A shared *absence* of a parent is not
partial credit, and scoring it as one would send you off to sharpen
definitions on a vocabulary that has no hierarchy to confuse.

## Run it yourself, with no API key

```bash
llm-expectations run examples/ticket-routing/run-offline.yml
```

`run-offline.yml` wires both judge jobs to `provider: fake`, which answers
from a hash of the prompt. No network, no key, no cost. Everything upstream
of the judge is real on real data — the free checks, the taxonomy health
checks, both grains, the confusion matrix, the annotator agreement — and
Gate 1 stops the run to say the rest is not:

```
  ┌ GATES ─────────────────────────────────────────────────────────────┐
  │  ✗ measurement is sound       STOP
  │  ✗ judge beats baselines      STOP
  └─────────────────────────────────────────────────────────────────────┘
    ✗ judge-a, judge-b, judge-c are scripted judges — they answer from a
      hash of the prompt and no model was asked. The free checks above are
      real; every judge-backed number in this run is invented and must not
      be read as a measurement.
```

Everything in this README that does not depend on a judge — the annotator
disagreement, the per-label floor, the settings merge, the confusion matrix,
the free checks — is reproducible from that one command.

Use `run.yml` and your own keys when you want the judge-backed numbers to
mean something.

## The other two kinds

| | |
|---|---|
| [`examples/release-notes`](../release-notes) | **free_text** — a sentence written *about* the item |
| [`examples/invoice-extraction`](../invoice-extraction) | **copied** — a value that is *in* the document |
| [`examples/jtbd`](../jtbd) | all three, thirteen items, sized to prove the refusals work |
