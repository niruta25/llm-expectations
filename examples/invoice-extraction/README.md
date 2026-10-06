# examples/invoice-extraction — a **copied** project

> Supplier invoices arrive as text. A model pulls out the invoice number, the
> date and the amount due, and the accounts payable system pays them. Before
> that runs without a person in the loop, somebody has to answer: **how often
> is the amount wrong, and would we know?**

This is the worked project for the **copied** kind of field — a value that is
supposed to be *in* the document. It is the only kind where a defect can be
**proved** for nothing, and the only kind where the opposite is impossible.

```
300 supplier invoices   3 copied fields   220 read by a person
```

```
run.yml        ties it together — note there is no taxonomy line
schema.yml     three copied fields, three different value types
judges.yml     one judge, one job, and why there is no panel
settings.yml   the audit rate
items.jsonl    the invoices, in four different layouts
outputs.jsonl  what the extractor pulled out
labels.jsonl   what a person read off the document — real values, not ratings
generate.py    rebuilds all three, deterministically
```

## Absence is proof. Presence is only evidence.

That sentence is the whole of this field kind.

If the document does not contain the amount, the model invented it. That is a
**proof**, it costs nothing, and no other kind of field gets one.

What the same check cannot establish is that a value it *found* is the right
one. Every invoice here carries a subtotal, a carriage charge, a VAT line and
a total, so there are four numbers any of which the model might have reported.
`inv-003` is in the corpus, exactly as printed:

```
   DOCUMENT   Marsh & Finch Legal — statement
              Ref TX-78931, raised 5 January 2026

              consultancy, 12 hours
                net        $6,110.93
                carriage   $89.96
                vat        $1,240.18
                ---------------------
                TOTAL      $7,441.07

   EXTRACTED  total: 89.96

   value is in the document   ✓ present
   the truth                  ✗ that is the carriage charge
```

So the report prints both numbers, and the gap between them is the finding:

```
    value is in the document       96.0%   ✗  12 of 300 failed
    grounding rate                  96.0%   ✓
    ...
    matches the human answer       89.5%   ✗  23 of 220 failed
```

**The grounding check passes and the field is still wrong one time in ten.**
Four per cent of totals are invented, and the free check catches every one of
them for nothing. Eight per cent are the carriage charge or the net — present
in the document, found by the check, and wrong. Roughly **two thirds of the
wrong totals pass every check that costs nothing**, and a tool that stopped
at "is it in the document" would have reported this field at 96% and been
believed.

## The gate, and how much it actually saves here

A copied field gates the expensive judge on free checks, like free text does.
But it also sends rows that *passed*, when finding the value there was luck:

```
  WHAT THE JUDGE WAS PAID FOR
    invoice_date
      12 failed a free check, 288 too many candidates, 0 audited
    invoice_number
      27 failed a free check, 27 audited, 246 not judged
    total
      12 failed a free check, 288 too many candidates, 0 audited
```

Read those three lines together, because they say something no library can
tell you in advance.

**On `total` and `invoice_date`, the gate saves nothing.** Every invoice holds
four amounts and two dates, so every single row is ambiguous and every single
row goes to the judge. That is not a failure of the gate — it is a property of
these documents, and the report states it rather than letting anyone assume
the cheerful case.

**On `invoice_number`, the gate saves 82%.** An invoice number is text with no
other candidate in the document, so finding it there is strong evidence, and
54 of 300 rows get judged instead of all of them.

If your receipts carry one amount each, `total` behaves like `invoice_number`
and the gate pays for itself. If they look like these invoices, it does not.
`ambiguous_above` is the dial, and the honest default is `1`.

## Matching is by meaning, not by characters

Three value types, three rules, all declared in `schema.yml`:

```yaml
invoice_number:  {value_type: text,   require_verbatim: true}
invoice_date:    {value_type: date}
total:           {value_type: number, ambiguous_above: 1}
```

`$1,234.50`, `1234.5` and `1,234.50` are **one amount**. `March 5, 2026` and
`2026-03-05` are **one day**. A checker comparing characters would report the
model as inventing a value every time it tidied a format, and the grounding
rate would be measuring formatting rather than fidelity.

`invoice_number` is the exception, and it is deliberate: `INV-10594` and
`INV10594` are the same invoice to a person and two different strings to the
ledger that has to match them against a payment. `require_verbatim: true` is
the field saying the characters really do matter — and the report duly flags
seventeen numbers that were correct apart from a dropped hyphen.

A date the library cannot read **safely** is reported unparseable rather than
guessed at:

```
    value has the right shape      99.0%   ✗  3 of 300 failed
      inv-034  '04/11/2026' is not a date
      inv-141  '17/07/2026' is not a date
```

`04/11/2026` is the 4th of November or the 11th of April depending on who
typed it, and a parser that picked one would be wrong about half of them
silently. Those three rows are **unscored**, not failed, in every downstream
number — `value_in_source` steps aside and says `value_shape` owns the row, and
so does the comparison against the human answer. One defect, charged once.

## The real-life workflow

### 1. Plan

```bash
llm-expectations plan examples/invoice-extraction/run.yml
```

```
PLAN
  triage :      0 calls
  gated  :    654 calls
           ------------------------
  total  :    654 calls, est. $1.15
```

`triage: 0` is correct and worth understanding. The triage judge ranks
**assigned labels**, and this project has none — so every call on the bill is
the gated groundedness judge, and the gate decided which 654 of the 900
possible rows were worth asking about.

### 2. Run

```bash
llm-expectations run examples/invoice-extraction/run.yml
```

### 3. Read the free checks before the judge

They are free, they are real, and on this kind of field they are most of the
value:

```
  ── invoice_number ──────────────────────────────────────────── copied ──
    value is in the document       91.0%   ✗  27 of 300 failed
      inv-002  'TX-32517' is not in the document
      inv-039  'SI48957' is not in the document
```

`inv-002` is a transcription slip — the document says `TX-32516`. `inv-039` is
the same number with the hyphen removed, which is only a defect because this
field asked for verbatim. Both are proved wrong without a single model call.

### 4. Have a person read 200 invoices

For a copied field, a human label is a **gold value** — the amount they read
off the document — not a defect rating:

```json
{"item_id": "inv-002", "field": "total", "label": "7329.58", "annotator": "ann-1"}
```

That is what unlocks `matches the human answer`, which is the only number in
the report that can see the carriage-charge mistake. It is also why there is
**no panel** in `judges.yml`: a panel measures quality over a sample and finds
fuzzy label boundaries, and a copied field has no label space to be fuzzy
about. "Which amount does this document support?" has one answer, and three
models agreeing on it tells you less than one person reading the invoice.

### 5. Watch what the gate costs you

```
  CALIBRATION
    invoice_number         not fitted
      40 labelled rows, below the 100 a two-parameter fit needs. Nothing
      was fitted; the ranking stays uncalibrated and says so.
```

This is the gate's hidden bill. Because `invoice_number` only sent 54 rows to
the judge, only 40 of them are labelled, and that is not enough to fit a
calibration curve. The saving on judge calls was real; so is the cost. Both
are printed, and neither is assumed.

### 6. Compare the next extractor against this one

```bash
llm-expectations compare out/invoices-v2 out/invoices-v3
```

## Run it yourself, with no API key

```bash
llm-expectations run examples/invoice-extraction/run-offline.yml
```

Everything above that does not mention the judge is reproducible from that one
command — the grounding rates, the unparseable dates, the verbatim failures,
the gate breakdown, and the gap between "in the document" and "the right one".
Gate 1 stops the run and says the judge numbers are invented:

```
  ┌ GATES ─────────────────────────────────────────────────────────────┐
  │  ✗ measurement is sound       STOP
  │  ✗ judge beats baselines      STOP
  └─────────────────────────────────────────────────────────────────────┘
    ✗ judge-a is a scripted judge — they answer from a hash of the prompt
      and no model was asked. The free checks above are real; every judge-
      backed number in this run is invented and must not be read as a
      measurement.
```

## A note on the corpus

The invoices are synthetic and `generate.py` is right there, with the defect
rates in plain sight. Four layouts, because an extractor that only works on
one layout is an extractor that has memorised a supplier. The numbers quoted
in this README are measured from the committed data, not from the generator's
nominal rates — those are what the dice actually came up with.

## The other two kinds

| | |
|---|---|
| [`examples/ticket-routing`](../ticket-routing) | **assigned** — a label from a versioned taxonomy |
| [`examples/release-notes`](../release-notes) | **free_text** — a sentence written *about* the item |
| [`examples/jtbd`](../jtbd) | all three, thirteen items, sized to prove the refusals work |
