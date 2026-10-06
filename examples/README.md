# examples

Four worked projects. Three of them are one per kind of field, sized like
real work; the fourth is a thirteen-item fixture sized to prove the library's
refusals fire.

Every one of them runs with **no API key and no cost**:

```bash
llm-expectations run examples/<project>/run-offline.yml
```

`provider: fake` answers from a hash of the prompt. Everything upstream of the
judge is real on real data, and Gate 1 stops the run to say the rest is not.

## One project per kind of field

| | kind | the question it answers |
|---|---|---|
| [**ticket-routing**](ticket-routing) | `assigned` | 320 support tickets into nine queues. Can we auto-route, and which tickets still need a person? |
| [**release-notes**](release-notes) | `free_text` | 260 merged pull requests into one customer-facing line each. Did any of them invent a feature? |
| [**invoice-extraction**](invoice-extraction) | `copied` | 300 supplier invoices. How often is the amount wrong, and would we know? |

Each one ships its data, its config, and the deterministic `generate.py` that
built the corpus, so nothing has to be taken on trust.

### What each one is for

**ticket-routing** is the kind with the weakest free checks: a well-formed
wrong label passes every check that costs nothing, so the judge has to see
every row. Its best finding is not about the model at all — two annotators
disagree with each other on one pair of queues 86% of the time, which is the
taxonomy's fault and no amount of prompting fixes it.

**release-notes** is the counter-intuitive one. Free text is **cheaper** to
check than an assigned label, because its free checks catch *content*
problems rather than format ones, and that makes them predictive enough to
decide what gets paid for. The gate removes 70% of the judge bill, and an
audit sample measures what it misses instead of assuming it misses nothing.

**invoice-extraction** is the only kind that can *prove* a defect for free: a
value the document does not contain was invented. It is also the kind that
most invites a false sense of security, and the two numbers it prints side by
side are why — 96% of totals are in the document, and 89.5% are the right
one.

## The fixture

[**jtbd**](jtbd) has all three kinds, thirteen items, and eight deliberately
planted defects with a manifest saying which check must catch each. It is the
acceptance test for every milestone, and it **fails Gate 2 on purpose**:
thirteen items cannot support a ranking claim, and the whole value of the
fixture is that the tool refuses to make one.

## Which to read first

Whichever matches the field you have. If you are not sure which you have:

```
   is the answer written down somewhere in the document?
     yes → copied            invoice-extraction
     no  ↓
   is it one of a fixed set of answers?
     yes → assigned          ticket-routing
     no  → free_text         release-notes
```

A field with a handful of possible values is `assigned`, not `free_text`,
even when a model writes it as a sentence. Same information, a fraction of
the cost, far better checks.
