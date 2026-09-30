# examples/jtbd

A worked project, and the fixture every milestone's acceptance test reads.

Thirteen support sessions, three fields, and eight deliberately planted
defects. `expected.yml` says which item carries which, which check has to catch
it, and when that check exists. The acceptance test is that the tool flags
**exactly** those — a check that also flags `s-01` is as broken as one that
misses `s-10`.

```
run.yml        ties it together
schema.yml     three fields: an assigned label, a free-text summary, an outcome
taxonomy.yml   jtbd@v4 — seven leaves under three branches
outcomes.yml   outcomes@v1 — a small vocabulary, so it is assigned, not free text
judges.yml     three judges, wired to two jobs
settings.yml   the middle layer of the merge; deletable
items.jsonl    the sessions
outputs.jsonl  what the model produced
labels.jsonl   human answers, including four second opinions from ann-2
expected.yml   what is planted, and what must catch it
```

Thirteen items is far below every sample floor in the library, and that is
deliberate. A fixture this size proves that checks fire on the right rows; it
proves nothing about whether a metric means anything, and the guardrails should
say so rather than printing a macro F1 off thirteen rows. If this example ever
reports a confident ranking number, that is a bug in the guardrails.

## What M1 finds here

```bash
llm-expectations run examples/jtbd/run.yml
```

Three of the eight plants — `s-07` sibling confusion, `s-08` too shallow,
`s-09` invented label — come out at the top of the queue, because a judge shown
the definitions catches all three. The other five need checks that do not exist
yet, and `expected.yml` says which milestone each waits for. M1 does not
pretend to have looked: every unchecked field gets an unscored finding naming
what is missing, and the report's closing box lists it.

The pair `billing.payment_failed` ↔ `billing.card_declined` is the intended
fuzzy boundary. `ann-2` disagrees with `ann-1` on exactly that pair in `s-07`
and `s-10`, which is the strongest evidence the fuzzy-pair detector takes: two
people who cannot separate two labels is not a model problem.
