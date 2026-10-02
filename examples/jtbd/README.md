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
proves nothing about whether a metric means anything.

So **this example fails Gate 2, on purpose**. Five errors in thirteen labelled
items is a degenerate target, the gate refuses to compute an AUC on it, and
the report lists what it withheld:

```
  ┌ GATES ─────────────────────────────────────────────────────────────┐
  │  ✓ measurement is sound       PASS                                  │
  │  ✗ judge beats baselines      STOP                                  │
  └─────────────────────────────────────────────────────────────────────┘
    ✗ only 5 errors in 13 labelled items (38.5%) — the floor is 30 and 5%.
```

If this example ever reports a confident ranking number, that is a bug in the
guardrails. Tests that need a corpus Gate 2 can actually run on build one —
see the `big_corpus` fixture in `tests/conftest.py`.

## What lands here today

```bash
llm-expectations run examples/jtbd/run.yml
```

The free checks catch three plants on their own — `s-08` stopped at a parent,
`s-09` is not a label at all, and `s-12`'s summary shares no vocabulary with
its label. Cross-field agreement also fires on `s-11`, whose summary is
generic enough to fit any session; `expected.yml` records that under
`also_caught_by`, because it is a real finding and not a false positive.

The triage judge then ranks `s-09`, `s-07` and `s-08` first. `s-07` is the one
nothing free can reach: `billing.payment_failed` and `billing.card_declined`
are both real leaves in the same branch, so only a judge or a human label can
separate them.

Four plants remain unreachable, and the tool says so rather than implying it
looked. Every unchecked field carries an unscored finding naming what is
missing, and the report's closing box lists it.

The pair `billing.payment_failed` ↔ `billing.card_declined` is the intended
fuzzy boundary, and all three sources of evidence for it are in this fixture:

- **Panel disagreement**, free and available today. When judges split on
  `s-07` or `s-10`, the dissenter names the label it would assign instead and
  the pair becomes a boundary rather than a guess.
- **A confusion matrix**, at M5, which adds direction — symmetric means fix
  the taxonomy, one-way means fix the prompt.
- **Two annotators disagreeing**, the strongest evidence of the three.
  `ann-2` splits from `ann-1` on exactly that pair in `s-07` and `s-10`. Two
  people who cannot separate two labels is not a model problem.
