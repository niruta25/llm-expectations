# JTBD session classification

Classifying support sessions by the Job To Be Done the user arrived with, and
measuring whether that classification can be trusted.

```bash
python demo_jtbd.py          # from the repository root, offline, no API key
```

## The point of the example

A classification task is the one case where a gold set answers "is this right?"
exactly. That makes it the right place to be strict about what a judge adds,
because you can measure the judge against the same humans it claims to stand in
for.

The demo runs the tiers in the order you would actually do the work:

| Tier | Check | Cost | Answers |
|---|---|---|---|
| 1 | `expect_field_matches_gold` | free | Is this label right? (where a human said) |
| 2 | `expect_label_distribution_stable` | free | Has the classifier collapsed onto one label? |
| 3 | `expect_extraction_label_trustworthy` | $$ | Is this label right? (where nobody said) |
| 4 | `expect_judge_agrees_with_gold` | free | Is the judge in tier 3 worth listening to? |

By the time tier 3 runs you already know the accuracy, the macro F1, the
confusion matrix and the label distribution, and none of it cost anything. The
judge exists for the sessions nobody labelled — which in production is all of
them.

## Swapping in your taxonomy

Everything reads the taxonomy from `taxonomy.py`: the synthetic corpus, the
judge's system prompt, the drift baseline. Replace `LABELS`, `JOB_STATEMENTS`
and `RUBRIC` there and nothing else needs to change.

The shipped taxonomy is a placeholder built to exercise the pipeline. It is not
a finding about your product.

## Swapping in your data

```python
from data import load_sessions

batch = load_sessions()                       # bundled synthetic corpus
batch = load_sessions("/private/sessions.jsonl")   # your own
batch = load_sessions(with_gold=False)        # simulate production
```

One seam, deliberately. The suite, the calibration and the expectations do not
know which they were handed.

**Real session logs do not belong in this repository.** Keep them outside the
tree or under a gitignored path and pass the path in. The loader expects one
JSON object per line with `doc_id`, `text`, `gold_jtbd`, `pred_jtbd`,
`gold_outcome` and `pred_outcome`.

## Two fields per session, on purpose

Each session carries `jtbd_label` and `outcome`. With a single field per
document, field grain and document grain are arithmetically identical and the
gap this framework exists to surface cannot appear. Two fields is the minimum
that makes document accuracy mean something.

## The A/B arm

The corpus carries a second variant (`pred_jtbd_v5`) that fixes three of v4's
six misclassifications and introduces one new one:

```python
batch_v4 = load_sessions()                  # 18/24 correct
batch_v5 = load_sessions(variant="v5")      # 20/24 correct
```

v5 is genuinely better, and the comparison says **p = 0.625** — nowhere near
enough evidence to act on. Three net wins across 24 sessions is a coin flip.
That is the most useful thing in the example: the free win rate reads 75% of
decided documents, and shipping on it would be a mistake.

## What the demo will not tell you

`MockLabelJudge` scores on how much of a label's vocabulary appears in the
transcript. That is a real signal and a weak one, chosen so the calibration
lands somewhere honest instead of at a flattering 1.00 — but every number the
demo prints is a property of that rule, not of any model.

The most interesting thing it prints is a failure: at a target precision of
0.90 **no threshold qualifies**, because every cut that catches a real error on
this corpus also catches a correct label. The framework declines to produce a
number rather than producing an undefendable one, and the demo refits at 0.80
and states the recall that costs. Expect the same shape of result with a real
judge, at different values.
