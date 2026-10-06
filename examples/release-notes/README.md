# examples/release-notes — a free_text project

Every merged pull request needs a one-line note a customer can read. A model
writes them from the commit and the diff. The release goes out on Thursday,
and nobody has time to read 260 lines — but a note that invents a feature, a
performance claim or a security fix is a line that ends up in a customer's
inbox.

This is the worked project for the **free_text** kind of field — a sentence
written *about* the item. There is no single right release note, so there is
no answer key. The question is not *is it right?* but **is anything wrong
with it?**, and the surprise is how much of that is free.

```
260 merged pull requests   1 free-text field   1 assigned field   210 rated
```

```
run.yml        ties it together
schema.yml     the note, and the area it has to agree with
areas.yml      areas@v1 — six areas, flat
judges.yml     three judges, two jobs, one of them gated
settings.yml   the audit rate, raised, and why
items.jsonl    the pull requests
outputs.jsonl  the notes the model wrote
labels.jsonl   defect ratings from a person — not better notes
generate.py    rebuilds all three, deterministically
```

The pull requests are synthetic and `generate.py` is right there. What it
plants is the five ways a written line goes wrong, in roughly the mix a real
note writer produces them.

## The one thing to understand about free text

**Free text is cheaper to check than an assigned label**, which is the
opposite of what you would guess.

For an assigned label, every free check is about *format* — is this a real
label, is it a leaf, did it abstain. A perfectly formed wrong label passes
all of them, so the judge has to see every single row.

For a sentence, the free checks are about *content*:

| check | catches | costs |
|---|---|---|
| `length_in_bounds` | a paragraph where a line was asked for | nothing |
| `specificity` | a line that would fit any release | nothing |
| `copy_ratio` | the commit message, pasted | nothing |
| `cross_field_agreement` | a note about something other than its area | nothing |
| `boilerplate` | the model falling into a template | nothing |
| `claims_supported` | **a claim the pull request does not support** | a judge call |

Four of the five defects are free. The fifth is the one that reaches a
customer — and because the free checks genuinely predict it, they are
allowed to decide which rows the judge sees.

## How much that saves, measured rather than claimed

```
  WHAT THE JUDGE WAS PAID FOR
    note
      45 failed a free check, 32 audited, 183 not judged
    of the flagged rows        36% were wrong
    5 of 32 audited rows that no free check flagged turned out to be wrong
    anyway. That is what the gate is missing, measured rather than
    assumed.
```

77 judge calls instead of 260: the gate removes **70% of the bill**. How much
it saves on *your* corpus is a property of your data, not of this library, so
the breakdown is printed rather than assumed.

The second half of that block is the part that makes the first half worth
believing. **The audit sample is the gate's own exam.** 32 rows that no free
check flagged were sent to the judge anyway, purely to find out what the gate
is missing — and five of them were wrong. Without that line, "the free checks
are predictive" would be a claim this library makes about itself.

`settings.yml` raises the audit rate from the default 5% to 15% for exactly
that reason, and says so:

```yaml
free_text_audit_rate: 0.15
```

The defect this project most fears is the one the free checks cannot see. The
audit is the only thing standing between *"the gate found nothing"* and
*"there is nothing there"*, and 26 extra judge calls is a cheap price for
knowing which of those you are looking at.

## The real-life workflow

### 1. Plan before you spend

```bash
llm-expectations plan examples/release-notes/run.yml
```

```
PLAN
  triage        :    260 calls
  panel:judge-b :    130 calls
  panel:judge-c :    130 calls
  gated         :     77 calls
                  ------------------------
  total         :    597 calls, est. $1.29
```

The free checks run during planning too — they have to, because they decide
how big the `gated` line is. No calls are made.

### 2. Run, and read the free checks first

```bash
llm-expectations run examples/release-notes/run.yml
```

```
  free checks
    agrees with other fields       90.4%   ✗  25 of 260 failed
    length in bounds               94.2%   ✗  15 of 260 failed
    specific, not filler           93.1%   ✗  18 of 260 failed
    copy ratio                     95.0%   ✗  13 of 260 failed
    boilerplate                  28 of 260   ✗  near-identical
```

Every one of those is real, on real data, with no API key and no cost. The
`boilerplate` line is the corpus-grain one and it is worth its own paragraph:
28 notes are near-identical to another note. That is the model falling back
on a template, and each of those 28 passes length, specificity *and* copy
ratio individually. Only a check that compares outputs to **each other** sees
it.

### 3. Rate some notes — but rate them, do not rewrite them

A human label for a free-text field is a **defect rating**, not a better
sentence:

```json
{"item_id": "pr-014", "field": "note", "label": "made_up",     "annotator": "ann-1"}
{"item_id": "pr-015", "field": "note", "label": "none",        "annotator": "ann-1"}
{"item_id": "pr-021", "field": "note", "label": "too_generic", "annotator": "ann-1"}
```

The boxes are `made_up`, `too_generic`, `contradicts`, `missing`, and `none`.
Asking someone to write a better note would score word choice — two notes can
be equally good and share almost no words. Asking which box to tick is a
question two people can answer the same way.

That buys the table this project exists for:

```
    defect                  n  agreement  tool only  human only
    made_up                64       0.70         18           1
    too_generic           210       1.00          0           0
    contradicts           210       0.93         14           0
    missing                64       0.95          0           3
```

`too_generic` is near-perfect, because "would this fit any release?" is a
question a rule can answer. `made_up` is the weak one — and the report says
so plainly rather than hiding it, because that is the expected result: an
invented claim is hard for a judge *and* hard for a person, and a number that
admits it is worth more than one that does not.

### 4. The arithmetic that tells you how many to rate

The first pass on this project rated 120 notes and Gate 2 refused:

```
    ✗ only 20 errors in 120 labelled items (16.7%) — the floor is 30 and 5%.
```

Not "the judge is bad" — **"you have not given me enough to tell"**. The fix
is arithmetic, not a lower floor: at a 17% defect rate, 30 errors needs about
180 rated notes, and the ranking floor wants 200 items on top of that. This
project rates 210, and the gate computes:

```
    ranked against 45 known errors in 210 labelled items
    strategy                 AUC  95% interval           n
    calibrated_risk         0.58  [0.48, 0.67]         210  ← the judge
    random                  0.54  [0.43, 0.64]         210  ← baseline
    output_length           0.48  [0.36, 0.59]         210  ← baseline
    majority_label          0.41  [0.34, 0.49]         210  ← baseline
    panel_disagreement      0.50  [0.39, 0.61]         134  ← baseline
    n = 134, below the floor of 200. This interval spans too much to
    support a conclusion.
```

Those are the offline numbers, which sit at chance because offline nothing
was asked. Note the last line: `panel_disagreement` is scored over the panel
sample only, which is 134 rows, and the report flags that one row of the
table as under the floor rather than quietly presenting five comparable
numbers.

**`output_length` is the baseline to watch on a free-text project.** If long
notes are the bad ones on your corpus, a judge that has quietly learned to
flag long notes will post a good AUC while knowing nothing at all. When that
row wins, the judge is not the thing that is working.

### 5. Compare the next prompt against this one

```bash
llm-expectations compare out/notes-v4 out/notes-v5
```

It shows improved, regressed and unchanged, and refuses to name a winner.

## Two thresholds this project moves, and why

Both are in `schema.yml`, on the field, and both are judgement calls about
this corpus rather than improvements on the library's defaults:

```yaml
max_copy_ratio: 0.6      # a note quotes component names; that is correct, not lazy
min_claim_support: 0.98  # an invented claim reaches a customer
```

At the default `0.5`, copy ratio flagged honest notes — a release note
naturally repeats "the pagination cursor" from the pull request that changed
it. Loosening it to `0.6` is a decision about release notes, and it is in the
file where anyone can see it, printed in the report beside every number it
produced.

## What the assigned field is doing here

`area` is not decoration. `note` declares `must_agree_with: [area]`, and
`cross_field_agreement` asks whether the sentence draws on the vocabulary of
the area it was tagged with. It is the cheapest useful second opinion there
is, and it fires **without knowing which of the two is wrong** — which is the
finding, not a weakness of it. Both fields are worth opening.

It is also why `areas.yml` lists component names inside the definitions. A
definition written as "things to do with importing" would give the check
nothing to match against, and it would pass everything.

## Run it yourself, with no API key

```bash
llm-expectations run examples/release-notes/run-offline.yml
```

Every free check, the gate's arithmetic, the defect-rating table, the
boilerplate sweep and both grains are real. Only the claim verdicts are
invented, and Gate 1 stops the run to say so.

## The other two kinds

| | |
|---|---|
| [`examples/ticket-routing`](../ticket-routing) | **assigned** — a label from a versioned taxonomy |
| [`examples/invoice-extraction`](../invoice-extraction) | **copied** — a value that is *in* the document |
| [`examples/jtbd`](../jtbd) | all three, thirteen items, sized to prove the refusals work |
