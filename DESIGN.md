# llm-expectations — Design

**A quality tool for LLM outputs that are judgements *about* a document, not values copied *out of* one.**

| | |
|---|---|
| Install | `pip install llm-expectations` |
| Import | `import llm_expectations` |
| Python | 3.10+ |
| Dependencies | `pyyaml`, `httpx`, `numpy` |

---

## Contents

1. [Metrics and settings](#1-metrics-and-settings)
2. [Core model](#2-core-model)
3. [Inputs and outputs](#3-inputs-and-outputs)
4. [The shared engine](#4-the-shared-engine)
5. [Assigned fields](#5-assigned-fields)
6. [Judges](#6-judges)
7. [Calibration, risk and triage](#7-calibration-risk-and-triage)
8. [Free text fields](#8-free-text-fields)
9. [Taxonomy](#9-taxonomy)
10. [Guardrails](#10-guardrails)
11. [Config and report](#11-config-and-report)
12. [Repo layout and build order](#12-repo-layout-and-build-order)
13. [v1 — what we are not building yet](#13-v1--what-we-are-not-building-yet)

---

## The problem

A model reads a document and produces structured output. Some of that output is
**copied from** the document — an amount, a date, a name. You can check those by
searching the text.

But most interesting LLM output is **about** the document:

```
   jtbd      billing.payment_failed        a label chosen from a taxonomy
   summary   "Maya's card was declined."    a sentence written about the item
   outcome   resolved                       a judgement on what happened
```

None of that appears in the source text. Searching for it tells you nothing.
Existing data-quality tools assume you can reconcile a value against something;
here there is nothing to reconcile against.

This tool covers those two kinds of field, and does it without pretending to
certainty it has not earned.

### Three kinds of field

```
   ┌─── ASSIGNED ──────────────┐  ┌─── FREE TEXT ─────────────┐
   │  a label from a taxonomy   │  │  a sentence about the item │
   │  jtbd, intent, outcome,    │  │  summary, reason,          │
   │  sentiment, priority       │  │  next action               │
   │                            │  │                            │
   │  one right answer          │  │  no single right answer    │
   │  → compare to it           │  │  → hunt for defects        │
   └────────────────────────────┘  └────────────────────────────┘

   ┌─── COPIED ────────────────┐
   │  a value in the document   │   v1 — the seam is ready,
   │  amount, date, vendor      │   not built yet
   └────────────────────────────┘
```

The kind is declared once per field. Everything else follows from it.

---

## 1. Metrics and settings

### Two gates before any quality number is reported

```
   ┌──────────────────────────────────────────────┐
   │  GATE 1   Can I trust the measurement?       │
   │           If this fails, every number below   │
   │           is meaningless. Stop and fix.       │
   └────────────────────┬─────────────────────────┘
                        │ pass
   ┌────────────────────▼─────────────────────────┐
   │  GATE 2   Is the judge better than nothing?  │
   │           If it loses to a trivial baseline,  │
   │           you are paying for noise.           │
   └────────────────────┬─────────────────────────┘
                        │ pass
   ┌────────────────────▼─────────────────────────┐
   │  QUALITY   Now the real numbers mean something│
   └──────────────────────────────────────────────┘
```

### Gate 1 — is the measurement trustworthy?

Always on, free, and each catches a failure that otherwise produces a confident
wrong number.

| Metric | Default setting | Why |
|---|---|---|
| Judge approval rate | warn outside **15–85%**, fail outside **2–98%** | Catches a rubber-stamp judge. A judge approving 99% passes every other check ever written. |
| Unparseable / empty replies | warn **> 2%**, fail **> 5%** | Above this you measured your fallback, not the model. Never default them — exclude and count. |
| Which items failed to parse | report the breakdown | Empty replies cluster on hard items. That biases everything. |
| Minority class size | need **≥ 30 items and ≥ 5%** | Below this any AUC is noise with a tight-looking interval. |
| Panel effective votes | flag when **n_eff / m < 0.5** | Three judges giving 1.1 opinions means you pay 3× for nothing. |
| Signal uses human labels? | boolean, must be **false** | A triage signal that peeks at the answer scores beautifully and is useless in production. |
| Sample size | see below | A small sample gives an interval so wide it cannot say anything. |

**Minimum sample sizes:**

| What you want | Minimum | Comfortable |
|---|---|---|
| Distribution / drift | 200 | 500 |
| Accuracy on one label | 30 per label | 100 per label |
| Any AUC or ranking claim | 200 | 500 |
| Judge comparison | 200 | 500 |

Below the minimum, the number is reported **with its interval and a line saying
it cannot support a conclusion.**

### Gate 2 — is the judge better than nothing?

Every judge is compared to the dumb options on the same target.

| Baseline | What it is |
|---|---|
| random | 0.5 |
| always-approve | a judge that never flags anything |
| majority label | always guess the most common label |
| item length | longer items = more errors? |

**Rule: the judge's number and every baseline appear side by side, always.**

```
   triage judge risk score     AUC 0.84    ← usable
   item length                 AUC 0.48
   random                      AUC 0.50
```

### Quality metrics — assigned fields

**Mode 0 — no human labels**

| Metric | Default | Reads as broken when |
|---|---|---|
| Label in taxonomy | expect 100% | anything below — the model invented a label |
| Largest label share | warn **> 50%** | the model collapsed onto one label |
| Abstention rate | warn outside **1–20%** | too high: taxonomy has gaps. too low: forcing labels. |
| Drift vs last run | warn on **> 10pp** move | something changed silently |
| Disagreement by label pair | top 5 reported | one pair carrying most disagreement = taxonomy bug |
| Stability on rerun | warn **< 90%** | the model flips its own answers |
| Triage ranking | the ordering output | **stamped uncalibrated in mode 0** — a ranking, not a validated one |

**Mode 1 — with human labels**

| Metric | Note |
|---|---|
| Macro F1 | the headline |
| Accuracy | never reported alone |
| Per-label recall + precision | flag any label below 0.5 recall |
| Tree buckets | exact / right parent / too shallow / wrong |
| Confusion matrix | the actionable output |
| Judge accuracy | must beat majority-label |
| Judge direction | approves-wrong % vs rejects-right % |
| **Error Recall@Budget** | **the primary metric** — errors found in the top K reviewed |
| Precision@Budget | of what you reviewed, how much was genuinely wrong |
| Strategy comparison | every ranker against every baseline, same budgets |
| Calibration quality | ECE, Brier, reliability — did the fit help? |
| Confidence calibration | does 0.9 mean 90%? |

**The operating point is the deliverable, not the AUC:**

```
   review budget    errors found    wasted review
      1%   (500)        18%             2%
      5% (2,500)        54%            11%
     10% (5,000)        71%            29%
```

### Quality metrics — free text fields

**Mode 0**

| Metric | Default | Cost |
|---|---|---|
| Has a specific detail from the source | warn **< 90%** | free |
| Copy ratio (longest verbatim run) | warn **> 50%** | free |
| Length within bounds | expect 100% | free |
| Boilerplate across the batch | warn on near-duplicates | free |
| Agrees with the assigned label | warn **< 90%** | free |
| Claim support rate | warn **< 95%** | judge |

**Mode 1** — a small human-rated sample, used to check the judge agrees with people.

### Settings that apply everywhere

| Setting | Default | Why |
|---|---|---|
| Bootstrap resamples | 1,000 | |
| Resample over | **items**, not (item, field) pairs | fields in one item move together; resampling them separately fakes independence |
| Judge temperature | 0 | a measuring instrument should not roll dice |
| Judge output budget | ~60 tokens | verdict + confidence + one sentence |
| Judge type | instruct | reasoning is opt-in, v1 |
| Confidence intervals | on every ranking metric | a number without one is not a result |

### The one-line answer to "can we use this?"

> **Yes, if a judge-based ranking beats random, majority-label, output-length
> and panel-disagreement on Error Recall@Budget at your actual review budget,
> at n ≥ 200, with an interval that clears the best baseline.**

If it does not, the honest answer is *"a judge is not buying you anything here"*
— and saying that is worth more than a dashboard.

---

## 2. Core model

### The seven things

| Thing | What it is |
|---|---|
| **Item** | The unit being processed — a session, a ticket, a document |
| **Schema** | Declares each field and its kind |
| **Taxonomy** | The label tree for assigned fields. Versioned, lives on its own |
| **Output** | What the model produced for one item + field |
| **Label** | A human answer. Optional. Kept **separate** from Output |
| **Verdict** | One judge's opinion on one output, with a reason |
| **Finding** | A check's result: pass / fail / unscored |

### Labels live apart, and that is deliberate

```
   Item  ──┐
           ├──►  judge  ──►  Verdict
   Output ─┘

   Label ─────►  never reaches a judge. Different path entirely.
```

A judge cannot grade itself against the answer key, because the code that calls
judges never receives labels. A wiring guarantee, not a rule someone has to
remember.

Labels also carry **who** wrote them, which is what makes two-annotator checks
possible:

```
   item_id   field   label                    annotator
   s-1042    jtbd    billing.payment_failed   ann-1
   s-1042    jtbd    billing.card_declined    ann-2      ← they disagree
```

### Three modes, decided per field, each run

```
   ┌─────────────────────────────────────────────────────────┐
   │  MODE 0   no labels                                     │
   │           screening only — distributions, drift,        │
   │           judge health, cross-field consistency,        │
   │           triage ranking                                │
   ├─────────────────────────────────────────────────────────┤
   │  MODE 1   ~100 items, one label each                    │
   │           + accuracy, confusion matrix, judge direction,│
   │             the operating-point table                   │
   ├─────────────────────────────────────────────────────────┤
   │  MODE 2   ~100 items, two independent labels            │
   │           + is the taxonomy actually crisp?             │
   │             everything above becomes trustworthy        │
   └─────────────────────────────────────────────────────────┘
```

Every report says which mode each field is in and what it therefore cannot tell
you:

```
   jtbd      MODE 1  (112 labels)
   summary   MODE 0  — accuracy not computable.
                       ~100 rated summaries would unlock it.
```

### Two grains

```
   FIELD grain    was this one field right?
   ITEM grain     was the WHOLE item right?   ← all fields, no exceptions
```

An item with a correct label but a made-up summary is not a usable item. Both
numbers are reported, always. The item number is always the worse one, and it is
the one a downstream consumer actually needs.

A third grain, **corpus**, covers what only exists across the batch — label
distribution, drift, boilerplate.

### Three finding states

```
   ✓  PASS       we checked it, it is fine
   ✗  FAIL       we checked it, it is wrong
   ○  UNSCORED   we did not check it — and here is why
```

Unscored never becomes a pass. Without the third state, a run that quietly
stopped checking reports green.

### How it flows

```
   Schema  +  Taxonomy@v4
        │
        ▼
   ┌──────────────────────────────────────┐
   │  Run:  Items + Outputs               │
   │  Labels (optional, separate)         │
   └───────────────┬──────────────────────┘
                   ▼
   ┌──────────────────────────────────────┐
   │  GATE 1 — is the measurement sound?  │  ← always, free
   └───────────────┬──────────────────────┘
                   ▼
   ┌──────────────────────────────────────┐
   │  Detect mode, per field              │
   └───────────────┬──────────────────────┘
                   ▼
        ┌──────────┴──────────┐
        ▼                     ▼
   ┌─────────────┐      ┌─────────────┐
   │  ASSIGNED   │      │  FREE TEXT  │
   │  checks     │      │  checks     │
   └──────┬──────┘      └──────┬──────┘
          └──────────┬─────────┘
                     ▼
   ┌──────────────────────────────────────┐
   │  GATE 2 — judge vs trivial baselines │
   └───────────────┬──────────────────────┘
                   ▼
   ┌──────────────────────────────────────┐
   │  Report: findings, both grains,      │
   │  mode banner, what you cannot conclude│
   └──────────────────────────────────────┘
```

---

## 3. Inputs and outputs

### Five inputs, each on its own

```
   items.jsonl       the sessions themselves
   outputs.jsonl     what the model produced
   taxonomy.yml      the label tree, versioned
   judges.yml        judges, panel and triage wiring
   labels.jsonl      human answers   (optional)
```

Separate, not one big file. Because:

- you re-run the model → new outputs, **same items**
- you compare two prompts → two output files, **one items file**
- you add labels later → drop in a file, nothing else changes
- the taxonomy evolves on its own schedule

### What each looks like

**items.jsonl**

```json
{"id": "s-1042", "text": "Maya wrote in March 3. Her card ending 4471..."}
```

**outputs.jsonl** — fields side by side, which is what your model already returns

```json
{"item_id": "s-1042", "jtbd": "billing.payment_failed",
 "summary": "Maya's card 4471 was declined at renewal; a new card fixed it.",
 "outcome": "resolved", "confidence": 0.82}
```

**taxonomy.yml**

```yaml
id: jtbd
version: 4

labels:
  billing:
    definition: "Anything about money moving, or failing to move."
    children:
      payment_failed:
        definition: "A charge was attempted and did not go through."
        examples: ["card was declined at renewal", "insufficient funds"]
        not_this:
          - "asking for money back → billing.refund_request"
          - "cannot reach the payment page → access.*"
```

**labels.jsonl**

```json
{"item_id": "s-1042", "field": "jtbd", "label": "billing.payment_failed", "annotator": "ann-1"}
```

**judges.yml**

```yaml
judges:
  - id: judge-a
    provider: anthropic
    model: claude-sonnet-5
    api_key_env: ANTHROPIC_API_KEY     # the NAME, never the key
    temperature: 0
    max_tokens: 60

  - id: judge-b
    provider: openai
    model: gpt-5-mini
    api_key_env: OPENAI_API_KEY

  - id: judge-c
    provider: openai_compatible        # local box, vLLM, LM Studio
    endpoint: http://gpu-01:1234/v1
    model: qwen2.5-14b-instruct
    api_key_env: null

# the two jobs, wired separately
panel:
  members: [judge-a, judge-b, judge-c]
  sample: 300                # configurable. the panel measures.

triage:
  judge: judge-a
  scope: all                 # the triage judge ranks.
```

API keys are referenced by environment-variable **name**. The config file gets
committed; the key never does.

### One file ties a run together

```yaml
# run.yml
run_id: jtbd-p8                 # a timestamp is added automatically

items:    items.jsonl
outputs:  outputs.jsonl
labels:   labels.jsonl
schema:   schema.yml
taxonomy: taxonomy.yml
judges:   judges.yml

produced_by:
  model: claude-sonnet-5
  prompt_version: p8

budget:
  max_usd: 5.00
  confirm: true
```

`produced_by` is stamped onto every result. Two runs from different prompt
versions are not comparable, and the tool says so rather than drawing you a
trend line across a prompt change.

### Getting data in

```
   ┌── files ──────────────────┐
   │  .jsonl    (default)      │
   │  .csv                     │──┐
   │  .parquet                 │  │
   └───────────────────────────┘  │
                                  ├──►  the same internal shape
   ┌── python ─────────────────┐  │
   │  list of dicts            │  │
   │  pandas / polars frame    │──┘
   └───────────────────────────┘
```

JSONL is the default because it streams — you can point at 500,000 items
without loading them. Readers are swappable. Warehouse connectors are v1.

### What comes out

```
   out/
     2026-09-09_0912_jtbd-p7/
     2026-09-09_1432_jtbd-p8/
       ├── run.json         what ran, settings, mode per field, gates, cost
       ├── verdicts.jsonl   every judge call — cached and reusable
       ├── findings.jsonl   every check result, one row each
       ├── metrics.json     the aggregate numbers with intervals
       ├── risk.jsonl       one row per item — triage score and its provenance
       ├── triage_eval.json strategy x budget comparison        (mode 1)
       ├── calibration.json the fitted calibrator and FitReport (mode 1)
       └── report.md        human readable
     index.jsonl
```

Timestamped folders, so several runs a day never collide. `index.jsonl` is one
line per run, which is what "show me every run this week" and the drift check
read:

```json
{"run_id":"2026-09-09_1432_jtbd-p8","prompt":"p8","taxonomy":"jtbd@v4",
 "gate1":"pass","gate2":"pass","macro_f1":0.71,"items":1204,"cost_usd":0.31}
```

**One finding row:**

```json
{"run_id": "2026-09-09_1432_jtbd-p8",
 "check": "expect_claims_supported",
 "grain": "field",
 "item_id": "s-1042",
 "field": "summary",
 "status": "fail",
 "score": 0.5,
 "threshold": 0.95,
 "threshold_from": "default",
 "evidence": {"unsupported": ["we issued a refund of $49"]},
 "judge": "judge-a",
 "cost_usd": 0.0002}
```

Every row carries **why**, not just a verdict.

### Per-judge visibility

Every judge call is its own row. Nothing is averaged away at write time.

```json
{"judge":"judge-a","item_id":"s-1042","field":"jtbd","verdict":true,
 "confidence":0.9,"reason":"the session is about a declined charge"}

{"judge":"judge-b","item_id":"s-1042","field":"jtbd","verdict":false,
 "confidence":0.7,"reason":"reads as a renewal problem, not a payment one"}
```

A panel finding shows the split and quotes the dissent:

```json
{"check":"expect_label_correct_panel",
 "item_id":"s-1042","field":"jtbd","status":"fail",
 "evidence":{
   "votes":{"judge-a":"correct","judge-b":"incorrect","judge-c":"correct"},
   "agreement":"2 of 3",
   "dissent":{"judge-b":"reads as a renewal problem, not a payment one"}
 }}
```

### Verdicts are cached, and analysis reads the cache

Judge calls are the only expensive thing here. They are written to disk the
moment they come back, and **every piece of analysis reads that file, not the
model.**

```
   ┌──────────────────┐
   │  COLLECT         │   costs money.  run once.
   │  judge calls  ───┼──►  verdicts.jsonl
   └──────────────────┘
                             │
   ┌──────────────────┐      │
   │  ANALYSE         │◄─────┘   free.  run a hundred times.
   │  metrics, checks │
   │  report          │
   └──────────────────┘
```

Change a threshold, add a check, fix a reporting bug, try a different target —
re-run instantly, pay nothing. Cheap reporting code that only runs after an
expensive collection is its own failure mode, and this removes it.

Cache key: `(judge, model, prompt hash, item, field, output value)`. Change any
of them and it is a miss. An explicit `tag` is available for when you
deliberately want two identical-prompt runs kept apart.

---

## 4. The shared engine

### The pipeline, stage by stage

```
   ┌────────────────────────────────────────────┬────────────┐
   │  1  load items, outputs, taxonomy, labels  │   SHARED   │
   │  2  validate config                        │   SHARED   │
   │  3  GATE 1 — measurement soundness         │   SHARED   │
   │  4  detect mode, per field                 │   SHARED   │
   │  5  plan: which checks, what cost          │   SHARED   │
   ├────────────────────────────────────────────┼────────────┤
   │  6  free checks                            │  DIFFERENT │
   │  7  ask the judges                         │  part/part │
   ├────────────────────────────────────────────┼────────────┤
   │  8  cache verdicts                         │   SHARED   │
   │  9  turn verdicts into findings            │   SHARED   │
   │ 10  aggregate metrics                      │  part/part │
   │ 11  GATE 2 — judge vs baselines            │   SHARED   │
   │ 12  report                                 │   SHARED   │
   └────────────────────────────────────────────┴────────────┘
```

### The seam

A field kind is three lists:

```python
class FieldKind:
    name           # "assigned" | "free_text" | "copied" (v1)
    checks         # which free checks apply
    judge_task     # how to ask a judge, and how to read the reply
    metrics        # which aggregates to compute
```

Every check has the same signature, whatever kind it belongs to:

```python
check(items, outputs, ctx) -> list[Finding]
```

So the runner walks a list and collects findings. **It never branches on field
kind.** If adding a kind requires touching the runner, the seam is in the wrong
place.

### One Verdict type

Both kinds return the same thing from a judge:

```python
Verdict:
    judge_id        "judge-b"
    check_id        "label_correct"
    item_id         "s-1042"
    field           "summary"
    status          PASS | FAIL | UNSCORED
    raw_confidence  0.7      # what the judge said. not a probability.
    reason          "claims a refund that never happened"
    detail          {...}    # free text puts its claim list here
    metadata        {...}    # model, prompt hash, tokens, latency, cache_hit
```

Note what is **absent**: no error probability and no triage score. Those are
computed downstream, from a verdict plus a fitted calibrator — see section 7.

Because the shape is common, all of this works for both kinds with no extra
code:

```
   judge approval rate        ← the rubber-stamp check
   parse failure rate
   panel voting and dissent
   relative leniency
   triage risk score
   per-judge health table
   verdict caching
```

If free text had its own verdict shape, every one of those would be written
twice.

### Three check grains

Checks differ by *what they look at*, not by field kind:

```
   FIELD    one item, one field
            "is this label in the taxonomy?"
            "does this summary mention anything specific?"

   ITEM     one item, several fields          ← the interesting one
            "does the summary agree with the label?"

   CORPUS   the whole batch
            "did one label swallow 60% of everything?"
            "are all the summaries suspiciously alike?"
```

Cross-field consistency is an **item** check, so it belongs to neither kind — it
sits above both and reads whatever fields the schema names.

### What is actually different

**Free checks:**

| Assigned | Free text |
|---|---|
| label in taxonomy | length in bounds |
| valid leaf of the tree | has a specific detail from the source |
| distribution / collapse | copy ratio |
| drift vs last run | boilerplate across the batch |

**Judge prompt and reply shape** — same transport, cache, retry, cost
accounting and panel logic; only the prompt text and the parser differ.

**Metrics:**

| Assigned only | Free text only | Shared |
|---|---|---|
| accuracy, macro F1 | claim support rate | bootstrap + intervals |
| confusion matrix | specificity rate | the trivial baselines |
| per-label precision/recall | copy ratio distribution | operating-point table |
| tree scoring | | triage AUC, drift |

### Rough size

| Part | Lines, roughly |
|---|---|
| Shared engine — I/O, config, gates, cache, judges, panel, sampling, bootstrap, baselines, operating point, report | **~1,200** |
| Assigned checks + metrics | ~450 |
| Free-text checks + metrics | ~470 |
| Item + corpus checks | ~200 |

A little under 60% shared. The shared part is all the fiddly infrastructure and
the kind-specific parts are small independent functions — which is the right way
round.

---

## 5. Assigned fields

### Config

```yaml
fields:
  jtbd:
    kind: assigned
    taxonomy: jtbd@v4
    require_leaf: true
    allow_abstain: true
    max_label_share: 0.5
    abstain_rate: [0.01, 0.20]
```

Every threshold is a default you can change.

### The check ladder

```
   ┌─ FREE, runs on everything ──────────────────────────────┐
   │                                                          │
   │  1  label exists in taxonomy@v4                          │
   │  2  label is a valid leaf                                │
   │  3  abstention rate in range           (corpus)          │
   │  4  distribution: no collapse, no drift (corpus)          │
   │  5  agrees with the other fields        (item)           │
   │                                                          │
   └──────────────────────┬───────────────────────────────────┘
                          │
   ┌─ COSTS MONEY ────────▼───────────────────────────────────┐
   │                                                          │
   │  6  PANEL     3 judges on ~300 items                     │
   │               → how good is this? which labels are fuzzy? │
   │                                                          │
   │  7  TRIAGE    1 judge on everything                      │
   │               → rank items for human review              │
   │                                                          │
   └──────────────────────┬───────────────────────────────────┘
                          │
   ┌─ FREE, only if you have labels ──────▼───────────────────┐
   │                                                          │
   │  8  compare to the human label                           │
   │  9  grade the judge against the human                    │
   │                                                          │
   └──────────────────────────────────────────────────────────┘
```

### An honest note about cost

For **copied** fields there is a strong free gate — a value not in the text was
invented, so you only pay to judge the suspicious ones.

**Assigned fields have no such gate.** Checks 1–5 catch *format* problems, not
*wrongness*. A perfectly formed label that is simply the wrong one passes every
free check.

So the triage judge has to see everything, or a large sample. The cost levers:

| Lever | Effect |
|---|---|
| Triage on all items | 1 call per item. The default. |
| Triage on a sample | Ranking only covers what you sampled. |
| Panel sample size | 3 judges × N. Default N = 300 → 900 calls. |
| Judge output budget | ~60 tokens. |
| Small instruct model for triage | The single biggest saving. |

A typical run: **50,000 triage calls + 900 panel calls.** The panel is a
rounding error; the triage judge is the whole bill.

### The free checks

**1. Label exists** — expect 100%. Anything else means the model invented a
label, or your taxonomy version moved under it.

**2. Valid leaf** — did it stop at the right depth?

```
   billing.payment_failed   ✓ leaf
   billing                  ✗ too shallow — stopped at the parent
   billing.payment.card     ✗ not a real path
```

**3. Abstention rate** — warn outside 1–20%.

```
   too high  → your taxonomy has a gap
   too low   → the model is forcing a label onto unclear items
```

**4. Distribution** — two things at once:

```
   COLLAPSE   biggest label share > 50%  →  warn
   DRIFT      vs the previous run; any label moving > 10pp  →  warn
```

Drift is what catches a prompt change nobody told you about.

**5. Agrees with the other fields** — item grain, free, underrated. If the
summary talks about logging in and the label says `payment_failed`, one of them
is wrong.

### Stability — the re-sampling hook

```yaml
produced_by:
  regenerate:                        # optional. no hook, no stability check.
    provider: anthropic
    model: claude-sonnet-5
    api_key_env: ANTHROPIC_API_KEY
    prompt_file: prompts/jtbd_p8.txt
    sample: 100                      # never the whole corpus
    runs: 2
```

Three rules keep it contained:

**Sample only.** A diagnostic, not a re-run.
**Never overwrites your outputs.** Writes to its own file.
**Two different numbers, labelled differently:**

```
   temperature 0     →  SERVING stability.  Expect ~100%.
                        Anything less means your serving stack is
                        nondeterministic, which is worth knowing.

   temperature > 0   →  DECISION stability.  How fragile is this label
                        when the model rolls the dice?
```

### With labels — Mode 1

**Tree scoring, four buckets:**

```
   human says:  billing.payment_failed

   EXACT           billing.payment_failed
   RIGHT PARENT    billing.refund_request     ← sibling. boundary problem.
   TOO SHALLOW     billing                    ← stopped at the parent
   WRONG           access.password_reset      ← different branch entirely
```

Each bucket points at a different fix. Sibling confusion means two definitions
need sharpening. Too-shallow means the model is hedging. Wrong means it is not
reading the item.

**The metrics:**

```
   macro F1                 the headline. not accuracy.
   per-label recall         flag any label under 0.5
   confusion matrix         the actionable output
   exact / parent / shallow / wrong
   confidence calibration   does 0.9 mean 90%?
```

Accuracy is reported but never alone — on a skewed taxonomy, always guessing the
biggest label scores well and knows nothing.

---

## 6. Judges

### Two jobs, wired separately

```
   ┌─ PANEL ──────────────────────────────────────────────┐
   │  3 judges  ×  ~300 items                             │
   │                                                       │
   │  answers:  how good is this overall?                  │
   │            which label pairs are fuzzy?               │
   │            is any judge broken?                       │
   └───────────────────────────────────────────────────────┘

   ┌─ TRIAGE ─────────────────────────────────────────────┐
   │  1 judge  ×  every item                              │
   │                                                       │
   │  answers:  which 500 of these 50,000 should a         │
   │            human open first?                          │
   └───────────────────────────────────────────────────────┘
```

### What a judge is asked

```
   SYSTEM
     You are checking a label another system assigned to an item.
     Decide whether the label is correct.
     If the item does not say enough to decide, answer cannot_decide
     rather than guessing.

     The permitted labels are:
       billing.payment_failed — a charge was attempted and did not go through
       billing.refund_request — the customer is asking for money back
       access.password_reset  — the customer cannot get into their account

   USER
     ITEM:
     Maya wrote in March 3. Her card ending 4471 was declined when
     the subscription auto-renewed...

     ASSIGNED LABEL: billing.payment_failed

     Reply with: correct (true/false/cannot_decide), confidence 0-1,
     and one sentence of reasoning.
```

Four things on purpose:

**The taxonomy definitions go in the prompt.** This is why the definitions in
`taxonomy.yml` are load-bearing and not documentation.

**A reason is always required.** ~60 tokens. It is what a reviewer reads, and how
you see *why* judges split.

**`cannot_decide` is allowed.** Forcing a verdict on an unclear item manufactures
noise. It maps to unscored, never to "wrong".

**Every panel judge gets the byte-identical prompt.** Otherwise panel
disagreement measures your prompt differences instead of your judges.

### The panel

```
   item s-1042 / jtbd

   judge-a   correct    0.9   "the session is about a declined charge"
   judge-b   incorrect  0.7   "reads as a renewal problem, not a payment one"
   judge-c   correct    0.8   "card declined is squarely a payment failure"

   ───────────────────────────────────────────────────────
   majority: correct (2 of 3)      ← reported, with the split visible
   dissent quoted in the finding
```

**What the panel is for** — three outputs:

```
   1  an accuracy estimate over the sample, with an interval
   2  which label pairs the judges keep splitting on   →  taxonomy bug
   3  a side-by-side health check of your judges
```

A per-item panel verdict *is* emitted, so you can derive your own insights from
it. It carries a warning rather than being withheld.

**Three things the panel deliberately does not do:**

| Not doing | Why |
|---|---|
| Route disagreements to review | Disagreement does not rank errors. It ranks near chance. |
| Auto-approve on unanimity | Judges agree constantly and are wrong on a lot of it. Consensus is not proof. |
| Report one blended score | Hiding a 2–1 split behind an average throws away the only interesting part |

**Effective votes** are reported:

```
   3 judges, agreement 92%  →  effective votes 1.1 of 3

   ⚠ you are paying for three opinions and receiving about one.
```

### The triage judge produces verdicts, not a ranking

A judge emits a status and a raw confidence. Turning those into an order to
review in is a **separate step with its own machinery**, because the obvious
shortcut — rank by `1 − confidence` and call it risk — quietly asserts that a
number the model emitted is a probability of error. It is not, and that
assertion is the failure this library exists to catch.

Ranking, calibration and how the two are evaluated are section 7.

What the judge owes the ranker is only this: a status, a raw confidence, a
reason, and honest metadata about how the call went.

### Judge screening — free, and the most important check

From the panel sample, per judge, no human labels needed:

```
   judge     approves   cannot_decide   parse fail
   judge-a     71%          4%            0.3%
   judge-b     58%          2%            0.7%
   judge-c     94%          0%            0.0%    ⚠ possible rubber stamp
```

A judge approving 94% of everything passes every format check ever written and
quietly destroys the panel. This table is the only thing that catches it.

**Relative leniency, also free.** When two judges disagree, note who approved:

```
   judge-c approves, judge-b rejects   →  188 times
   judge-b approves, judge-c rejects   →   14 times

   judge-c is the pushover.
```

With human labels you also get the absolute version — how often each judge waves
through a wrong label versus rejects a right one.

### Choosing judges

**Three panel members is enough.** More buys very little.

**Do not expect independence from using different providers.** Two models from
different companies routinely make the same mistakes.

**What helps is a judge that fails the other way.** If your judges all wave
through wrong labels, add a strict one that rejects too much. Two lenient judges
are nearly one judge.

**Instruct models, temperature 0, ~60 output tokens.** Reasoning models are v1 —
more accurate, far more expensive, and they return empty replies when thinking
eats the token budget. Those empties land on the hard items.

### Settings

| Setting | Default |
|---|---|
| Panel size | 3 |
| Panel sample | 300 items (configurable) |
| Triage scope | all items |
| Temperature | 0 |
| Max output tokens | 60 |
| Approval-rate warning band | 15–85% |
| Parse failure warning | 2% |
| Retries | 3, with backoff |

---

## 7. Calibration, risk and triage

### Three numbers, never silently interconverted

The mistake this section exists to prevent is treating a number a language
model emitted as a probability. It is not one. A judge reporting "confidence
0.9" is stating a feeling; ranking on it and calling the result a risk score is
exactly the trust-laundering this library is built to catch.

So three separately named things:

| Name | What it is |
|---|---|
| `raw_confidence` | What the judge said. A number in [0,1]. **Not a probability of anything.** |
| `calibrated_error_probability` | P(this verdict is wrong), from a calibrator fitted on human-labelled data. Exists only in mode 1+. |
| `triage_score` | The ranking signal. Derived from one of the above, and always stamped with which. |

```
   judge  ──►  raw_confidence  ──┐
                                 ├──►  calibrator  ──►  calibrated_error_probability
   labelled sample ──────────────┘                              │
                                                                ▼
                                                          triage_score
                                                  stamped calibrated / uncalibrated
```

### The Verdict model

```python
Verdict:
    judge_id        "judge-b"
    check_id        "label_correct"
    item_id         "s-1042"
    field           "summary"
    status          PASS | FAIL | UNSCORED
    raw_confidence  0.7        # what the judge said. not a probability.
    reason          "claims a refund that never happened"
    detail          {...}      # claim lists, per-field breakdowns
    metadata        {...}      # model, prompt hash, tokens, latency, cache_hit
```

`calibrated_error_probability` is deliberately **not** on the Verdict. A judge
does not produce it; a calibrator computes it later from a verdict plus a fitted
model. Putting the field here would invite writing it at judge time, which is
the bug this whole section exists to prevent.

It lives on a derived row instead:

```python
RiskRow:
    item_id
    triage_score                    # 0–1, higher = review sooner
    calibrated_error_probability    # float | None
    raw_confidence_mean             # float
    strategy                        # "calibrated_risk" | "raw_confidence" | ...
    calibrated                      # bool
    calibration_id                  # str | None
```

### The calibration interface

```python
class Calibrator(Protocol):
    id: str
    is_calibrated: bool

    def fit(self, verdicts: list[Verdict], labels: list[Label]) -> FitReport: ...
    def error_probability(self, verdict: Verdict) -> float: ...
```

Two ship in v0:

| Calibrator | `is_calibrated` | What it does |
|---|---|---|
| `identity` | **False** | Returns `1 − raw_confidence`. The default with no labels. Everything it touches is stamped uncalibrated. |
| `platt` | True | Logistic regression of the human verdict on raw confidence. Needs ≥ 100 labelled rows. |

Isotonic regression is the obvious third and is deferred: it needs more data
than most teams have at the start and overfits badly below a few hundred labels.

**The identity calibrator is not a calibration.** It exists so the pipeline has
one shape in both modes, and it says so on every number it produces:

```
   ⚠ triage ranking is UNCALIBRATED
       ranked by raw judge confidence, which is not a probability of error
       ~100 labelled rows would let it be fitted and measured
```

### Whether the calibration worked

Fitting one is not the same as it helping. `FitReport` carries:

| Metric | Reads as broken when |
|---|---|
| Expected calibration error | above ~0.10 — stated probabilities do not match observed rates |
| Brier score | no better than the base rate — the calibration adds nothing |
| Reliability curve | bins deviate from the diagonal in one direction |
| n, and n per bin | below the floor; no bin under 20 rows gets its own number |

### Calibrations go stale

Fingerprinted on `(judge_id, model, prompt hash, check_id, taxonomy version)` —
the same guard as the taxonomy hash.

```
   ERROR  calibration `jtbd_v4_judge-a` was fitted on prompt p7.
          This run uses p8. Refit, or run uncalibrated and say so.
```

### Triage strategies are a plug point

Ranking is not a hardcoded formula:

```python
class TriageStrategy(Protocol):
    id: str
    requires: frozenset[str]        # verdicts | labels | panel | outputs
    def rank(self, ctx: TriageContext) -> dict[str, float]: ...
```

Six ship in v0, and four of them exist to be beaten:

| Strategy | Needs | Role |
|---|---|---|
| `random` | — | baseline, seeded |
| `output_length` | outputs | baseline — catches a target confounded with length |
| `majority_label` | outputs | baseline — the trivial classifier |
| `panel_disagreement` | panel | **a candidate, and a baseline** |
| `raw_confidence` | verdicts | default when uncalibrated |
| `calibrated_risk` | verdicts + calibration | default when calibrated |

### Disagreement is a hypothesis, not a conclusion

Published results — on invoice extraction, and on contested label spaces — find
panel disagreement ranks errors close to chance and below any single judge. That
is why it is not the default.

But those are other corpora. Asserting the finding holds for yours without
measuring it would be the same unearned confidence this library exists to
prevent. So `panel_disagreement` ships as a **selectable strategy and a scored
baseline**, and the comparison table settles it on your data.

If it wins on your corpus, use it. The framework does not hold an opinion it
will not let you check.

### Error Recall@Budget is the primary metric

> **Error Recall@K** = true errors found in the top K reviewed rows ÷ all true errors

That is the question a review budget actually poses. AUC and the other aggregate
ranking measures stay, and stay secondary.

```yaml
triage:
  strategy: auto                 # calibrated_risk if fitted, else raw_confidence
  budgets: [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
```

Per strategy, per budget:

```
   review_budget
   n_reviewed
   errors_found
   error_recall          errors_found / total_true_errors
   precision_at_budget   errors_found / n_reviewed
   ci_low, ci_high       bootstrap over items, when n allows
```

**This needs labels.** You cannot count true errors without them, so
recall@budget is a **mode 1** capability. In mode 0 you still get the ranking —
you just do not get to claim it has been validated, and the report says exactly
that rather than implying otherwise.

### The comparison table

```
   strategy               budget   reviewed   found   recall   precision
   calibrated_risk          1%        500       92    18.4%      18.4%
   raw_confidence           1%        500       71    14.2%      14.2%
   panel_disagreement       1%        500       34     6.8%       6.8%
   output_length            1%        500       12     2.4%       2.4%
   random                   1%        500       10     2.0%       2.0%
```

`llm-expectations triage-eval out/<run>/` produces it, reading verdicts and
labels off disk and issuing zero model calls.

### The two loops

Calibration is where they meet, and they must not blur into one:

```
   EVALUATOR VALIDATION LOOP            rare · offline · labels eventually
   ─────────────────────────
   sample ─► panel ─► verdicts ─► agreement, judge health
                               └─► calibration fit ─► FitReport
                                            │
                                            │  a fitted calibrator, as data
                                            ▼
   PRODUCTION QUALITY LOOP              every run · no labels required
   ───────────────────────
   outputs ─► judge ─► raw_confidence ─► calibrated_error_probability
                                      ─► triage_score ─► human review
```

Labels enter the first loop and reach a judge in neither. Downstream of the
judge they are used for exactly four things: fitting a calibration, validating
it, computing metrics, and evaluating triage.

---

## 8. Free text fields

### Why it is different

For an **assigned** field you can hold up the right answer:

```
   model said:     billing_problem
   right answer:   payment_failed
   →  wrong. done.
```

For **free text** there is no single right answer:

```
   model said:      "Maya's card was declined at renewal; a new card fixed it."
   equally good:    "Payment failed on auto-renew, resolved by updating the card."

   Both correct. They share almost no words.
```

So you cannot ask *"is it right?"* — comparing to one gold summary would score
word choice. You ask *"is anything wrong with it?"* and hunt for specific
defects.

### What goes wrong

Source: *"Maya wrote in March 3. Her card ending 4471 was declined when the
subscription auto-renewed. She gave us a different card and the $49 charge went
through."*

| # | Bad output | What is wrong |
|---|---|---|
| 1 | "Her card was declined so we **issued a refund of $49**." | **Made up.** No refund happened. |
| 2 | "The customer had a problem and it was resolved." | **Generic.** True of every session. |
| 3 | *pastes the whole session back* | **Copied**, not summarised. |
| 4 | "Maya couldn't **log in** to her account." | **Contradicts the label.** |
| 5 | *three paragraphs* | **Wrong shape.** |

Four of the five are caught for free. Only "made up" needs a judge — and it is
the one that matters most, so it is worth paying for.

### Config

```yaml
fields:
  summary:
    kind: free_text
    style: descriptive          # descriptive | judgement | proposal
    min_words: 5
    max_words: 30
    must_agree_with: [jtbd]
    min_specificity: 0.9
    max_copy_ratio: 0.5
    min_claim_support: 0.95
```

`style` decides how the expensive check works:

```
   descriptive   "Maya's card 4471 was declined; a new card fixed it."
                 many small facts  →  split into claims, check each

   judgement     "Resolved on first contact."
                 one conclusion    →  don't split. one question over
                                      the whole item.

   proposal      "Confirm her new card is saved as default."
                 about the future  →  don't ground it at all.
                                      ask: does this follow from the facts?
```

### The check ladder

```
   ┌─ FREE, runs on everything ──────────────────────────────┐
   │                                                          │
   │  1  length in bounds                                     │
   │  2  is it specific, or filler?                           │
   │  3  copy ratio — is it pasting the source back?          │
   │  4  boilerplate across the batch        (corpus)         │
   │  5  agrees with the other fields         (item)          │
   │                                                          │
   └──────────────────────┬───────────────────────────────────┘
                          │  flagged items + a small audit sample
                          ▼
   ┌─ COSTS MONEY ────────────────────────────────────────────┐
   │                                                          │
   │  6  are the claims supported by the item?                │
   │                                                          │
   └──────────────────────────────────────────────────────────┘
```

**Free text is cheaper than assigned**, which is the opposite of what you would
guess. Checks 2 and 5 are genuinely predictive — a summary with no specific
detail, or one that contradicts the label, is usually the one that has invented
something. So they *do* gate the expensive check.

### The free checks

**2. Specific, or filler?** — no NLP library needed.

```
   Find tokens that appear in the item AND are rare across the batch
   (in fewer than 5% of items). Then: does the output contain any?

   "the customer had a problem"           → 0 rare tokens   ✗ filler
   "card 4471 declined at renewal"        → "4471"          ✓ specific
```

**3. Copy ratio** — longest run copied word-for-word, as a share of the output.
Over 50% means it is pasting, not summarising.

**4. Boilerplate** — how many outputs are near-duplicates of another output.

```
   47 of 1,204 summaries are near-identical to another summary
   →  the model has fallen into a template
```

**5. Agrees with the other fields** — free, and it uses the taxonomy.

```
   label:  billing.payment_failed
   its definition mentions: charge, payment, declined, card

   summary: "Maya couldn't log in to her account."
   overlap with those terms: none
   →  flag. one of these two fields is wrong.
```

### The expensive check

The judge splits the claims — it is better at that than a regex, and it is one
call:

```
   ITEM:    Maya wrote in March 3. Her card ending 4471 was declined
            when the subscription auto-renewed...

   TEXT:    "Her card was declined so we issued a refund of $49."

   →  claim 1: "her card was declined"        supported     ✓
      claim 2: "we issued a refund of $49"    NOT supported ✗

      support rate: 1/2 = 0.50    threshold 0.95   →  FAIL
```

The claim list rides in `detail`. This needs a bigger output budget than the
assigned judge — around 200 tokens instead of 60.

### With human ratings — Mode 1

Do not ask humans to write a better summary. **Ask them to mark defects** — the
same boxes the library checks:

```
   s-1042   "Her card was declined so we issued a refund of $49."

     ☑  something made up
     ☐  too generic
     ☐  contradicts another field
     ☐  missing something important
```

About 30 seconds per item, and it lines up exactly with what the judge said:

```
   defect              judge vs human agreement
   made up                    0.86
   too generic                0.91
   contradicts                0.94
   missing something          0.52     ← the weak one, as expected
```

That last row is the honest outcome. *"Did it miss something?"* is hard for the
judge and for the human, and the number says so instead of hiding it.

### Cost

| | Assigned | Free text |
|---|---|---|
| Free gate is predictive? | no | **yes** |
| Judge calls | 1 per item | ~1 per flagged item + audit |
| Output tokens per call | ~60 | ~200 |

If 15% of summaries get flagged and you audit 5% of the rest, you pay for about
19% of the corpus instead of 100%.

---

## 9. Taxonomy

### One file, four readers

```
                    ┌──► the judge prompt
                    │     (definitions + not_this)
                    │
   taxonomy.yml ────┼──► the validity check
      @v4           │     (is this label real? is it a leaf?)
                    │
                    ├──► tree scoring
                    │     (parent / child relationships)
                    │
                    ├──► cross-field consistency
                    │     (does the summary use this label's vocabulary?)
                    │
                    └──► the human annotator
                          (the same definitions, word for word)
```

The last one matters most. If the judge and the human read different
definitions, judge-versus-human disagreement tells you nothing about either.

**`not_this` is the part that makes a taxonomy work.** Definitions tell you what
a label covers; boundary cases tell you where it stops. Those boundaries are
exactly where a model and a human both get confused.

### Versioning, and the guardrail

Any content change bumps the version. The library hashes the file and stamps it
into every run:

```
   ERROR  taxonomy jtbd@v4 has changed since it was last used.
          Recorded hash: 8f3a...   Current: c1d9...
          Bump to v5, or restore the file.
```

This stops the worst silent failure — someone tightens a definition, nobody
bumps, and a month of results quietly stop being comparable.

### Migration between versions

```yaml
# jtbd_v4_to_v5.yml
billing.payment_failed:  billing.charge_failed     # renamed
billing.card_declined:   billing.charge_failed     # merged into one
access.password_reset:   access.password_reset     # unchanged
billing.refund_request:  null                      # no equivalent
```

With a mapping, old runs can be compared to new ones. Without one, the library
refuses and says why.

### The fuzzy-pair detector

The check nothing else gives you. Three sources of evidence, increasing in
strength:

**Free, no labels — from panel disagreement:**

```
   87 items where the judges split.
   62 of them involve one pair:

       billing.payment_failed  ↔  billing.card_declined

   →  71% of all disagreement sits on a single boundary.
```

**With labels — from the confusion matrix, and the direction matters:**

```
   SYMMETRIC                          ASYMMETRIC
   payment_failed → card_declined 23   payment_failed → refund 31
   card_declined → payment_failed 19   refund → payment_failed  2

   The boundary is unclear.            The model is biased one way.
   → fix the TAXONOMY                  → fix the PROMPT
```

**With two annotators — the strongest evidence:**

```
   Two humans disagreed on 18% of items.
   Two thirds of those disagreements are this same pair.

   →  Your annotators cannot separate these labels either.
      This is not a model problem. Merge them or rewrite both definitions.
```

### Static health checks on the file

Run before you spend anything, no data required:

| Check | Catches |
|---|---|
| Every label has a definition | A label the judge can only guess at |
| Every label has at least one example | Same |
| No duplicate label names across branches | `billing.other` and `access.other` |
| Two definitions nearly identical | Two labels that are really one |
| Leaf depth consistent | A branch that stops early by accident |
| Boundary cases on sibling labels | The pairs most likely to become fuzzy |

### Settings

| Setting | Default |
|---|---|
| Require definition on every label | on |
| Require examples | warn only |
| Require leaf | on, per field |
| Definition similarity warning | 0.8 |
| Fuzzy-pair report | top 5 pairs |
| Version hash enforcement | on |

---

## 10. Guardrails

Checks on the **measurement**, not on your data. They run first, cost nothing,
and are always on.

### Three severities — and they suppress numbers, not the run

```
   STOP   this specific number cannot be computed honestly.
          It is not reported. The reason is.

   WARN   reported, with a flag attached.

   NOTE   worth knowing.
```

A guardrail firing does not kill the run. It removes the numbers it invalidates
and says which and why:

```
   ┌─────────────────────────────────────────────────────┐
   │  WHAT THIS RUN CANNOT TELL YOU                      │
   │                                                     │
   │  ✗ triage AUC for `summary`                         │
   │      only 12 items in the minority class (need 30)  │
   │                                                     │
   │  ✗ judge-c's contribution to panel agreement        │
   │      judge-c approved 97% of items — excluded       │
   │                                                     │
   │  ✗ drift vs the previous run                        │
   │      taxonomy changed from v4 to v5, no mapping     │
   └─────────────────────────────────────────────────────┘
```

### Tier 1 — always on, each catches a silent disaster

**Rubber-stamp judge**

```
   WARN   outside 15–85%
   STOP   outside  2–98%   → excluded from panel aggregates,
                             raw verdicts still written to disk
```

**Unreadable replies — counted, never defaulted**

```
   WARN  > 2%     STOP  > 5%
```

A coin-flip default is uncorrelated by construction, which quietly changes every
agreement number in the run. And report *which* items failed:

```
   ⚠ 31 empty replies. They are not random:
       mean item length of failures : 4,200 words
       mean item length overall     :   900 words

     Your hardest items are the ones going unjudged.
```

**Degenerate target** — need ≥ 30 items and ≥ 5% in the minority class. An AUC
computed while ranking three negatives looks fine and has a tight-looking
interval.

**No label leakage** — structural. The code that builds triage signals never
receives labels; they arrive on a separate path and the function signature does
not accept them. Backed by one runtime assertion.

### Tier 2 — cheap, needs a little data

**Sample too small** — the number is shown with its interval and a plain
statement:

```
   triage AUC   0.68  [0.46, 0.87]   n = 28
   ⚠ This interval spans "useless" to "excellent". It cannot support
     a conclusion. 200 items minimum.
```

A wide interval is not a weak result — it is no result.

**The trivial baselines, run automatically**, next to every ranking metric.

**Resampling over items**, never over `(item, field)` pairs. Fields within one
item move together.

### Tier 3 — needs labels

| Guardrail | What it catches |
|---|---|
| Judge direction | A judge that only fails one way, adding nothing to a panel that already fails that way |
| Judge vs majority-label | A judge that cannot beat "always guess the biggest label" |
| Confidence calibration | A judge whose 0.9 means 60% |

### Tier 4 — v1

Outcome-selected exclusion. Threshold provenance audit.

### Exclusions are logged, always

```
   excluded   reason                          n
   ────────────────────────────────────────────
   filtered   item text empty                12
   filtered   user filter: language != en   340
   dropped    judge gave no readable reply    31
```

Dropping items is normal. Dropping them *because they scored badly* is how
numbers get manufactured, and it is invisible unless every exclusion states a
reason.

### Cost guard

```
   PLAN   1,204 items
          triage : 1,204 calls
          panel  :   900 calls  (300 items × 3 judges)
          claims :   228 calls  (flagged summaries + audit)
          ─────────────────────
          2,332 calls, est. $0.34

   Budget is $0.25. Continue? [y/N]
```

Over budget, items past the cap are marked **unscored**, never passed.

### Settings

| Guardrail | Default | Change it |
|---|---|---|
| Judge approval warn band | 15–85% | yes |
| Judge approval stop band | 2–98% | yes |
| Parse failure warn / stop | 2% / 5% | yes |
| Minority class floor | 30 items and 5% | yes |
| Minimum n for a ranking claim | 200 | yes |
| Bootstrap resamples | 1,000 | yes |
| Cost confirmation prompt | on | yes |
| Label-leak assertion | on | **no** |

Everything is tunable except the label-leak assertion. That is not a threshold,
it is a bug check.

---

## 11. Config and report

### The files

```
   project/
     run.yml           ties everything together
     schema.yml        fields and their kinds
     taxonomy.yml      the label tree, versioned
     judges.yml        judges, panel and triage wiring
     settings.yml      thresholds        (optional — all have defaults)

     items.jsonl       the sessions
     outputs.jsonl     what your model produced
     labels.jsonl      human answers     (optional)
```

### Settings override in three layers

```
   built-in defaults
        ↓  overridden by
   settings.yml            project-wide
        ↓  overridden by
   schema.yml, per field   the specific case
```

Nothing has to be configured to start. Every default is printed in the report
next to the number it produced.

### `schema.yml`

```yaml
item: support_session

fields:
  jtbd:
    kind: assigned
    taxonomy: jtbd@v4
    require_leaf: true
    allow_abstain: true

  summary:
    kind: free_text
    style: descriptive
    max_words: 30
    must_agree_with: [jtbd]

  outcome:
    kind: assigned          # small vocabulary → a label, not free text
    taxonomy: outcomes@v1
```

**If a free-text field only ever takes a handful of values, make it assigned.**
Same information, a fraction of the cost, far better checks.

### CLI

```bash
llm-expectations run run.yml           # collect + analyse
llm-expectations plan run.yml          # cost estimate, no calls
llm-expectations analyse out/<run>/    # re-analyse from cache, free
llm-expectations compare out/a out/b   # two runs, head to head
llm-expectations check taxonomy.yml    # static taxonomy health
```

`analyse` is the one you will use most. Change a threshold, add a check, fix a
bug — re-run and pay nothing.

### The report

```
llm-expectations   2026-09-09_1432_jtbd-p8

  1,204 items      $0.34      2,332 calls      4m 12s

  ┌ GATES ────────────────────────────────────────────────────┐
  │  measurement sound       PASS                              │
  │  judge beats baselines   PASS    0.84  vs  0.50 best       │
  └────────────────────────────────────────────────────────────┘

  MODE
    jtbd       MODE 1    112 labels
    summary    MODE 0    no ratings
    outcome    MODE 1    112 labels

  ┌ WHAT THIS RUN CANNOT TELL YOU ────────────────────────────┐
  │  ✗ whether the judge agrees with people on `summary`       │
  │      no human defect ratings.                              │
  │      ~100 rated summaries would unlock it.                 │
  └────────────────────────────────────────────────────────────┘


  ── jtbd ─────────────────────────────── assigned · jtbd@v4 ──

  free checks
    label in taxonomy            100.0%   ✓
    valid leaf                    99.7%   ✗  4 stopped at a parent
    abstention rate                6.2%   ✓  band 1–20%
    largest label share           31.0%   ✓  max 50%
    drift vs previous run         +4pp on billing.payment_failed
    agrees with summary           94.1%   ✗  71 conflicts

  panel                          3 judges × 300 items
    majority says correct         78.3%   [74.1, 82.2]
    judges agree                  86.0%
    effective votes               1.4 of 3   ⚠ paying 3× for ~1.4
    fuzziest pair                 payment_failed ↔ card_declined
                                  62 of 87 splits (71%)  → taxonomy bug

  judges
    judge        approves   cannot_decide   unreadable
    judge-a         71%          4%            0.3%
    judge-b         58%          2%            0.7%
    judge-c         94%          0%            0.0%   ⚠ rubber stamp?

  vs humans                      112 labels
    macro F1                      0.71
    accuracy                      0.79    majority-label baseline 0.34
    exact / parent / shallow / wrong    0.79 / 0.11 / 0.03 / 0.07
    weakest label                 access.sso_issue   recall 0.42  n=31

    review budget    errors found    wasted effort
        1%               18%              2%
        5%               54%             11%
       10%               71%             29%


  ── summary ────────────────────── free_text · descriptive ──

  free checks
    length in bounds              98.9%   ✓
    specific, not filler          91.2%   ✓  min 90%
    copy ratio under 50%          99.1%   ✓
    boilerplate                   47 near-duplicates   ⚠
    agrees with jtbd              94.1%   ✗  71 conflicts

  claim support                  228 judged  (185 flagged + 43 audit)
    support rate                  0.87    ✗  min 0.95
    items with an invented claim  31
    worst   s-2201   "we issued a refund of $49"  — not in the item


  ── EXCLUSIONS ──────────────────────────────────────────────
    12   item text was empty
    31   judge gave no readable reply

  ── FILES ───────────────────────────────────────────────────
    out/2026-09-09_1432_jtbd-p8/
```

### Three things the report always does

**Says which mode each field is in, at the top.** You should never have to guess
whether a number is backed by human answers.

**Prints the threshold next to every result.** `91.2% ✓ min 90%`.

**Prints the baseline next to every metric that has one.** A number without its
baseline is not a result.

---

## 12. Repo layout and build order

### Layout

```
llm-expectations/
  pyproject.toml
  README.md
  DESIGN.md

  llm_expectations/
    types.py          Item, Output, Label, Verdict, Finding, enums
    schema.py         Schema, FieldSpec, FieldKind
    taxonomy.py       tree, versioning, hash, health checks
    config.py         run.yml, settings layering

    read.py           jsonl / csv / parquet / python objects
    write.py          findings, verdicts, metrics, index
    report.py         the text report

    plan.py           what will run, cost estimate
    run.py            the orchestrator
    cache.py          verdict cache
    budget.py

    judges/
      base.py         Judge protocol, Verdict
      providers.py    anthropic, openai, openai-compatible
      prompts.py      prompt builders per task
      panel.py        voting, dissent, agreement, effective votes
      screening.py    approval rate, leniency, health table

    calibration/
      base.py         Calibrator protocol, FitReport, fingerprinting
      identity.py     uncalibrated passthrough — stamps everything
      platt.py        logistic fit on labelled verdicts
      quality.py      ECE, Brier, reliability curve

    triage/
      base.py         TriageStrategy protocol, TriageContext, RiskRow
      strategies.py   random, length, majority, disagreement,
                      raw_confidence, calibrated_risk
      evaluate.py     Error Recall@Budget, precision@budget, comparison

    checks/
      base.py         Check protocol, three grains
      assigned.py
      free_text.py
      item.py         cross-field consistency
      corpus.py       distribution, drift, boilerplate
      guardrails.py   Gate 1

    metrics/
      classification.py   F1, confusion matrix, tree buckets
      ranking.py          AUC and the other secondary aggregates
      agreement.py        agreement, effective votes, leniency
      stats.py            bootstrap, intervals, baselines

    cli.py

  tests/
  examples/jtbd/
```

Three rules that keep it from rotting:

**`types.py` imports nothing** from the rest of the package. If something needs
to import upward, the layering is wrong.

**`checks/` never asks what field kind it is in.** A check is a function with one
signature; the schema decides which ones run.

**`metrics/` never calls a model.** It reads findings and verdicts off disk.
That is what makes `analyse` free.

### Build order

```
   M0  skeleton
       types, schema, taxonomy, readers, config
       ~2 days

   M1  one judge, end to end                    ◄── FIRST SHIPPABLE
       providers, prompts, cache, triage risk score,
       judge health (approval rate, parse rate), minimal report
       → "here are the 500 sessions to open first"
       ~4 days

   M2  free checks + full report
       assigned, corpus, item checks
       ~3 days

   M3  panel
       voting, dissent, agreement, effective votes,
       screening, fuzzy-pair detector
       ~2 days

   M4  guardrails + stats                       ◄── SHIPPABLE
       Gate 1 in full, Gate 2, bootstrap, intervals, baselines
       ~2 days

   M5  labels (Mode 1)
       F1, confusion matrix, tree buckets, judge direction
       ~3 days

   M5b calibration + triage evaluation          <-- SHIPPABLE
       Platt fit, FitReport quality metrics, staleness fingerprint,
       Error Recall@Budget, the strategy comparison table
       ~3 days

   M6  free text                                ◄── SHIPPABLE
       free checks, claim judge, defect ratings
       ~4 days

   M7  across runs — compare, drift, stability hook
       ~2 days

   M8  polish, docs, worked example
       ~2 days
```

**Judge health goes into M1, not M4.** Approval rate and parse rate are about
twenty lines each, and without them the first shippable version could be quietly
ranking items with a rubber-stamp judge. That guardrail cannot wait.

### What depends on what

```
   M0 ──► M1 ──► M2 ──► M3 ──► M4 ──► M5 ──► M7 ──► M8
                  │                    │
                  └────► M6 ───────────┘
```

### Testing

**Everything offline.** A `FakeJudge` returning scripted verdicts is the
backbone — deterministic, free, and it lets CI run the whole pipeline including
the panel.

**Golden fixtures with deliberately planted errors.** A small corpus where you
know exactly which items are wrong and how:

| Seeded error | Example |
|---|---|
| sibling confusion | `payment_failed` where it should be `card_declined` |
| too shallow | `billing` instead of a leaf |
| invented label | a label not in the taxonomy at all |
| collapse | one label swallowing everything |
| invented claim | a summary asserting something not in the item |
| filler | a summary with no specific detail |
| contradiction | summary and label disagreeing |

The acceptance test is that the tool flags **exactly** those — no more, no
fewer.

### Tests that must exist

Each pins a mistake that is easy to make and hard to see:

1. Unscored never counts as a pass
2. A judge never receives a label — asserted at the type level
3. A rubber-stamp judge is excluded from panel aggregates, but its verdicts are still written
4. Unreadable replies are counted, never defaulted
5. A metric below its sample floor is not reported at all
6. `analyse` makes zero model calls
7. Editing a taxonomy without bumping the version is an error
8. Item-grain pass rate is never higher than field-grain pass rate
9. Two runs on different taxonomy versions refuse to compare

### Dependencies

```
   pyyaml          config
   httpx           provider calls
   numpy           stats

   optional:
     pandas / polars    dataframe input
     pyarrow            parquet input
```

---

## 13. v1 — what we are not building yet

Each has a reason to wait and a seam already in place, so none requires a
rewrite.

### Copied fields — the third kind

Values that live *in* the item: amounts, dates, names, IDs.

```python
COPIED = FieldKind(
    name="copied",
    checks=[value_in_source, span_matches, ...],
    judge_task=GroundednessTask(),
    metrics=[grounding_rate],
)
```

The engine, gates, cache, panel and report do not change. This is the test of
whether the seam in section 4 is in the right place.

**Why later:** the work here is assigned and free text. Copied is the
well-trodden case and other tools cover it.

### Derived thresholds

Every threshold is currently a default you can change by hand. The honest
version derives it from labelled data at a target precision:

```
   "we want to be right 90% of the time when we flag something"
   → sweep, find the cut, report what recall it costs
```

**Why later:** it needs Mode 1 data. The `threshold_from` field is already on
every finding, so it slots in.

### A/B between two prompt versions, with significance

M7 gives a plain comparison. The full version adds an exact test over the items
where the two runs actually differ, both win rates (including and excluding
ties, conservative first), and a pairwise judge that grades each item **twice
with the order swapped**, scoring a tie when the verdict flips.

Most items tie in a real A/B. Three net wins over 200 items is a coin flip, and
quoting a win rate without the test is how underpowered changes get shipped.

**Why later:** you need two real runs before it is worth anything.

### Multi-label

One item, two labels. Precision and recall become set operations, the confusion
matrix becomes a co-occurrence matrix, tree scoring gets harder.

**Why later:** v0 is scoped to single label. `Output.value` becoming a list is
contained; the metrics are the real work.

### Reasoning-model judges

More accurate and better at ranking, but far more expensive, and they return
empty replies when thinking eats the token budget — and those empties land on
the hard items.

**Why later:** cost. When it lands, it is opt-in per judge and ships with the
guardrail that checks *which* items came back empty.

### Cache at scale

`verdicts.jsonl` is fine to a few hundred thousand rows. Beyond that it wants
SQLite with an index on the cache key. The read/write path is behind one
interface, so it is a swap, not a rewrite.

### Warehouse in and out

Reading items from a warehouse, writing findings back as tables. Files cover the
first year; readers are already pluggable.

### Multi-annotator workflow

Assigning items to annotators, tracking who did what, adjudicating
disagreements. Mode 2 only needs the *data* — two labels with annotator IDs,
which `labels.jsonl` already carries. Producing that data is a different
product.

### Deliberately never

| Not doing | Why |
|---|---|
| Generating the labels | This tool checks; it does not classify |
| Prompt management | Separate concern, mature tools exist |
| A dashboard | Files feed whatever you already use |
| Streaming / per-item | Batch by design; wrap it if you need to |
| Pydantic runtime decorators | A decorator validating one call inline is a **guardrail library** — a different product sharing almost no machinery with a batch quality job. Doing both is how you do both badly. |
| DataFrame accessor methods | Dataframes are already accepted as input. A `.llm_expectations` accessor is sugar over a function call. |
| A fluent assertion DSL | Chained matchers are per-item assertions; this is a batch job whose output is a ranked corpus and a metrics table. Forcing one shape into the other makes both worse, and it would stand up a second config surface beside YAML. The matcher **names** were worth borrowing — check ids are now short verb phrases, since `expect_` is noise in a library called expectations. The chaining was not. |
