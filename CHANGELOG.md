# Changelog

## 0.1.0

The first version that does anything. Every milestone in DESIGN.md §14 is
built, and all three kinds of field with it: free checks on each, a triage
judge, a panel, both gates, label-backed metrics, a fitted calibration, Error
Recall@Budget, and comparison across runs.

### What it does

- **Free checks, on every row, before anything is spent.** Label validity and
  leaf depth, abstention rate, label collapse, drift against the previous run,
  cross-field agreement; for free text: length, specificity, copy ratio and
  boilerplate; for copied values: shape, presence in the source document, and
  the grounding rate.
- **A gate on the expensive judge**, for the two kinds whose free checks are
  predictive enough to decide what gets paid for — plus an audit sample that
  measures what the gate misses rather than assuming it misses nothing. How
  much it saves is a property of your data, so the breakdown is printed.
- **Two jobs for judges, wired separately.** One ranks what a human should
  open first; a panel of two or more measures quality over a sample, reports
  what its votes are actually worth, and locates the label boundaries it keeps
  splitting on.
- **Two gates before any quality number.** Can the measurement be trusted, and
  does the judge beat random, output length, majority label and panel
  disagreement on the same rows with an interval that clears the best of them.
- **With labels:** macro F1 beside the baseline it has to beat, a confusion
  matrix, four tree buckets saying *which kind* of wrong, judge direction, and
  whether a stated confidence of 0.9 means 90%.
- **With two annotators:** whether two people can separate two labels at all.
- **A fitted calibration**, per field, with its quality measured out of fold.
- **Error Recall@Budget** for the judge and every baseline at once.
- **`compare`** across two runs, which refuses to say which is better.

### What it refuses to do

Each of these is a deliberate absence, and most were harder to build than the
affirmative version would have been.

- A guardrail that fires withholds the numbers it invalidates and names them,
  rather than printing them with a caveat.
- Unscored never becomes a pass. An unreadable judge reply is counted, never
  defaulted.
- No ranking is called validated without labels and a target large enough to
  rank on. A metric below its sample floor is printed with its interval and a
  line saying it cannot support a conclusion.
- `compare` shows improved, regressed and unchanged, and withholds the
  verdict: most items tie in a real A/B, and a net delta without a test is how
  underpowered changes get shipped. The test is v1.
- A calibration fitted against one prompt version refuses to be reused against
  another. Two runs on different taxonomy versions refuse to compare without a
  migration mapping.
- A judge never sees a human label. The context type that reaches a triage
  strategy has no field for one, and an assertion confirms it at runtime.

### Values copied out of the document

The third kind of field, and the test of whether the seam in DESIGN.md §4 was
in the right place. Mostly it was: the checks and the judge task are new
files that register against the kind.

**Absence is proof; presence is only evidence.** A value the document does
not contain was invented, and establishing that costs nothing — it is the one
defect any kind of field can prove for free. What the same check cannot
establish is that a value it *found* is the right one, because a document
listing a subtotal, a carriage charge and a total contains the number you
extracted whichever of the three you meant. So `value_in_source` carries how
many candidates the document held, and the gate sends ambiguous rows to a
judge even though they passed.

**Matching is by meaning.** `$1,234.50` and `1234.5` are one amount;
`March 3, 2026` and `2026-03-03` are one day. `require_verbatim` is there for
invoice numbers and SKUs, where the characters really do matter. A date in a
shape the library cannot read safely — `04/05/2026` — is reported unparseable
rather than guessed at, and is unscored rather than wrong in every number
downstream.

### Three places the implementation corrected the design

DESIGN.md has been updated in each case, with the reasoning kept inline.

- **`calibrated_error_probability` is P(the output is wrong)**, not P(the
  verdict is wrong). The latter reading sorts a judge's most confident
  rejections to the bottom of the review queue.
- **A single Platt curve cannot improve a ranking**, because it is monotone.
  The calibration is fitted per field, which can reorder; when only one field
  has enough rows, the report says the order is unchanged rather than letting
  a reader infer a win.
- **A yes/no verdict cannot name a fuzzy pair.** A rejecting judge is now
  asked which label it would assign instead, which makes the free-tier
  fuzzy-pair detector real rather than inferred from sibling structure.

### Five things the worked projects found

Building one realistic project per kind of field turned up five defects that
the thirteen-item fixture was too small to show.

- **`plan` never counted the panel.** On a three-judge panel that is not a
  rounding error; the estimate was a third of the bill. Every job the
  collector performs is costed now, each on its own line.
- **`plan` and `run` disagreed on free-text and copied projects.** The CLI's
  `plan` skipped the free checks, so it never knew which rows the gate would
  send. It runs them now — they cost nothing — and quotes the same number
  the run bills for.
- **The gate's own audit was lost on re-analysis.** Which rows were flagged
  and which were sampled lived in verdict metadata, which does not survive
  the cache, so `analyse` reported a measured gate as a gate nobody measured.
  It is recomputed from the plan, which is deterministic from the data.
- **A free-text project could never build a ranking target.** A free-text
  label is a defect rating, not a gold value, so it was excluded from the
  target entirely — correctly refusing to compare a box name against a
  sentence, and thereby refusing to rank the one field the project was about.
  "The rater ticked a box" is a perfectly good definition of an error, and it
  is one now.
- **A copied value was compared to its human answer character by character**,
  so a model that tidied `1234.5` into `$1,234.50` was marked wrong. One
  definition of "the same value" is now shared by the free check and the
  scoring.

### Running it without an API key

`provider: fake` answers from a hash of the prompt, reaches no network and
costs nothing, so every project in `examples/` has a `run-offline.yml` that
runs for anyone. Everything upstream of the judge is real; Gate 1 stops the
run and says every judge-backed number in it is invented.

### Worked projects

One per kind of field, each with its data, its config and the deterministic
script that built the corpus:

- `examples/ticket-routing` — **assigned**, 320 support tickets into nine queues
- `examples/release-notes` — **free_text**, 260 pull requests into one line each
- `examples/invoice-extraction` — **copied**, 300 supplier invoices
- `examples/jtbd` — the fixture: thirteen items, eight planted defects, and a
  manifest naming the check that must catch each

### Requires

Python 3.10+, `pyyaml`, `httpx`, `numpy`.
