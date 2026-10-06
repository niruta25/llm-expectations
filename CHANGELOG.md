# Changelog

## 0.1.0

The first version that does anything. Every milestone in DESIGN.md §13 is
built: free checks on both kinds of field, a triage judge, a panel, both
gates, label-backed metrics, a fitted calibration, Error Recall@Budget, and
comparison across runs.

### What it does

- **Free checks, on every row, before anything is spent.** Label validity and
  leaf depth, abstention rate, label collapse, drift against the previous run,
  cross-field agreement, and for free text: length, specificity, copy ratio
  and boilerplate.
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

### Running it without an API key

`provider: fake` answers from a hash of the prompt, reaches no network and
costs nothing, so `examples/jtbd/run-offline.yml` runs for anyone. Everything
upstream of the judge is real; Gate 1 stops the run and says every
judge-backed number in it is invented.

### Requires

Python 3.10+, `pyyaml`, `httpx`, `numpy`.
