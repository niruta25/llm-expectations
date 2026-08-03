"""The JTBD taxonomy, in one place.

>>> REPLACE THIS WITH YOUR REAL CATEGORIES. <<<

Everything downstream — the synthetic corpus, the judge's system prompt, the
calibration, the distribution baseline — reads the taxonomy from here, so
swapping it is a single edit rather than a search across the example.

A Job To Be Done is what the user came to accomplish, not what they said or
what the agent did about it. `LABELS` are stable machine ids; `JOB_STATEMENTS`
are the human phrasings a judge is shown, because "regain access to my locked
account" gives a model far more to work with than `password_reset` does.
"""

from __future__ import annotations

LABELS = [
    "password_reset",
    "billing_dispute",
    "cancel_subscription",
    "data_export",
    "integration_setup",
    "bug_report",
]

JOB_STATEMENTS = {
    "password_reset": "regain access to an account they are locked out of",
    "billing_dispute": "get an unexpected or incorrect charge explained or reversed",
    "cancel_subscription": "stop paying and leave the product",
    "data_export": "get their own data out of the product",
    "integration_setup": "connect the product to another system they already use",
    "bug_report": "report something that is broken and get it fixed",
}

RUBRIC = """A Job To Be Done is what the user came to accomplish, not the
surface topic of the conversation and not what the agent ended up doing.

Judge against these jobs:
""" + "\n".join(f"  {k}: {v}" for k, v in JOB_STATEMENTS.items()) + """

Two rules that decide most of the hard cases:

  - A user who mentions billing while trying to leave is doing
    `cancel_subscription`, not `billing_dispute`. The job is the outcome they
    want, not the topic they raised.
  - A user who cannot do something because the product is broken is doing
    `bug_report` only if they want it fixed; if they want the underlying task
    done some other way, label the underlying task.

If the transcript genuinely does not say what the user wanted, score near 0.5.
An unlabelled ambiguity is more useful than a confident guess."""

# Share of each label in a healthy corpus, for the drift check. Derived from
# the synthetic corpus below; replace with your production mix.
BASELINE_DISTRIBUTION = {
    "password reset": 0.25,
    "billing dispute": 0.167,
    "cancel subscription": 0.167,
    "data export": 0.167,
    "integration setup": 0.125,
    "bug report": 0.125,
}
