"""Loading sessions into a Batch.

One seam, deliberately: `load_sessions()` reads the bundled synthetic corpus by
default and any JSONL you point it at otherwise. Nothing else in the example —
not the suite, not the calibration, not the expectations — knows or cares which
it got. That is what makes "prove it on synthetic, then run it on real data"
a path argument rather than a rewrite.

Real session logs do not belong in a public repository. Keep them outside the
tree (or under a gitignored path) and pass the path in.
"""

from __future__ import annotations

import json
from pathlib import Path

from llmex import Batch, ExtractionRecord, SourceDoc

HERE = Path(__file__).parent
SYNTHETIC = HERE / "sessions.jsonl"

FIELDS = ("jtbd_label", "outcome")
"""Two extracted fields per session, so document grain means something.

With a single field per document the two grains are arithmetically identical
and the gap this framework exists to surface cannot appear.
"""


def load_rows(path: str | Path | None = None) -> list[dict]:
    source = Path(path) if path else SYNTHETIC
    return [json.loads(line) for line in source.read_text().splitlines() if line.strip()]


VARIANTS = {"v4": "", "v5": "_v5"}
"""Two agent variants over the same sessions, for the A/B arm.

v5 fixes three of v4's six misclassifications and introduces one new one. That
is a real improvement and — over 24 sessions — nowhere near enough evidence to
say so, which is the point the comparison makes.
"""


def load_sessions(
    path: str | Path | None = None,
    with_gold: bool = True,
    variant: str = "v4",
    prompt_version: str | None = None,
) -> Batch:
    """A Batch of session extractions.

    `with_gold=False` simulates production, where nobody has labelled anything.
    The same suite runs either way — checks that need gold report themselves
    unscored rather than passing, which is how you can tell the difference.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; known: {sorted(VARIANTS)}")
    suffix = VARIANTS[variant]
    rows = load_rows(path)
    texts = {r["doc_id"]: r["text"] for r in rows}
    records: list[ExtractionRecord] = []

    for row in rows:
        for field_name in FIELDS:
            meta = {}
            if with_gold:
                # Gold rides on the record itself. It never reaches a judge:
                # ScorePayload has no field to carry it.
                meta["gold_label"] = row[f"gold_{_key(field_name)}"]
            records.append(
                ExtractionRecord(
                    doc_id=row["doc_id"],
                    field_name=field_name,
                    value=row[f"pred_{_key(field_name)}{suffix}"],
                    prompt_version=prompt_version or f"jtbd-classify@{variant}",
                    generator_model=f"agent-{variant}",
                    meta=meta,
                )
            )

    return Batch(
        records,
        source_resolver=lambda d: SourceDoc(d, texts[d]),
        schema={f: "string" for f in FIELDS},
    )


def _key(field_name: str) -> str:
    return "jtbd" if field_name == "jtbd_label" else field_name
