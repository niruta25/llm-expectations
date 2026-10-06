#!/usr/bin/env python3
"""Build the release-notes corpus. Deterministic — rerunning changes nothing.

Each item is a merged pull request: title, body, and the files it touched.
Each output is the one-line customer-facing note a model wrote from it, plus
the product area it tagged.

The planted defects are the ways a written sentence goes wrong, in roughly
the mix a real note writer produces them:

    made up      a claim the pull request does not support  (judge only)
    too generic  a line that would fit any release          (free)
    contradicts  a note about something other than its area (free)
    pasted       the commit message, lightly reflowed       (free)
    wrong length a paragraph where a line was asked for     (free)

Four of the five are free to find. The first is not, and it is the one that
reaches a customer, which is the whole argument for paying a judge at all.

Note the two phrasings of every symptom, trigger and capability. The pull
request says "when the file is larger than 50,000 rows"; the note says "on
very large files". That gap is not decoration — a corpus where the note
repeated the commit message word for word would make `copy_ratio` fire on
every honest row, and the check would be measuring the fixture rather than
the model.

    python examples/release-notes/generate.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).parent
SEED = 20260214
N_ITEMS = 260
N_RATED = 210

AREAS = ["importer", "reports", "billing", "api", "mobile", "admin"]

COMPONENTS = {
    "importer": ["CSV reader", "column mapper", "row validator", "staging queue",
                 "duplicate detector", "encoding sniffer"],
    "reports": ["usage report", "revenue chart", "export scheduler", "pivot builder",
                "date-range picker", "saved view"],
    "billing": ["proration engine", "invoice renderer", "dunning job", "tax resolver",
                "credit note flow", "seat counter"],
    "api": ["records endpoint", "rate limiter", "pagination cursor", "webhook signer",
            "token exchange", "bulk writer"],
    "mobile": ["offline cache", "push registration", "photo uploader", "sync indicator",
               "deep link handler", "session restore"],
    "admin": ["role editor", "audit log", "invite flow", "SSO settings",
              "retention policy", "bulk deactivate"],
}

#: (how the pull request puts it, how a note puts it).
SYMPTOMS = {
    "importer": [
        ("silently drop the last row", "skips rows"),
        ("reject the whole file", "refuses a whole file"),
        ("mangle accented characters", "garbles accents"),
        ("stop at the first blank line", "stops early"),
    ],
    "reports": [
        ("show yesterday's figure", "lags a day behind"),
        ("lose the sort order", "re-sorts itself"),
        ("round the wrong way", "rounds badly"),
        ("leave the chart empty", "comes back blank"),
    ],
    "billing": [
        ("double-count refunded seats", "over-counts seats"),
        ("charge the old plan", "bills the previous plan"),
        ("skip the tax line", "omits tax"),
        ("round the proration down", "under-prorates"),
    ],
    "api": [
        ("return an empty page", "answers with nothing"),
        ("time out", "hangs"),
        ("re-send the same event", "duplicates events"),
        ("drop the cursor", "loses its place"),
    ],
    "mobile": [
        ("lose unsent changes", "discards a draft"),
        ("spin without finishing", "hangs mid-sync"),
        ("show a stale badge count", "shows the wrong badge"),
        ("sign the user out", "logs you out"),
    ],
    "admin": [
        ("grant the wrong role", "assigns the wrong permissions"),
        ("omit an entry from the log", "misses an entry"),
        ("send the invite twice", "doubles up invitations"),
        ("ignore the retention setting", "keeps data too long"),
    ],
}

TRIGGERS = {
    "importer": [
        ("the file is larger than 50,000 rows", "very large files"),
        ("a column header repeats", "duplicated headers"),
        ("the file is saved as UTF-16", "unusual encodings"),
        ("the first row is blank", "a blank first line"),
    ],
    "reports": [
        ("the window crosses a month boundary", "ranges spanning two months"),
        ("two reports share a saved view", "shared views"),
        ("the account has more than one currency", "multi-currency accounts"),
        ("a filter matches nothing", "empty filters"),
    ],
    "billing": [
        ("a plan changes mid-cycle", "mid-cycle plan changes"),
        ("the account has more than one currency", "multi-currency accounts"),
        ("a credit note is issued the same day", "same-day credit notes"),
        ("seats drop to zero", "accounts emptied of seats"),
    ],
    "api": [
        ("a retry arrives out of order", "out-of-order retries"),
        ("the request carries no page cursor", "uncursored requests"),
        ("the token is refreshed mid-request", "a token refresh"),
        ("more than 500 records are written at once", "large bulk writes"),
    ],
    "mobile": [
        ("the device is offline", "patchy connections"),
        ("the app is backgrounded during a sync", "an interrupted sync"),
        ("two people edit at once", "simultaneous edits"),
        ("the photo is larger than 20 MB", "very large photos"),
    ],
    "admin": [
        ("two administrators edit at once", "simultaneous edits"),
        ("the group is nested", "nested groups"),
        ("the user has never signed in", "people who never signed in"),
        ("the policy was set before the upgrade", "older policies"),
    ],
}

CAPABILITIES = {
    "importer": [
        ("undo an import in a single step", "undo an import"),
        ("retry one failed row rather than the whole file", "retry a single row"),
        ("preview the first hundred rows before committing", "preview before committing"),
        ("map a column by hand when detection fails", "map a column yourself"),
    ],
    "reports": [
        ("schedule a monthly send to a mailing list", "schedule a monthly send"),
        ("filter by tag across saved views", "filter by tag"),
        ("compare a range against the previous period", "compare two periods"),
        ("pin a view to the sidebar for everyone", "pin a view"),
    ],
    "billing": [
        ("search invoices by their number", "search by invoice number"),
        ("split one charge across two cards", "split a charge"),
        ("download a year of invoices in one archive", "download a year at once"),
        ("name a separate billing contact", "set a billing contact"),
    ],
    "api": [
        ("page backwards through a result set", "page backwards"),
        ("request a partial response with a field mask", "ask for fewer fields"),
        ("replay a failed delivery from the dashboard", "replay a failed delivery"),
        ("scope a token to a single project", "scope a token"),
    ],
    "mobile": [
        ("work offline for a full day before syncing", "work offline for a day"),
        ("scan a barcode instead of typing a code", "scan a barcode"),
        ("switch accounts without signing out", "switch accounts"),
        ("keep a draft on the device when the network drops", "keep a draft locally"),
    ],
    "admin": [
        ("copy a role from one person to another", "copy a role"),
        ("require two-factor for a single group", "require two-factor per group"),
        ("see who exported what, and when", "see who exported what"),
        ("schedule a deactivation for a future date", "schedule a deactivation"),
    ],
}

CAUSES = [
    "an off-by-one in the cursor", "a stale read from the replica",
    "an unescaped delimiter", "a timezone applied twice",
    "a shared mutable default", "a rounding step in the wrong order",
]

#: (kind, pull-request body, the notes a careful writer produces from it).
#:
#: Three phrasings per shape, because a real writer does not reach for the
#: same sentence every time. One phrasing per shape would make most of the
#: corpus near-duplicates of each other, `boilerplate` would fire on the
#: whole thing, and the only honest reading of that finding would be "the
#: fixture has four templates in it".
SHAPES = [
    (
        "fix",
        "Fixes an issue where the {component} would {symptom} when {trigger}. "
        "Root cause was {cause}. Added a regression test covering {trigger}.",
        (
            "The {component} no longer {short_symptom} on {short_trigger}.",
            "We fixed the {component}, which {short_symptom} whenever you had "
            "{short_trigger}.",
            "On {short_trigger}, the {component} would sometimes {short_symptom}. "
            "It does not any more.",
        ),
    ),
    (
        "fix",
        "Since the {version} release the {component} would {symptom} whenever "
        "{trigger}. This restores the previous behaviour and leaves the {cause} "
        "path alone.",
        (
            "Fixed a regression in the {component} affecting {short_trigger}, "
            "introduced back in {version}.",
            "The {component} has behaved oddly with {short_trigger} since {version}. "
            "That is now back to normal.",
            "A {version} regression in the {component} is fixed: {short_trigger} "
            "work the way they used to.",
        ),
    ),
    (
        "feature",
        "Adds the ability to {capability}, built on the {component}. Opt-in behind "
        "the existing account setting; no change for anyone who leaves it off.",
        (
            "The {component} can now {short_capability}. It stays off until you "
            "switch it on in your account settings.",
            "New in the {component}: {short_capability}. Turn it on in your account "
            "settings whenever you are ready.",
            "You can {short_capability} with the {component} now. It is opt-in, so "
            "nothing changes until you ask for it.",
        ),
    ),
    (
        "feature",
        "Extends the {component} so you can {capability} without leaving the page. "
        "Touches {files} files; the underlying {cause} is unchanged.",
        (
            "You can {short_capability} straight from the {component} now, without "
            "a round trip through another screen.",
            "The {component} grew a way to {short_capability} in place, so there is "
            "no second screen to visit.",
            "No more hopping between screens: {short_capability} right there in the "
            "{component}.",
        ),
    ),
    (
        "perf",
        "Reduces work in the {component} by caching {cause} between calls. On the "
        "{version} fixture this takes it from {before} to {after} when {trigger}.",
        (
            "The {component} is quicker on {short_trigger} — roughly {before} down "
            "to {after} against our own test data.",
            "{short_trigger} got faster in the {component}: about {before} before, "
            "about {after} now, on our measurements.",
            "We cut the {component} from around {before} to around {after} on "
            "{short_trigger}.",
        ),
    ),
    (
        "chore",
        "Upgrades the {component} to drop a deprecated dependency. No behaviour "
        "change intended; {files} files touched, all internal.",
        (
            "Behind-the-scenes upgrade to the {component}. You should not notice "
            "any difference in how it works.",
            "Housekeeping in the {component} — a dependency upgrade, with no change "
            "you can see from outside.",
            "We upgraded some internals behind the {component}. Nothing you do with "
            "it should change.",
        ),
    ),
]


#: The generic lines. Each is true of every release ever shipped, and none of
#: them reaches for anything in its own pull request.
GENERIC = [
    "Various improvements and bug fixes have landed in this part of the product.",
    "We made a number of updates to make things more reliable and a little faster.",
    "This release includes several small enhancements and a handful of corrections.",
    "General stability work across the product, with no change to how anything looks.",
    "Several long-standing issues have been addressed in this update.",
]

#: The claims nothing in the pull request supports. Specific, plausible, the
#: right length, and about something that did not happen.
INVENTED = [
    "It is also about three times faster for everyone.",
    "This closes a security issue reported by an external researcher.",
    "There is no longer any upper limit on the number of rows.",
    "It settles the long-standing complaint about how this is priced.",
    "The old behaviour has been removed for every customer today.",
    "The {component} is now certified against the new audit requirements.",
]


def _slots(rng: random.Random, area: str) -> dict[str, str]:
    before = rng.randrange(4, 40)
    symptom, short_symptom = rng.choice(SYMPTOMS[area])
    trigger, short_trigger = rng.choice(TRIGGERS[area])
    capability, short_capability = rng.choice(CAPABILITIES[area])
    return {
        "component": rng.choice(COMPONENTS[area]),
        "symptom": symptom,
        "short_symptom": short_symptom,
        "trigger": trigger,
        "short_trigger": short_trigger,
        "capability": capability,
        "short_capability": short_capability,
        "cause": rng.choice(CAUSES),
        "version": f"{rng.randrange(3, 9)}.{rng.randrange(0, 12)}",
        "files": str(rng.randrange(2, 40)),
        "before": f"{before}s",
        "after": f"{max(1, before // rng.randrange(2, 6))}s",
    }


def build() -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(SEED)  # noqa: S311 — a fixture, not cryptography
    items: list[dict] = []
    outputs: list[dict] = []
    planted: dict[str, str] = {}

    for index in range(1, N_ITEMS + 1):
        item_id = f"pr-{index:03d}"
        area = AREAS[index % len(AREAS)]
        kind, body_template, note_templates = rng.choice(SHAPES)
        slots = _slots(rng, area)
        body = body_template.format(**slots)
        aspect = rng.choice(["handling", "behaviour", "path", "flow"])
        items.append(
            {
                "id": item_id,
                "text": f"{kind}({area}): {slots['component']} {aspect}\n\n{body}",
                "kind": kind,
                "files_changed": int(slots["files"]),
                "merged": f"2026-0{rng.randrange(1, 10)}-{rng.randrange(10, 29)}",
            }
        )

        note = rng.choice(note_templates).format(**slots)
        tagged = area
        defect = "none"

        roll = rng.random()
        if roll < 0.09:
            # Made up. Everything free passes: it is specific, the right
            # length, agrees with its area, and is not pasted.
            note = f"{note} {rng.choice(INVENTED).format(**slots)}"
            defect = "made_up"
        elif roll < 0.16:
            note = rng.choice(GENERIC)
            defect = "too_generic"
        elif roll < 0.20:
            # The note is fine; the area tag is not. Free to catch, and it
            # fires without knowing which of the two is wrong.
            tagged = rng.choice([a for a in AREAS if a != area])
            defect = "contradicts"
        elif roll < 0.24:
            note = body                                    # the commit message, pasted
            defect = "pasted"
        elif roll < 0.265:
            note = " ".join([body, note] * 3)              # a changelog, not a line
            defect = "too_long"
        elif roll < 0.285:
            note = f"Fixed the {slots['component']}."
            defect = "too_short"

        outputs.append(
            {
                "item_id": item_id,
                "area": tagged,
                "note": note,
                "confidence": round(min(0.99, max(0.4, rng.gauss(0.86, 0.08))), 2),
            }
        )
        planted[item_id] = defect

    # Humans rate the notes for defects rather than rewriting them. There is
    # no single right release note, so asking someone to write a better one
    # would score word choice; asking which box to tick is a question two
    # people can answer the same way.
    #
    # They are not perfect raters, and that is deliberate. A human who agrees
    # with the tool by construction measures nothing, and the expected result
    # — that *missing something* scores worst, because it is hard for the
    # judge and hard for the person — only appears if the people can miss it.
    rated = sorted(rng.sample([i["id"] for i in items], N_RATED))
    labels: list[dict] = []
    for item_id in rated:
        defect = planted[item_id]
        human = defect if defect in {"made_up", "too_generic", "contradicts"} else "none"
        if defect == "made_up" and rng.random() < 0.25:
            human = "none"              # the invented claim read as plausible
        elif defect == "none" and rng.random() < 0.05:
            human = "missing"           # they wanted something the note left out
        labels.append(
            {"item_id": item_id, "field": "note", "label": human, "annotator": "ann-1"}
        )
        labels.append(
            {
                "item_id": item_id,
                "field": "area",
                "label": AREAS[int(item_id.split("-")[1]) % len(AREAS)],
                "annotator": "ann-1",
            }
        )

    return items, outputs, labels


def write(rows: list[dict], name: str) -> None:
    path = HERE / name
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    print(f"{path.relative_to(HERE.parent.parent)}: {len(rows)} rows")


if __name__ == "__main__":
    items, outputs, labels = build()
    write(items, "items.jsonl")
    write(outputs, "outputs.jsonl")
    write(labels, "labels.jsonl")
