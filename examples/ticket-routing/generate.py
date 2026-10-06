#!/usr/bin/env python3
"""Build the ticket-routing corpus. Deterministic — rerunning changes nothing.

The tickets are synthetic, and the README says so. What is *not* synthetic is
the shape of the error: the planted mistakes are the ones a real router makes,
in roughly the proportions a real router makes them, and they are planted
without ever telling the tool where they are.

    python examples/ticket-routing/generate.py

Writes items.jsonl, outputs.jsonl and labels.jsonl beside this file.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).parent
SEED = 20260105
N_ITEMS = 320
N_LABELLED = 280
N_SECOND_OPINION = 70

NAMES = [
    "Maya", "Devan", "Priya", "Tomas", "Aoife", "Rafael", "Hana", "Oskar",
    "Noor", "Jonas", "Lena", "Mateo", "Ingrid", "Kwame", "Sofia", "Ravi",
    "Elin", "Hugo", "Yuki", "Nadia", "Pietro", "Anja", "Caleb", "Freya",
    "Dmitri", "Imani", "Lars", "Mira", "Sanjay", "Tove", "Bruno", "Zara",
    "Emeka", "Clara", "Oliver", "Rin", "Sven", "Thandi", "Viktor", "Wren",
]
COMPANIES = [
    "Northwind Freight", "Belmont Dental", "Arcadia Labs", "Pellet & Sons",
    "Brightline Tutors", "Quay Street Books", "Harlow Logistics", "Volta Energy",
    "Marsh & Finch", "Cedarhouse Clinic", "Orbit Travel", "Pinewood Studios",
    "Salter Robotics", "Tidewater Marine", "Umber Design", "Verso Publishing",
    "Westgate Legal", "Yarrow Nutrition", "Zephyr Textiles", "Anvil Fabrication",
    "Bramble Coffee", "Copperfield Events", "Dunmore Properties", "Eastcote Vets",
    "Fenwick Surveying", "Granite Peak Gear", "Hollis Insurance", "Ironbark Timber",
]


def _slots(rng: random.Random, index: int) -> dict[str, str]:
    return {
        "name": rng.choice(NAMES),
        "company": rng.choice(COMPANIES),
        "ref": f"{rng.choice('ABCDEFGH')}{rng.randrange(1000, 9999)}",
        "amount": f"{rng.randrange(40, 9000)}.{rng.randrange(0, 100):02d}",
        "day": str(rng.randrange(1, 28)),
        "month": rng.choice(
            ["January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"]
        ),
        "count": str(rng.randrange(2, 400)),
        "code": f"{rng.randrange(400, 600)}",
        "seat": str(rng.randrange(3, 90)),
        "index": str(index),
    }


#: leaf -> (severity, template). The severity is what the ticket actually is,
#: which is what a human annotator would mark; the model's answer is derived
#: from it and then corrupted at the rates below.
SCENARIOS: dict[str, list[tuple[str, str]]] = {
    "billing.invoicing": [
        ("question",
         "{name} at {company} says the {month} invoice never arrived in their inbox. "
         "Their finance team needs it before the {day}th to get it into this quarter. "
         "Purchase order {ref} is already approved on their side."),
        ("question",
         "{company} received invoice {ref} addressed to a legal entity they closed last "
         "year. {name} is asking for it to be reissued to the new entity, same amount, "
         "${amount}."),
        ("degraded",
         "{name} reports that the invoice PDF for {ref} shows {count} seats but their "
         "contract says {seat}. Nothing has been charged yet — they want the invoice "
         "corrected before the run on the {day}th."),
    ],
    "billing.payment_failure": [
        ("blocker",
         "The renewal charge of ${amount} for {company} did not go through on the "
         "{day}th. {name} says nothing changed on their side and the account is now "
         "showing as past due."),
        ("degraded",
         "{name} at {company} saw three failed charge attempts against reference {ref} "
         "overnight. The card has funds; the attempts simply never completed."),
        ("blocker",
         "Autopay for {company} has not collected since {month}. {name} only noticed "
         "when the account dropped to read-only. Outstanding is ${amount}."),
    ],
    "billing.refunds": [
        ("question",
         "{company} was billed twice for {month} after {name} changed plans on the "
         "{day}th. They have asked for the duplicate ${amount} charge to be returned."),
        ("question",
         "{name} cancelled on the {day}th but the annual charge of ${amount} had already "
         "gone out under {ref}. They are asking for the unused portion back."),
        ("question",
         "A trial account at {company} converted by mistake and {name} is asking us to "
         "reverse the ${amount} charge. No usage at all since the conversion."),
    ],
    "access.sso": [
        ("blocker",
         "Single sign-on for {company} started returning an error after their identity "
         "provider rotated certificates on the {day}th. {name} says all {seat} of their "
         "users are locked out."),
        ("blocker",
         "{name} reports the SAML assertion from their Okta tenant is being rejected with "
         "error {code}. This began after they renamed the application in Okta."),
        ("degraded",
         "Users at {company} who signed in before {month} are fine, but anyone new gets "
         "bounced back to the identity provider in a loop. {name} has {count} new starters "
         "waiting."),
    ],
    "access.account_recovery": [
        ("degraded",
         "{name} at {company} cannot sign in and says the password reset email never "
         "arrives. We can see it bouncing off their mail server."),
        ("blocker",
         "The only administrator at {company} has left and {name} cannot get into the "
         "account at all. They have the invoice under {ref} as proof of ownership."),
        ("degraded",
         "{name} is stuck in a reset loop — the link in the email says it has already "
         "been used, every time, across {count} attempts since the {day}th."),
    ],
    "data.import_export": [
        ("degraded",
         "{company}'s nightly import of {count} rows failed on the {day}th with a column "
         "mismatch. {name} says the file format has not changed in months."),
        ("question",
         "{name} is trying to export the {month} ledger for {company} and the download "
         "stops at around {count} rows every time."),
        ("blocker",
         "The bulk upload for {company} has been rejecting every file since the {day}th "
         "with error {code}. {name} has {count} records queued and nothing is landing."),
    ],
    "data.reporting": [
        ("degraded",
         "The {month} usage report for {company} shows {count} active seats where the "
         "admin console shows {seat}. {name} needs the two to agree before their board "
         "meeting."),
        ("question",
         "{name} asks whether the revenue figure on the dashboard for {company} includes "
         "the ${amount} credit note issued in {month}."),
        ("degraded",
         "Scheduled reports for {company} stopped arriving after the {day}th. {name} can "
         "still generate them by hand, so the data is there — the schedule is not firing."),
    ],
    "integrations.api_errors": [
        ("blocker",
         "{company} is getting error {code} on every call to the records endpoint since "
         "the {day}th. {name} says their client and credentials are unchanged, and {count} "
         "jobs have now backed up."),
        ("degraded",
         "{name} reports intermittent {code} responses from our API — roughly {count} in "
         "every thousand calls — starting around the {day}th of {month}."),
        ("blocker",
         "Every authenticated request from {company} is coming back {code} after they "
         "rotated their key. {name} has confirmed the new key is in the header."),
    ],
    "integrations.webhooks": [
        ("degraded",
         "{company} stopped receiving delivery callbacks on the {day}th. {name} can see "
         "{count} events listed as sent on our side and nothing arriving at their endpoint."),
        ("blocker",
         "Every webhook to {company} is being retried and failing with {code}. {name} says "
         "their endpoint is up and answering other traffic normally."),
        ("question",
         "{name} wants to know why {company} received the same event {count} times on the "
         "{day}th, all with the same reference {ref}."),
    ],
}

LEAVES = list(SCENARIOS)
PARENT = {leaf: leaf.split(".")[0] for leaf in LEAVES}
SIBLINGS = {
    leaf: [other for other in LEAVES if PARENT[other] == PARENT[leaf] and other != leaf]
    for leaf in LEAVES
}

#: The one boundary this taxonomy cannot hold: a charge that failed and an
#: invoice that is wrong both arrive as "the money is not right". Annotators
#: split on it too, which is the evidence that it is the taxonomy's fault.
FUZZY = ("billing.payment_failure", "billing.invoicing")

INVENTED = [
    "access.login_broken",
    "billing.chargeback",
    "integrations.rate_limit",
    "data.sync_error",
]


def build() -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(SEED)  # noqa: S311 — a fixture, not cryptography
    items: list[dict] = []
    outputs: list[dict] = []
    truth: dict[str, dict[str, str]] = {}

    for index in range(1, N_ITEMS + 1):
        item_id = f"t-{index:03d}"
        leaf = LEAVES[index % len(LEAVES)]
        severity, template = rng.choice(SCENARIOS[leaf])
        text = template.format(**_slots(rng, index))
        items.append(
            {
                "id": item_id,
                "text": text,
                "channel": rng.choice(["email", "chat", "portal"]),
                "received": f"2026-0{rng.randrange(1, 10)}-{rng.randrange(10, 29)}",
            }
        )
        truth[item_id] = {"queue": leaf, "severity": severity}

        # What the router answered. The defect mix below is the point of the
        # fixture: mostly right, wrong in the ways routers are actually wrong.
        roll = rng.random()
        queue = leaf
        if roll < 0.085 and SIBLINGS[leaf]:
            queue = rng.choice(SIBLINGS[leaf])          # a sibling: the hard one
        elif roll < 0.115:
            queue = PARENT[leaf]                        # hedged at the parent
        elif roll < 0.135:
            queue = rng.choice(INVENTED)                # not a label at all
        elif roll < 0.165:
            queue = rng.choice([x for x in LEAVES if PARENT[x] != PARENT[leaf]])
        elif roll < 0.205:
            queue = "abstain"

        sev_roll = rng.random()
        answered_severity = severity
        if sev_roll < 0.12:
            answered_severity = rng.choice([s for s in ("blocker", "degraded", "question")
                                            if s != severity])

        # Confidence is the producing model's own feeling, and it is only
        # loosely related to being right — which is the thing calibration is
        # for. A wrong answer here is often a confident one.
        stated = rng.gauss(0.84 if queue == leaf else 0.76, 0.09)
        confidence = round(min(0.99, max(0.35, stated)), 2)
        outputs.append(
            {
                "item_id": item_id,
                "queue": queue,
                "severity": answered_severity,
                "confidence": confidence,
            }
        )

    labelled = sorted(rng.sample([i["id"] for i in items], N_LABELLED))
    labels: list[dict] = []
    for item_id in labelled:
        for field in ("queue", "severity"):
            labels.append(
                {
                    "item_id": item_id,
                    "field": field,
                    "label": truth[item_id][field],
                    "annotator": "ann-1",
                }
            )

    # A second annotator on a quarter of the labelled set. They mostly agree
    # with ann-1. Where they do not, it is overwhelmingly on the one boundary
    # this taxonomy cannot hold — which is the strongest evidence there is
    # that the boundary, and not the model, is the problem.
    for item_id in sorted(rng.sample(labelled, N_SECOND_OPINION)):
        gold = truth[item_id]["queue"]
        second = gold
        if gold in FUZZY and rng.random() < 0.7:
            second = FUZZY[1] if gold == FUZZY[0] else FUZZY[0]
        elif rng.random() < 0.03 and SIBLINGS[gold]:
            second = rng.choice(SIBLINGS[gold])     # ordinary annotator noise
        labels.append(
            {"item_id": item_id, "field": "queue", "label": second, "annotator": "ann-2"}
        )
        labels.append(
            {
                "item_id": item_id,
                "field": "severity",
                "label": truth[item_id]["severity"],
                "annotator": "ann-2",
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
