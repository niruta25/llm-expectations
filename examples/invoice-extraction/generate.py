#!/usr/bin/env python3
"""Build the invoice-extraction corpus. Deterministic — rerunning changes nothing.

Each item is a supplier invoice as it arrives: a few lines of text with a
number, a date, and three amounts on it. Each output is what an extraction
model pulled out.

The planted defects are the ones that matter for a copied field, and they
split into two groups that no single check can tell apart:

    invented     a value the document does not contain at all
    misread      a value the document *does* contain — just not that one

The first is free to find and the whole corpus is checked for it. The second
is the one that looks like success: a document listing a subtotal, a shipping
charge and a total contains the number you extracted whichever of the three
you meant. Every tool that stops at "is it in the document" reports those
rows as clean.

So the corpus is built so the two numbers come apart. Roughly 6% of totals
are invented and roughly 11% are the shipping charge or the subtotal, and
the gap between the grounding rate and the match rate is the point of the
whole example.

    python examples/invoice-extraction/generate.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).parent
SEED = 20260331
N_ITEMS = 300
N_LABELLED = 220

SUPPLIERS = [
    "Northwind Freight", "Belmont Dental Supply", "Arcadia Labs",
    "Pellet & Sons Joinery", "Quay Street Books", "Harlow Logistics",
    "Volta Energy", "Marsh & Finch Legal", "Cedarhouse Clinical",
    "Orbit Travel Management", "Pinewood Studios", "Salter Robotics",
    "Tidewater Marine", "Umber Design Co", "Verso Publishing",
    "Westgate Facilities", "Yarrow Nutrition", "Zephyr Textiles",
    "Anvil Fabrication", "Bramble Coffee Roasters", "Copperfield Events",
    "Dunmore Property Services", "Eastcote Veterinary", "Fenwick Surveying",
]

LINE_ITEMS = [
    "consultancy, 12 hours", "replacement filters, box of 40",
    "annual licence renewal", "site visit and report", "courier, next day",
    "printed materials", "equipment hire, 3 days", "storage, one quarter",
    "calibration service", "training, half day", "spare parts kit",
    "monthly retainer",
]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

#: Four layouts, because an extraction model that only works on one layout
#: is a model that has memorised a supplier. Each one puts the date in a
#: different shape and the amounts in a different order.
LAYOUTS = [
    "{supplier}\nInvoice {number}    Issued {iso_date}    Due {iso_due}\n\n"
    "{line} .......... {subtotal}\nShipping .......... {shipping}\n"
    "VAT at 20% .......... {vat}\nTotal due .......... {total}\n",

    "INVOICE {number}\n{supplier}\nDate of issue: {long_date}. Payment due by "
    "{long_due}.\n\nFor {line}: {subtotal}. Carriage {shipping}. Tax {vat}. "
    "Amount payable {total}.\n",

    "{supplier} — statement\nRef {number}, raised {uk_date}\n\n"
    "{line}\n  net        {subtotal}\n  carriage   {shipping}\n"
    "  vat        {vat}\n  ---------------------\n  TOTAL      {total}\n"
    "Settlement due {uk_due}.\n",

    "Remittance advice\nSupplier: {supplier}\nOur reference: {number}\n"
    "Invoice date {iso_date} (due {long_due})\n\n{line} at {subtotal}, plus "
    "{shipping} delivery and {vat} VAT, giving {total} to pay.\n",
]


def _money(value: float) -> str:
    return f"${value:,.2f}"


def build() -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(SEED)  # noqa: S311 — a fixture, not cryptography
    items: list[dict] = []
    outputs: list[dict] = []
    truth: dict[str, dict[str, str]] = {}

    for index in range(1, N_ITEMS + 1):
        item_id = f"inv-{index:03d}"
        supplier = rng.choice(SUPPLIERS)
        number = f"{rng.choice(['INV', 'SI', 'A', 'TX'])}-{rng.randrange(10000, 99999)}"

        subtotal = round(rng.uniform(40, 9000), 2)
        shipping = round(rng.uniform(4, 95), 2)
        vat = round((subtotal + shipping) * 0.2, 2)
        total = round(subtotal + shipping + vat, 2)

        month = rng.randrange(1, 13)
        day = rng.randrange(1, 29)
        due_month = month % 12 + 1
        due_day = rng.randrange(1, 29)
        iso_date = f"2026-{month:02d}-{day:02d}"
        iso_due = f"2026-{due_month:02d}-{due_day:02d}"

        text = rng.choice(LAYOUTS).format(
            supplier=supplier,
            number=number,
            line=rng.choice(LINE_ITEMS),
            subtotal=_money(subtotal),
            shipping=_money(shipping),
            vat=_money(vat),
            total=_money(total),
            iso_date=iso_date,
            iso_due=iso_due,
            long_date=f"{MONTHS[month - 1]} {day}, 2026",
            long_due=f"{MONTHS[due_month - 1]} {due_day}, 2026",
            uk_date=f"{day} {MONTHS[month - 1]} 2026",
            uk_due=f"{due_day} {MONTHS[due_month - 1]} 2026",
        )
        items.append({"id": item_id, "text": text, "supplier": supplier})
        truth[item_id] = {
            "invoice_number": number,
            "invoice_date": iso_date,
            "total": f"{total:.2f}",
        }

        # ---- what the extraction model answered -------------------------
        got_total = f"{total:.2f}"
        roll = rng.random()
        if roll < 0.04:
            # Invented: an amount the document does not contain. Free to
            # prove, and the one defect this kind of field can prove.
            got_total = f"{round(total * rng.uniform(0.6, 1.4), 2):.2f}"
        elif roll < 0.12:
            got_total = f"{shipping:.2f}"       # present, and the wrong one
        elif roll < 0.17:
            got_total = f"{subtotal:.2f}"       # present, and the wrong one
        elif roll < 0.21:
            # Correct, reformatted. A character-by-character check would call
            # this invented; it is the same amount written the way a person
            # writes it.
            got_total = _money(total)

        got_date = iso_date
        date_roll = rng.random()
        if date_roll < 0.05:
            got_date = iso_due                  # the due date: present, wrong
        elif date_roll < 0.09:
            got_date = f"{MONTHS[month - 1]} {day}, 2026"   # same date, other shape
        elif date_roll < 0.11:
            got_date = f"{day:02d}/{month:02d}/2026"        # a shape nobody can read safely
        elif date_roll < 0.13:
            got_date = f"2026-{month:02d}-{(day % 28) + 1:02d}"   # off by a day: invented

        got_number = number
        number_roll = rng.random()
        if number_roll < 0.03:
            got_number = number.replace("-", "")            # fails verbatim, same value
        elif number_roll < 0.05:
            got_number = f"{number[:-1]}{(int(number[-1]) + 1) % 10}"   # transcription slip

        outputs.append(
            {
                "item_id": item_id,
                "invoice_number": got_number,
                "invoice_date": got_date,
                "total": got_total,
                "confidence": round(min(0.99, max(0.5, rng.gauss(0.9, 0.06))), 2),
            }
        )

    # A person reads the document and writes down the value. For a copied
    # field a human answer *is* a gold value, unlike free text — so these are
    # amounts and dates, not defect boxes.
    labelled = sorted(rng.sample([i["id"] for i in items], N_LABELLED))
    labels = [
        {"item_id": item_id, "field": field, "label": value, "annotator": "ann-1"}
        for item_id in labelled
        for field, value in sorted(truth[item_id].items())
    ]
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
