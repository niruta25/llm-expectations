# Golden fixture: invoices

Nine documents with deliberately seeded errors, one of each production error
type, plus an answer key. Every test that touches it runs offline.

| doc | field | error type | Which tier catches it |
|---|---|---|---|
| g1 | — | none | — |
| g2 | `vendor` | `fabrication` | deterministic — value absent from source |
| g3 | `total_amount` | `transposition` | deterministic — `7,430.10` for `7,340.10` |
| g4 | `invoice_date` | `wrong_instance` | model tier — the wrong date, but a real one |
| g5 | `vendor` | `misattribution` | model tier — ship-to company, present in source |
| g6 | `vendor` | `span_drift` | neither yet — right value, wrong offsets |
| g7 | `currency` | `omission` | statistical — null where the source has a value |
| g8 | `invoice_date` | `format_drift` | deterministic — `2024-02-31` is well-formed and impossible |
| g9 | `total_num` | `derived_error` | cross-field rule — inputs right, total wrong |

The split matters more than the count. Grounding flags exactly three of the
eight seeded errors (g2, g3, g8) because the other five put a value in the
output that genuinely appears in the source, or no value at all. That boundary
is the argument for the model tier, and `test_expectations.py` pins it: if
grounding ever starts flagging g4 or g5, something has changed semantics.

`g6` is caught by nothing today. Exact substring match resolves before the
span comparison, so a right-value/wrong-offset extraction passes. Fixing it
means checking the span first when one is declared, which would make the check
stricter for everyone; it is recorded here rather than silently ignored.

`g9` demonstrates the design guidance that derived values do not belong in the
model's job: `expect_fields_to_satisfy` catches it for free, where an expensive
judge would be needed to notice the same thing.
