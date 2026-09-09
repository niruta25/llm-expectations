"""llm-expectations — quality checks for LLM outputs that are judgements *about*
a document, not values copied *out of* one.

Pre-alpha. This release exists to reserve the name; there is no functionality
yet. The design is settled and written up in DESIGN.md:

  - assigned fields   a label chosen from a versioned taxonomy
  - free text fields  a sentence written about the item

Two gates run before any quality number is reported: can the measurement be
trusted, and is the judge better than guessing. Results carry three states —
pass, fail, and *not checked* — and the third never silently becomes a pass.

See https://github.com/niruta25/llm-expectations
"""

from __future__ import annotations

__version__ = "0.0.0"

__all__ = ["__version__"]
