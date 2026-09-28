"""Command line entry point.

A placeholder that reports the truth: the skeleton is in place and nothing runs
from a terminal yet. The subcommands (``run``, ``plan``, ``analyse``,
``compare``, ``check``) arrive with M1 onwards — see DESIGN.md §11 and §12.
"""

from __future__ import annotations

import sys

from . import __version__

REPO = "https://github.com/niruta25/llm-expectations"

_MESSAGE = f"""llm-expectations {__version__} — pre-alpha, nothing to run yet.

M0 has landed: types, schema, taxonomy, readers and config. A project loads and
validates in Python (``llm_expectations.load_run("run.yml")``); no checks run
and nothing calls a model. Planned commands:

  run       collect judge verdicts and analyse
  plan      estimate cost, make no calls
  analyse   re-analyse a finished run from cache, free
  compare   two runs, head to head
  check     static health check on a taxonomy file

Design and build order: {REPO}/blob/main/DESIGN.md
"""


def main(argv: list[str] | None = None) -> int:
    """Print status and exit. Always succeeds; there is nothing to fail at yet."""
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in {"-V", "--version"}:
        print(__version__)
        return 0
    print(_MESSAGE, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
