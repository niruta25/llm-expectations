"""Command line entry point.

A placeholder that reports the truth: nothing is implemented yet. The real
subcommands (``run``, ``plan``, ``analyse``, ``compare``, ``check``) arrive with
M1 onwards — see DESIGN.md §10.
"""

from __future__ import annotations

import sys

from . import __version__

REPO = "https://github.com/niruta25/llm-expectations"

_MESSAGE = f"""llm-expectations {__version__} — pre-alpha, no functionality yet.

This release reserves the name. The design is finished; the build has not
started. Planned commands:

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
