"""Command line entry point.

``analyse`` is the one you will use most. Change a threshold, add a check, fix
a bug — re-run it and pay nothing, because every number downstream of the judge
is computed from ``verdicts.jsonl`` rather than from the model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .calibration.base import StaleCalibration
from .compare import ComparisonError
from .config import ConfigError, RunConfig
from .read import ReadError
from .taxonomy import TaxonomyError, load_taxonomy
from .types import Severity

REPO = "https://github.com/niruta25/llm-expectations"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-expectations",
        description="Quality checks for LLM outputs that are judgements about a document.",
    )
    parser.add_argument("-V", "--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="collect judge verdicts and analyse")
    run.add_argument("config", type=Path, help="path to run.yml")
    run.add_argument("--out", type=Path, default=Path("out"), help="output directory")
    run.add_argument(
        "--reuse",
        type=Path,
        default=None,
        help="a previous run directory whose verdicts may be reused instead of re-paid for",
    )
    run.add_argument("--yes", action="store_true", help="skip the cost confirmation")

    plan = sub.add_parser("plan", help="estimate cost, make no calls")
    plan.add_argument("config", type=Path)

    analyse = sub.add_parser("analyse", help="re-analyse a finished run from cache, free")
    analyse.add_argument("directory", type=Path, help="a run directory under out/")

    evaluate = sub.add_parser(
        "triage-eval", help="Error Recall@Budget per strategy, from a finished run"
    )
    evaluate.add_argument("directory", type=Path, help="a run directory under out/")

    check = sub.add_parser("check", help="static health check on a taxonomy file")
    check.add_argument("taxonomy", type=Path)

    head_to_head = sub.add_parser("compare", help="two runs, head to head")
    head_to_head.add_argument("a", type=Path, help="the earlier run directory")
    head_to_head.add_argument("b", type=Path, help="the later run directory")
    head_to_head.add_argument(
        "--migration",
        type=Path,
        default=None,
        help="a taxonomy migration file, required when the two runs used different versions",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = _parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            return _run(args)
        if args.command == "plan":
            return _plan(args)
        if args.command == "analyse":
            return _analyse(args)
        if args.command == "triage-eval":
            return _triage_eval(args)
        if args.command == "check":
            return _check(args)
        if args.command == "compare":
            return _compare(args)
    except (
        ComparisonError,
        ConfigError,
        ReadError,
        StaleCalibration,
        TaxonomyError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    parser.print_help()
    print(f"\nDesign and build order: {REPO}/blob/main/DESIGN.md")
    return 0


def _run(args: argparse.Namespace) -> int:
    from .config import load_run
    from .run import run

    config = load_run(args.config)
    if args.yes:
        config = _without_confirmation(config)
    result = run(config, out=args.out, reuse=args.reuse)
    if result is None:
        return 1
    print(result.report)
    print(f"  written to {result.directory}")
    return 0


def _without_confirmation(config: RunConfig) -> RunConfig:
    import dataclasses

    from .config import Budget

    return dataclasses.replace(
        config, budget=Budget(max_usd=config.budget.max_usd, confirm=False)
    )


def _plan(args: argparse.Namespace) -> int:
    from .config import load_run
    from .plan import plan_run
    from .run import load_dataset

    config = load_run(args.config)
    dataset = load_dataset(config)
    plan = plan_run(config, dataset.items, dataset.outputs)
    print("PLAN")
    for line in plan.lines():
        print(line)
    for reason, count in sorted(plan.skipped.items()):
        print(f"  skipped {count:,}: {reason}")
    if plan.unpriced_models:
        print(f"\n  ⚠ no published price for {', '.join(plan.unpriced_models)} — no estimate.")
    print("\n  No calls were made. The free checks cost nothing and run either way;")
    print("  this is only what the judge would add.")
    return 0


def _analyse(args: argparse.Namespace) -> int:
    from .run import analyse_run

    result = analyse_run(args.directory)
    print(result.report)
    print("  re-analysed from cache. Zero model calls, zero cost.")
    return 0


def _compare(args: argparse.Namespace) -> int:
    """Two finished runs, read off disk. Zero model calls."""
    from .compare import load_and_compare
    from .report import comparison_report

    comparison = load_and_compare(args.a, args.b, migration_path=args.migration)
    for line in comparison_report(comparison):
        print(line)
    print("  Read from disk. Zero model calls, zero cost.")
    return 0


def _triage_eval(args: argparse.Namespace) -> int:
    """The comparison table alone, from disk. Zero model calls."""
    import json

    from .report import operating_point_table

    path = Path(args.directory) / "triage_eval.json"
    if not path.exists():
        print(
            f"no triage_eval.json in {args.directory}. It is written when a run has "
            "human labels — without them true errors cannot be counted.",
            file=sys.stderr,
        )
        return 2
    payload = json.loads(path.read_text(encoding="utf-8"))
    for line in operating_point_table(payload):
        print(line)
    print("  Read from disk. Zero model calls, zero cost.")
    return 0


def _check(args: argparse.Namespace) -> int:
    taxonomy = load_taxonomy(args.taxonomy)
    issues = taxonomy.health()
    print(f"{taxonomy.ref}   {len(taxonomy)} labels, {len(taxonomy.leaves())} leaves")
    print(f"content hash {taxonomy.content_hash[:12]}")
    if not issues:
        print("\nno issues.")
        return 0
    print()
    marks = {Severity.STOP: "✗", Severity.WARN: "⚠", Severity.NOTE: "·"}
    for issue in issues:
        print(f"  {marks[issue.severity]} {issue.check}")
        print(f"      {issue.message}")
        if issue.paths:
            print(f"      {', '.join(issue.paths[:6])}")
    return 1 if any(i.severity is Severity.STOP for i in issues) else 0


if __name__ == "__main__":
    raise SystemExit(main())
