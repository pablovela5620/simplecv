"""Append a single experiment row to ``bench/experiments.jsonl``.

Schema (one JSON object per line, append-only):
    id            : str   E000, E001, ...
    hypothesis    : str   one-line description of what we tried
    files         : list  files touched (relative to worktree)
    before_s      : float median seconds before this experiment
    after_s       : float median seconds after this experiment (or None)
    tests_pass    : bool  whether `pixi run pytest tests/test_exoego_catalog.py -x` passed
    accepted      : bool  whether this experiment was kept
    reason        : str   short reason for accept/reject
    timestamp     : str   ISO-8601 UTC
    notes         : str?  optional free-form

Usage:
    pixi run python bench/log_experiment.py \\
        --id E001 \\
        --hypothesis "skip duplicate discover_rrd_paths in main()" \\
        --files simplecv/apis/exoego_forge_catalog.py \\
        --before-s 240.5 --after-s 238.1 \\
        --tests-pass --accepted --reason "small win, low risk"
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

LOG_PATH: Path = Path("/home/pablo/0Dev/personal/simplecv-perf-catalog/bench/experiments.jsonl")


def main() -> int:
    """CLI entry."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--files", nargs="*", default=[])
    parser.add_argument("--before-s", type=float, default=None)
    parser.add_argument("--after-s", type=float, default=None)
    parser.add_argument("--tests-pass", action="store_true")
    parser.add_argument("--tests-fail", action="store_true")
    parser.add_argument("--accepted", action="store_true")
    parser.add_argument("--rejected", action="store_true")
    parser.add_argument("--reason", default="")
    parser.add_argument("--notes", default=None)
    args: argparse.Namespace = parser.parse_args()

    tests_pass: bool | None
    if args.tests_pass and not args.tests_fail:
        tests_pass = True
    elif args.tests_fail and not args.tests_pass:
        tests_pass = False
    else:
        tests_pass = None
    accepted: bool | None
    if args.accepted and not args.rejected:
        accepted = True
    elif args.rejected and not args.accepted:
        accepted = False
    else:
        accepted = None

    row: dict[str, object] = {
        "id": args.id,
        "hypothesis": args.hypothesis,
        "files": list(args.files),
        "before_s": args.before_s,
        "after_s": args.after_s,
        "tests_pass": tests_pass,
        "accepted": accepted,
        "reason": args.reason,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if args.notes is not None:
        row["notes"] = args.notes
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
