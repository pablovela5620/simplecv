"""Benchmark harness for `pixi run catalog` registration time.

Spawns the catalog task, streams its stdout, parses the
`SIMPLECV_CATALOG_READY ts=...` marker line emitted by
`simplecv.apis.exoego_forge_catalog.main`, and records elapsed
seconds from process spawn to marker. Then sends SIGINT to the
process group to shut down the server cleanly.

Usage:
    pixi run python bench/bench.py --n 3
    pixi run python bench/bench.py --n 3 --profile py-spy --profile-out bench/baseline/profile.svg
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

WORKTREE: Path = Path("/home/pablo/0Dev/personal/simplecv-perf-catalog")
READY_TOKEN: str = "SIMPLECV_CATALOG_READY"
SHUTDOWN_TIMEOUT_S: float = 15.0
HARD_TIMEOUT_S: float = 1800.0
RRD_ROOT: str = "/mnt/8tb/data/exoego-forge-catalog"
BENCH_PORT: int = 19988
"""Off-default port so the bench never collides with a user-running catalog."""


def catalog_argv(*, port: int = BENCH_PORT) -> list[str]:
    """Build the catalog argv used by the bench (bypasses the pixi task)."""
    return [
        "pixi", "run", "python", "tools/catalog.py",
        "--rrd-root", RRD_ROOT,
        "--no-optimize-for-catalog",
        "--port", str(port),
    ]


@dataclass(frozen=True, slots=True)
class RunResult:
    """One catalog-run measurement.

    Attributes:
        elapsed_s: Wall-clock seconds from spawn to the
            ``SIMPLECV_CATALOG_READY`` marker. ``None`` if the marker
            never appeared (server crashed or timed out).
        log_path: Path to the captured stdout for this run.
        phases_s: Mapping of phase name to seconds, parsed from
            ``[catalog-phase] <name> <seconds>s`` lines emitted by the
            instrumented catalog. The synthetic ``cli_and_imports``
            entry is computed as ``main_entered_at_epoch - spawn``.
    """

    elapsed_s: float | None
    log_path: Path
    phases_s: dict[str, float]


def run_once(log_dir: Path, run_idx: int, profile_cmd: list[str] | None = None) -> RunResult:
    """Spawn `pixi run catalog`, time to ready, then shut down.

    Args:
        log_dir: Directory to capture stdout to (one file per run).
        run_idx: 0-based run index used to name the log file.
        profile_cmd: If set, replaces ``pixi run catalog`` with this
            argv. The replacement must produce the same ready marker
            on stdout. Used for py-spy / cProfile wrapping.

    Returns:
        RunResult with elapsed seconds and log path.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path: Path = log_dir / f"run_{run_idx:02d}.log"
    argv: list[str] = profile_cmd if profile_cmd is not None else catalog_argv()
    print(f"[bench] run {run_idx}: spawning {argv!r}", flush=True)
    t_start_epoch: float = time.time()
    t_start: float = time.monotonic()
    proc: subprocess.Popen[str] = subprocess.Popen(
        argv,
        cwd=str(WORKTREE),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )
    ready_at: float | None = None
    deadline: float = t_start + HARD_TIMEOUT_S
    phases: dict[str, float] = {}
    main_entered_epoch: float | None = None
    try:
        with log_path.open("w") as log:
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if "[catalog-phase]" in line:
                    parts: list[str] = line.split()
                    try:
                        marker_idx: int = parts.index("[catalog-phase]")
                        name: str = parts[marker_idx + 1]
                        value_token: str = parts[marker_idx + 2]
                        seconds: float = float(value_token.rstrip("s"))
                    except (ValueError, IndexError):
                        pass
                    else:
                        if name == "main_entered_at_epoch":
                            main_entered_epoch = seconds
                        else:
                            phases[name] = phases.get(name, 0.0) + seconds
                if READY_TOKEN in line and ready_at is None:
                    ready_at = time.monotonic()
                    print(
                        f"[bench] run {run_idx}: ready in {ready_at - t_start:.2f}s",
                        flush=True,
                    )
                    break
                if time.monotonic() > deadline:
                    print(f"[bench] run {run_idx}: HARD TIMEOUT after {HARD_TIMEOUT_S}s", flush=True)
                    break
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
    elapsed_s: float | None = (ready_at - t_start) if ready_at is not None else None
    if main_entered_epoch is not None:
        phases["cli_and_imports"] = max(0.0, main_entered_epoch - t_start_epoch)
    return RunResult(elapsed_s=elapsed_s, log_path=log_path, phases_s=phases)


def main() -> int:
    """CLI entry: run N warm catalog runs and print min/median seconds."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3, help="Number of warm runs.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=WORKTREE / "bench" / "logs",
        help="Directory to write per-run stdout logs.",
    )
    parser.add_argument(
        "--warmup/",
        dest="warmup",
        action="store_true",
        default=True,
        help="Run one untimed warmup before the N timed runs (default true).",
    )
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.add_argument(
        "--profile",
        choices=["none", "py-spy", "cprofile"],
        default="none",
        help="Wrap a single run with a profiler.",
    )
    parser.add_argument(
        "--profile-out",
        type=Path,
        default=None,
        help="Output path for the profiler (svg for py-spy, .prof for cProfile).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="If set, write {min, median, runs[]} JSON to this path.",
    )
    args: argparse.Namespace = parser.parse_args()

    if args.warmup:
        print("[bench] warmup run (untimed)", flush=True)
        _: RunResult = run_once(args.log_dir / "warmup", run_idx=0)

    results: list[RunResult] = [
        run_once(args.log_dir / "warm", run_idx=i) for i in range(args.n)
    ]
    elapsed: list[float] = [r.elapsed_s for r in results if r.elapsed_s is not None]
    if not elapsed:
        print("[bench] ERROR: no successful runs", flush=True)
        return 2
    print("[bench] elapsed seconds:", [f"{x:.2f}" for x in elapsed], flush=True)
    print(f"[bench]   min:    {min(elapsed):.2f}s", flush=True)
    print(f"[bench]   median: {statistics.median(elapsed):.2f}s", flush=True)
    print(f"[bench]   max:    {max(elapsed):.2f}s", flush=True)

    median_phases: dict[str, float] = {}
    if results and results[0].phases_s:
        phase_keys: list[str] = sorted({k for r in results for k in r.phases_s})
        for key in phase_keys:
            values: list[float] = [r.phases_s.get(key, 0.0) for r in results]
            median_phases[key] = statistics.median(values)
        print("[bench] phase medians (s):", flush=True)
        for k in sorted(median_phases, key=lambda k: -median_phases[k]):
            print(f"  {median_phases[k]:>8.3f}  {k}", flush=True)

    if args.profile != "none":
        prof_out: Path = args.profile_out or (
            WORKTREE / "bench" / "baseline" / (
                "profile.svg" if args.profile == "py-spy" else "profile.prof"
            )
        )
        prof_out.parent.mkdir(parents=True, exist_ok=True)
        print(f"[bench] profile run with {args.profile} -> {prof_out}", flush=True)
        if args.profile == "py-spy":
            argv: list[str] = [
                "pixi", "run", "py-spy", "record",
                "--format", "flamegraph",
                "-o", str(prof_out),
                "--subprocesses",
                "--",
                *catalog_argv(),
            ]
        else:
            argv = [
                "pixi", "run", "python", "-m", "cProfile",
                "-o", str(prof_out),
                "tools/catalog.py",
                "--rrd-root", RRD_ROOT,
                "--no-optimize-for-catalog",
                "--port", str(BENCH_PORT),
            ]
        run_once(args.log_dir / "profile", run_idx=99, profile_cmd=argv)

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "n": args.n,
            "min_s": min(elapsed),
            "median_s": statistics.median(elapsed),
            "max_s": max(elapsed),
            "runs_s": elapsed,
            "warmup": args.warmup,
            "phase_medians_s": median_phases,
            "runs_phases_s": [r.phases_s for r in results],
        }
        args.out_json.write_text(json.dumps(payload, indent=2))
        print(f"[bench] wrote {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
