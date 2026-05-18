"""Capture catalog/dataset/segment screenshots of the running catalog.

Launches the catalog, waits for the background table-build to finish, then
spawns three headless ``rerun --screenshot-to`` invocations under
``xvfb-run`` to capture the catalog object-model views:

- ``catalog.png``                       — catalog root listing all datasets
- ``dataset_epfl-smart-kitchen.png``    — EPFL Smart Kitchen dataset entry
- ``segment.png``                       — one segment opened (camera entities)

We use ``--screenshot-to`` rather than wiring up
``rr.experimental.ViewerClient.save_screenshot`` directly because the
``--screenshot-to`` path waits for the viewer to render the supplied URL
and then exits cleanly — friendlier to a headless overnight pipeline than
keeping a Viewer process alive plus a client. Either approach renders the
same Rust-side scene; the artifact is what matters for verification.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

WORKTREE: Path = Path("/home/pablo/0Dev/personal/simplecv-perf-catalog")
RRD_ROOT: str = "/mnt/8tb/data/exoego-forge-catalog"
CATALOG_PORT: int = 19989  # off-default to avoid colliding with user / bench
SCREENSHOTS_DIR: Path = WORKTREE / "bench" / "screenshots" / "final"
WINDOW_SIZE: str = "1600x900"
SETTLE_AFTER_READY_S: float = 60.0
"""Sleep after the catalog prints its ready marker but before screenshots,
to give Rerun's prefix walker a chance to populate at least the small
datasets (aria-gen2 has 12 RRDs, hot3d-quest3 has 20)."""
SCREENSHOT_TIMEOUT_S: float = 120.0
READY_TOKEN: str = "SIMPLECV_CATALOG_READY"


def _spawn_catalog() -> subprocess.Popen[str]:
    """Spawn `pixi run python tools/catalog.py ...` on a private port."""
    argv: list[str] = [
        "pixi", "run", "python", "tools/catalog.py",
        "--rrd-root", RRD_ROOT,
        "--no-optimize-for-catalog",
        "--port", str(CATALOG_PORT),
    ]
    print(f"[shots] spawning catalog: {argv!r}", flush=True)
    proc: subprocess.Popen[str] = subprocess.Popen(
        argv,
        cwd=str(WORKTREE),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )
    return proc


def _wait_for_marker(
    proc: subprocess.Popen[str],
    *,
    token: str,
    timeout_s: float,
    log_path: Path,
) -> bool:
    """Stream catalog stdout until ``token`` is seen or the timeout passes."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    deadline: float = time.monotonic() + timeout_s
    assert proc.stdout is not None
    with log_path.open("a") as log:
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if token in line:
                return True
            if time.monotonic() > deadline:
                return False
            if proc.poll() is not None:
                return False
    return False


def _extract_table_urls(log_path: Path) -> dict[str, str]:
    """Parse `    <table_name>: rerun+http://.../entry/<id>` lines from the catalog log."""
    pattern: re.Pattern[str] = re.compile(r"^\s+(\S+_table): (rerun\+http\S+)\s*$")
    urls: dict[str, str] = {}
    for line in log_path.read_text().splitlines():
        m: re.Match[str] | None = pattern.match(line)
        if m is not None:
            urls[m.group(1)] = m.group(2)
    return urls


def _run_screenshot(url: str, out_path: Path) -> tuple[bool, str]:
    """Spawn xvfb-run + rerun --screenshot-to, return (ok, tail-of-stderr)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    argv: list[str] = [
        "xvfb-run", "-a",
        "--server-args=-screen 0 1920x1080x24",
        "pixi", "run", "rerun",
        url,
        "--screenshot-to", str(out_path),
        "--window-size", WINDOW_SIZE,
    ]
    print(f"[shots] {url} -> {out_path.name}", flush=True)
    try:
        completed: subprocess.CompletedProcess[str] = subprocess.run(
            argv,
            cwd=str(WORKTREE),
            capture_output=True,
            text=True,
            timeout=SCREENSHOT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return False, "screenshot subprocess timed out"
    # The viewer reliably writes the screenshot to disk before exiting,
    # but sometimes panics during teardown (after the screenshot is
    # saved). Treat "file exists with non-trivial content" as success
    # regardless of returncode.
    if not out_path.exists() or out_path.stat().st_size < 100_000:
        return False, (completed.stderr or completed.stdout)[-2000:]
    return True, ""


def main() -> int:
    """Run end-to-end screenshot capture."""
    log_path: Path = WORKTREE / "bench" / "screenshots" / "catalog_run.log"
    if log_path.exists():
        log_path.unlink()

    proc: subprocess.Popen[str] = _spawn_catalog()
    try:
        if not _wait_for_marker(proc, token=READY_TOKEN, timeout_s=120.0, log_path=log_path):
            print("[shots] ERROR: catalog never printed ready marker", flush=True)
            return 2
        print(
            f"[shots] catalog ready; sleeping {SETTLE_AFTER_READY_S}s so segment "
            "registration can start populating the smaller datasets",
            flush=True,
        )
        time.sleep(SETTLE_AFTER_READY_S)
        catalog_url: str = f"rerun+http://127.0.0.1:{CATALOG_PORT}"

        # Query the catalog via the Rerun client to discover entry IDs and
        # a real segment URL. The viewer's URL parser accepts entry/segment
        # routes that include these IDs but rejects "/dataset/<name>".
        import rerun as rr  # late import — keep startup cost out of the bench

        dataset_entry_url: str = catalog_url
        segment_target_url: str = catalog_url
        try:
            client = rr.catalog.CatalogClient(catalog_url)
            epfl_entry = client.get_dataset("epfl-smart-kitchen")
            dataset_entry_url = f"{catalog_url}/entry/{epfl_entry.id}"
            print(f"[shots] epfl-smart-kitchen entry id={epfl_entry.id}", flush=True)
            for probe_name in ("aria-gen2", "hot3d-quest3", "epfl-smart-kitchen"):
                try:
                    probe_entry = client.get_dataset(probe_name)
                    segment_ids: list[str] = probe_entry.segment_ids()
                except Exception as exc:  # noqa: BLE001 - keep probing
                    print(f"[shots] {probe_name} segment_ids() error: {exc!r}", flush=True)
                    continue
                if segment_ids:
                    seg_id: str = segment_ids[0]
                    segment_target_url = f"{catalog_url}/entry/{probe_entry.id}?segment_id={seg_id}"
                    print(
                        f"[shots] using {probe_name} segment {seg_id} for segment screenshot",
                        flush=True,
                    )
                    break
            else:
                print(
                    "[shots] no segments registered yet; falling back to catalog URL for segment shot",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            print(f"[shots] catalog client probe failed: {exc!r}", flush=True)

        targets: list[tuple[str, str, Path]] = [
            ("catalog", catalog_url, SCREENSHOTS_DIR / "catalog.png"),
            (
                "dataset_epfl-smart-kitchen",
                dataset_entry_url,
                SCREENSHOTS_DIR / "dataset_epfl-smart-kitchen.png",
            ),
            (
                "segment",
                segment_target_url,
                SCREENSHOTS_DIR / "segment.png",
            ),
        ]
        results: list[dict[str, object]] = []
        for label, url, out_path in targets:
            ok, err = _run_screenshot(url, out_path)
            results.append({"label": label, "url": url, "path": str(out_path), "ok": ok, "error": err})
            if not ok:
                print(f"[shots] {label} failed: {err}", flush=True)
        manifest_path: Path = SCREENSHOTS_DIR / "manifest.json"
        manifest_path.write_text(json.dumps(results, indent=2))
        print(f"[shots] manifest -> {manifest_path}", flush=True)
        ok_count: int = sum(1 for r in results if r["ok"])
        print(f"[shots] captured {ok_count}/{len(results)} screenshots", flush=True)
        return 0 if ok_count == len(results) else 1
    finally:
        if proc.poll() is None:
            print("[shots] shutting down catalog", flush=True)
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()


if __name__ == "__main__":
    sys.exit(main())
