"""Ingestion-speed parity test for ``log_video``.

The bit-preserving VideoStream path (demux + bsf, no pixel decode, no
re-encode) must not be substantially slower than ``method="asset_video"``
(which is a blob copy). A previous decode + libx264 re-encode
implementation was ~291× slower; the goal is ≤ 2× the AssetVideo time.

Uses the ego/hololens hocap MP4 — the highest-resolution case (1280×720)
and the most representative for performance.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import rerun as rr

from simplecv.rerun_log_utils import log_video


_HOCAP_BASE = Path("data/hocap/sample")
_SLOWDOWN_BUDGET: float = 2.0  # VideoStream / AssetVideo wall-time ratio.
_TRIALS: int = 3  # Median across trials to dampen jitter.


def _find_hololens_mp4() -> Path | None:
    """Locate the ego/hololens hocap MP4."""
    if not _HOCAP_BASE.exists():
        return None
    candidates: list[Path] = sorted(_HOCAP_BASE.rglob("hololens*/output.mp4"))
    return candidates[0] if candidates else None


def _time_log(mp4: Path, method: str, tmp_path: Path, trial: int) -> float:
    rrd_path: Path = tmp_path / f"{method}-{trial}.rrd"
    rec: rr.RecordingStream = rr.RecordingStream(
        application_id=f"speed-{method}",
        recording_id=f"speed-{method}-{trial}",
    )
    rec.save(str(rrd_path))
    t0: float = time.perf_counter()
    log_video(mp4, Path("/v"), method=method, recording=rec)  # type: ignore[arg-type]
    # Force RecordingStream finalize so all data is on disk before timing stops.
    del rec
    return time.perf_counter() - t0


def test_log_video_stream_speed_within_budget_of_asset_video(tmp_path: Path) -> None:
    """VideoStream ingestion is ≤ 2× AssetVideo on the hololens (1280×720) MP4."""
    mp4: Path | None = _find_hololens_mp4()
    if mp4 is None:
        pytest.skip("hocap sample not downloaded (run pixi _download-hocap-sample)")

    asset_times: list[float] = [
        _time_log(mp4, "asset_video", tmp_path, t) for t in range(_TRIALS)
    ]
    stream_times: list[float] = [
        _time_log(mp4, "video_stream", tmp_path, t) for t in range(_TRIALS)
    ]
    asset_median: float = sorted(asset_times)[_TRIALS // 2]
    stream_median: float = sorted(stream_times)[_TRIALS // 2]
    ratio: float = stream_median / asset_median if asset_median > 0 else float("inf")

    print(
        f"AssetVideo  median: {asset_median*1000:.1f} ms (trials: "
        f"{[f'{x*1000:.0f}ms' for x in asset_times]})"
    )
    print(
        f"VideoStream median: {stream_median*1000:.1f} ms (trials: "
        f"{[f'{x*1000:.0f}ms' for x in stream_times]})"
    )
    print(f"VideoStream / AssetVideo ratio: {ratio:.2f}×")

    assert ratio <= _SLOWDOWN_BUDGET, (
        f"VideoStream is {ratio:.2f}× slower than AssetVideo on the hololens MP4 "
        f"(budget {_SLOWDOWN_BUDGET}×). "
        f"AssetVideo median {asset_median*1000:.1f} ms, "
        f"VideoStream median {stream_median*1000:.1f} ms."
    )
