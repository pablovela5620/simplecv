"""Ingestion-speed sanity test for ``log_video``.

The bit-preserving VideoStream path (demux + bsf, no pixel decode, no
re-encode) should ingest fast: a previous decode + libx264 re-encode
implementation took ~9.8s on this clip — orders of magnitude slower than
the demux-only path. Asserting an absolute upper bound (100ms for the
hololens 1280×720 clip, ~1085 frames) catches the re-encode regression
while tolerating jitter at the sub-20ms scale where ratio comparisons
against the few-ms AssetVideo path become noisy.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import rerun as rr

from simplecv.rerun_log_utils import log_video


_HOCAP_BASE = Path("data/hocap/sample")
_MAX_STREAM_TIME_S: float = 0.1  # 100 ms ceiling for hololens 720p, ~1085 frames.
_TRIALS: int = 5  # Median across trials to dampen jitter.


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


def test_log_video_stream_ingestion_under_budget(tmp_path: Path) -> None:
    """VideoStream ingestion ≤ 100 ms on the hololens (1280×720, ~1085 frames) MP4."""
    mp4: Path | None = _find_hololens_mp4()
    if mp4 is None:
        pytest.skip("hocap sample not downloaded (run pixi _download-hocap-sample)")

    # Warm up the OS file cache.
    _time_log(mp4, "video_stream", tmp_path, -1)

    stream_times: list[float] = [
        _time_log(mp4, "video_stream", tmp_path, t) for t in range(_TRIALS)
    ]
    stream_median: float = sorted(stream_times)[_TRIALS // 2]

    print(
        f"VideoStream median: {stream_median*1000:.1f} ms (trials: "
        f"{[f'{x*1000:.0f}ms' for x in stream_times]}, budget: "
        f"{_MAX_STREAM_TIME_S*1000:.0f} ms)"
    )

    assert stream_median <= _MAX_STREAM_TIME_S, (
        f"VideoStream median {stream_median*1000:.1f} ms exceeds "
        f"{_MAX_STREAM_TIME_S*1000:.0f} ms budget on the hololens MP4. "
        f"A previous re-encode implementation took ~9.8 s on this clip — "
        f"a regression at this scale would indicate accidental re-introduction "
        f"of the decode + re-encode pipeline."
    )
