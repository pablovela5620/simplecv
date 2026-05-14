"""Automated screenshot validation: aria-gen2 RGB via VideoStream vs AssetVideo.

AssetVideo is the known-good ground truth (user confirmed it renders).
We log the SAME aria-gen2 RGB MP4 via both methods to sibling entities,
screenshot each panel via ``ViewerClient.save_screenshot``, and compute
PSNR. Anything ≥ 40 dB means VideoStream renders effectively the same
pixels as AssetVideo. Pure black VideoStream panel → very low PSNR vs
AssetVideo → failure.

Per https://github.com/rerun-io/rerun/blob/main/docs/snippets/all/howto/screenshot.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image
from rerun.blueprint.components import AbsoluteTimeRange
from rerun.experimental import ViewerClient

from simplecv.rerun_log_utils import log_video


_RGB_MP4 = Path("/mnt/8tb/data/aria-gen2-pilot/eat_0/_simplecv/rgb.mp4")
_OUT_DIR = Path("/tmp/aria_rgb_validate")
_STREAM_ENTITY = "/world/ego/camera-rgb/pinhole/video"
_ASSET_ENTITY = "/world/ego/camera-rgb/pinhole/video_assetvideo"
_CURSOR_NS: int = 5_000_000_000  # 5 s into the video.


def _wait_for_file(path: Path, timeout_s: float = 60.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if path.exists() and path.stat().st_size > 0:
            time.sleep(0.5)
            return
        time.sleep(0.5)
    raise RuntimeError(f"Screenshot not produced within {timeout_s}s: {path}")


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR over the common crop of two RGB images."""
    n = min(a.shape[0], b.shape[0])
    m = min(a.shape[1], b.shape[1])
    a32 = a[:n, :m, :3].astype(np.float64)
    b32 = b[:n, :m, :3].astype(np.float64)
    mse = float(np.mean((a32 - b32) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * float(np.log10(255.0)) - 10.0 * float(np.log10(mse))


def main() -> int:
    if not _RGB_MP4.exists():
        print(f"missing source: {_RGB_MP4}", file=sys.stderr)
        return 2

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in _OUT_DIR.glob("*.png"):
        f.unlink()

    rr.init("aria_rgb_validate", spawn=True)

    stream_view = rrb.Spatial2DView(
        origin=_STREAM_ENTITY,
        name="VideoStream",
        contents=[f"+ {_STREAM_ENTITY}/**"],
    )
    asset_view = rrb.Spatial2DView(
        origin=_ASSET_ENTITY,
        name="AssetVideo",
        contents=[f"+ {_ASSET_ENTITY}/**"],
    )
    blueprint = rrb.Blueprint(
        rrb.Horizontal(contents=[stream_view, asset_view]),
        rrb.TimePanel(
            timeline="video_time",
            time_selection=AbsoluteTimeRange(min=_CURSOR_NS, max=_CURSOR_NS),
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)

    print(f"Logging {_RGB_MP4} via VideoStream + AssetVideo …")
    log_video(_RGB_MP4, Path(_STREAM_ENTITY), method="video_stream")
    log_video(_RGB_MP4, Path(_ASSET_ENTITY), method="asset_video")

    rr.set_time("video_time", duration=np.timedelta64(_CURSOR_NS, "ns"))
    rr.log("/_cursor_marker", rr.AnyValues(at=_CURSOR_NS))

    rr.get_global_data_recording().flush()
    print("Waiting 30s for ingestion + decode …")
    time.sleep(30.0)

    viewer = ViewerClient()
    stream_png = _OUT_DIR / "stream.png"
    asset_png = _OUT_DIR / "asset.png"
    full_png = _OUT_DIR / "full.png"
    print("Capturing screenshots …")
    viewer.save_screenshot(str(stream_png), view_id=stream_view.id)
    viewer.save_screenshot(str(asset_png), view_id=asset_view.id)
    viewer.save_screenshot(str(full_png))
    _wait_for_file(stream_png)
    _wait_for_file(asset_png)
    _wait_for_file(full_png)

    stream_img = np.asarray(Image.open(stream_png).convert("RGB"))
    asset_img = np.asarray(Image.open(asset_png).convert("RGB"))
    full_img = np.asarray(Image.open(full_png).convert("RGB"))

    s_mean, s_std = float(stream_img.mean()), float(stream_img.std())
    a_mean, a_std = float(asset_img.mean()), float(asset_img.std())
    f_mean = float(full_img.mean())
    print(f"VideoStream panel: shape={stream_img.shape}, mean={s_mean:.2f}, std={s_std:.2f}")
    print(f"AssetVideo  panel: shape={asset_img.shape}, mean={a_mean:.2f}, std={a_std:.2f}")
    print(f"Full viewer:       shape={full_img.shape}, mean={f_mean:.2f}")
    print(f"Screenshots saved to {_OUT_DIR}")

    if a_mean < 5.0:
        print(
            "⚠  AssetVideo panel is also empty — viewer didn't decode/render. "
            "Cursor placement, ingestion timing, or screenshot timing issue, not a VideoStream bug.",
            file=sys.stderr,
        )
        return 3

    psnr = _psnr(asset_img, stream_img)
    print(f"PSNR (AssetVideo vs VideoStream): {psnr:.2f} dB")
    if psnr < 30.0:
        print(f"❌ FAIL: VideoStream diverges from AssetVideo (PSNR={psnr:.2f} dB < 30 dB)")
        return 1
    print(f"✓ PASS: VideoStream matches AssetVideo (PSNR={psnr:.2f} dB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
