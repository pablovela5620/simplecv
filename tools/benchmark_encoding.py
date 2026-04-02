"""Benchmark encoding methods for HOT3D VRS preprocessing.

Compares three approaches on one VRS stream:
  A) Current: TurboJPEG→BGR→RGB flip→av container (libsvtav1)
  B) YUV + CPU: TurboJPEG→YUV planes→MP4Writer (libsvtav1)
  C) YUV + NVENC: TurboJPEG→YUV planes→MP4Writer (av1_nvenc)

Usage:
    pixi run python tools/benchmark_encoding.py
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import av
import numpy as np
import pyvrs
from tqdm import tqdm
from turbojpeg import TurboJPEG

from simplecv.video_encoder import MP4Writer, VideoCodecChoice

_TJ: TurboJPEG = TurboJPEG()

VRS_PATH: Path = Path("/mnt/8tb/data/hot3d/aria/P0001_10a27bf7/recording.vrs")
STREAM_ID: str = "214-1"  # RGB camera
MAX_FRAMES: int = 500  # Enough to benchmark, not too slow


def read_jpeg_frames(vrs_path: Path, stream_id: str, max_frames: int) -> list[bytes]:
    """Read JPEG bytes from VRS."""
    reader: pyvrs.SyncVRSReader = pyvrs.SyncVRSReader(str(vrs_path))
    filtered = reader.filtered_by_fields(stream_ids=stream_id, record_types="data")

    frames: list[bytes] = []
    for record in filtered:
        if record.n_image_blocks == 0:
            continue
        frames.append(record.image_blocks[0].tobytes())
        if len(frames) >= max_frames:
            break
    return frames


def benchmark_method_a(jpeg_frames: list[bytes], output_path: Path) -> dict:
    """Current method: TurboJPEG→BGR→RGB→av container (libsvtav1)."""
    t_decode: float = 0.0
    t_encode: float = 0.0

    # Decode all
    t0: float = time.perf_counter()
    bgr_frames: list[np.ndarray] = [_TJ.decode(f) for f in jpeg_frames]
    t_decode = time.perf_counter() - t0

    # Encode
    first: np.ndarray = bgr_frames[0]
    h, w = first.shape[:2]
    fps: float = 30.0

    t0 = time.perf_counter()
    container: av.container.OutputContainer = av.open(str(output_path), mode="w")
    stream: av.video.stream.VideoStream = container.add_stream("libsvtav1", rate=round(fps))
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "30", "preset": "6"}

    for bgr in bgr_frames:
        rgb: np.ndarray = bgr[:, :, ::-1]
        frame: av.VideoFrame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    t_encode = time.perf_counter() - t0

    size: int = output_path.stat().st_size
    return {"decode": t_decode, "encode": t_encode, "total": t_decode + t_encode, "size": size}


def benchmark_method_b(jpeg_frames: list[bytes], output_path: Path) -> dict:
    """YUV + CPU: TurboJPEG→YUV→MP4Writer (libsvtav1)."""
    t_decode: float = 0.0
    t_encode: float = 0.0

    # Decode to YUV
    t0 = time.perf_counter()
    yuv_frames: list[list[np.ndarray]] = [_TJ.decode_to_yuv_planes(f) for f in jpeg_frames]
    t_decode = time.perf_counter() - t0

    # Encode via MP4Writer forcing CPU
    t0 = time.perf_counter()
    writer: MP4Writer = MP4Writer(output_path, codec=VideoCodecChoice.AV1, fps=30.0)
    # Force CPU by monkey-patching candidates temporarily
    from simplecv import video_encoder
    orig: list[str] = video_encoder._ENCODER_CANDIDATES[VideoCodecChoice.AV1]
    video_encoder._ENCODER_CANDIDATES[VideoCodecChoice.AV1] = ["libsvtav1"]
    try:
        for planes in yuv_frames:
            if len(planes) >= 3:
                writer.write_yuv_planes(planes[0], planes[1], planes[2])
            else:
                writer.write_yuv_planes(planes[0])
        writer.close()
    finally:
        video_encoder._ENCODER_CANDIDATES[VideoCodecChoice.AV1] = orig
    t_encode = time.perf_counter() - t0

    size = output_path.stat().st_size
    return {"decode": t_decode, "encode": t_encode, "total": t_decode + t_encode, "size": size, "encoder": "libsvtav1"}


def benchmark_method_c(jpeg_frames: list[bytes], output_path: Path) -> dict:
    """YUV + NVENC: TurboJPEG→YUV→MP4Writer (av1_nvenc auto-select)."""
    t_decode: float = 0.0
    t_encode: float = 0.0

    # Decode to YUV
    t0 = time.perf_counter()
    yuv_frames: list[list[np.ndarray]] = [_TJ.decode_to_yuv_planes(f) for f in jpeg_frames]
    t_decode = time.perf_counter() - t0

    # Encode via MP4Writer (will auto-select NVENC if available)
    t0 = time.perf_counter()
    writer = MP4Writer(output_path, codec=VideoCodecChoice.AV1, fps=30.0)
    for planes in yuv_frames:
        if len(planes) >= 3:
            writer.write_yuv_planes(planes[0], planes[1], planes[2])
        else:
            writer.write_yuv_planes(planes[0])
    writer.close()
    t_encode = time.perf_counter() - t0

    size = output_path.stat().st_size
    return {"decode": t_decode, "encode": t_encode, "total": t_decode + t_encode, "size": size, "encoder": writer.encoder_name}


def main() -> None:
    print(f"Reading {MAX_FRAMES} JPEG frames from VRS...")
    t0: float = time.perf_counter()
    jpeg_frames: list[bytes] = read_jpeg_frames(VRS_PATH, STREAM_ID, MAX_FRAMES)
    print(f"  Read {len(jpeg_frames)} frames in {time.perf_counter() - t0:.1f}s")

    # Check first frame dimensions
    img: np.ndarray = _TJ.decode(jpeg_frames[0])
    print(f"  Frame size: {img.shape[1]}x{img.shape[0]} ({img.shape})")
    print()

    results: dict[str, dict] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp: Path = Path(tmpdir)

        print("Method A: BGR→RGB→libsvtav1 (current)")
        results["A"] = benchmark_method_a(jpeg_frames, tmp / "a.mp4")
        print(f"  decode: {results['A']['decode']:.1f}s, encode: {results['A']['encode']:.1f}s, "
              f"total: {results['A']['total']:.1f}s, size: {results['A']['size']/1e6:.1f}MB")

        print("\nMethod B: YUV→MP4Writer (libsvtav1 CPU)")
        results["B"] = benchmark_method_b(jpeg_frames, tmp / "b.mp4")
        print(f"  decode: {results['B']['decode']:.1f}s, encode: {results['B']['encode']:.1f}s, "
              f"total: {results['B']['total']:.1f}s, size: {results['B']['size']/1e6:.1f}MB, "
              f"encoder: {results['B']['encoder']}")

        print("\nMethod C: YUV→MP4Writer (NVENC auto)")
        try:
            results["C"] = benchmark_method_c(jpeg_frames, tmp / "c.mp4")
            print(f"  decode: {results['C']['decode']:.1f}s, encode: {results['C']['encode']:.1f}s, "
                  f"total: {results['C']['total']:.1f}s, size: {results['C']['size']/1e6:.1f}MB, "
                  f"encoder: {results['C']['encoder']}")
        except RuntimeError as e:
            print(f"  FAILED (no NVENC): {e}")

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Method':<30} {'Decode':>8} {'Encode':>8} {'Total':>8} {'Size':>8} {'Speedup':>8}")
    print("-" * 60)
    baseline: float = results["A"]["total"]
    for name, r in results.items():
        speedup: float = baseline / r["total"]
        label: str = {"A": "BGR→RGB→libsvtav1", "B": "YUV→libsvtav1", "C": f"YUV→{r.get('encoder', '?')}"}[name]
        print(f"{label:<30} {r['decode']:>7.1f}s {r['encode']:>7.1f}s {r['total']:>7.1f}s "
              f"{r['size']/1e6:>6.1f}MB {speedup:>7.1f}x")


if __name__ == "__main__":
    main()
