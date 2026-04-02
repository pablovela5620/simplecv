"""One-time VRS → AV1 MP4 conversion for HOT3D Aria sequences.

Reads each VRS file with pyvrs, decodes JPEG frames per camera stream,
re-encodes to AV1 MP4 using PyAV, and extracts calibration from the
MPS online_calibration.jsonl.

Usage:
    pixi run preprocess-hot3d --root /mnt/8tb/data/hot3d/aria
    pixi run preprocess-hot3d --root /mnt/8tb/data/hot3d/aria --sequence P0003_c701bd11
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyvrs
import tyro
from tqdm import tqdm
from turbojpeg import TurboJPEG

from simplecv.data.hot3d_utils import (
    ARIA_STREAM_ID_TO_LABEL,
    Hot3dSequenceCalibration,
    parse_online_calibration_first,
    save_calibration,
)
from simplecv.video_encoder import MP4Writer, VideoCodecChoice

# Shared TurboJPEG instance (thread-safe)
_TJ: TurboJPEG = TurboJPEG()

# VRS stream IDs for Aria image streams
ARIA_IMAGE_STREAM_IDS: list[str] = ["214-1", "1201-1", "1201-2"]

# Output filenames per stream label
STREAM_LABEL_TO_FILENAME: dict[str, str] = {
    "camera-rgb": "rgb.mp4",
    "camera-slam-left": "slam_left.mp4",
    "camera-slam-right": "slam_right.mp4",
}

OUTPUT_DIR_NAME: str = "_simplecv"


@dataclass
class PreprocessConfig:
    """Configuration for HOT3D VRS preprocessing."""

    root: Path = Path("/mnt/8tb/data/hot3d/aria")
    """Root directory containing sequence folders."""
    sequence: str = ""
    """Process a single sequence (empty = all sequences with recording.vrs)."""
    num_decode_workers: int = 8
    """Number of parallel JPEG decode threads."""
    skip_existing: bool = True
    """Skip sequences that already have _simplecv/ output."""
    streams: list[str] = field(default_factory=lambda: ["214-1", "1201-1", "1201-2"])
    """VRS stream IDs to extract. Default: RGB + both SLAM cameras."""


def decode_jpeg_to_yuv(jpeg_bytes: bytes) -> list[np.ndarray]:
    """Decode JPEG bytes to YUV420 planes using TurboJPEG (fastest path)."""
    return _TJ.decode_to_yuv_planes(jpeg_bytes)


def extract_stream_to_mp4(
    vrs_path: Path,
    stream_id: str,
    output_path: Path,
    num_workers: int,
) -> list[int]:
    """Extract a single VRS image stream to AV1 MP4.

    Returns list of frame timestamps in nanoseconds.
    """
    # pyvrs API follows rerun-io/examples-monorepo/packages/pyvrs-viewer patterns
    reader: pyvrs.SyncVRSReader = pyvrs.SyncVRSReader(str(vrs_path))

    assert stream_id in reader.stream_ids, f"Stream {stream_id} not found in {vrs_path}. Available: {reader.stream_ids}"
    assert reader.might_contain_images(stream_id), f"Stream {stream_id} does not contain images"

    info: dict = reader.get_stream_info(stream_id)
    n_frames: int = info["data_records_count"]
    label: str = ARIA_STREAM_ID_TO_LABEL.get(stream_id, stream_id)

    # Filter to data records for this stream (pyvrs filtered iteration pattern)
    filtered = reader.filtered_by_fields(stream_ids=stream_id, record_types="data")

    # ── Phase 1: Read JPEG frames from VRS ──────────────────────────────
    t_read_start: float = time.perf_counter()
    jpeg_frames: list[bytes] = []
    timestamps_ns: list[int] = []

    for record in tqdm(filtered, total=n_frames, desc=f"Reading {label}", leave=False):
        if record.n_image_blocks == 0:
            continue

        # image_blocks[0] is a 1D uint8 ndarray of raw JPEG bytes (pyvrs convention)
        jpeg_bytes: bytes = record.image_blocks[0].tobytes()
        timestamp_sec: float = float(record.timestamp)
        timestamp_ns: int = int(timestamp_sec * 1e9)

        jpeg_frames.append(jpeg_bytes)
        timestamps_ns.append(timestamp_ns)

    t_read_elapsed: float = time.perf_counter() - t_read_start

    if not jpeg_frames:
        print(f"  [WARN] No image frames found in stream {stream_id}")
        return []

    # ── Phase 2+3: Overlapped decode + encode ───────────────────────────
    # Parallel JPEG→YUV decode feeds directly into NVENC encode.
    # ThreadPoolExecutor.map() returns a lazy iterator — decode runs ahead
    # while the main thread encodes, overlapping CPU decode with GPU encode.
    t_pipeline_start: float = time.perf_counter()

    # Calculate FPS from timestamps
    if len(timestamps_ns) > 1:
        dt_ns: float = float(timestamps_ns[-1] - timestamps_ns[0]) / (len(timestamps_ns) - 1)
        fps: float = 1e9 / dt_ns
    else:
        fps = 30.0

    n_frames: int = len(jpeg_frames)
    width: int = 0
    height: int = 0
    writer: MP4Writer = MP4Writer(output_path, codec=VideoCodecChoice.AV1, fps=fps)
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for planes in tqdm(pool.map(decode_jpeg_to_yuv, jpeg_frames), total=n_frames, desc=f"Decode+Encode {label}", leave=False):
            if width == 0:
                height, width = planes[0].shape
            if len(planes) >= 3:
                writer.write_yuv_planes(planes[0], planes[1], planes[2])
            else:
                writer.write_yuv_planes(planes[0])
    writer.close()

    t_pipeline_elapsed: float = time.perf_counter() - t_pipeline_start
    t_total: float = t_read_elapsed + t_pipeline_elapsed

    encoder_name: str = writer.encoder_name
    print(
        f"  {label}: {n_frames} frames, {width}x{height}, {fps:.1f}fps → {output_path.name} "
        f"[{encoder_name}] ({t_total:.1f}s total: read {t_read_elapsed:.1f}s, decode+encode {t_pipeline_elapsed:.1f}s)"
    )
    return timestamps_ns


def preprocess_sequence(seq_dir: Path, config: PreprocessConfig) -> None:
    """Preprocess a single HOT3D sequence."""
    vrs_path: Path = seq_dir / "recording.vrs"
    if not vrs_path.exists():
        print(f"  [SKIP] No recording.vrs in {seq_dir}")
        return

    output_dir: Path = seq_dir / OUTPUT_DIR_NAME
    if config.skip_existing and output_dir.exists():
        # Check if all expected outputs exist
        expected_files: list[str] = ["calibration.json", "timestamps_ns.json"]
        for sid in config.streams:
            label: str = ARIA_STREAM_ID_TO_LABEL.get(sid, sid)
            expected_files.append(STREAM_LABEL_TO_FILENAME.get(label, f"{label}.mp4"))
        if all((output_dir / f).exists() for f in expected_files):
            print(f"  [SKIP] Already preprocessed: {seq_dir.name}")
            return

    output_dir.mkdir(parents=True, exist_ok=True)
    t_seq_start: float = time.perf_counter()

    # Extract calibration from MPS online_calibration.jsonl
    cal_jsonl: Path = seq_dir / "mps" / "slam" / "online_calibration.jsonl"
    if cal_jsonl.exists():
        cal: Hot3dSequenceCalibration = parse_online_calibration_first(cal_jsonl)
        save_calibration(cal, output_dir / "calibration.json")
        print(f"  Calibration: {len(cal.streams)} streams extracted")
    else:
        print("  [WARN] No online_calibration.jsonl found")

    # Extract video streams
    all_timestamps: dict[str, list[int]] = {}
    for stream_id in config.streams:
        label = ARIA_STREAM_ID_TO_LABEL.get(stream_id, stream_id)
        filename: str = STREAM_LABEL_TO_FILENAME.get(label, f"{label}.mp4")
        output_path: Path = output_dir / filename

        timestamps: list[int] = extract_stream_to_mp4(
            vrs_path=vrs_path,
            stream_id=stream_id,
            output_path=output_path,
            num_workers=config.num_decode_workers,
        )
        all_timestamps[label] = timestamps

    # Save timestamps
    ts_path: Path = output_dir / "timestamps_ns.json"
    ts_path.write_text(json.dumps(all_timestamps))

    t_seq_elapsed: float = time.perf_counter() - t_seq_start
    print(f"  Done in {t_seq_elapsed:.1f}s ({len(config.streams)} streams)")


def main(config: PreprocessConfig) -> None:
    """Preprocess HOT3D VRS files."""
    root: Path = config.root
    assert root.exists(), f"Root directory not found: {root}"

    if config.sequence:
        # Single sequence
        seq_dir: Path = root / config.sequence
        assert seq_dir.exists(), f"Sequence not found: {seq_dir}"
        print(f"Processing: {config.sequence}")
        preprocess_sequence(seq_dir, config)
    else:
        # All sequences with recording.vrs
        seq_dirs: list[Path] = sorted([d for d in root.iterdir() if d.is_dir() and (d / "recording.vrs").exists()])
        print(f"Found {len(seq_dirs)} sequences with recording.vrs")
        for i, seq_dir in enumerate(seq_dirs):
            print(f"\n[{i + 1}/{len(seq_dirs)}] {seq_dir.name}")
            preprocess_sequence(seq_dir, config)

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    cfg: PreprocessConfig = tyro.cli(PreprocessConfig)
    main(cfg)
