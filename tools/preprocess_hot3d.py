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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import av
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
    codec: str = "libsvtav1"
    """Video codec for encoding. 'libsvtav1' (AV1 CPU), 'av1_nvenc' (AV1 GPU)."""
    crf: int = 30
    """Constant rate factor for encoding quality (lower = better, 0-63)."""
    num_decode_workers: int = 4
    """Number of parallel JPEG decode threads."""
    skip_existing: bool = True
    """Skip sequences that already have _simplecv/ output."""
    streams: list[str] = field(default_factory=lambda: ["214-1"])
    """VRS stream IDs to extract. Default: RGB only. Add '1201-1', '1201-2' for SLAM cameras."""


def decode_jpeg_to_bgr(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes to BGR numpy array using TurboJPEG."""
    return _TJ.decode(jpeg_bytes)


def extract_stream_to_mp4(
    vrs_path: Path,
    stream_id: str,
    output_path: Path,
    codec: str,
    crf: int,
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

    # Collect all JPEG frames + timestamps from VRS
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

    if not jpeg_frames:
        print(f"  [WARN] No image frames found in stream {stream_id}")
        return []

    # Parallel JPEG decode
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        bgr_frames: list[np.ndarray] = list(
            tqdm(
                pool.map(decode_jpeg_to_bgr, jpeg_frames),
                total=len(jpeg_frames),
                desc=f"Decoding {label}",
                leave=False,
            )
        )

    # Encode to MP4
    first_frame: np.ndarray = bgr_frames[0]
    height: int = first_frame.shape[0]
    width: int = first_frame.shape[1]

    # Calculate FPS from timestamps
    if len(timestamps_ns) > 1:
        dt_ns: float = float(timestamps_ns[-1] - timestamps_ns[0]) / (len(timestamps_ns) - 1)
        fps: float = 1e9 / dt_ns
    else:
        fps = 30.0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    container: av.container.OutputContainer = av.open(str(output_path), mode="w")
    stream: av.video.stream.VideoStream = container.add_stream(codec, rate=round(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    # Set CRF for quality control
    if "nvenc" in codec:
        stream.options = {"preset": "p4", "rc": "constqp", "qp": str(crf)}
    else:
        stream.options = {"crf": str(crf), "preset": "6"}

    for bgr_frame in tqdm(bgr_frames, desc=f"Encoding {label}", leave=False):
        # Convert BGR to RGB for av
        rgb_frame: np.ndarray = bgr_frame[:, :, ::-1]
        video_frame: av.VideoFrame = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")
        for packet in stream.encode(video_frame):
            container.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)

    container.close()

    print(f"  {label}: {len(bgr_frames)} frames, {width}x{height}, {fps:.1f}fps → {output_path.name}")
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
            codec=config.codec,
            crf=config.crf,
            num_workers=config.num_decode_workers,
        )
        all_timestamps[label] = timestamps

    # Save timestamps
    ts_path: Path = output_dir / "timestamps_ns.json"
    ts_path.write_text(json.dumps(all_timestamps))
    print(f"  Timestamps saved: {ts_path}")


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
