"""One-time VRS → MP4 conversion for Aria Gen2 Pilot sequences.

Aria Gen2 encodes video streams as H.265 inside VRS (unlike Gen1 which uses
JPEG). This script extracts H.265 NAL units from VRS and **remuxes** them
into MP4 containers without decode/encode — ~130x faster than transcoding.

For the RGB stream, the download provides a preview MP4 which is used directly
(just copied + timestamps extracted from VRS).

Usage:
    pixi run preprocess-aria-gen2-pilot --root /mnt/8tb/data/aria-gen2-pilot
    pixi run preprocess-aria-gen2-pilot --root /mnt/8tb/data/aria-gen2-pilot --sequence walk_1
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import pyvrs
from tqdm import tqdm

from simplecv.data.hot3d_utils import (
    Hot3dSequenceCalibration,
    Hot3dStreamCalibration,
    parse_online_calibration_first,
    save_calibration,
)

# ── Aria Gen2 stream mapping ──────────────────────────────────────────── #
# Gen2 has 5 cameras vs Gen1's 3, with different SLAM camera labels.

ARIA_GEN2_STREAM_ID_TO_LABEL: dict[str, str] = {
    "214-1": "camera-rgb",
    "1201-1": "slam-front-left",
    "1201-2": "slam-front-right",
    "1201-3": "slam-side-left",
    "1201-4": "slam-side-right",
}

ARIA_GEN2_STREAM_LABEL_TO_FILENAME: dict[str, str] = {
    "camera-rgb": "rgb.mp4",
    "slam-front-left": "slam_front_left.mp4",
    "slam-front-right": "slam_front_right.mp4",
    "slam-side-left": "slam_side_left.mp4",
    "slam-side-right": "slam_side_right.mp4",
}

# Default streams to extract
ARIA_GEN2_STREAM_IDS: list[str] = ["214-1", "1201-1", "1201-2", "1201-3", "1201-4"]

OUTPUT_DIR_NAME: str = "_simplecv"
VRS_FILENAME: str = "video.vrs"


@dataclass
class PreprocessConfig:
    """Configuration for Aria Gen2 Pilot VRS preprocessing."""

    root: Path = Path("/mnt/8tb/data/aria-gen2-pilot")
    """Root directory containing sequence folders."""
    sequence: str = ""
    """Process a single sequence (empty = all sequences with video.vrs)."""
    skip_existing: bool = True
    """Skip sequences that already have _simplecv/ output."""
    streams: list[str] = field(default_factory=list)
    """VRS stream IDs to extract. Empty = default Aria Gen2 streams."""


def _extract_vrs_timestamps(vrs_path: Path, stream_id: str) -> list[int]:
    """Read VRS timestamps for a stream without decoding images.

    Returns list of nanosecond timestamps in device-time domain.
    """
    reader: pyvrs.SyncVRSReader = pyvrs.SyncVRSReader(str(vrs_path))
    filtered = reader.filtered_by_fields(stream_ids=stream_id, record_types="data")
    timestamps_ns: list[int] = []
    for rec in filtered:
        timestamps_ns.append(int(rec.timestamp * 1e9))
    return timestamps_ns


def _get_stream_dimensions(vrs_path: Path, stream_id: str) -> tuple[int, int]:
    """Read image dimensions from VRS stream image_spec.

    Returns (width, height).
    """
    reader: pyvrs.SyncVRSReader = pyvrs.SyncVRSReader(str(vrs_path))
    filtered = reader.filtered_by_fields(stream_ids=stream_id, record_types="data")
    for rec in filtered:
        if rec.image_specs:
            spec = rec.image_specs[0]
            return spec.width, spec.height
        break
    raise ValueError(f"Could not determine dimensions for stream {stream_id}")


def _pick_ffmpeg_encoder() -> str:
    """Return the best available ffmpeg AV1/H.265 encoder.

    Prefers NVENC GPU (av1_nvenc > hevc_nvenc) then CPU fallback (libsvtav1).
    Note: NVDEC (hevc_cuvid) is NOT used for decode because it cannot handle
    raw Annex-B input (``-f hevc``). CPU H.265 decode is fast enough (~380fps
    for 512x512).
    """
    import subprocess

    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True, text=True, timeout=10,
    )
    encoders: str = result.stdout

    for candidate in ["av1_nvenc", "hevc_nvenc"]:
        if candidate in encoders:
            return candidate
    return "libsvtav1"


def transcode_h265_stream_to_mp4(
    vrs_path: Path,
    stream_id: str,
    output_path: Path,
) -> list[int]:
    """Extract H.265 NAL units from VRS and transcode to yuv420p AV1 MP4.

    The VRS stores monochrome (gray8) H.265 which Rerun cannot decode
    (H.265 Rext profile). This function converts gray → yuv420p AV1
    via ffmpeg with full GPU pipeline: NVDEC decode + NVENC AV1 encode.

    Approach:
    1. Read raw H.265 Annex-B NAL units + timestamps from VRS
    2. Write concatenated bitstream to temp file
    3. Run ``ffmpeg -c:v hevc_cuvid -i tmp.h265 -c:v av1_nvenc -pix_fmt yuv420p out.mp4``

    Returns list of VRS timestamps in nanoseconds.
    """
    import subprocess
    import tempfile

    reader: pyvrs.SyncVRSReader = pyvrs.SyncVRSReader(str(vrs_path))
    label: str = ARIA_GEN2_STREAM_ID_TO_LABEL.get(stream_id, stream_id)
    info: dict = reader.get_stream_info(stream_id)
    n_frames: int = info["data_records_count"]

    # Phase 1: Read raw H.265 NAL units + timestamps from VRS
    t0: float = time.perf_counter()
    filtered = reader.filtered_by_fields(stream_ids=stream_id, record_types="data")
    nal_units: list[bytes] = []
    timestamps_ns: list[int] = []
    for rec in tqdm(filtered, total=n_frames, desc=f"Reading {label}", leave=False):
        if rec.n_image_blocks > 0:
            nal_units.append(rec.image_blocks[0].tobytes())
            timestamps_ns.append(int(rec.timestamp * 1e9))
    t_read: float = time.perf_counter() - t0

    if not nal_units:
        print(f"  [WARN] No frames in stream {stream_id}")
        return []

    # Phase 2: Write Annex-B bitstream to temp file
    tmp_h265 = tempfile.NamedTemporaryFile(suffix=".h265", delete=False)
    tmp_h265.write(b"".join(nal_units))
    tmp_h265.close()
    tmp_path: str = tmp_h265.name

    # Phase 3: ffmpeg GPU transcode gray H.265 → yuv420p AV1 MP4
    t1: float = time.perf_counter()
    if len(timestamps_ns) > 1:
        dt_ns: float = float(timestamps_ns[-1] - timestamps_ns[0]) / (len(timestamps_ns) - 1)
        fps: int = max(1, round(1e9 / dt_ns))
    else:
        fps = 30

    encoder: str = _pick_ffmpeg_encoder()
    cmd: list[str] = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "hevc", "-i", tmp_path,
        "-c:v", encoder, "-pix_fmt", "yuv420p",
        "-r", str(fps),
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    Path(tmp_path).unlink()

    if result.returncode:
        raise RuntimeError(f"ffmpeg failed for {label}: {result.stderr[-300:]}")

    t_transcode: float = time.perf_counter() - t1
    t_total: float = t_read + t_transcode
    print(
        f"  {label}: {len(nal_units)} frames → {output_path.name} "
        f"[{encoder}] ({t_total:.1f}s: read {t_read:.1f}s, transcode {t_transcode:.1f}s)"
    )
    return timestamps_ns


def _find_preview_mp4(seq_dir: Path) -> Path | None:
    """Find the preview RGB MP4 downloaded alongside the VRS."""
    candidates: list[Path] = list(seq_dir.glob("*_preview_rgb.mp4"))
    if candidates:
        return candidates[0]
    return None


def preprocess_sequence(seq_dir: Path, config: PreprocessConfig) -> None:
    """Preprocess a single Aria Gen2 Pilot sequence."""
    vrs_path: Path = seq_dir / VRS_FILENAME
    if not vrs_path.exists():
        print(f"  [SKIP] No {VRS_FILENAME} in {seq_dir}")
        return

    streams: list[str] = config.streams if config.streams else ARIA_GEN2_STREAM_IDS

    output_dir: Path = seq_dir / OUTPUT_DIR_NAME
    if config.skip_existing and output_dir.exists():
        expected_files: list[str] = ["calibration.json", "timestamps_ns.json"]
        for sid in streams:
            label: str = ARIA_GEN2_STREAM_ID_TO_LABEL.get(sid, sid)
            expected_files.append(ARIA_GEN2_STREAM_LABEL_TO_FILENAME.get(label, f"{label}.mp4"))
        if all((output_dir / f).exists() for f in expected_files):
            print(f"  [SKIP] Already preprocessed: {seq_dir.name}")
            return

    output_dir.mkdir(parents=True, exist_ok=True)
    t_seq_start: float = time.perf_counter()
    print(f"  Streams: {streams}")

    # ── Extract calibration with correct dimensions ──────────────────────
    cal_jsonl: Path = seq_dir / "mps" / "slam" / "online_calibration.jsonl"
    if not cal_jsonl.exists():
        print("  [WARN] No online_calibration.jsonl found, skipping")
        return

    cal: Hot3dSequenceCalibration = parse_online_calibration_first(cal_jsonl)

    # Fix image dimensions from VRS (online_calibration uses approximate cx*2/cy*2)
    vrs_dims: dict[str, tuple[int, int]] = {}
    for sid in streams:
        label: str = ARIA_GEN2_STREAM_ID_TO_LABEL.get(sid, sid)
        try:
            w, h = _get_stream_dimensions(vrs_path, sid)
            vrs_dims[label] = (w, h)
        except ValueError:
            pass

    for stream_cal in cal.streams:
        if stream_cal.stream_label in vrs_dims:
            stream_cal.width, stream_cal.height = vrs_dims[stream_cal.stream_label]

    save_calibration(cal, output_dir / "calibration.json")
    print(f"  Calibration: {len(cal.streams)} streams, dims patched from VRS")

    # ── Extract video streams ─────────────────────────────────────────────
    all_timestamps: dict[str, list[int]] = {}

    for stream_id in streams:
        label: str = ARIA_GEN2_STREAM_ID_TO_LABEL.get(stream_id, stream_id)
        filename: str = ARIA_GEN2_STREAM_LABEL_TO_FILENAME.get(label, f"{label}.mp4")
        output_path: Path = output_dir / filename

        if label == "camera-rgb":
            # RGB: use preview MP4 if available (much faster than VRS transcode)
            preview: Path | None = _find_preview_mp4(seq_dir)
            if preview is not None:
                shutil.copy2(str(preview), str(output_path))
                timestamps: list[int] = _extract_vrs_timestamps(vrs_path, stream_id)
                all_timestamps[label] = timestamps
                print(f"  {label}: copied preview MP4 ({preview.name}), {len(timestamps)} VRS timestamps")
                continue

        # SLAM cameras (or RGB fallback): gray H.265 → yuv420p via ffmpeg+NVENC
        timestamps = transcode_h265_stream_to_mp4(
            vrs_path=vrs_path,
            stream_id=stream_id,
            output_path=output_path,
        )
        all_timestamps[label] = timestamps

    # Save timestamps
    ts_path: Path = output_dir / "timestamps_ns.json"
    ts_path.write_text(json.dumps(all_timestamps))

    t_seq_elapsed: float = time.perf_counter() - t_seq_start
    print(f"  Done in {t_seq_elapsed:.1f}s ({len(streams)} streams)")


def main(config: PreprocessConfig) -> None:
    """Preprocess Aria Gen2 Pilot VRS files."""
    root: Path = config.root
    assert root.exists(), f"Root directory not found: {root}"

    if config.sequence:
        seq_dir: Path = root / config.sequence
        assert seq_dir.exists(), f"Sequence not found: {seq_dir}"
        print(f"Processing: {config.sequence}")
        preprocess_sequence(seq_dir, config)
    else:
        seq_dirs: list[Path] = sorted([d for d in root.iterdir() if d.is_dir() and (d / VRS_FILENAME).exists()])
        print(f"Found {len(seq_dirs)} sequences with {VRS_FILENAME}")
        for i, seq_dir in enumerate(seq_dirs):
            print(f"\n[{i + 1}/{len(seq_dirs)}] {seq_dir.name}")
            preprocess_sequence(seq_dir, config)

    print("\nPreprocessing complete.")
