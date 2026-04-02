"""Download HOT3D Aria sequences from CDN URL JSON.

Reads the download URL JSON (e.g. Hot3DAria_download_urls.json) obtained from
projectaria.com and fetches selected data types for each sequence.

Usage:
    pixi run download-hot3d \\
        --urls-json /mnt/8tb/data/hot3d/Hot3DAria_download_urls.json \\
        --output-dir /mnt/8tb/data/hot3d/aria \\
        --max-sequences 20
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import requests
import tyro
from tqdm import tqdm

# Data types we want for simplecv integration. Skip mps_slam_points (huge, 400MB+)
# and mps_artifacts (huge, 460MB+) and mps_slam_summary (trivial).
DESIRED_DATA_TYPES: list[str] = [
    "main_vrs",
    "hand_data",
    "mps_slam_trajectories",
    "mps_slam_calibration",
    "ground_truth",
    "video_main_rgb",
    "mps_eye_gaze",
]

# Mapping from data type to subdirectory inside the sequence folder.
# Mirrors HOT3D's DATA_TYPE_TO_SAVE_PATH convention.
DATA_TYPE_SAVE_PATHS: dict[str, str] = {
    "main_vrs": ".",
    "video_main_rgb": ".",
    "hand_data": ".",
    "ground_truth": ".",
    "mps_slam_trajectories": "mps/slam",
    "mps_slam_calibration": "mps/slam",
    "mps_slam_points": "mps/slam",
    "mps_slam_summary": "mps/slam",
    "mps_eye_gaze": "mps/eye_gaze",
    "mps_artifacts": "mps",
}

# VRS files get renamed to this canonical name after download.
VRS_CANONICAL_NAME: str = "recording.vrs"


@dataclass
class DownloadConfig:
    """Configuration for HOT3D download."""

    urls_json: Path
    """Path to the Hot3DAria_download_urls.json file."""
    output_dir: Path = Path("/mnt/8tb/data/hot3d/aria")
    """Output directory for downloaded sequences."""
    max_sequences: int | None = 20
    """Maximum number of sequences to download (None = all)."""
    data_types: list[str] = field(default_factory=lambda: DESIRED_DATA_TYPES)
    """Data types to download per sequence."""
    verify_sha1: bool = True
    """Verify SHA1 checksums after download."""


def download_file(url: str, dest_path: Path, expected_size: int | None = None) -> None:
    """Download a file with progress bar, skipping if already complete."""
    if dest_path.exists() and expected_size is not None and dest_path.stat().st_size == expected_size:
        return  # Already downloaded
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    response: requests.Response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    total_size: int = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=dest_path.name,
        leave=False,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def verify_sha1(file_path: Path, expected_sha1: str) -> bool:
    """Verify SHA1 checksum of a downloaded file."""
    sha1 = hashlib.sha1()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha1.update(chunk)
    return sha1.hexdigest() == expected_sha1


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """Extract a ZIP file and remove the archive."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    zip_path.unlink()


def download_sequence(
    sequence_name: str,
    sequence_data: dict[str, dict],
    output_dir: Path,
    data_types: list[str],
    verify: bool,
) -> bool:
    """Download all requested data types for a single sequence."""
    seq_dir: Path = output_dir / sequence_name
    seq_dir.mkdir(parents=True, exist_ok=True)

    success: bool = True
    for dtype in data_types:
        if dtype not in sequence_data:
            continue

        info: dict = sequence_data[dtype]
        filename: str = info["filename"]
        url: str = info["download_url"]
        expected_size: int = info["file_size_bytes"]
        expected_sha1: str = info["sha1sum"]

        # Determine save location
        save_subdir: str = DATA_TYPE_SAVE_PATHS.get(dtype, ".")
        save_dir: Path = seq_dir / save_subdir
        save_dir.mkdir(parents=True, exist_ok=True)
        dest_path: Path = save_dir / filename

        # Download
        try:
            download_file(url, dest_path, expected_size)
        except Exception as exc:
            print(f"  [FAIL] {dtype}: {exc}")
            success = False
            continue

        # Verify checksum
        if verify and not verify_sha1(dest_path, expected_sha1):
            print(f"  [FAIL] {dtype}: SHA1 mismatch for {filename}")
            dest_path.unlink(missing_ok=True)
            success = False
            continue

        # Extract ZIPs
        if filename.endswith(".zip"):
            extract_zip(dest_path, save_dir)

        # Rename VRS to canonical name
        if dtype == "main_vrs" and dest_path.exists():
            canonical: Path = seq_dir / VRS_CANONICAL_NAME
            if not canonical.exists():
                shutil.move(str(dest_path), str(canonical))

    return success


def main(config: DownloadConfig) -> None:
    """Download HOT3D Aria sequences."""
    assert config.urls_json.exists(), f"URL JSON not found: {config.urls_json}"

    with open(config.urls_json) as f:
        data: dict = json.load(f)

    sequences: dict[str, dict] = data["sequences"]
    seq_names: list[str] = list(sequences.keys())

    if config.max_sequences is not None:
        seq_names = seq_names[: config.max_sequences]

    print(f"Downloading {len(seq_names)} sequences to {config.output_dir}")
    print(f"Data types: {config.data_types}")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    for i, seq_name in enumerate(seq_names):
        print(f"\n[{i + 1}/{len(seq_names)}] {seq_name}")
        ok: bool = download_sequence(
            sequence_name=seq_name,
            sequence_data=sequences[seq_name],
            output_dir=config.output_dir,
            data_types=config.data_types,
            verify=config.verify_sha1,
        )
        if ok:
            print(f"  [OK] {seq_name}")
        else:
            print(f"  [PARTIAL] {seq_name}")

    print(f"\nDone. Downloaded {len(seq_names)} sequences to {config.output_dir}")


if __name__ == "__main__":
    cfg: DownloadConfig = tyro.cli(DownloadConfig)
    main(cfg)
