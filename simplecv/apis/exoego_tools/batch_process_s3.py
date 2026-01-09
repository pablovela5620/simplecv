"""Batch process ExoEgo sequences from S3: download, cut, and ingest to RRD."""

import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal

from serde import serde
from serde.json import from_json, to_json
from tqdm.auto import tqdm
from upath import UPath

from simplecv.apis.exoego_tools.cut_synced_sequences import (
    EpisodeInfo,
    cut_episode,
)
from simplecv.apis.ingest_exoego_recording import IngestConfig
from simplecv.apis.ingest_exoego_recording import main as ingest_main
from simplecv.rerun_log_utils import RerunTyroConfig

# =============================================================================
# Type Aliases & Constants
# =============================================================================

SequenceStatusLiteral = Literal[
    "pending", "downloading", "cutting", "ingesting", "cut_complete", "complete", "failed"
]


class ProcessMode(Enum):
    """Processing mode for the batch pipeline."""

    NORMAL = "normal"
    """Full pipeline: download → cut → ingest."""
    CUT_ONLY = "cut_only"
    """Download and cut only, skip ingestion."""
    REINGEST = "reingest"
    """Re-ingest all episodes, regenerating RRDs."""


# =============================================================================
# Data Structures
# =============================================================================


@serde
@dataclass
class EpisodeStatus:
    """Status of a single episode."""

    cut: bool = False
    """Whether the episode has been cut."""
    rrd: bool = False
    """Whether the episode has been ingested to RRD."""


@serde
@dataclass
class SequenceStatus:
    """Status of a sequence and its episodes."""

    status: SequenceStatusLiteral = "pending"
    """Overall status: pending | downloading | cutting | ingesting | cut_complete | complete | failed"""
    date_prefix: str = ""
    """Date prefix from S3 path (e.g., 2025-11-24)."""
    downloaded_at: str | None = None
    """ISO timestamp when download completed."""
    error: str | None = None
    """Error message if failed."""
    episodes: dict[str, EpisodeStatus] = field(default_factory=dict)
    """Per-episode status."""


@serde
@dataclass
class ProgressManifest:
    """Tracks progress of batch processing."""

    s3_bucket: str
    """S3 bucket being processed."""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    """When the manifest was created."""
    sequences: dict[str, SequenceStatus] = field(default_factory=dict)
    """Per-sequence status."""


@dataclass
class Config:
    """Configuration for batch S3 processing."""

    s3_bucket: str
    """S3 bucket containing ExoEgo sequences."""
    output_dir: Path
    """Local directory for downloaded and processed data."""
    profile: str = "pablo-sso"
    """AWS profile name for S3 access."""
    parallel_workers: int = 4
    """Number of parallel FFmpeg processes for video cutting."""
    dry_run: bool = False
    """If True, print what would be done without actually processing."""
    cleanup_synced: bool = False
    """If True, delete synced/ folder after cutting to save disk space."""
    reingest_only: bool = False
    """If True, skip downloading/cutting and only re-run ingestion on existing cut episodes."""
    cut_only: bool = False
    """If True, only download and cut videos; skip RRD ingestion."""


# =============================================================================
# Manifest Helpers
# =============================================================================


def load_or_create_manifest(output_dir: Path, s3_bucket: str) -> ProgressManifest:
    """Load existing manifest or create a new one.

    Args:
        output_dir: Directory containing manifest.json.
        s3_bucket: S3 bucket being processed.

    Returns:
        Loaded or newly created manifest.
    """
    manifest_path: Path = output_dir / "manifest.json"
    if manifest_path.exists():
        return from_json(ProgressManifest, manifest_path.read_text())
    return ProgressManifest(s3_bucket=s3_bucket)


def save_manifest(manifest: ProgressManifest, output_dir: Path) -> None:
    """Save manifest to disk.

    Args:
        manifest: Manifest to save.
        output_dir: Directory to save manifest.json in.
    """
    manifest_path: Path = output_dir / "manifest.json"
    manifest_path.write_text(to_json(manifest, indent=2))


def print_summary(manifest: ProgressManifest) -> None:
    """Print status summary for all sequences.

    Args:
        manifest: Progress manifest containing sequence statuses.
    """
    complete: int = sum(1 for s in manifest.sequences.values() if s.status == "complete")
    cut_complete: int = sum(1 for s in manifest.sequences.values() if s.status == "cut_complete")
    failed: int = sum(1 for s in manifest.sequences.values() if s.status == "failed")
    pending: int = len(manifest.sequences) - complete - cut_complete - failed
    print(f"  Complete: {complete} | Cut Complete: {cut_complete} | Failed: {failed} | Pending: {pending}")


# =============================================================================
# S3 Operations
# =============================================================================


def discover_sequences(s3_bucket: str, profile: str) -> dict[str, str]:
    """Discover all sequences with episode_info.json on S3.

    Args:
        s3_bucket: S3 bucket to search.
        profile: AWS profile for authentication.

    Returns:
        Dict mapping sequence_id to date prefix (e.g., {"a8e0...": "2025-11-24"}).
    """
    base_path: UPath = UPath(f"s3://{s3_bucket}", profile=profile)
    episode_info_paths: list[UPath] = list(base_path.glob("**/episode_info.json"))

    sequences: dict[str, str] = {}
    for path in episode_info_paths:
        # Path structure: s3://bucket/date/sequence_id/episode_info.json
        sequence_id: str = path.parent.name
        date_prefix: str = path.parent.parent.name
        sequences[sequence_id] = date_prefix

    return sequences


def download_sequence(
    s3_bucket: str,
    sequence_id: str,
    date_prefix: str,
    output_dir: Path,
    profile: str,
) -> Path:
    """Download synced/ and episode_info.json for a sequence.

    Args:
        s3_bucket: S3 bucket.
        sequence_id: UUID of the sequence.
        date_prefix: Date prefix from S3 (e.g., 2025-11-24).
        output_dir: Local output directory.
        profile: AWS profile.

    Returns:
        Path to the downloaded sequence directory.
    """
    base_s3: UPath = UPath(f"s3://{s3_bucket}", profile=profile)

    # Find the sequence path
    episode_info_paths: list[UPath] = list(base_s3.glob(f"**/{sequence_id}/episode_info.json"))
    if not episode_info_paths:
        raise FileNotFoundError(f"Sequence {sequence_id} not found on S3")

    s3_sequence_path: UPath = episode_info_paths[0].parent
    local_sequence_dir: Path = output_dir / date_prefix / sequence_id
    local_sequence_dir.mkdir(parents=True, exist_ok=True)

    # Download episode_info.json
    episode_info_local: Path = local_sequence_dir / "episode_info.json"
    if not episode_info_local.exists():
        episode_info_s3: UPath = s3_sequence_path / "episode_info.json"
        episode_info_local.write_bytes(episode_info_s3.read_bytes())

    # Download metadata.json (session metadata: collector, time, location, weather)
    metadata_local: Path = local_sequence_dir / "metadata.json"
    if not metadata_local.exists():
        metadata_s3: UPath = s3_sequence_path / "metadata.json"
        if metadata_s3.exists():
            metadata_local.write_bytes(metadata_s3.read_bytes())

    # Download synced/ directory using aws s3 sync
    synced_s3: str = str(s3_sequence_path / "synced")
    synced_local: Path = local_sequence_dir / "synced"
    synced_local.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        "aws", "s3", "sync",
        synced_s3, str(synced_local),
        "--profile", profile,
        "--quiet",
    ]
    result: subprocess.CompletedProcess[str] = subprocess.run(
        cmd, check=True, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"S3 sync failed: {result.stderr}")

    return local_sequence_dir


def ensure_downloaded(
    config: Config,
    manifest: ProgressManifest,
    seq_id: str,
    seq_status: SequenceStatus,
) -> Path:
    """Download sequence if synced/ doesn't exist.

    Args:
        config: Batch processing configuration.
        manifest: Progress manifest for tracking.
        seq_id: Sequence UUID.
        seq_status: Status object for the sequence.

    Returns:
        Path to the sequence directory.
    """
    sequence_dir: Path = config.output_dir / seq_status.date_prefix / seq_id

    if not (sequence_dir / "synced").exists():
        print(f"\nDownloading {seq_id}...")
        seq_status.status = "downloading"
        save_manifest(manifest, config.output_dir)

        download_sequence(
            s3_bucket=config.s3_bucket,
            sequence_id=seq_id,
            date_prefix=seq_status.date_prefix,
            output_dir=config.output_dir,
            profile=config.profile,
        )
        seq_status.downloaded_at = datetime.now().isoformat()
        save_manifest(manifest, config.output_dir)

    return sequence_dir


# =============================================================================
# Episode Processing
# =============================================================================


def ingest_episode(
    ep_dir: Path,
    ep_name: str,
    seq_status: SequenceStatus,
    output_dir: Path,
    manifest: ProgressManifest,
    force: bool = False,
) -> None:
    """Ingest a single episode to RRD.

    Args:
        ep_dir: Path to the episode directory.
        ep_name: Name of the episode (e.g., "episode-001").
        seq_status: Sequence status object to update.
        output_dir: Base output directory.
        manifest: Progress manifest for tracking.
        force: If True, delete existing RRD first (for reingest mode).
    """
    rrd_path: Path = ep_dir / f"{ep_name}.rrd"

    if force and rrd_path.exists():
        rrd_path.unlink()

    ingest_config: IngestConfig = IngestConfig(
        exoego_dir=ep_dir,
        rr_config=RerunTyroConfig(save=rrd_path),
    )
    ingest_main(ingest_config)

    # Update episode status
    if ep_name not in seq_status.episodes:
        seq_status.episodes[ep_name] = EpisodeStatus(cut=True, rrd=True)
    else:
        seq_status.episodes[ep_name].rrd = True
    save_manifest(manifest, output_dir)


# =============================================================================
# Sequence Processing
# =============================================================================


def cut_sequence(
    sequence_dir: Path,
    output_dir: Path,
    manifest: ProgressManifest,
    sequence_id: str,
    date_prefix: str,
    parallel_workers: int,
    cleanup_synced: bool = False,
) -> None:
    """Cut all episodes for a sequence (respects episode-level cut status).

    Args:
        sequence_dir: Path to downloaded sequence (contains synced/ and episode_info.json).
        output_dir: Base output directory for episodes.
        manifest: Progress manifest to update.
        sequence_id: ID of sequence being processed.
        date_prefix: Date prefix from S3 (e.g., 2025-11-24).
        parallel_workers: Number of parallel FFmpeg workers.
        cleanup_synced: If True, delete synced/ folder after cutting.
    """
    episode_info_path: Path = sequence_dir / "episode_info.json"
    episode_info: EpisodeInfo = from_json(EpisodeInfo, episode_info_path.read_text())
    synced_dir: Path = sequence_dir / "synced"

    seq_status: SequenceStatus = manifest.sequences[sequence_id]

    # Initialize episode statuses
    for episode in episode_info.episodes:
        ep_name: str = f"episode-{episode.episode_number:03d}"
        if ep_name not in seq_status.episodes:
            seq_status.episodes[ep_name] = EpisodeStatus()

    seq_status.status = "cutting"
    save_manifest(manifest, output_dir)

    for episode in tqdm(episode_info.episodes, desc=f"Cutting {sequence_id[:8]}"):
        ep_name: str = f"episode-{episode.episode_number:03d}"
        ep_status: EpisodeStatus = seq_status.episodes[ep_name]

        if not ep_status.cut:
            cut_episode(
                synced_dir=synced_dir,
                output_dir=output_dir,
                session_id=f"{date_prefix}/{sequence_id}",
                episode=episode,
                parallel_workers=parallel_workers,
            )
            ep_status.cut = True
            save_manifest(manifest, output_dir)

    # Cleanup synced folder
    if cleanup_synced and synced_dir.exists():
        print(f"  Cleaning up synced/ folder for {sequence_id[:8]}...")
        shutil.rmtree(synced_dir)


def ingest_sequence(
    output_dir: Path,
    manifest: ProgressManifest,
    sequence_id: str,
    date_prefix: str,
    force: bool = False,
) -> None:
    """Ingest episodes for a sequence.

    Args:
        output_dir: Output directory for processed data.
        manifest: Progress manifest for tracking.
        sequence_id: The sequence UUID.
        date_prefix: Date prefix (e.g., "2025-11-24").
        force: If True, regenerate all RRDs (reingest mode).
               If False, only ingest episodes where rrd=False (incremental).
    """
    seq_status: SequenceStatus = manifest.sequences[sequence_id]
    episodes_dir: Path = output_dir / date_prefix / sequence_id / "episodes"

    if not episodes_dir.exists():
        print(f"  No episodes directory found for {sequence_id[:8]}, skipping.")
        return

    episode_dirs: list[Path] = sorted(episodes_dir.glob("episode-*"))
    if not episode_dirs:
        print(f"  No episode directories found for {sequence_id[:8]}, skipping.")
        return

    # Filter to episodes that need ingestion (unless force=True)
    if force:
        episodes_to_ingest: list[Path] = episode_dirs
    else:
        episodes_to_ingest = [
            ep_dir for ep_dir in episode_dirs
            if not seq_status.episodes.get(ep_dir.name, EpisodeStatus()).rrd
        ]

    if not episodes_to_ingest:
        print(f"  All episodes already ingested for {sequence_id[:8]}.")
        seq_status.status = "complete"
        save_manifest(manifest, output_dir)
        return

    if not force:
        print(f"  {len(episodes_to_ingest)}/{len(episode_dirs)} episodes need ingestion.")

    seq_status.status = "ingesting"
    save_manifest(manifest, output_dir)

    desc: str = f"Re-ingesting {sequence_id[:8]}" if force else f"Ingesting {sequence_id[:8]}"
    for ep_dir in tqdm(episodes_to_ingest, desc=desc):
        ingest_episode(
            ep_dir=ep_dir,
            ep_name=ep_dir.name,
            seq_status=seq_status,
            output_dir=output_dir,
            manifest=manifest,
            force=force,
        )

    seq_status.status = "complete"
    save_manifest(manifest, output_dir)


def process_sequence(
    sequence_dir: Path,
    output_dir: Path,
    manifest: ProgressManifest,
    sequence_id: str,
    date_prefix: str,
    parallel_workers: int,
    cleanup_synced: bool = False,
    cut_only: bool = False,
) -> None:
    """Cut and optionally ingest all episodes for a sequence.

    Args:
        sequence_dir: Path to downloaded sequence.
        output_dir: Base output directory for episodes.
        manifest: Progress manifest to update.
        sequence_id: ID of sequence being processed.
        date_prefix: Date prefix from S3 (e.g., 2025-11-24).
        parallel_workers: Number of parallel FFmpeg workers.
        cleanup_synced: If True, delete synced/ folder after cutting.
        cut_only: If True, skip RRD ingestion after cutting.
    """
    # Cut phase
    cut_sequence(
        sequence_dir=sequence_dir,
        output_dir=output_dir,
        manifest=manifest,
        sequence_id=sequence_id,
        date_prefix=date_prefix,
        parallel_workers=parallel_workers,
        cleanup_synced=cleanup_synced,
    )

    seq_status: SequenceStatus = manifest.sequences[sequence_id]

    if cut_only:
        seq_status.status = "cut_complete"
        save_manifest(manifest, output_dir)
        return

    # Ingest phase
    ingest_sequence(
        output_dir=output_dir,
        manifest=manifest,
        sequence_id=sequence_id,
        date_prefix=date_prefix,
        force=False,
    )


# =============================================================================
# Main Entry Point
# =============================================================================


def process_sequences_loop(
    sequences: list[str],
    manifest: ProgressManifest,
    config: Config,
    mode: ProcessMode,
) -> None:
    """Process a list of sequences with unified error handling.

    Args:
        sequences: List of sequence IDs to process.
        manifest: Progress manifest for tracking.
        config: Batch processing configuration.
        mode: Processing mode (normal, cut_only, reingest).
    """
    desc_map: dict[ProcessMode, str] = {
        ProcessMode.NORMAL: "Processing sequences",
        ProcessMode.CUT_ONLY: "Cutting sequences",
        ProcessMode.REINGEST: "Re-ingesting sequences",
    }

    for seq_id in tqdm(sequences, desc=desc_map[mode]):
        seq_status: SequenceStatus = manifest.sequences[seq_id]

        try:
            if mode == ProcessMode.REINGEST:
                print(f"\nRe-ingesting {seq_id}...")
                ingest_sequence(
                    output_dir=config.output_dir,
                    manifest=manifest,
                    sequence_id=seq_id,
                    date_prefix=seq_status.date_prefix,
                    force=True,
                )

            elif mode == ProcessMode.CUT_ONLY:
                sequence_dir: Path = ensure_downloaded(config, manifest, seq_id, seq_status)
                print(f"\nCutting {seq_id}...")
                process_sequence(
                    sequence_dir=sequence_dir,
                    output_dir=config.output_dir,
                    manifest=manifest,
                    sequence_id=seq_id,
                    date_prefix=seq_status.date_prefix,
                    parallel_workers=config.parallel_workers,
                    cleanup_synced=config.cleanup_synced,
                    cut_only=True,
                )

            else:  # NORMAL mode
                if seq_status.status == "cut_complete":
                    # Already cut, just ingest
                    print(f"\nIngesting {seq_id} (already cut)...")
                    ingest_sequence(
                        output_dir=config.output_dir,
                        manifest=manifest,
                        sequence_id=seq_id,
                        date_prefix=seq_status.date_prefix,
                        force=False,
                    )
                else:
                    # Full pipeline
                    sequence_dir = ensure_downloaded(config, manifest, seq_id, seq_status)
                    print(f"\nProcessing {seq_id}...")
                    process_sequence(
                        sequence_dir=sequence_dir,
                        output_dir=config.output_dir,
                        manifest=manifest,
                        sequence_id=seq_id,
                        date_prefix=seq_status.date_prefix,
                        parallel_workers=config.parallel_workers,
                        cleanup_synced=config.cleanup_synced,
                        cut_only=False,
                    )

        except Exception as e:
            seq_status.status = "failed"
            seq_status.error = str(e)
            save_manifest(manifest, config.output_dir)
            print(f"ERROR processing {seq_id}: {e}")
            continue


def main(config: Config) -> None:
    """Main entry point for batch S3 processing."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load or create manifest
    manifest: ProgressManifest = load_or_create_manifest(config.output_dir, config.s3_bucket)

    # Discover sequences
    print(f"Discovering sequences in s3://{config.s3_bucket}...")
    sequences_with_dates: dict[str, str] = discover_sequences(config.s3_bucket, config.profile)
    print(f"Found {len(sequences_with_dates)} sequences with episode_info.json")

    # Initialize sequence statuses
    for seq_id, date_prefix in sequences_with_dates.items():
        if seq_id not in manifest.sequences:
            manifest.sequences[seq_id] = SequenceStatus(date_prefix=date_prefix)
        elif not manifest.sequences[seq_id].date_prefix:
            manifest.sequences[seq_id].date_prefix = date_prefix
    save_manifest(manifest, config.output_dir)

    # Determine mode and filter sequences
    if config.reingest_only:
        mode: ProcessMode = ProcessMode.REINGEST
        sequences: list[str] = [
            seq_id for seq_id, status in manifest.sequences.items()
            if (config.output_dir / status.date_prefix / seq_id / "episodes").exists()
        ]
        print(f"Sequences to re-ingest: {len(sequences)}")

    elif config.cut_only:
        mode = ProcessMode.CUT_ONLY
        sequences = [
            seq_id for seq_id, status in manifest.sequences.items()
            if status.status not in ("complete", "cut_complete")
        ]
        print(f"Sequences to cut: {len(sequences)}")

    else:
        mode = ProcessMode.NORMAL
        sequences = [
            seq_id for seq_id, status in manifest.sequences.items()
            if status.status != "complete"
        ]
        print(f"Sequences to process: {len(sequences)}")

    print_summary(manifest)

    # Dry run
    if config.dry_run:
        print(f"\n[DRY RUN] Would process {len(sequences)} sequences")
        for _ in tqdm(sequences, desc="Sequences", leave=True):
            pass
        return

    # Process
    process_sequences_loop(sequences, manifest, config, mode)

    # Final summary
    complete: int = sum(1 for s in manifest.sequences.values() if s.status == "complete")
    failed: int = sum(1 for s in manifest.sequences.values() if s.status == "failed")
    print(f"\nDone! Complete: {complete}, Failed: {failed}, Total: {len(manifest.sequences)}")
