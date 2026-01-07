"""Batch process ExoEgo sequences from S3: download, cut, and ingest to RRD."""

import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

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

    status: str = "pending"
    """Overall status: pending | downloading | cutting | ingesting | complete | failed"""
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


# =============================================================================
# Core Functions
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
        manifest: ProgressManifest = from_json(ProgressManifest, manifest_path.read_text())
        return manifest
    return ProgressManifest(s3_bucket=s3_bucket)


def save_manifest(manifest: ProgressManifest, output_dir: Path) -> None:
    """Save manifest to disk.

    Args:
        manifest: Manifest to save.
        output_dir: Directory to save manifest.json in.
    """
    manifest_path: Path = output_dir / "manifest.json"
    manifest_path.write_text(to_json(manifest, indent=2))


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

    # Find the sequence path (includes date prefix)
    episode_info_paths: list[UPath] = list(base_s3.glob(f"**/{sequence_id}/episode_info.json"))
    if not episode_info_paths:
        raise FileNotFoundError(f"Sequence {sequence_id} not found on S3")

    s3_sequence_path: UPath = episode_info_paths[0].parent
    local_sequence_dir: Path = output_dir / date_prefix / sequence_id

    # Create local directory
    local_sequence_dir.mkdir(parents=True, exist_ok=True)

    # Download episode_info.json
    episode_info_s3: UPath = s3_sequence_path / "episode_info.json"
    episode_info_local: Path = local_sequence_dir / "episode_info.json"
    if not episode_info_local.exists():
        episode_info_local.write_bytes(episode_info_s3.read_bytes())

    # Download synced/ directory using aws s3 sync (faster for many files)
    synced_s3: str = str(s3_sequence_path / "synced")
    synced_local: Path = local_sequence_dir / "synced"
    synced_local.mkdir(parents=True, exist_ok=True)

    # Use AWS CLI for efficient sync
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


def process_sequence(
    sequence_dir: Path,
    output_dir: Path,
    manifest: ProgressManifest,
    sequence_id: str,
    date_prefix: str,
    parallel_workers: int,
    cleanup_synced: bool = False,
) -> None:
    """Cut all episodes and ingest to RRD for a sequence.

    Args:
        sequence_dir: Path to downloaded sequence (contains synced/ and episode_info.json).
        output_dir: Base output directory for episodes.
        manifest: Progress manifest to update.
        sequence_id: ID of sequence being processed.
        date_prefix: Date prefix from S3 (e.g., 2025-11-24).
        parallel_workers: Number of parallel FFmpeg workers.
        cleanup_synced: If True, delete synced/ folder after cutting to save disk space.
    """
    # Load episode info
    episode_info_path: Path = sequence_dir / "episode_info.json"
    episode_info: EpisodeInfo = from_json(EpisodeInfo, episode_info_path.read_text())
    synced_dir: Path = sequence_dir / "synced"

    # Initialize episode statuses if not present
    seq_status: SequenceStatus = manifest.sequences[sequence_id]
    for episode in episode_info.episodes:
        ep_name: str = f"episode-{episode.episode_number:03d}"
        if ep_name not in seq_status.episodes:
            seq_status.episodes[ep_name] = EpisodeStatus()

    # Cut all episodes
    seq_status.status = "cutting"
    save_manifest(manifest, output_dir)

    for episode in tqdm(episode_info.episodes, desc=f"Cutting {sequence_id[:8]}"):
        ep_name: str = f"episode-{episode.episode_number:03d}"
        ep_status: EpisodeStatus = seq_status.episodes[ep_name]

        if not ep_status.cut:
            # session_id includes date_prefix for proper path structure
            cut_episode(
                synced_dir=synced_dir,
                output_dir=output_dir,
                session_id=f"{date_prefix}/{sequence_id}",
                episode=episode,
                parallel_workers=parallel_workers,
            )
            ep_status.cut = True
            save_manifest(manifest, output_dir)

    # Cleanup synced folder after cutting (saves disk space)
    if cleanup_synced and synced_dir.exists():
        print(f"  Cleaning up synced/ folder for {sequence_id[:8]}...")
        shutil.rmtree(synced_dir)

    # Ingest all episodes to RRD
    seq_status.status = "ingesting"
    save_manifest(manifest, output_dir)

    for episode in tqdm(episode_info.episodes, desc=f"Ingesting {sequence_id[:8]}"):
        ep_name: str = f"episode-{episode.episode_number:03d}"
        ep_status: EpisodeStatus = seq_status.episodes[ep_name]

        if not ep_status.rrd:
            ep_dir: Path = output_dir / date_prefix / sequence_id / "episodes" / ep_name
            rrd_path: Path = ep_dir / f"{ep_name}.rrd"

            # Run ingestion via direct function call
            ingest_config: IngestConfig = IngestConfig(
                exoego_dir=ep_dir,
                rr_config=RerunTyroConfig(save=rrd_path),
            )
            ingest_main(ingest_config)

            ep_status.rrd = True
            save_manifest(manifest, output_dir)

    seq_status.status = "complete"
    save_manifest(manifest, output_dir)


def reingest_sequence(
    sequence_dir: Path,
    output_dir: Path,
    manifest: ProgressManifest,
    sequence_id: str,
    date_prefix: str,
) -> None:
    """Re-ingest all episodes for a sequence (skip cutting, regenerate RRDs).

    This is used when the ingestion code has been fixed and we want to
    regenerate RRDs without re-cutting the videos.

    Args:
        sequence_dir: Path to the sequence directory.
        output_dir: Output directory for processed data.
        manifest: Progress manifest for tracking.
        sequence_id: The sequence UUID.
        date_prefix: Date prefix (e.g., "2025-11-24").
    """
    seq_status: SequenceStatus = manifest.sequences[sequence_id]
    episodes_dir: Path = output_dir / date_prefix / sequence_id / "episodes"

    if not episodes_dir.exists():
        print(f"  No episodes directory found for {sequence_id[:8]}, skipping.")
        return

    # Find all episode directories
    episode_dirs: list[Path] = sorted(episodes_dir.glob("episode-*"))
    if not episode_dirs:
        print(f"  No episode directories found for {sequence_id[:8]}, skipping.")
        return

    seq_status.status = "ingesting"
    save_manifest(manifest, output_dir)

    for ep_dir in tqdm(episode_dirs, desc=f"Re-ingesting {sequence_id[:8]}"):
        ep_name: str = ep_dir.name
        rrd_path: Path = ep_dir / f"{ep_name}.rrd"

        # Delete existing RRD if present
        if rrd_path.exists():
            rrd_path.unlink()

        # Run ingestion
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

    seq_status.status = "complete"
    save_manifest(manifest, output_dir)

def main(config: Config) -> None:
    """Main entry point for batch S3 processing.

    Args:
        config: Configuration with S3 and output settings.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load or create manifest
    manifest: ProgressManifest = load_or_create_manifest(config.output_dir, config.s3_bucket)

    # Discover sequences
    print(f"Discovering sequences in s3://{config.s3_bucket}...")
    sequences_with_dates: dict[str, str] = discover_sequences(config.s3_bucket, config.profile)
    print(f"Found {len(sequences_with_dates)} sequences with episode_info.json")

    # Initialize sequence statuses (with date prefix)
    for seq_id, date_prefix in sequences_with_dates.items():
        if seq_id not in manifest.sequences:
            manifest.sequences[seq_id] = SequenceStatus(date_prefix=date_prefix)
        elif not manifest.sequences[seq_id].date_prefix:
            # Update date_prefix if it was missing
            manifest.sequences[seq_id].date_prefix = date_prefix
    save_manifest(manifest, config.output_dir)

    # Handle reingest-only mode
    if config.reingest_only:
        # Process ALL sequences that have episodes directories (including complete ones)
        sequences_to_reingest: list[str] = [
            seq_id for seq_id, status in manifest.sequences.items()
            if (config.output_dir / status.date_prefix / seq_id / "episodes").exists()
        ]
        print(f"Sequences to re-ingest: {len(sequences_to_reingest)}")
        print("  (Re-ingesting all sequences with existing episodes)")

        if config.dry_run:
            print("\n[DRY RUN] Would re-ingest:")
            total_episodes: int = 0
            for seq_id in sequences_to_reingest:
                episodes_dir: Path = config.output_dir / manifest.sequences[seq_id].date_prefix / seq_id / "episodes"
                ep_count: int = len(list(episodes_dir.glob("episode-*")))
                total_episodes += ep_count
                print(f"  - {seq_id} ({ep_count} episodes)")
            # ~13 seconds per episode based on observed ingestion times
            est_minutes: float = total_episodes * 13 / 60
            print(f"\nTotal: {len(sequences_to_reingest)} sequences, {total_episodes} episodes")
            print(f"Estimated time: ~{est_minutes:.0f} minutes ({est_minutes / 60:.1f} hours)")
            return

        for seq_id in tqdm(sequences_to_reingest, desc="Re-ingesting sequences"):
            seq_status: SequenceStatus = manifest.sequences[seq_id]
            date_prefix: str = seq_status.date_prefix
            sequence_dir: Path = config.output_dir / date_prefix / seq_id

            try:
                print(f"\nRe-ingesting {seq_id}...")
                reingest_sequence(
                    sequence_dir=sequence_dir,
                    output_dir=config.output_dir,
                    manifest=manifest,
                    sequence_id=seq_id,
                    date_prefix=date_prefix,
                )
            except Exception as e:
                seq_status.status = "failed"
                seq_status.error = str(e)
                save_manifest(manifest, config.output_dir)
                print(f"ERROR re-ingesting {seq_id}: {e}")
                continue

        complete: int = sum(1 for s in manifest.sequences.values() if s.status == "complete")
        failed: int = sum(1 for s in manifest.sequences.values() if s.status == "failed")
        print(f"\nDone! Complete: {complete}, Failed: {failed}, Total: {len(manifest.sequences)}")
        return

    # Normal mode: Filter to pending/failed sequences
    sequences_to_process: list[str] = [
        seq_id for seq_id, status in manifest.sequences.items()
        if status.status not in ("complete",)
    ]
    print(f"Sequences to process: {len(sequences_to_process)}")

    # Show summary
    complete: int = sum(1 for s in manifest.sequences.values() if s.status == "complete")
    failed: int = sum(1 for s in manifest.sequences.values() if s.status == "failed")
    pending: int = len(sequences_to_process)
    print(f"  Complete: {complete} | Failed: {failed} | Pending: {pending}")

    if config.dry_run:
        print("\n[DRY RUN] Would process:")
        for _seq_id in tqdm(sequences_to_process, desc="Sequences to process", leave=True):
            pass  # tqdm shows progress bar
        print(f"\nEstimated time: ~{pending * 5.4:.0f} minutes ({pending * 5.4 / 60:.1f} hours)")
        return

    # Process each sequence
    for seq_id in tqdm(sequences_to_process, desc="Processing sequences"):
        seq_status: SequenceStatus = manifest.sequences[seq_id]
        date_prefix: str = seq_status.date_prefix

        try:
            # Download if needed
            sequence_dir: Path = config.output_dir / date_prefix / seq_id
            if not (sequence_dir / "synced").exists():
                print(f"\nDownloading {seq_id}...")
                seq_status.status = "downloading"
                save_manifest(manifest, config.output_dir)

                download_sequence(
                    s3_bucket=config.s3_bucket,
                    sequence_id=seq_id,
                    date_prefix=date_prefix,
                    output_dir=config.output_dir,
                    profile=config.profile,
                )
                seq_status.downloaded_at = datetime.now().isoformat()
                save_manifest(manifest, config.output_dir)

            # Process (cut + ingest)
            print(f"\nProcessing {seq_id}...")
            process_sequence(
                sequence_dir=sequence_dir,
                output_dir=config.output_dir,
                manifest=manifest,
                sequence_id=seq_id,
                date_prefix=date_prefix,
                parallel_workers=config.parallel_workers,
                cleanup_synced=config.cleanup_synced,
            )

        except Exception as e:
            seq_status.status = "failed"
            seq_status.error = str(e)
            save_manifest(manifest, config.output_dir)
            print(f"ERROR processing {seq_id}: {e}")
            continue

    # Summary
    complete: int = sum(1 for s in manifest.sequences.values() if s.status == "complete")
    failed: int = sum(1 for s in manifest.sequences.values() if s.status == "failed")
    print(f"\nDone! Complete: {complete}, Failed: {failed}, Total: {len(manifest.sequences)}")
