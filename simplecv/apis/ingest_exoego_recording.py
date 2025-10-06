import json
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess
from typing import cast

import rerun as rr
import rerun.blueprint as rrb
from natsort import natsorted
from rerun.blueprint import ContainerLike
from tqdm.auto import tqdm

from simplecv.rerun_log_utils import RerunTyroConfig, log_video
from simplecv.video_utils import Resolution, reencode_video_optimal


@dataclass
class IngestConfig:
    """Structured configuration for running the exo/ego visualization CLI."""

    rr_config: RerunTyroConfig
    """Command-line options for spawning and configuring the Rerun viewer."""
    exoego_dir: Path
    """Path to the directory containing 'exo' and/or 'ego' subdirectories with video files."""
    verbose: bool = False
    """Enable verbose logging with progress bars during ingestion."""


def validate_exoego_dir(exoego_dir: Path) -> tuple[Path | None, Path | None]:
    if not exoego_dir.exists():
        raise ValueError(f"The provided directory does not exist: {exoego_dir}")
    if not exoego_dir.is_dir():
        raise ValueError(f"The provided path is not a directory: {exoego_dir}")

    # make sure that either "exo" or "ego" subdirectory exists
    if not (exoego_dir / "exo").exists() and not (exoego_dir / "ego").exists():
        raise ValueError(f"The provided directory does not contain 'exo' or 'ego' subdirectory: {exoego_dir}")

    return (exoego_dir / "exo") if (exoego_dir / "exo").exists() else None, (exoego_dir / "ego") if (
        exoego_dir / "ego"
    ).exists() else None


@dataclass(frozen=True, slots=True)
class VideoProbeResult:
    """Metadata extracted via ffprobe for the primary video stream."""

    codec_name: str
    width: int
    height: int
    format_name: str


@dataclass(frozen=True, slots=True)
class PrepareVideoForLoggingResult:
    """Prepared video asset details suitable for Rerun logging."""

    prepared_path: Path
    """Filesystem path that will be logged (original or temporary)."""
    metadata: VideoProbeResult
    """ffprobe metadata gathered after any conversions."""
    should_cleanup: bool
    """Flag indicating whether ``prepared_path`` is a temporary file to delete."""


@dataclass(frozen=True, slots=True)
class VideoIngestEntry:
    """Tuple-like container mapping a source video to its Rerun entity path."""

    source_path: Path
    """Filesystem path of the input video on disk."""
    log_entity_path: Path
    """Rerun entity path where the processed video will be logged."""


def probe_video_stream(video_path: Path) -> VideoProbeResult:
    """
    Inspect ``video_path`` using ffprobe and return the first video stream metadata.

    Raises:
        RuntimeError: If ffprobe fails during inspection.
        ValueError: If no valid video stream metadata is available.
    """

    ffprobe_cmd: list[str] = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height",
        "-show_entries",
        "format=format_name",
        "-of",
        "json",
        str(video_path),
    ]
    process: CompletedProcess[str] = subprocess.run(
        ffprobe_cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        stderr_output: str = process.stderr.strip()
        raise RuntimeError(f"ffprobe failed when inspecting {video_path}: {stderr_output}")

    probe_data: dict[str, object] = json.loads(process.stdout)
    streams_raw: object = probe_data.get("streams", [])
    if not isinstance(streams_raw, list) or not streams_raw:
        raise ValueError(f"No video streams found in {video_path}")

    stream_info_obj: object = streams_raw[0]
    if not isinstance(stream_info_obj, dict):
        raise ValueError(f"Malformed ffprobe stream metadata for {video_path}")

    codec_name_obj: object | None = stream_info_obj.get("codec_name")
    codec_name: str | None = codec_name_obj if isinstance(codec_name_obj, str) else None
    width_obj: object | None = stream_info_obj.get("width")
    width: int | None = _coerce_int(width_obj)
    height_obj: object | None = stream_info_obj.get("height")
    height: int | None = _coerce_int(height_obj)
    format_obj: object | None = probe_data.get("format", {})
    format_name_obj: object | None = format_obj.get("format_name") if isinstance(format_obj, dict) else None
    format_name: str | None = format_name_obj if isinstance(format_name_obj, str) else None

    if codec_name is None or width is None or height is None or format_name is None:
        raise ValueError(f"Missing stream metadata for {video_path}")

    return VideoProbeResult(
        codec_name=codec_name,
        width=width,
        height=height,
        format_name=format_name,
    )


def _format_tokens(format_name: str) -> set[str]:
    """Normalize the comma-separated ffprobe format name list."""

    return {token.strip().lower() for token in format_name.split(",") if token}


def _coerce_int(value: object) -> int | None:
    """Attempt to convert ffprobe numeric fields to integers."""

    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def prepare_video_for_logging(video_path: Path) -> PrepareVideoForLoggingResult:
    """
    Ensure ``video_path`` is AV1 encoded, stored as MP4, and respects the 1280x720 ceiling.

    Returns:
        A ``PrepareVideoForLoggingResult`` capturing the prepared path, metadata, and
        whether the prepared asset is temporary.
    """

    initial_probe: VideoProbeResult = probe_video_stream(video_path)
    needs_reencode: bool = False
    resize_resolution: Resolution | None = None

    if "mp4" not in _format_tokens(initial_probe.format_name):
        needs_reencode = True
    if initial_probe.codec_name != "av1":
        needs_reencode = True
    if initial_probe.width > 1280 or initial_probe.height > 720:
        resize_resolution = "720p"
        needs_reencode = True

    if not needs_reencode:
        if initial_probe.width > 1280 or initial_probe.height > 720:
            raise ValueError(
                f"{video_path} exceeds the maximum resolution (found {initial_probe.width}x{initial_probe.height})."
            )
        return PrepareVideoForLoggingResult(
            prepared_path=video_path,
            metadata=initial_probe,
            should_cleanup=False,
        )

    prepared_video_path: Path = reencode_video_optimal(
        input_video_path=video_path,
        resize=resize_resolution,
    )
    prepared_probe: VideoProbeResult = probe_video_stream(prepared_video_path)

    if "mp4" not in _format_tokens(prepared_probe.format_name):
        prepared_video_path.unlink(missing_ok=True)
        raise RuntimeError(f"Re-encoded video is not MP4: {prepared_video_path}")
    if prepared_probe.codec_name != "av1":
        prepared_video_path.unlink(missing_ok=True)
        raise RuntimeError(f"Re-encoded video is not AV1: {prepared_video_path}")
    if prepared_probe.width > 1280 or prepared_probe.height > 720:
        prepared_video_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Re-encoded video exceeds the maximum resolution: {prepared_probe.width}x{prepared_probe.height}"
        )

    return PrepareVideoForLoggingResult(
        prepared_path=prepared_video_path,
        metadata=prepared_probe,
        should_cleanup=True,
    )


def collect_video_entries(
    video_dir: Path,
    *,
    log_root: Path,
) -> list[VideoIngestEntry]:
    """Gather source/log path pairs for every MP4 inside ``video_dir``."""

    all_video_paths: list[Path] = natsorted(video_dir.glob("*.mp4"))
    assert all_video_paths, f"No .mp4 files found in directory: {video_dir}"

    video_entries: list[VideoIngestEntry] = [
        VideoIngestEntry(
            source_path=video_path,
            log_entity_path=log_root / video_path.stem,
        )
        for video_path in all_video_paths
    ]
    return video_entries


def ingest_video_directory(
    video_entries: list[VideoIngestEntry],
    *,
    timeline: str,
    verbose: bool,
    progress_label: str,
) -> list[Path]:
    """Ingest the provided videos ensuring uniform encoding and resolution constraints.

    Returns:
        list[Path]: Entity paths where the prepared videos were logged.
    """

    assert video_entries, "No video entries provided for ingestion."

    expected_resolution: tuple[int, int] | None = None
    logged_video_entities: list[Path] = []
    iterator: Iterable[VideoIngestEntry] = (
        cast(
            Iterable[VideoIngestEntry],
            tqdm(
                video_entries,
                desc=progress_label,
                leave=False,
            ),
        )
        if verbose
        else cast(Iterable[VideoIngestEntry], video_entries)
    )

    for entry in iterator:
        prepared_video_result: PrepareVideoForLoggingResult = prepare_video_for_logging(video_path=entry.source_path)
        prepared_path: Path = prepared_video_result.prepared_path
        metadata: VideoProbeResult = prepared_video_result.metadata
        should_cleanup: bool = prepared_video_result.should_cleanup
        actual_resolution: tuple[int, int] = (metadata.width, metadata.height)

        if expected_resolution is None:
            expected_resolution = actual_resolution
        elif actual_resolution != expected_resolution:
            if should_cleanup:
                prepared_path.unlink(missing_ok=True)
            raise ValueError(
                f"Video {entry.source_path} has resolution {actual_resolution} which does not match "
                f"the expected resolution {expected_resolution}."
            )

        log_video(
            video_path=prepared_path,
            video_log_path=entry.log_entity_path,
            timeline=timeline,
        )
        logged_entity: Path = entry.log_entity_path
        logged_video_entities.append(logged_entity)

        if should_cleanup:
            prepared_path.unlink(missing_ok=True)

    return logged_video_entities


def create_ingest_view(
    *,
    exo_video_log_paths: list[Path] | None,
    ego_video_log_paths: list[Path] | None,
) -> ContainerLike:
    """
    Assemble a Rerun container/view showing exo videos along the bottom row and ego videos on the right column.
    """

    main_view = rrb.Spatial3DView(origin="/")

    if ego_video_log_paths:
        ego_views = [
            rrb.Tabs(
                rrb.Spatial2DView(origin=str(video_log_path)),
            )
            for video_log_path in ego_video_log_paths
        ]
        main_view = rrb.Horizontal(
            contents=[
                main_view,
                rrb.Vertical(contents=ego_views),
            ],
            column_shares=[4, 1],
        )

    if exo_video_log_paths:
        exo_views = [
            rrb.Tabs(
                rrb.Spatial2DView(origin=str(video_log_path)),
            )
            for video_log_path in exo_video_log_paths
        ]
        main_view = rrb.Vertical(
            contents=[
                main_view,
                rrb.Horizontal(contents=exo_views),
            ],
            row_shares=[4, 1],
        )

    return main_view


def main(config: IngestConfig) -> None:
    validate_exoego_dir(config.exoego_dir)
    print(f"Ingesting data from {config.exoego_dir} to RRD at {config.exoego_dir}")

    parent_log_path: Path = Path("world")
    timeline: str = "video_time"
    dir_tuple: tuple[Path | None, Path | None] = validate_exoego_dir(config.exoego_dir)
    exo_dir: Path | None = dir_tuple[0]
    ego_dir: Path | None = dir_tuple[1]

    exo_entries: list[VideoIngestEntry] = (
        collect_video_entries(
            video_dir=exo_dir,
            log_root=parent_log_path / "exo",
        )
        if exo_dir is not None
        else []
    )
    ego_entries: list[VideoIngestEntry] = (
        collect_video_entries(
            video_dir=ego_dir,
            log_root=parent_log_path / "ego",
        )
        if ego_dir is not None
        else []
    )

    ingest_view: ContainerLike = create_ingest_view(
        exo_video_log_paths=[entry.log_entity_path for entry in exo_entries] or None,
        ego_video_log_paths=[entry.log_entity_path for entry in ego_entries] or None,
    )
    rr.send_blueprint(rrb.Blueprint(ingest_view, collapse_panels=True))

    if exo_entries:
        ingest_video_directory(
            video_entries=exo_entries,
            timeline=timeline,
            verbose=config.verbose,
            progress_label="Ingesting exo videos",
        )

    if ego_entries:
        ingest_video_directory(
            video_entries=ego_entries,
            timeline=timeline,
            verbose=config.verbose,
            progress_label="Ingesting ego videos",
        )
