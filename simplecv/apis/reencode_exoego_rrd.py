import atexit
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import rerun as rr
import tyro
from rerun_bindings import ComponentColumnDescriptor, Recording, Schema

from simplecv.rerun_log_utils import (
    RerunTyroConfig,
    mux_h264_to_mp4,
    read_h264_samples_from_rrd,
    write_asset_video_blob,
)

from tqdm import tqdm


@dataclass(slots=True)
class _RRDCameraStream:
    """Metadata describing an exo camera stream discovered in an RRD file."""

    name: str
    video_entity: str
    pinhole_entity: str
    transform_entity: str
    data_kind: Literal["video_stream", "asset_video"]


def _discover_camera_streams(schema: Schema, group: Literal["exo", "ego"]) -> list[_RRDCameraStream]:
    component_columns: list[ComponentColumnDescriptor] = schema.component_columns()
    descriptors: list[Any] = (
        list(component_columns.keys()) if isinstance(component_columns, dict) else list(component_columns)
    )

    stream_map: dict[str, _RRDCameraStream] = {}
    for descriptor in descriptors:
        entity_path = getattr(descriptor, "entity_path", None)
        component_name = getattr(descriptor, "component", None)
        if not isinstance(component_name, str) or entity_path is None:
            continue

        entity_str = str(entity_path).lstrip("/")
        if not entity_str.startswith(f"world/{group}"):
            continue

        data_kind: Literal["video_stream", "asset_video"] | None = None
        if component_name.endswith("VideoStream:sample"):
            data_kind = "video_stream"
        elif component_name.endswith("AssetVideo:blob"):
            data_kind = "asset_video"

        if data_kind is None:
            continue

        video_entity = entity_str
        pinhole_entity = str(PurePosixPath(video_entity).parent)
        transform_entity = str(PurePosixPath(pinhole_entity).parent)
        camera_name = PurePosixPath(transform_entity).name

        stream = stream_map.get(video_entity)
        if stream is None:
            stream_map[video_entity] = _RRDCameraStream(
                name=camera_name,
                video_entity=video_entity,
                pinhole_entity=pinhole_entity,
                transform_entity=transform_entity,
                data_kind=data_kind,
            )
        else:
            # Prefer video streams if both exist; otherwise keep detected kind.
            if stream.data_kind != data_kind and data_kind == "video_stream":
                stream_map[video_entity] = _RRDCameraStream(
                    name=camera_name,
                    video_entity=video_entity,
                    pinhole_entity=pinhole_entity,
                    transform_entity=transform_entity,
                    data_kind=data_kind,
                )

    camera_streams: list[_RRDCameraStream] = sorted(stream_map.values(), key=lambda stream: stream.name)
    return camera_streams


@dataclass
class ReEncodeExoEgoRRDConfig:
    """Structured configuration for running the exo/ego visualization CLI."""

    rr_config: RerunTyroConfig
    """Command-line options for spawning and configuring the Rerun viewer."""
    exoego_rrd_path: Path
    """Path to the directory containing 'exo' and/or 'ego' subdirectories with video files."""
    verbose: bool = False
    """Enable verbose console logging during reencoding."""


def main(config: ReEncodeExoEgoRRDConfig) -> None:
    print(config.exoego_rrd_path)

    recording: Recording = rr.dataframe.load_recording(str(config.exoego_rrd_path))

    schema: Schema = recording.schema()
    timeline = "video_time"
    exo_streams: list[_RRDCameraStream] = _discover_camera_streams(schema, "exo")
    ego_streams: list[_RRDCameraStream] = _discover_camera_streams(schema, "ego")
    # make sure either exo or ego streams were found
    assert len(exo_streams) > 0 or len(ego_streams) > 0, "No exo or ego camera streams found in the provided RRD."
    # Extract MP4 Files to a temp directory for re-encoding
    _remux_tmpdir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(prefix="rrd_reencode_remux_")
    atexit.register(_remux_tmpdir.cleanup)

    video_paths: list[Path] = []
    for camera_stream in tqdm(exo_streams + ego_streams, desc="Re-encoding camera streams", unit="stream"):
        match camera_stream.data_kind:
            case "video_stream":
                times, samples = read_h264_samples_from_rrd(
                    str(config.exoego_rrd_path), camera_stream.video_entity, timeline
                )
                mp4_path: Path = Path(_remux_tmpdir.name) / f"{camera_stream.name}.mp4"
                mux_h264_to_mp4(times, samples, str(mp4_path))
            case "asset_video":
                mp4_path = Path(_remux_tmpdir.name) / f"{camera_stream.name}.mp4"
                write_asset_video_blob(
                    recording=recording,
                    timeline=timeline,
                    video_entity=camera_stream.video_entity,
                    output_path=mp4_path,
                )
            case _:
                raise ValueError(f"Unsupported data kind for RRD camera stream: {camera_stream.data_kind}")

        assert mp4_path.exists(), f"Expected remuxed video at {mp4_path}"
        video_paths.append(mp4_path)
        print(f"Re-encoded {camera_stream.data_kind} '{camera_stream.name}' to {mp4_path}")


def entrypoint() -> None:
    """Entrypoint leveraging Tyro to expose the reencoding workflow via CLI."""

    tyro.extras.set_accent_color("bright_cyan")
    config: ReEncodeExoEgoRRDConfig = tyro.cli(
        ReEncodeExoEgoRRDConfig,
        description="Given an rrd with ego/exo, convert them to av1 mp4 format.",
    )
    main(config=config)


if __name__ == "__main__":
    entrypoint()
