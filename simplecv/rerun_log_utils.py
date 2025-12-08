import hashlib
import io
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any
from uuid import UUID

import av
import rerun as rr
from jaxtyping import Int
from numpy import ndarray
from pyarrow import ChunkedArray
from rerun_bindings import Recording, RecordingView

from simplecv.camera_parameters import Fisheye62Parameters, PinholeParameters
from simplecv.rerun_custom_types import PinholeWithDistortion


def _default_cache_root() -> Path:
    env_override: str | None = os.environ.get("SIMPLECV_VIDEO_CACHE")
    if env_override:
        return Path(env_override).expanduser()
    return Path.home() / ".cache" / "simplecv" / "exoego_videos"


@dataclass(slots=True)
class _VideoCacheMetadata:
    rrd_mtime_ns: int
    rrd_size: int


class VideoCache:
    """Filesystem-backed cache for remuxed AssetVideo blobs."""

    def __init__(self, root: Path | None = None) -> None:
        self.root: Path = (root or _default_cache_root()).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def _bucket_dir(self, rrd_path: Path) -> Path:
        resolved: Path = rrd_path.resolve()
        sha1: str = hashlib.sha1(str(resolved).encode(), usedforsecurity=False).hexdigest()
        bucket: Path = self.root / sha1
        bucket.mkdir(parents=True, exist_ok=True)
        return bucket

    def _fingerprint(self, rrd_path: Path) -> tuple[int, int]:
        stat_result: os.stat_result = rrd_path.stat()
        return stat_result.st_mtime_ns, stat_result.st_size

    def _metadata_path(self, mp4_path: Path) -> Path:
        return mp4_path.with_suffix(mp4_path.suffix + ".json")

    def _load_metadata(self, metadata_path: Path) -> _VideoCacheMetadata | None:
        try:
            payload: Any = json.loads(metadata_path.read_text())
            return _VideoCacheMetadata(
                rrd_mtime_ns=int(payload["rrd_mtime_ns"]),
                rrd_size=int(payload["rrd_size"]),
            )
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def get(self, *, rrd_path: Path, camera_name: str) -> Path | None:
        bucket: Path = self._bucket_dir(rrd_path)
        cached_mp4: Path = bucket / f"{camera_name}.mp4"
        metadata_path: Path = self._metadata_path(cached_mp4)
        metadata: _VideoCacheMetadata | None = self._load_metadata(metadata_path)
        if metadata is None or not cached_mp4.exists():
            return None
        current_mtime, current_size = self._fingerprint(rrd_path)
        if metadata.rrd_mtime_ns != current_mtime or metadata.rrd_size != current_size:
            try:
                cached_mp4.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
            finally:
                return None
        return cached_mp4

    def store(self, *, rrd_path: Path, camera_name: str, source_path: Path) -> None:
        bucket: Path = self._bucket_dir(rrd_path)
        dest: Path = bucket / f"{camera_name}.mp4"
        metadata_path: Path = self._metadata_path(dest)
        tmp_dest: Path = dest.with_suffix(dest.suffix + ".tmp")
        shutil.copy2(source_path, tmp_dest)
        os.replace(tmp_dest, dest)
        mtime, size = self._fingerprint(rrd_path)
        metadata_payload: dict[str, int] = {"rrd_mtime_ns": mtime, "rrd_size": size}
        metadata_path.write_text(json.dumps(metadata_payload))


_CACHE_DISABLED: bool = os.environ.get("SIMPLECV_VIDEO_CACHE_DISABLE", "0") in {"1", "true", "True"}
_VIDEO_CACHE: VideoCache | None = None


def get_video_cache() -> VideoCache | None:
    """Return process-wide video cache unless disabled via env."""

    global _VIDEO_CACHE
    if _CACHE_DISABLED:
        return None
    if _VIDEO_CACHE is None:
        _VIDEO_CACHE = VideoCache()
    return _VIDEO_CACHE


def get_safe_application_id() -> str:
    """Get application ID safely, with fallback if __main__.__file__ doesn't exist"""
    try:
        main = sys.modules.get("__main__")
        if main:
            file_attr = getattr(main, "__file__", None)
            if isinstance(file_attr, str):
                return Path(file_attr).stem
    except Exception:
        pass
    return "rerun-application"  # Default fallback


@dataclass
class RerunTyroConfig:
    application_id: str = field(default_factory=get_safe_application_id)
    """Name of the application"""
    recording_id: str | UUID | None = None
    """Recording ID"""
    connect: bool = False
    """Wether to connect to an existing rerun instance or not"""
    save: Path | None = None
    """Path to save the rerun data, this will make it so no data is visualized but saved"""
    serve: bool = False
    """Serve the rerun data"""
    headless: bool = False
    """Run rerun in headless mode"""

    def __post_init__(self):
        rr.init(
            application_id=self.application_id,
            recording_id=self.recording_id,
            default_enabled=True,
            strict=True,
        )
        self.rec_stream: rr.RecordingStream = rr.get_global_data_recording()  # type: ignore[assignment]

        if self.serve:
            rr.serve_web()
        elif self.connect:
            # Send logging data to separate `rerun` process.
            # You can omit the argument to connect to the default address,
            # which is `127.0.0.1:9876`.
            rr.connect_grpc()
        elif self.save is not None:
            rr.save(self.save)
        elif not self.headless:
            rr.spawn()


def log_pinhole(
    camera: PinholeParameters | Fisheye62Parameters,
    cam_log_path: Path,
    image_plane_distance: int | float = 0.5,
    static: bool = False,
    *,
    recording: rr.RecordingStream | None = None,
    include_distortion: bool = True,
) -> None:
    """
    Logs the pinhole camera parameters and transformation data.

    Parameters:
    camera (PinholeParameters): The pinhole camera parameters including intrinsics and extrinsics.
    cam_log_path (Path): The path where the camera log will be saved.
    image_plane_distance (float, optional): The distance of the image plane from the camera. Defaults to 0.5.
    static (bool, optional): If True, the log data will be marked as static. Defaults to False.

    Returns:
    None
    """
    # camera intrinsics
    rr.log(
        f"{cam_log_path}/pinhole",
        PinholeWithDistortion.from_camera(
            camera,
            image_plane_distance=image_plane_distance,
            include_distortion=include_distortion,
        ),
        static=static,
        recording=recording,
    )
    # camera extrinsics
    rr.log(
        f"{cam_log_path}",
        rr.Transform3D(
            translation=camera.extrinsics.cam_t_world,
            mat3x3=camera.extrinsics.cam_R_world,
            from_parent=True,
        ),
        static=static,
        recording=recording,
    )


def log_video(
    video_path: Path,
    video_log_path: Path,
    timeline: str = "video_time",
    *,
    recording: rr.RecordingStream | None = None,
) -> Int[ndarray, "num_frames"]:
    """
    Logs a video asset and its frame timestamps.

    Parameters:
    video_path (Path): The path to the video file.
    video_log_path (Path): The path where the video log will be saved.

    Returns:
    None
    """
    # Log video asset which is referred to by frame references.
    video_asset = rr.AssetVideo(path=video_path)
    rr.log(str(video_log_path), video_asset, static=True, recording=recording)

    # Send automatically determined video frame timestamps.
    frame_timestamps_ns: Int[ndarray, "num_frames"] = video_asset.read_frame_timestamps_nanos()

    rr.send_columns(
        f"{video_log_path}",
        # Note timeline values don't have to be the same as the video timestamps.
        indexes=[rr.TimeColumn(timeline, duration=1e-9 * frame_timestamps_ns)],
        columns=rr.VideoFrameReference.columns_nanos(frame_timestamps_ns),
        recording=recording,
    )
    return frame_timestamps_ns


def read_h264_samples_from_rrd(rrd_path: str, video_entity: str, timeline: str) -> tuple[ChunkedArray, ChunkedArray]:
    """Load recording data and query video stream."""

    recording: Recording = rr.dataframe.load_recording(rrd_path)
    normalized_entity: str = video_entity.lstrip("/")
    view: RecordingView = recording.view(index=timeline, contents=normalized_entity)

    # Make sure this is H.264 encoded.
    # For that we just read out the first codec value batch and check whether it's H.264.
    codec = view.select(f"{normalized_entity}:VideoStream:codec")
    first_codec_batch = codec.read_next_batch()
    if first_codec_batch is None:
        raise ValueError(f"There's no video stream codec specified at {video_entity} for timeline {timeline}.")
    codec_value = first_codec_batch.column(0)[0][0].as_py()
    if codec_value != rr.VideoCodec.H264.value:
        raise ValueError(
            f"Video stream codec is not H.264 at {video_entity} for timeline {timeline}. "
            f"Got {hex(codec_value)}, but the value for H.264 is {hex(rr.VideoCodec.H264.value)}."
        )
    else:
        print(f"Video stream codec is H.264 at {video_entity} for timeline {timeline}.")

    # Get the video stream
    timestamps_and_samples = view.select(timeline, f"{normalized_entity}:VideoStream:sample").read_all()
    times = timestamps_and_samples[0]
    samples = timestamps_and_samples[1]

    print(f"Retrieved {len(samples)} video samples.")

    return times, samples


def write_asset_video_blob(
    recording: Recording,
    *,
    timeline: str,
    video_entity: str,
    output_path: Path,
) -> Path:
    """Persist an AssetVideo blob from ``recording`` to ``output_path``.

    Args:
        recording: Loaded Rerun recording containing the asset.
        timeline: Timeline used to index the recording view.
        video_entity: Entity path (without a leading ``/``) holding the ``AssetVideo`` component.
        output_path: Destination path to write the extracted video bytes.

    Returns:
        Path: The provided ``output_path`` after writing the bytes.

    Raises:
        ValueError: If no asset video data is present for ``video_entity``.
    """

    view = recording.view(index=timeline, contents=video_entity)
    reader = view.select(f"{video_entity}:AssetVideo:blob")

    batch = reader.read_next_batch()
    while batch is not None:
        column = batch.column(0)
        for row_idx in range(batch.num_rows):
            value = column[row_idx]
            if value is None:
                continue
            data_list = value.as_py()
            if isinstance(data_list, list) and len(data_list) == 1 and isinstance(data_list[0], list):
                data_list = data_list[0]
            video_bytes = bytes(data_list)
            output_path.write_bytes(video_bytes)
            return output_path
        batch = reader.read_next_batch()

    raise ValueError(f"No AssetVideo data found for entity {video_entity}")


def mux_h264_to_mp4(times: ChunkedArray, samples: ChunkedArray, output_path: str) -> None:
    """Mux H.264 Annex B samples to an mp4 file using PyAV."""
    # See https://pyav.basswood-io.com/docs/stable/cookbook/basics.html#remuxing

    # Flatten out sample list into a single byte buffer.
    sample_bytes = samples.combine_chunks().flatten(recursive=True)
    sample_bytes = io.BytesIO(sample_bytes.buffers()[1])

    # Setup samples as input container.
    input_container = av.open(sample_bytes, mode="r", format="h264")  # Input is AnnexB H.264 stream.
    input_stream = input_container.streams.video[0]

    # Setup output container.
    output_container = av.open(output_path, mode="w")
    output_stream = output_container.add_stream_from_template(input_stream)
    # Preserve nanosecond timeline from the recording to avoid fps skew when
    # remuxing. Without this, ffmpeg/pyav may infer a default time_base that
    # snaps frames to a different cadence than the recorded timestamps.
    output_stream.time_base = Fraction(1, 1_000_000_000)
    if output_stream.codec_context is not None:
        output_stream.codec_context.time_base = output_stream.time_base

    # Timestamps are made relative to the first timestamp.
    start_time = times.chunk(0)[0]
    print(f"Offsetting timestamps with start time: {start_time}")

    # Demux and mux packets.
    ns_time_base = Fraction(1, 1_000_000_000)
    for packet, time in zip(input_container.demux(input_stream), times, strict=False):
        packet.time_base = ns_time_base  # timestamps stored in nanoseconds
        packet.pts = int(time.value - start_time.value)
        packet.dts = packet.pts  # dts == pts since there's no B-frames.
        packet.stream = output_stream
        output_container.mux(packet)

    input_container.close()
    output_container.close()
