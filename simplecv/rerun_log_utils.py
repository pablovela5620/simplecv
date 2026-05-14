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
from pyarrow import ChunkedArray, LargeListArray, ListArray

from simplecv.camera_parameters import Fisheye62Parameters, PinholeParameters
from simplecv.rrd_query_utils import RRDQuerySession, first_valid_value, unwrap_singleton_lists
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
            except OSError:
                pass
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
    executable_name: str = "rerun"
    """Executable name passed to ``rerun.spawn`` when launching the viewer."""
    executable_path: str | None = None
    """Optional absolute or relative path to the Rerun executable."""

    def __post_init__(self):
        rr.init(
            application_id=self.application_id,
            recording_id=self.recording_id,
            default_enabled=True,
            strict=True,
        )
        self.rec_stream: rr.RecordingStream = rr.get_global_data_recording()  # type: ignore[assignment]

        if self.serve:
            rr.serve_grpc()
            rr.serve_web_viewer(open_browser=not self.headless)
        elif self.connect:
            # Send logging data to separate `rerun` process.
            # You can omit the argument to connect to the default address,
            # which is `127.0.0.1:9876`.
            rr.connect_grpc()
        elif self.save is not None:
            rr.save(self.save)
        elif not self.headless:
            rr.spawn(
                executable_name=self.executable_name,
                executable_path=self.executable_path,
            )


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
    video_source: Path | bytes,
    video_log_path: Path,
    timeline: str = "video_time",
    *,
    recording: rr.RecordingStream | None = None,
) -> Int[ndarray, "num_frames"]:
    """
    Logs a video asset and its frame timestamps.

    Args:
        video_source: Path to video file or raw video bytes.
        video_log_path: The entity path where the video log will be saved.
        timeline: Timeline name for frame timestamps.
        recording: Optional specific recording stream to log to.

    Returns:
        Frame timestamps in nanoseconds.
    """
    # Create AssetVideo from path or bytes. When the source is bytes we have
    # no filesystem suffix to infer the MIME type from, so default to
    # ``video/mp4`` (matches the GT catalog).
    video_asset = (
        rr.AssetVideo(contents=video_source, media_type="video/mp4")
        if isinstance(video_source, bytes)
        else rr.AssetVideo(path=video_source)
    )

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

    normalized_entity: str = video_entity.lstrip("/")
    query_session = RRDQuerySession(Path(rrd_path))

    # Make sure this is H.264 encoded.
    codec_table = query_session.read_arrow(
        contents=normalized_entity,
        selectors=[f"{normalized_entity}:VideoStream:codec"],
        index=None,
    )
    if codec_table.num_rows == 0:
        codec_table = query_session.read_arrow(
            contents=normalized_entity,
            selectors=[f"{normalized_entity}:VideoStream:codec"],
            index=timeline,
        )
        codec_column = codec_table.column(1) if codec_table.num_columns > 1 else codec_table.column(0)
    else:
        codec_column = codec_table.column(0)

    if codec_table.num_rows == 0:
        raise ValueError(f"There's no video stream codec specified at {video_entity} for timeline {timeline}.")

    codec_value_raw = first_valid_value(
        codec_column,
        component_name=f"{normalized_entity}:VideoStream:codec",
    )
    codec_value = int(np.asarray(codec_value_raw).reshape(-1)[0])
    if codec_value != rr.VideoCodec.H264.value:
        raise ValueError(
            f"Video stream codec is not H.264 at {video_entity} for timeline {timeline}. "
            f"Got {hex(codec_value)}, but the value for H.264 is {hex(rr.VideoCodec.H264.value)}."
        )
    else:
        print(f"Video stream codec is H.264 at {video_entity} for timeline {timeline}.")

    # Get the video stream
    timestamps_and_samples = query_session.read_arrow(
        contents=normalized_entity,
        selectors=[f"{normalized_entity}:VideoStream:sample"],
        index=timeline,
    )
    if timestamps_and_samples.num_rows == 0:
        raise ValueError(f"No H.264 samples found at {video_entity} for timeline {timeline}.")

    times = timestamps_and_samples.column(0)
    samples = timestamps_and_samples.column(1)

    print(f"Retrieved {len(samples)} video samples.")

    return times, samples

def extract_asset_video_blob_fast(
    video_entity: str,
    timeline: str = "video_time",
    *,
    query_session: RRDQuerySession | None = None,
    rrd_path: Path | str | None = None,
) -> bytes:
    """Extract AssetVideo blob bytes from a Rerun recording using fast pyarrow buffer access.

    This method is ~680x faster than the slow as_py() approach for large videos.
    It directly accesses the underlying pyarrow buffer without creating
    intermediate Python objects.

    Args:
        video_entity: Entity path (without leading ``/``) containing the AssetVideo component.
        timeline: Timeline used to index the recording view.
        query_session: Optional shared RRD query session for catalog reads.
        rrd_path: Optional RRD path used to create a temporary query session.

    Returns:
        Video bytes suitable for TorchCodec VideoDecoder.

    Raises:
        ValueError: If no AssetVideo blob found.
    """
    import pyarrow as pa

    normalized_entity: str = video_entity.lstrip("/")
    blob_column: str = f"{normalized_entity}:AssetVideo:blob"

    active_session = query_session
    if active_session is None and rrd_path is not None:
        active_session = RRDQuerySession(Path(rrd_path))
    if active_session is None:
        raise ValueError("extract_asset_video_blob_fast requires either query_session or rrd_path.")

    table = active_session.read_arrow(
        contents=normalized_entity,
        selectors=[blob_column],
        index=None,
    )
    blob_column_idx = 0
    if table.num_rows == 0:
        table = active_session.read_arrow(
            contents=normalized_entity,
            selectors=[blob_column],
            index=timeline,
        )
        blob_column_idx = 1
    if table.num_rows == 0:
        raise ValueError(f"No AssetVideo blob found for entity {video_entity}")
    column: pa.Array | pa.ChunkedArray = table.column(blob_column_idx)

    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()

    # FAST PATH: Access pyarrow buffer directly without Python list intermediate
    # Structure: list<list<uint8>> -> values -> list<uint8> -> values -> uint8[]
    try:
        inner_list: pa.ListArray = column.values  # type: ignore[assignment]  # Inner list<uint8>
        uint8_values: pa.UInt8Array = inner_list.values  # type: ignore[assignment]  # The actual uint8 array
        buffers = uint8_values.buffers()
        # Buffer 0 is validity bitmap (null), Buffer 1 is data
        if len(buffers) >= 2 and buffers[1] is not None:
            blob: bytes = buffers[1].to_pybytes()
            return blob
    except Exception:
        pass  # Fall back to slow path

    # SLOW FALLBACK: Use as_py() if buffer access fails
    first_row = column[0].as_py()
    first_row = unwrap_singleton_lists(first_row)
    return bytes(first_row)


def mux_h264_to_mp4(times: ChunkedArray, samples: ChunkedArray, output_path: str) -> None:
    """Mux H.264 Annex B samples to an mp4 file using PyAV."""
    # See https://pyav.basswood-io.com/docs/stable/cookbook/basics.html#remuxing

    # Flatten out sample list into a single byte buffer.
    sample_array = samples.combine_chunks()
    if isinstance(sample_array, ListArray | LargeListArray):
        sample_array = sample_array.flatten(recursive=True)
    buffer = sample_array.buffers()[1]
    if buffer is None:
        raise ValueError("Missing H.264 sample buffer.")
    sample_bytes = io.BytesIO(buffer.to_pybytes())

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
