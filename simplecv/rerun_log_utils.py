import io
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from uuid import UUID

import av
import rerun as rr
from jaxtyping import Int
from numpy import ndarray
from pyarrow import ChunkedArray
from rerun_bindings import Recording, RecordingView

from simplecv.camera_parameters import PinholeParameters


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
            rr.connect_grpc(flush_timeout_sec=None)
        elif self.save is not None:
            rr.save(self.save)
        elif not self.headless:
            rr.spawn()


def log_pinhole(
    camera: PinholeParameters,
    cam_log_path: Path,
    image_plane_distance: int | float = 0.5,
    static: bool = False,
    *,
    recording: rr.RecordingStream | None = None,
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
        rr.Pinhole(
            image_from_camera=camera.intrinsics.k_matrix,
            height=camera.intrinsics.height,
            width=camera.intrinsics.width,
            camera_xyz=getattr(
                rr.ViewCoordinates,
                camera.intrinsics.camera_conventions,
            ),
            image_plane_distance=image_plane_distance,
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
        raise ValueError(
            f"There's no video stream codec specified at {video_entity} for timeline {timeline}."
        )
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

    # Timestamps are made relative to the first timestamp.
    start_time = times.chunk(0)[0]
    print(f"Offsetting timestamps with start time: {start_time}")

    # Demux and mux packets.
    for packet, time in zip(input_container.demux(input_stream), times, strict=False):
        packet.time_base = Fraction(1, 1_000_000_000)  # Assuming duration timestamps in nanoseconds.
        packet.pts = int(time.value - start_time.value)
        packet.dts = packet.pts  # dts == pts since there's no B-frames.
        packet.stream = output_stream
        output_container.mux(packet)

    input_container.close()
    output_container.close()
