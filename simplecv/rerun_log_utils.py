import io
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Self
from uuid import UUID

import av
import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Float, Int, UInt8
from numpy import ndarray
from pyarrow import ChunkedArray
from rerun_bindings import Recording, RecordingView

from simplecv.camera_parameters import PinholeParameters


def get_safe_application_id() -> str:
    """Get application ID safely, with fallback if __main__.__file__ doesn't exist"""
    try:
        main = sys.modules.get("__main__")
        if main and hasattr(main, "__file__"):
            return Path(main.__file__).stem
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


def confidence_scores_to_rgb(
    confidence_scores: Float[ndarray, "n_frames n_kpts 1"],
) -> UInt8[ndarray, "n_frames n_kpts 3"]:
    """Converts confidence scores to RGB colors using a Red-Yellow-Green gradient.

    The color mapping is as follows:
    - A confidence score of 0.0 is mapped to Red (255, 0, 0).
    - A confidence score of 0.5 is mapped to Yellow (255, 255, 0).
    - A confidence score of 1.0 is mapped to Green (0, 255, 0).
    Scores are linearly interpolated between these points. Values outside the
    [0.0, 1.0] range will be clipped by the function.

        confidence_scores (Float32[ndarray, "n_frames n_kpts 1"]):
            A NumPy array of shape (n_frames, n_kpts, 1) containing
            confidence values. Values are typically between 0.0 and 1.0.

        UInt8[ndarray, "n_frames n_kpts 3"]:
            A NumPy array of shape (n_frames, n_kpts, 3) containing
            the corresponding RGB colors as uint8 values. Each color is
            represented as an array of three integers [R, G, B]."""
    n_frames, n_kpts, _ = confidence_scores.shape
    clipped_confidences: Float[ndarray, "n_frames n_kpts 1"] = np.clip(confidence_scores, a_min=0.0, a_max=1.0)
    clipped_confidences: Float[ndarray, "n_frames n_kpts"] = np.squeeze(clipped_confidences, axis=-1)
    safe_confidences: Float[ndarray, "n_frames n_kpts"] = np.nan_to_num(clipped_confidences, nan=0.0)

    colors: UInt8[ndarray, "n_frames n_kpts 3"] = np.zeros((n_frames, n_kpts, 3), dtype=np.uint8)
    # Segment A: red → yellow for conf ≤ 0.5
    mask_low = safe_confidences <= 0.5
    if mask_low.any():
        t_low = safe_confidences[mask_low] * 2.0  # 0‥1
        colors[..., 0][mask_low] = 255  # red fixed
        colors[..., 1][mask_low] = (t_low * 255).astype(np.uint8)

    # Segment B: yellow → green for conf > 0.5
    mask_high = ~mask_low
    if mask_high.any():
        t_high = (safe_confidences[mask_high] - 0.5) * 2.0
        colors[..., 0][mask_high] = ((1.0 - t_high) * 255).astype(np.uint8)
        colors[..., 1][mask_high] = 255  # green fixed

    # blue channel remains 0
    return colors


def _confidence_component_descriptor(
    archetype_name: str,
    field_name: str = "confidences",
    *,
    component_type: str = "simplecv.components.KeypointConfidence",
) -> rr.ComponentDescriptor:
    component: str = f"{archetype_name}:{field_name}"
    return rr.ComponentDescriptor(
        component=component,
        archetype=archetype_name,
        component_type=component_type,
    )


class ConfidenceBatch(rr.ComponentBatchMixin):
    """A batch of confidence data."""

    def __init__(self, confidence: Float[ndarray, "..."]) -> None:
        self.confidence = confidence

    def component_descriptor(self) -> rr.ComponentDescriptor:
        """The descriptor of the custom component."""
        return rr.ComponentDescriptor(
            "simplecv.components.KeypointConfidence",
            component_type="simplecv.components.KeypointConfidence",
        )

    def as_arrow_array(self) -> pa.Array:
        """The arrow batch representing the custom component."""
        return pa.array(self.confidence, type=pa.float32())


class AverageConfidenceBatch(rr.ComponentBatchMixin):
    """A batch containing one or more average confidence values."""

    def __init__(
        self,
        average_confidence: float | Float[ndarray, "n"],
    ) -> None:
        average_conf_array: Float[ndarray, "n"] = np.atleast_1d(
            np.asarray(average_confidence, dtype=np.float32)
        )
        self.average_confidence: Float[ndarray, "n"] = average_conf_array

    def component_descriptor(self) -> rr.ComponentDescriptor:
        """Descriptor for the average confidence component."""
        return rr.ComponentDescriptor(
            "simplecv.components.KeypointConfidenceMean",
            component_type="simplecv.components.KeypointConfidenceMean",
        )

    def as_arrow_array(self) -> pa.Array:
        """Arrow representation of the average confidence value."""
        return pa.array(self.average_confidence, type=pa.float32())


def _flatten_confidences(confidences: Float[ndarray, "..."]) -> Float[ndarray, "n"]:
    conf_array: Float[ndarray, "..."] = np.asarray(confidences, dtype=np.float32)
    if conf_array.ndim > 1:
        conf_array = conf_array.reshape(-1)
    return conf_array.astype(np.float32, copy=False)


class _ConfidenceAwareColumnList(rr.ComponentColumnList):
    """Extend a column list with confidence and per-frame averages for send_columns.

    The helper mirrors the built-in archetype column helpers while quietly
    appending the extra confidence-related components before partitioning.
    """

    def __init__(
        self,
        base_columns: rr.ComponentColumnList,
        confidences: Float[ndarray, "n"] | None,
        confidence_descriptor: rr.ComponentDescriptor,
        average_descriptor: rr.ComponentDescriptor,
        *,
        average_confidences: Float[ndarray, "m"] | None = None,
    ) -> None:
        base_list: list[rr.ComponentColumn] = list(base_columns)
        self._average_descriptor: rr.ComponentDescriptor = average_descriptor
        self._provided_average: Float[ndarray, "m"] | None = (
            None
            if average_confidences is None
            else np.asarray(average_confidences, dtype=np.float32).reshape(-1)
        )
        if confidences is None:
            self._raw_confidences: Float[ndarray, "0"] | None = None
            super().__init__(base_list)
            return

        raw_conf: Float[ndarray, "n"] = _flatten_confidences(confidences)
        self._raw_confidences = raw_conf
        base_list.append(
            rr.ComponentColumn(confidence_descriptor, ConfidenceBatch(raw_conf))
        )
        super().__init__(base_list)

    def partition(self, lengths: Int[ndarray, "m"]) -> rr.ComponentColumnList:
        partitioned: rr.ComponentColumnList = super().partition(lengths)
        if self._raw_confidences is None:
            return partitioned

        lengths_arr: Int[ndarray, "m"] = np.asarray(lengths, dtype=np.int64)
        total_length: int = int(lengths_arr.sum())
        if total_length != int(self._raw_confidences.size):
            raise ValueError(
                "Sum of partition lengths does not match number of confidences."
            )

        if self._provided_average is not None:
            if self._provided_average.shape[0] != lengths_arr.shape[0]:
                raise ValueError(
                    "Provided average confidences must match number of partitions."
                )
            averages = self._provided_average.astype(np.float32, copy=False)
        else:
            offsets: Int[ndarray, "m_plus_one"] = np.concatenate(
                (np.array([0], dtype=np.int64), np.cumsum(lengths_arr, dtype=np.int64))
            )
            averages = np.empty(lengths_arr.shape[0], dtype=np.float32)
            for idx, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=False)):
                if end <= start:
                    averages[idx] = np.nan
                    continue
                segment: Float[ndarray, "k"] = self._raw_confidences[start:end]
                valid = segment[~np.isnan(segment)]
                averages[idx] = (
                    float(np.nanmean(valid)) if valid.size > 0 else np.nan
                )

        avg_column = rr.ComponentColumn(
            self._average_descriptor,
            AverageConfidenceBatch(averages),
            lengths=np.ones_like(averages, dtype=np.int32),
        )

        columns_with_average: list[rr.ComponentColumn] = list(partitioned)
        columns_with_average.append(avg_column)
        return rr.ComponentColumnList(columns_with_average)


class Points2DWithConfidence(rr.AsComponents):
    """Custom Points2D archetype with per-keypoint and average confidences."""

    def __init__(
        self: Any,
        positions: Float[ndarray, "n_kpts 2"],
        confidences: Float[ndarray, "n_kpts"],
        class_ids: int,
        keypoint_ids: list[int],
        show_labels: bool = False,
        colors: UInt8[ndarray, "n_kpts 3"] | None = None,
        radii: float | None = None,
    ) -> None:
        self.points2d = rr.Points2D(
            positions=positions,
            class_ids=class_ids,
            keypoint_ids=keypoint_ids,
            show_labels=show_labels,
            colors=colors,
            radii=radii,
        )
        self._include_confidence: bool = True
        confidence_descriptor: rr.ComponentDescriptor = _confidence_component_descriptor(
            "simplecv.KeypointConfidence2D"
        )
        average_descriptor: rr.ComponentDescriptor = _confidence_component_descriptor(
            "simplecv.KeypointConfidence2D",
            field_name="average_confidence",
            component_type="simplecv.components.KeypointConfidenceMean",
        )
        confidences_arr: Float[ndarray, "n_kpts"] = np.asarray(confidences, dtype=np.float32)
        mean_confidence: float = (
            float(np.nanmean(confidences_arr)) if confidences_arr.size else float("nan")
        )
        self.confidences = ConfidenceBatch(confidences_arr).described(confidence_descriptor)
        self.average_confidence = AverageConfidenceBatch(mean_confidence).described(average_descriptor)

    def as_component_batches(self) -> list[rr.DescribedComponentBatch]:
        batches: list[rr.DescribedComponentBatch] = list(self.points2d.as_component_batches())
        if self._include_confidence:
            batches.extend([self.confidences, self.average_confidence])
        return batches

    @classmethod
    def columns(
        cls,
        *,
        positions: Float[ndarray, "n 2"] | None = None,
        confidences: Float[ndarray, "n"] | None = None,
        average_confidences: Float[ndarray, "m"] | None = None,
        class_ids: Int[ndarray, "n"] | int | None = None,
        keypoint_ids: Int[ndarray, "n"] | list[int] | None = None,
        show_labels: bool | ndarray | None = None,
        colors: UInt8[ndarray, "n 3"] | None = None,
        radii: float | Float[ndarray, "n"] | None = None,
        ) -> rr.ComponentColumnList:
        """Return column components mirroring `rr.Points2D.columns` plus confidences.

        Inputs are already flattened to `(n_frames * n_kpts)` by the caller so the
        returned list can be partitioned with per-frame lengths before handing it
        to `rr.send_columns`.
        """
        base_columns: rr.ComponentColumnList = rr.Points2D.columns(
            positions=positions,
            radii=radii,
            colors=colors,
            show_labels=show_labels,
            class_ids=class_ids,
            keypoint_ids=keypoint_ids,
        )

        confidence_descriptor: rr.ComponentDescriptor = _confidence_component_descriptor(
            "simplecv.KeypointConfidence2D"
        )
        average_descriptor: rr.ComponentDescriptor = _confidence_component_descriptor(
            "simplecv.KeypointConfidence2D",
            field_name="average_confidence",
            component_type="simplecv.components.KeypointConfidenceMean",
        )

        if confidences is None and average_confidences is None:
            return base_columns

        return _ConfidenceAwareColumnList(
            base_columns,
            confidences,
            confidence_descriptor,
            average_descriptor,
            average_confidences=average_confidences,
        )

    @classmethod
    def from_fields(cls, **fields: Any) -> Self:
        """Create a static placeholder matching `rr.Points2D.from_fields`.

        The returned instance excludes confidence data so it can be safely logged
        as a static archetype while columnar updates provide the dynamic values.
        """
        instance = cls.__new__(cls)
        instance.points2d = rr.Points2D.from_fields(**fields)
        instance._include_confidence = False
        confidence_descriptor: rr.ComponentDescriptor = _confidence_component_descriptor(
            "simplecv.KeypointConfidence2D"
        )
        average_descriptor: rr.ComponentDescriptor = _confidence_component_descriptor(
            "simplecv.KeypointConfidence2D",
            field_name="average_confidence",
            component_type="simplecv.components.KeypointConfidenceMean",
        )
        empty_conf: Float[ndarray, "0"] = np.empty(0, dtype=np.float32)
        instance.confidences = ConfidenceBatch(empty_conf).described(confidence_descriptor)
        instance.average_confidence = AverageConfidenceBatch(empty_conf).described(average_descriptor)
        return instance


class Points3DWithConfidence(rr.ComponentColumn):
    """Custom Points3D archetype with per-keypoint and average confidences."""

    def __init__(
        self: Any,
        positions: Float[ndarray, "n_kpts 3"],
        confidences: Float[ndarray, "n_kpts"],
        class_ids: int,
        keypoint_ids: list[int],
        show_labels: bool = False,
        colors: UInt8[ndarray, "n_kpts 3"] | None = None,
        radii: float | None = None,
    ) -> None:
        self.points3d = rr.Points3D(
            positions=positions,
            class_ids=class_ids,
            keypoint_ids=keypoint_ids,
            show_labels=show_labels,
            colors=colors,
            radii=radii,
        )
        self._include_confidence: bool = True
        confidence_descriptor: rr.ComponentDescriptor = _confidence_component_descriptor(
            "simplecv.KeypointConfidence3D"
        )
        average_descriptor: rr.ComponentDescriptor = _confidence_component_descriptor(
            "simplecv.KeypointConfidence3D",
            field_name="average_confidence",
            component_type="simplecv.components.KeypointConfidenceMean",
        )
        confidences_arr: Float[ndarray, "n_kpts"] = np.asarray(confidences, dtype=np.float32)
        mean_confidence: float = (
            float(np.nanmean(confidences_arr)) if confidences_arr.size else float("nan")
        )
        self.confidences = ConfidenceBatch(confidences_arr).described(confidence_descriptor)
        self.average_confidence = AverageConfidenceBatch(mean_confidence).described(average_descriptor)

    def as_component_batches(self) -> list[rr.DescribedComponentBatch]:
        batches: list[rr.DescribedComponentBatch] = list(self.points3d.as_component_batches())
        if self._include_confidence:
            batches.extend([self.confidences, self.average_confidence])
        return batches

    @classmethod
    def columns(
        cls,
        *,
        positions: Float[ndarray, "n 3"] | None = None,
        confidences: Float[ndarray, "n"] | None = None,
        average_confidences: Float[ndarray, "m"] | None = None,
        class_ids: Int[ndarray, "n"] | int | None = None,
        keypoint_ids: Int[ndarray, "n"] | list[int] | None = None,
        show_labels: bool | ndarray | None = None,
        colors: UInt8[ndarray, "n 3"] | None = None,
        radii: float | Float[ndarray, "n"] | None = None,
    ) -> rr.ComponentColumnList:
        """Return column components mirroring `rr.Points3D.columns` plus confidences."""
        base_columns: rr.ComponentColumnList = rr.Points3D.columns(
            positions=positions,
            colors=colors,
            radii=radii,
            show_labels=show_labels,
            class_ids=class_ids,
            keypoint_ids=keypoint_ids,
        )

        confidence_descriptor: rr.ComponentDescriptor = _confidence_component_descriptor(
            "simplecv.KeypointConfidence3D"
        )
        average_descriptor: rr.ComponentDescriptor = _confidence_component_descriptor(
            "simplecv.KeypointConfidence3D",
            field_name="average_confidence",
            component_type="simplecv.components.KeypointConfidenceMean",
        )

        if confidences is None and average_confidences is None:
            return base_columns

        return _ConfidenceAwareColumnList(
            base_columns,
            confidences,
            confidence_descriptor,
            average_descriptor,
            average_confidences=average_confidences,
        )

    @classmethod
    def from_fields(cls, **fields: Any) -> Self:
        """Create a static placeholder for `Points3DWithConfidence` headers."""
        instance = cls.__new__(cls)
        instance.points3d = rr.Points3D.from_fields(**fields)
        instance._include_confidence = False
        confidence_descriptor: rr.ComponentDescriptor = _confidence_component_descriptor(
            "simplecv.KeypointConfidence3D"
        )
        average_descriptor: rr.ComponentDescriptor = _confidence_component_descriptor(
            "simplecv.KeypointConfidence3D",
            field_name="average_confidence",
            component_type="simplecv.components.KeypointConfidenceMean",
        )
        empty_conf: Float[ndarray, "0"] = np.empty(0, dtype=np.float32)
        instance.confidences = ConfidenceBatch(empty_conf).described(confidence_descriptor)
        instance.average_confidence = AverageConfidenceBatch(empty_conf).described(average_descriptor)
        return instance


def read_h264_samples_from_rrd(rrd_path: str, video_entity: str, timeline: str) -> tuple[ChunkedArray, ChunkedArray]:
    """Load recording data and query video stream."""

    recording: Recording = rr.dataframe.load_recording(rrd_path)
    view: RecordingView = recording.view(index=timeline, contents=video_entity)

    # Make sure this is H.264 encoded.
    # For that we just read out the first codec value batch and check whether it's H.264.
    codec = view.select(f"{video_entity}:VideoStream:codec")
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
    timestamps_and_samples = view.select(timeline, f"{video_entity}:VideoStream:sample").read_all()
    times = timestamps_and_samples[0]
    samples = timestamps_and_samples[1]

    print(f"Retrieved {len(samples)} video samples.")

    return times, samples


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
