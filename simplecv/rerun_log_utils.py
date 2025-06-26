import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Float, Int, UInt8
from numpy import ndarray

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
    )


def log_video(video_path: Path, video_log_path: Path, timeline: str = "video_time") -> Int[ndarray, "num_frames"]:
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
    rr.log(str(video_log_path), video_asset, static=True)

    # Send automatically determined video frame timestamps.
    frame_timestamps_ns: Int[ndarray, "num_frames"] = (  # noqa: UP037
        video_asset.read_frame_timestamps_ns()
    )
    rr.send_columns(
        f"{video_log_path}",
        # Note timeline values don't have to be the same as the video timestamps.
        indexes=[rr.TimeNanosColumn(timeline, frame_timestamps_ns)],
        columns=rr.VideoFrameReference.columns_nanoseconds(frame_timestamps_ns),
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

    colors: UInt8[ndarray, "n_frames n_kpts 3"] = np.zeros((n_frames, n_kpts, 3), dtype=np.uint8)
    # Segment A: red → yellow for conf ≤ 0.5
    mask_low = clipped_confidences <= 0.5
    if mask_low.any():
        t_low = clipped_confidences[mask_low] * 2.0  # 0‥1
        colors[..., 0][mask_low] = 255  # red fixed
        colors[..., 1][mask_low] = (t_low * 255).astype(np.uint8)

    # Segment B: yellow → green for conf > 0.5
    mask_high = ~mask_low
    if mask_high.any():
        t_high = (clipped_confidences[mask_high] - 0.5) * 2.0
        colors[..., 0][mask_high] = ((1.0 - t_high) * 255).astype(np.uint8)
        colors[..., 1][mask_high] = 255  # green fixed

    # blue channel remains 0
    return colors


class ConfidenceBatch(rr.ComponentBatchMixin):
    """A batch of confidence data."""

    def __init__(self, confidence: Float[ndarray, "..."]) -> None:
        self.confidence = confidence

    def component_descriptor(self) -> rr.ComponentDescriptor:
        """The descriptor of the custom component."""
        return rr.ComponentDescriptor("user.Confidence")

    def as_arrow_array(self) -> pa.Array:
        """The arrow batch representing the custom component."""
        return pa.array(self.confidence, type=pa.float32())


class Points2DWithConfidence(rr.AsComponents):
    """A custom archetype that extends Rerun's builtin `Points3D` archetype with a custom component."""

    def __init__(
        self: Any,
        positions: Float[ndarray, "n_kpts 2"],
        confidences: Float[ndarray, "n_kpts"],  # Confidence values for each point
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
            colors=colors,  # Optional colors for the points
            radii=radii,
        )
        self.confidences = ConfidenceBatch(confidences).or_with_descriptor_overrides(
            archetype_name="user.CustomPoints3D", archetype_field_name="confidences"
        )

    def as_component_batches(self) -> list[rr.DescribedComponentBatch]:
        return (
            list(self.points2d.as_component_batches())  # The components from Points2D
            + [self.confidences]  # Custom confidence data
        )


class Points3DWithConfidence(rr.ComponentColumn):
    """A custom archetype that extends Rerun's builtin `Points3D` archetype with a custom component."""

    def __init__(
        self: Any,
        positions: Float[ndarray, "n_kpts 3"],
        confidences: Float[ndarray, "n_kpts"],  # Confidence values for each point
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
            colors=colors,  # Optional colors for the points
            radii=radii,
        )
        self.confidences = ConfidenceBatch(confidences).or_with_descriptor_overrides(
            archetype_name="user.CustomPoints3D", archetype_field_name="confidences"
        )

    def as_component_batches(self) -> list[rr.DescribedComponentBatch]:
        return (
            list(self.points3d.as_component_batches())  # The components from Points3D
            + [self.confidences]  # Custom confidence data
        )
