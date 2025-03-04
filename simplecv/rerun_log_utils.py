import sys
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import rerun as rr
from jaxtyping import Int
from numpy import ndarray

from simplecv.camera_parameters import PinholeParameters

def get_safe_application_id():
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
        rec: rr.RecordingStream = rr.get_global_data_recording()  # type: ignore[assignment]

        if self.serve:
            rec.serve()
        elif self.connect:
            # Send logging data to separate `rerun` process.
            # You can omit the argument to connect to the default address,
            # which is `127.0.0.1:9876`.
            rec.connect()
        elif self.save is not None:
            rec.save(self.save)
        elif not self.headless:
            rec.spawn()


def log_pinhole(
    camera: PinholeParameters,
    cam_log_path: Path,
    image_plane_distance: float = 0.5,
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


def log_video(video_path: Path, video_log_path: Path) -> Int[ndarray, "num_frames"]:
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
        str(video_log_path),
        # Note timeline values don't have to be the same as the video timestamps.
        times=[rr.TimeNanosColumn("video_time", frame_timestamps_ns)],
        components=[
            rr.VideoFrameReference.indicator(),
            rr.components.VideoTimestamp.nanoseconds(frame_timestamps_ns),
        ],
    )
    return frame_timestamps_ns
