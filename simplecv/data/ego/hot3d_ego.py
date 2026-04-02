"""Egocentric view loader for HOT3D Aria with per-frame MPS extrinsics."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rerun as rr
from jaxtyping import Float32, Int64
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates

from simplecv.camera_parameters import Extrinsics, Fisheye62Parameters, Intrinsics, KannalaBrandtDistortion
from simplecv.data.ego.base_ego import BaseEgoSequence, EgoData
from simplecv.data.hot3d_utils import (
    Hot3dSequenceCalibration,
    Hot3dStreamCalibration,
    load_calibration,
    lookup_nearest_poses,
    parse_mps_closed_loop_trajectory,
)

if TYPE_CHECKING:
    from simplecv.data.exoego.hot3d import Hot3dConfig
else:  # pragma: no cover - runtime alias to avoid circular import
    from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig as Hot3dConfig

# Camera stream name used for the RGB ego camera
RGB_STREAM_NAME: str = "camera-rgb"

# Preprocessed output directory name
SIMPLECV_DIR: str = "_simplecv"


class Hot3dEgoSequence(BaseEgoSequence[Hot3dConfig]):
    """Egocentric view loader for HOT3D Aria with per-frame MPS extrinsics in meters."""

    def __len__(self) -> int:
        assert len(self.ego_video_readers) > 0, "No videos found."
        return len(self.ego_video_readers)

    def __getitem__(self, idx: int) -> EgoData:
        raise NotImplementedError("Hot3D ego data loading not implemented yet.")

    def _sequence_dir(self) -> Path:
        """Get the sequence directory path."""
        return Path(self.config.root_directory) / self.config.sequence_name

    def load_video_paths(self) -> list[Path]:
        """Load paths to preprocessed AV1 MP4 videos."""
        seq_dir: Path = self._sequence_dir()
        simplecv_dir: Path = seq_dir / SIMPLECV_DIR

        rgb_path: Path = simplecv_dir / "rgb.mp4"
        assert rgb_path.exists(), (
            f"Preprocessed RGB video not found at {rgb_path}. "
            f"Run `pixi run preprocess-hot3d --root {self.config.root_directory} --sequence {self.config.sequence_name}`"
        )
        return [rgb_path]

    def load_ego_cams(self) -> dict[str, list[Fisheye62Parameters]]:
        """Load per-frame fisheye camera parameters from calibration + MPS trajectory.

        Intrinsics come from ``_simplecv/calibration.json`` (extracted from online_calibration.jsonl).
        Per-frame extrinsics come from ``mps/slam/closed_loop_trajectory.csv``:
            ``world_T_camera = world_T_device @ device_T_camera``
            ``cam_T_world = inv(world_T_camera)``
        """
        seq_dir: Path = self._sequence_dir()

        # ── Load calibration ─────────────────────────────────────────────
        cal_path: Path = seq_dir / SIMPLECV_DIR / "calibration.json"
        assert cal_path.exists(), f"Calibration not found at {cal_path}"
        cal: Hot3dSequenceCalibration = load_calibration(cal_path)

        # Find the RGB stream calibration
        rgb_cal: Hot3dStreamCalibration | None = None
        for stream_cal in cal.streams:
            if stream_cal.stream_label == RGB_STREAM_NAME:
                rgb_cal = stream_cal
                break
        assert rgb_cal is not None, f"No calibration found for stream '{RGB_STREAM_NAME}'"

        device_T_camera: Float32[ndarray, "4 4"] = np.array(rgb_cal.device_T_camera, dtype=np.float32)

        # ── Load MPS trajectory ──────────────────────────────────────────
        trajectory_path: Path = seq_dir / "mps" / "slam" / "closed_loop_trajectory.csv"
        assert trajectory_path.exists(), f"MPS trajectory not found at {trajectory_path}"

        traj_ts_ns: Int64[ndarray, "n_poses"]
        world_T_device_all: Float32[ndarray, "n_poses 4 4"]
        traj_ts_ns, world_T_device_all, _quality = parse_mps_closed_loop_trajectory(trajectory_path)

        # ── Get video frame timestamps ───────────────────────────────────
        rgb_path: Path = seq_dir / SIMPLECV_DIR / "rgb.mp4"
        video_ts_ns: Int64[ndarray, "n_frames"] = rr.AssetVideo(path=rgb_path).read_frame_timestamps_nanos()
        n_frames: int = len(video_ts_ns)

        # ── Look up nearest device pose for each video frame ─────────────
        world_T_device_frames: Float32[ndarray, "n_frames 4 4"] = lookup_nearest_poses(
            query_ts_ns=video_ts_ns.astype(np.int64),
            trajectory_ts_ns=traj_ts_ns,
            world_T_device=world_T_device_all,
        )

        # ── Build Fisheye62Parameters per frame ──────────────────────────
        intrinsics: Intrinsics = Intrinsics(
            camera_conventions="RDF",
            fl_x=rgb_cal.fl_x,
            fl_y=rgb_cal.fl_y,
            cx=rgb_cal.cx,
            cy=rgb_cal.cy,
            height=rgb_cal.height,
            width=rgb_cal.width,
        )
        distortion: KannalaBrandtDistortion = KannalaBrandtDistortion(
            k1=rgb_cal.k1,
            k2=rgb_cal.k2,
            k3=rgb_cal.k3,
            k4=rgb_cal.k4,
            k5=rgb_cal.k5,
            k6=rgb_cal.k6,
            p1=rgb_cal.p1,
            p2=rgb_cal.p2,
        )

        cam_list: list[Fisheye62Parameters] = []
        prev_cam_T_world: Float32[ndarray, "4 4"] = np.eye(4, dtype=np.float32)

        for frame_idx in range(n_frames):
            world_T_device: Float32[ndarray, "4 4"] = world_T_device_frames[frame_idx]
            world_T_camera: Float32[ndarray, "4 4"] = world_T_device @ device_T_camera

            # Handle singular matrices (reuse last valid)
            try:
                cam_T_world: Float32[ndarray, "4 4"] = np.linalg.inv(world_T_camera)
            except np.linalg.LinAlgError:
                cam_T_world = prev_cam_T_world
            else:
                if np.all(np.isfinite(cam_T_world)):
                    prev_cam_T_world = cam_T_world
                else:
                    cam_T_world = prev_cam_T_world

            cam_R_world: Float32[ndarray, "3 3"] = cam_T_world[:3, :3]
            cam_t_world: Float32[ndarray, "3"] = cam_T_world[:3, 3]

            extrinsics: Extrinsics = Extrinsics(
                cam_R_world=cam_R_world,
                cam_t_world=cam_t_world,
            )

            cam_params: Fisheye62Parameters = Fisheye62Parameters(
                name=RGB_STREAM_NAME,
                intrinsics=intrinsics,
                distortion=distortion,
                extrinsics=extrinsics,
            )
            cam_list.append(cam_params)

        return {RGB_STREAM_NAME: cam_list}

    def align_cams_and_videos(
        self, video_path_list: list[Path], ego_cam_dict: dict[str, list[Fisheye62Parameters]]
    ) -> tuple[dict[str, list[Fisheye62Parameters]], dict[str, Path]]:
        """Align cameras and videos — straightforward for single RGB stream."""
        assert len(video_path_list) == 1, f"Expected 1 ego video, got {len(video_path_list)}"
        assert RGB_STREAM_NAME in ego_cam_dict, f"Camera {RGB_STREAM_NAME} missing from ego camera dictionary"

        aligned_cam_dict: dict[str, list[Fisheye62Parameters]] = {RGB_STREAM_NAME: ego_cam_dict[RGB_STREAM_NAME]}
        aligned_video_map: dict[str, Path] = {RGB_STREAM_NAME: video_path_list[0]}

        return aligned_cam_dict, aligned_video_map

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        """Aria MPS uses gravity-aligned world with Z pointing down."""
        return rr.ViewCoordinates.RIGHT_HAND_Z_DOWN

    @property
    def image_plane_distance(self) -> int | float:
        """Image plane distance for camera visualization in meters."""
        return 0.035
