"""Egocentric view loader for HOT3D Aria with per-frame MPS extrinsics.

Supports all three Aria ego cameras: RGB (1408x1408) and two SLAM
monochrome cameras (640x480 each).
"""

from __future__ import annotations

import json
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

# Aria ego camera streams: label → MP4 filename stem.
# Order matters: videos and cameras are paired positionally.
HOT3D_EGO_STREAMS: dict[str, str] = {
    "camera-rgb": "rgb",
    "camera-slam-left": "slam_left",
    "camera-slam-right": "slam_right",
}

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
        """Load paths to preprocessed AV1 MP4 videos for all Aria ego cameras."""
        seq_dir: Path = self._sequence_dir()
        simplecv_dir: Path = seq_dir / SIMPLECV_DIR

        video_paths: list[Path] = []
        for label, stem in HOT3D_EGO_STREAMS.items():
            video_path: Path = simplecv_dir / f"{stem}.mp4"
            assert video_path.exists(), (
                f"Preprocessed video for '{label}' not found at {video_path}. "
                f"Run: pixi run preprocess-hot3d --root {self.config.root_directory} "
                f"--sequence {self.config.sequence_name} --streams 214-1 1201-1 1201-2 --skip-existing false"
            )
            video_paths.append(video_path)

        return video_paths

    def load_ego_cams(self) -> dict[str, list[Fisheye62Parameters]]:
        """Load per-frame fisheye camera parameters for all ego cameras.

        Intrinsics come from ``_simplecv/calibration.json``.
        Per-frame extrinsics come from ``mps/slam/closed_loop_trajectory.csv``:
            ``world_T_camera = world_T_device @ device_T_camera``
            ``cam_T_world = inv(world_T_camera)``
        """
        seq_dir: Path = self._sequence_dir()

        # ── Load calibration ─────────────────────────────────────────────
        cal_path: Path = seq_dir / SIMPLECV_DIR / "calibration.json"
        assert cal_path.exists(), f"Calibration not found at {cal_path}"
        cal: Hot3dSequenceCalibration = load_calibration(cal_path)

        # Index calibration by stream label
        cal_by_label: dict[str, Hot3dStreamCalibration] = {s.stream_label: s for s in cal.streams}

        # ── Load MPS trajectory ──────────────────────────────────────────
        trajectory_path: Path = seq_dir / "mps" / "slam" / "closed_loop_trajectory.csv"
        assert trajectory_path.exists(), f"MPS trajectory not found at {trajectory_path}"

        traj_ts_ns: Int64[ndarray, "n_poses"]
        world_T_device_all: Float32[ndarray, "n_poses 4 4"]
        traj_ts_ns, world_T_device_all, _quality = parse_mps_closed_loop_trajectory(trajectory_path)

        # ── Load VRS device-time timestamps per stream ───────────────────
        vrs_ts_path: Path = seq_dir / SIMPLECV_DIR / "timestamps_ns.json"
        assert vrs_ts_path.exists(), f"VRS timestamps not found at {vrs_ts_path}"
        vrs_ts_data: dict = json.loads(vrs_ts_path.read_text())

        # ── Build per-frame camera params for each stream ────────────────
        all_cam_dict: dict[str, list[Fisheye62Parameters]] = {}

        for label in HOT3D_EGO_STREAMS:
            stream_cal: Hot3dStreamCalibration | None = cal_by_label.get(label)
            assert stream_cal is not None, f"No calibration found for stream '{label}'"
            assert label in vrs_ts_data, (
                f"No timestamps for stream '{label}' in timestamps_ns.json. "
                f"Re-run preprocessing with --streams 214-1 1201-1 1201-2"
            )

            device_T_camera: Float32[ndarray, "4 4"] = np.array(stream_cal.device_T_camera, dtype=np.float32)

            # Per-stream device-time timestamps for MPS pose lookup
            stream_device_ts: Int64[ndarray, "n_frames"] = np.array(vrs_ts_data[label], dtype=np.int64)
            n_frames: int = len(stream_device_ts)

            # Look up nearest device pose for each frame
            world_T_device_frames: Float32[ndarray, "n_frames 4 4"] = lookup_nearest_poses(
                query_ts_ns=stream_device_ts,
                trajectory_ts_ns=traj_ts_ns,
                world_T_device=world_T_device_all,
            )

            # Build intrinsics + distortion (static per stream)
            intrinsics: Intrinsics = Intrinsics(
                camera_conventions="RDF",
                fl_x=stream_cal.fl_x,
                fl_y=stream_cal.fl_y,
                cx=stream_cal.cx,
                cy=stream_cal.cy,
                height=stream_cal.height,
                width=stream_cal.width,
            )
            distortion: KannalaBrandtDistortion = KannalaBrandtDistortion(
                k1=stream_cal.k1,
                k2=stream_cal.k2,
                k3=stream_cal.k3,
                k4=stream_cal.k4,
                k5=stream_cal.k5,
                k6=stream_cal.k6,
                p1=stream_cal.p1,
                p2=stream_cal.p2,
            )

            # Build per-frame Fisheye62Parameters
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
                    name=label,
                    intrinsics=intrinsics,
                    distortion=distortion,
                    extrinsics=extrinsics,
                )
                cam_list.append(cam_params)

            all_cam_dict[label] = cam_list

        return all_cam_dict

    def align_cams_and_videos(
        self, video_path_list: list[Path], ego_cam_dict: dict[str, list[Fisheye62Parameters]]
    ) -> tuple[dict[str, list[Fisheye62Parameters]], dict[str, Path]]:
        """Align cameras and videos by stream label order."""
        stream_labels: list[str] = list(HOT3D_EGO_STREAMS.keys())
        assert len(video_path_list) == len(stream_labels), (
            f"Expected {len(stream_labels)} ego videos, got {len(video_path_list)}"
        )

        aligned_cam_dict: dict[str, list[Fisheye62Parameters]] = {}
        aligned_video_map: dict[str, Path] = {}

        for idx, label in enumerate(stream_labels):
            assert label in ego_cam_dict, f"Camera '{label}' missing from ego camera dictionary"
            aligned_cam_dict[label] = ego_cam_dict[label]
            aligned_video_map[label] = video_path_list[idx]

        return aligned_cam_dict, aligned_video_map

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        """Aria MPS world frame: gravity = [0,0,-9.81] so +Z is up."""
        return rr.ViewCoordinates.RIGHT_HAND_Z_UP

    @property
    def image_plane_distance(self) -> int | float:
        """Image plane distance for camera visualization in meters."""
        return 0.035
