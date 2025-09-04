from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from jaxtyping import Float32, Int64
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates
from scipy.spatial.transform import Rotation as R
from serde import serde
from serde.json import from_json

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.data.ego.base_ego import BaseEgoSequence, EgoData

if TYPE_CHECKING:
    from simplecv.data.exoego.stereo import StereoConfig
CameraName = Literal["left", "right"]


@serde
class _StereoCamCalib:
    socket: str
    width: int
    height: int
    intrinsics: Float32[ndarray, "3 3"]
    distortion: Float32[ndarray, "..."]
    fov_deg: float


@serde
class _StereoLeftToRight:
    R: Float32[ndarray, "3 3"]
    T: Float32[ndarray, "3"]
    matrix_4x4: Float32[ndarray, "4 4"]


@serde
class StereoCalibration:
    device_mxid: str
    left: _StereoCamCalib
    right: _StereoCamCalib
    extrinsics_left_to_right: _StereoLeftToRight


@serde
class PoseWorldRow:
    timestamp_ns: int
    tx: float
    ty: float
    tz: float
    qx: float
    qy: float
    qz: float
    qw: float


def _quat_trans_to_mat4(row: PoseWorldRow) -> Float32[ndarray, "4 4"]:
    q = np.array([row.qx, row.qy, row.qz, row.qw], dtype=np.float32)
    t = np.array([row.tx, row.ty, row.tz], dtype=np.float32)
    rot = R.from_quat(q)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot.as_matrix().astype(np.float32)
    T[:3, 3] = t
    return T.astype(np.float32)


class StereoEgoSequence(BaseEgoSequence):
    """
    Ego stereo sequence loader for left/right cameras.

    Expects directory structure:
      <root>/<sequence_name>/ego/
        - calibration.json
        - left.mp4, right.mp4
        - poses_world_coordinates.csv  # applies to LEFT camera
    """

    config: "StereoConfig"

    def load_video_paths(self) -> list[Path]:
        ego_dir: Path = self.config.root_directory / self.config.sequence_name / "ego"
        assert ego_dir.exists(), f"Directory {ego_dir} does not exist"
        left_mp4: Path = ego_dir / "left.mp4"
        right_mp4: Path = ego_dir / "right.mp4"
        assert left_mp4.exists(), f"File {left_mp4} does not exist"
        assert right_mp4.exists(), f"File {right_mp4} does not exist"
        return [left_mp4, right_mp4]

    def load_ego_cams(self) -> dict[CameraName, list[PinholeParameters]]:
        ego_dir: Path = self.config.root_directory / self.config.sequence_name / "ego"
        calib_path: Path = ego_dir / "calibration.json"
        assert calib_path.exists(), f"File {calib_path} does not exist"
        calib: StereoCalibration = from_json(StereoCalibration, calib_path.read_text())

        # Build intrinsics
        left_intri = Intrinsics(
            camera_conventions="RDF",
            fl_x=float(calib.left.intrinsics[0, 0]),
            fl_y=float(calib.left.intrinsics[1, 1]),
            cx=float(calib.left.intrinsics[0, 2]),
            cy=float(calib.left.intrinsics[1, 2]),
            width=int(calib.left.width),
            height=int(calib.left.height),
        )
        right_intri = Intrinsics(
            camera_conventions="RDF",
            fl_x=float(calib.right.intrinsics[0, 0]),
            fl_y=float(calib.right.intrinsics[1, 1]),
            cx=float(calib.right.intrinsics[0, 2]),
            cy=float(calib.right.intrinsics[1, 2]),
            width=int(calib.right.width),
            height=int(calib.right.height),
        )

        # Load left camera cam_T_world per-frame aligned to video frames using left.csv
        poses_csv: Path = ego_dir / "poses_world_coordinates.csv"
        assert poses_csv.exists(), f"File {poses_csv} does not exist"
        left_index_csv: Path = ego_dir / "left.csv"
        assert left_index_csv.exists(), f"File {left_index_csv} does not exist"

        # parse pose rows (timestamp -> cam_T_world)
        pose_rows: list[PoseWorldRow] = []
        with open(poses_csv, "r", newline="") as f_pose:
            reader_pose = csv.DictReader(f_pose)
            for row_dict in reader_pose:
                pose_rows.append(
                    PoseWorldRow(**{k: (int(v) if k == "timestamp_ns" else float(v)) for k, v in row_dict.items()})
                )
        # ensure sorted by timestamp
        pose_rows.sort(key=lambda r: r.timestamp_ns)

        # parse left frame timestamps and indices
        left_frames: list[tuple[int, int]] = []  # (ts_ns, frame_idx)
        with open(left_index_csv, "r", newline="") as f_left:
            reader_left = csv.DictReader(f_left)
            for row in reader_left:
                ts_ns = int(row["ts_ns"]) if row.get("ts_ns") is not None else int(row["timestamp_ns"])  # robustness
                fi = int(row["frame_idx"]) if row.get("frame_idx") is not None else int(row["frame"])  # robustness
                left_frames.append((ts_ns, fi))
        # sort by frame index ascending
        left_frames.sort(key=lambda x: x[1])

        # clamp to actual video frame count to match timestamps used by Rerun
        try:
            import cv2

            left_mp4 = ego_dir / "left.mp4"
            cap = cv2.VideoCapture(str(left_mp4))
            frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else len(left_frames)
            cap.release()
            if len(left_frames) > frame_cnt:
                left_frames = left_frames[:frame_cnt]
        except Exception:
            # If OpenCV is unavailable or any error occurs, fall back to existing list
            pass

        # Build per-frame transforms using nearest previous pose timestamp
        right_T_left: Float32[ndarray, "4 4"] = calib.extrinsics_left_to_right.matrix_4x4.astype(np.float32)
        left_cam_list: list[PinholeParameters] = []
        right_cam_list: list[PinholeParameters] = []

        pose_i = 0
        n_pose = len(pose_rows)
        current_left_T_world: Float32[ndarray, "4 4"] | None = None
        for ts_ns, _frame_idx in left_frames:
            # advance pose index while next pose timestamp <= current frame timestamp
            while pose_i + 1 < n_pose and pose_rows[pose_i + 1].timestamp_ns <= ts_ns:
                pose_i += 1
            # use current pose_i; if none yet, use the first pose
            row = pose_rows[pose_i] if n_pose > 0 else None
            if row is not None:
                current_left_T_world = _quat_trans_to_mat4(row)
            # fallback if file empty (shouldn't happen)
            if current_left_T_world is None:
                current_left_T_world = np.eye(4, dtype=np.float32)

            left_cam_T_world = current_left_T_world
            right_cam_T_world: Float32[ndarray, "4 4"] = right_T_left @ left_cam_T_world

            # Fill Extrinsics using camera->world (preferred) to match project conventions
            left_extri = Extrinsics(cam_R_world=left_cam_T_world[:3, :3], cam_t_world=left_cam_T_world[:3, 3])
            right_extri = Extrinsics(cam_R_world=right_cam_T_world[:3, :3], cam_t_world=right_cam_T_world[:3, 3])

            left_cam_list.append(PinholeParameters(name="left", intrinsics=left_intri, extrinsics=left_extri))
            right_cam_list.append(PinholeParameters(name="right", intrinsics=right_intri, extrinsics=right_extri))

        ego_cam_dict: dict[CameraName, list[PinholeParameters]] = {
            "left": left_cam_list,
            "right": right_cam_list,
        }
        return ego_cam_dict

    def align_cams_and_videos(
        self, video_path_list: list[Path], ego_cam_dict: dict[CameraName, list[PinholeParameters]]
    ) -> tuple[dict[CameraName, list[PinholeParameters]], dict[CameraName, Path]]:
        assert len(video_path_list) == 2, f"Expected two videos, got {len(video_path_list)}"
        assert set(ego_cam_dict.keys()) == {"left", "right"}, "Expected left/right cameras in calibration"

        mapping: dict[CameraName, Path] = {}
        for p in video_path_list:
            stem = p.stem.lower()
            if "left" in stem:
                mapping["left"] = p
            elif "right" in stem:
                mapping["right"] = p
        assert set(mapping.keys()) == {"left", "right"}, (
            f"Video names must include 'left' and 'right': {video_path_list}"
        )
        # Keep deterministic ordering
        ego_cam_dict = dict(sorted(ego_cam_dict.items()))
        mapping = dict(sorted(mapping.items()))
        return ego_cam_dict, mapping

    def __getitem__(self, idx: int) -> EgoData:
        # Note: downstream visualization reads directly from video readers when logging
        return EgoData(
            cam_params_list=[self._ego_cam_dict["left"][idx], self._ego_cam_dict["right"][idx]],
            bgr_list=self.ego_video_readers[idx],
        )

    def __len__(self) -> int:  # type: ignore[override]
        if not self._ego_cam_dict:
            return 0
        cams_len = len(next(iter(self._ego_cam_dict.values())))
        vids_len = len(self.ego_video_readers)
        return min(cams_len, vids_len)

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        return ViewCoordinates.RIGHT_HAND_Z_DOWN

    @property
    def image_plane_distance(self) -> int | float:
        return 0.1
