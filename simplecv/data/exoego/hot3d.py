"""HOT3D Aria dataset adapter for the ExoEgo visualization pipeline.

Ego-only (no exo cameras), following the UmeTrack pattern. Uses preprocessed
AV1 MP4 videos (from VRS), MPS SLAM trajectories for camera poses, and
UmeTrack-format hand annotations.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import rerun as rr
from jaxtyping import Float32, Int, Int64
from natsort import natsorted
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates

from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.ego.hot3d_ego import Hot3dEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels, ExoEgoSample
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig
from simplecv.data.hot3d_utils import build_4x4, load_timecode_to_devicetime_mapping, quat_wxyz_to_matrix
from simplecv.data.skeleton.assembly_hands import assembly21_to_coco133
from simplecv.umetrack_temp.generic_hand_model_numpy import HandModelNumpy, SingleHandPose, landmarks_from_hand_pose


@dataclass
class Hot3dConfig(BaseExoEgoDatasetConfig):
    """Configuration for HOT3D Aria sequences."""

    _target: type = field(default_factory=lambda: Hot3dSequence)
    root_directory: Path = Path("/mnt/8tb/data/hot3d/aria")
    """Root directory containing HOT3D Aria sequence folders."""
    sequence_name: str = "P0001_10a27bf7"
    """Sequence folder name (e.g. 'P0001_10a27bf7')."""


class Hot3dSequence(BaseExoEgoSequence[Hot3dConfig]):
    """HOT3D Aria dataset adapter emitting 3D hand annotations in meters."""

    def __init__(self, cfg: Hot3dConfig) -> None:
        self._ego_stream_names: list[str] = []
        self._exo_stream_names: list[str] = []
        super().__init__(cfg)

    def _sequence_dir(self) -> Path:
        return Path(self.config.root_directory) / self.config.sequence_name

    def __getitem__(self, idx: int | None = None, ts_nano: np.timedelta64 | None = None) -> ExoEgoSample:
        canonical_idx, ts_ns = self._resolve_canonical(idx=idx, ts_nano=ts_nano)
        ego_cam_params_list, ego_bgr_list = self._sample_ego(ts_ns)
        labels: ExoEgoLabels | None = self._sample_labels(canonical_idx, ts_ns)

        return ExoEgoSample(
            canonical_index=canonical_idx,
            canonical_timestamp_ns=ts_ns,
            ego_cam_params_list=ego_cam_params_list,
            ego_bgr_list=ego_bgr_list,
            exo_cam_params_list=None,
            exo_bgr_list=None,
            labels=labels,
        )

    def _build_ego(self) -> BaseEgoSequence[Hot3dConfig] | None:
        return Hot3dEgoSequence(cfg=self.config)

    def _build_exo(self) -> BaseExoSequence[Hot3dConfig] | None:
        return None  # ego-only (like UmeTrack)

    def load_stream_timestamps_ns(self) -> dict[str, Int[ndarray, "n_frames"]]:
        """Return per-stream timestamps for ego videos (and labels if available)."""
        stream_ts: dict[str, Int[ndarray, "n_frames"]] = {}
        self._ego_stream_names.clear()

        if self.ego_sequence is not None:
            for name, video_path in zip(
                self.ego_sequence.ego_video_names,
                self.ego_sequence.ego_video_paths,
                strict=True,
            ):
                stream_name: str = f"ego/{name}"
                timestamps: Int[ndarray, "n_frames"] = rr.AssetVideo(path=video_path).read_frame_timestamps_nanos()
                stream_ts[stream_name] = timestamps
                self._ego_stream_names.append(stream_name)

        labels: ExoEgoLabels | None = self.exoego_labels
        if labels is not None and labels.timestamps_ns is not None:
            stream_ts["labels"] = labels.timestamps_ns

        return stream_ts

    def load_labels(self) -> ExoEgoLabels:
        """Load COCO-133 hand keypoints in meters from HOT3D UmeTrack annotations.

        HOT3D stores hand annotations in UmeTrack JSONL format:
        - ``umetrack_hand_pose_trajectory.jsonl``: per-frame hand poses
        - ``umetrack_hand_user_profile.json``: hand model definition

        The hand model rest positions are in millimeters (same as UmeTrack).
        The wrist_xform translation in HOT3D is in meters, so we convert it to
        mm before FK, then scale the output back to meters.
        """
        seq_dir: Path = self._sequence_dir()

        # ── Load hand model ──────────────────────────────────────────────
        profile_path: Path = seq_dir / "umetrack_hand_user_profile.json"
        assert profile_path.exists(), f"Hand user profile not found at {profile_path}"

        from serde import from_dict

        # HOT3D wraps the hand model in {"hand_model": {...}} envelope
        profile_data: dict = json.loads(profile_path.read_text())
        hand_model: HandModelNumpy = from_dict(HandModelNumpy, profile_data["hand_model"])

        # ── Parse JSONL hand annotations ─────────────────────────────────
        annotations_path: Path = seq_dir / "umetrack_hand_pose_trajectory.jsonl"
        assert annotations_path.exists(), f"Hand annotations not found at {annotations_path}"

        frame_data: list[dict] = []
        with open(annotations_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    frame_data.append(json.loads(line))

        # ── Load timecode → device-time mapping ─────────────────────────
        # HOT3D JSONL timestamps are in "timecode" domain, but VRS/MPS use
        # "device time".  The mapping CSV provides the 1:1 translation.
        mapping_path: Path = seq_dir / "timecode_devicetime_mapping.csv"
        assert mapping_path.exists(), f"Timecode mapping not found at {mapping_path}"
        devicetime_ns_all: Int64[ndarray, "n_entries"] = load_timecode_to_devicetime_mapping(mapping_path)
        assert len(devicetime_ns_all) == len(frame_data), (
            f"Timecode mapping has {len(devicetime_ns_all)} entries but JSONL has {len(frame_data)}"
        )

        # ── Filter to RGB-frame-aligned entries only ─────────────────────
        # The JSONL has entries for all camera streams (~2-3x more than RGB
        # frames).  Only keep the entry closest to each RGB video frame to
        # ensure 1:1 alignment between label frames and video/camera frames.
        vrs_ts_path: Path = seq_dir / "_simplecv" / "timestamps_ns.json"
        assert vrs_ts_path.exists(), f"VRS timestamps not found at {vrs_ts_path}"
        vrs_ts_data: dict = json.loads(vrs_ts_path.read_text())
        vrs_rgb_ts: Int64[ndarray, "n_video"] = np.array(vrs_ts_data["camera-rgb"], dtype=np.int64)

        # For each RGB frame, find the nearest label by device-time
        rgb_label_indices: Int64[ndarray, "n_video"] = np.searchsorted(devicetime_ns_all, vrs_rgb_ts, side="left")
        rgb_label_indices = np.clip(rgb_label_indices, 0, len(devicetime_ns_all) - 1).astype(np.int64)

        frame_data_filtered: list[dict] = [frame_data[int(i)] for i in rgb_label_indices]
        devicetime_ns_filtered: Int64[ndarray, "n_video"] = devicetime_ns_all[rgb_label_indices]

        num_frames: int = len(frame_data_filtered)
        xyzc_stack: Float32[ndarray, "num_frames 133 4"] = np.full((num_frames, 133, 4), np.nan, dtype=np.float32)
        xyzc_stack[:, :, 3] = np.float32(0.0)

        prev_landmarks_lr: Float32[ndarray, "2 21 3"] = np.full((2, 21, 3), np.nan, dtype=np.float32)

        for frame_idx, entry in enumerate(frame_data_filtered):
            hand_poses: dict = entry.get("hand_poses", {})
            landmarks_lr: Float32[ndarray, "2 21 3"] = np.full((2, 21, 3), np.nan, dtype=np.float32)
            hand_confidences: Float32[ndarray, "2"] = np.zeros(2, dtype=np.float32)

            # HOT3D JSONL key "0" = Left hand, "1" = Right hand
            # (verified by comparing wrist positions against HOT3D clips GT)
            # simplecv: LEFT_HAND_INDEX=0, RIGHT_HAND_INDEX=1
            for hand_key, hand_idx in [("0", 0), ("1", 1)]:
                if hand_key not in hand_poses:
                    continue

                pose_data: dict = hand_poses[hand_key]
                confidence: float = float(pose_data.get("hand_confidence", 0.0))
                hand_confidences[hand_idx] = np.float32(confidence)

                if confidence > 0.0:
                    # Build wrist 4x4 transform from quaternion + translation
                    wrist_data: dict = pose_data["wrist_xform"]
                    q_wxyz: list[float] = wrist_data["q_wxyz"]
                    t_xyz: list[float] = wrist_data["t_xyz"]
                    R_wrist: Float32[ndarray, "3 3"] = quat_wxyz_to_matrix(q_wxyz)
                    # Hand model rest positions are in mm; wrist translation
                    # from HOT3D is in meters.  Convert to mm for FK.
                    t_xyz_mm: list[float] = [v * 1000.0 for v in t_xyz]
                    wrist_xform: Float32[ndarray, "4 4"] = build_4x4(R_wrist, t_xyz_mm)

                    joint_angles: Float32[ndarray, "22"] = np.array(
                        pose_data["joint_angles"], dtype=np.float32
                    )

                    hand_pose: SingleHandPose = SingleHandPose(
                        joint_angles=joint_angles,
                        wrist_xform=wrist_xform,
                        hand_confidence=confidence,
                    )
                    landmarks_mm: Float32[ndarray, "21 3"] = landmarks_from_hand_pose(
                        hand_model, hand_pose, hand_idx
                    ).astype(np.float32, copy=False)
                    scale_to_meters: float = 1e-3
                    landmarks_world: Float32[ndarray, "21 3"] = landmarks_mm * scale_to_meters
                    landmarks_lr[hand_idx] = landmarks_world
                    prev_landmarks_lr[hand_idx] = landmarks_world
                else:
                    landmarks_lr[hand_idx] = prev_landmarks_lr[hand_idx]

            xyzc_stack[frame_idx] = assembly21_to_coco133(landmarks_lr)

            # Set confidence for hand keypoints
            visible_hands: Float32[ndarray, "2"] = np.maximum(hand_confidences, np.float32(0.0))
            adjustments: tuple[tuple[int, int], ...] = ((0, 91), (1, 112))
            wrist_indices: tuple[int, int] = (9, 10)
            thumb_base_indices: tuple[int, int] = (92, 113)
            for hand_idx, coco_offset in adjustments:
                conf: float = float(visible_hands[hand_idx])
                if conf <= 0.0:
                    xyzc_stack[frame_idx, coco_offset : coco_offset + 21, 3] = np.float32(0.0)
                    xyzc_stack[frame_idx, wrist_indices[hand_idx], 3] = np.float32(0.0)
                    xyzc_stack[frame_idx, thumb_base_indices[hand_idx], 3] = np.float32(0.0)
                else:
                    xyzc_stack[frame_idx, coco_offset : coco_offset + 21, 3] = np.float32(conf)
                    xyzc_stack[frame_idx, wrist_indices[hand_idx], 3] = np.float32(conf)
                    if not np.isnan(xyzc_stack[frame_idx, thumb_base_indices[hand_idx], :3]).all():
                        xyzc_stack[frame_idx, thumb_base_indices[hand_idx], 3] = np.float32(conf)

        # Normalize label timestamps to the video container's 0-based timeline.
        vrs_start_ns: np.int64 = np.int64(vrs_rgb_ts[0])
        normalized_label_ts: Int64[ndarray, "num_frames"] = devicetime_ns_filtered - vrs_start_ns

        return ExoEgoLabels(
            xyzc_stack=xyzc_stack,
            timestamps_ns=normalized_label_ts,
        )

    @classmethod
    def iter_episode_sequences(cls, cfg: Hot3dConfig) -> Generator["Hot3dSequence", None, None]:
        """Iterate over all sequences in the HOT3D aria root directory.

        Yields one ``Hot3dSequence`` per sequence folder that contains
        a preprocessed ``_simplecv/`` directory.
        """
        root: Path = cfg.root_directory
        assert root.exists(), f"HOT3D root directory {root} does not exist."

        seq_dirs: list[Path] = natsorted([
            d for d in root.iterdir()
            if d.is_dir() and (d / "_simplecv" / "rgb.mp4").exists()
        ])

        for seq_dir in seq_dirs:
            episode_cfg: Hot3dConfig = replace(
                cfg,
                sequence_name=seq_dir.name,
            )
            try:
                yield cls(episode_cfg)
            except Exception as exc:  # pragma: no cover
                print(f"[skip] {seq_dir.name}: {exc}")

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        """Aria MPS uses gravity-aligned world with Z pointing down."""
        return rr.ViewCoordinates.RIGHT_HAND_Z_DOWN

    @property
    def image_plane_distance(self) -> int | float:
        """Image plane distance for camera visualization in meters."""
        return 0.035
