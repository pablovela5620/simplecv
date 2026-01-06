from __future__ import annotations

import json
import warnings
from collections.abc import Iterator, Sequence
from csv import DictReader
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
import rerun as rr
from jaxtyping import Bool, Float32, Float64, Int, Int64
from numpy import ndarray
from rerun import AnnotationInfo, ClassDescription
from serde import coerce, from_dict, serde
from serde import field as serde_field

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.data.skeleton.assembly_hands import assembly21_to_coco133
from simplecv.data.skeleton.coco_133 import COCO_133_ID2NAME, COCO_133_IDS, COCO_133_LINKS
from simplecv.ops.triangulate import proj_3d_vectorized
from simplecv.rerun_custom_types import Points2DWithConfidence, Points3DWithConfidence
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole, log_video
from simplecv.umetrack_temp.generic_hand_model_numpy import LANDMARK


# ---- Quest body skeleton metadata ----

_QUEST_BODY_JOINT_NAMES: tuple[str, ...] = (
    "root",
    "hips",
    "spine_lower",
    "spine_middle",
    "spine_upper",
    "chest",
    "neck",
    "head",
    "left_shoulder",
    "left_scapula",
    "left_arm_upper",
    "left_arm_lower",
    "left_hand_wrist_twist",
    "right_shoulder",
    "right_scapula",
    "right_arm_upper",
    "right_arm_lower",
    "right_hand_wrist_twist",
    "left_upper_leg",
    "left_lower_leg",
    "left_foot_ankle_twist",
    "left_foot_ankle",
    "left_foot_subtalar",
    "left_foot_transverse",
    "left_foot_ball",
    "right_upper_leg",
    "right_lower_leg",
    "right_foot_ankle_twist",
    "right_foot_ankle",
    "right_foot_subtalar",
    "right_foot_transverse",
    "right_foot_ball",
)

_QUEST_BODY_NAME_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(_QUEST_BODY_JOINT_NAMES)}

_QUEST_BODY_LINK_NAMES: tuple[tuple[str, str], ...] = (
    ("root", "hips"),
    ("hips", "spine_lower"),
    ("spine_lower", "spine_middle"),
    ("spine_middle", "spine_upper"),
    ("spine_upper", "chest"),
    ("chest", "neck"),
    ("neck", "head"),
    ("chest", "left_shoulder"),
    ("left_shoulder", "left_scapula"),
    ("left_scapula", "left_arm_upper"),
    ("left_arm_upper", "left_arm_lower"),
    ("left_arm_lower", "left_hand_wrist_twist"),
    ("chest", "right_shoulder"),
    ("right_shoulder", "right_scapula"),
    ("right_scapula", "right_arm_upper"),
    ("right_arm_upper", "right_arm_lower"),
    ("right_arm_lower", "right_hand_wrist_twist"),
    ("hips", "left_upper_leg"),
    ("left_upper_leg", "left_lower_leg"),
    ("left_lower_leg", "left_foot_ankle_twist"),
    ("left_foot_ankle_twist", "left_foot_ankle"),
    ("left_foot_ankle", "left_foot_subtalar"),
    ("left_foot_subtalar", "left_foot_transverse"),
    ("left_foot_transverse", "left_foot_ball"),
    ("hips", "right_upper_leg"),
    ("right_upper_leg", "right_lower_leg"),
    ("right_lower_leg", "right_foot_ankle_twist"),
    ("right_foot_ankle_twist", "right_foot_ankle"),
    ("right_foot_ankle", "right_foot_subtalar"),
    ("right_foot_subtalar", "right_foot_transverse"),
    ("right_foot_transverse", "right_foot_ball"),
)

_QUEST_BODY_LINKS: tuple[tuple[int, int], ...] = tuple(
    (_QUEST_BODY_NAME_TO_IDX[src], _QUEST_BODY_NAME_TO_IDX[dst]) for src, dst in _QUEST_BODY_LINK_NAMES
)

_QUEST_BODY_KEYPOINT_IDS: list[int] = list(range(len(_QUEST_BODY_JOINT_NAMES)))
_QUEST_BODY_CLASS_ID: int = 1

# Map a subset of Quest body joints into COCO-133 IDs so body logging can share the same stream
_QUEST_BODY_TO_COCO_ID: dict[str, int] = {
    # Torso/shoulders: use scapula joints for proper lateral spread
    "left_scapula": 5,
    "right_scapula": 6,
    # Arms
    "left_arm_lower": 7,  # elbow
    "right_arm_lower": 8,  # elbow
    "left_hand_wrist_twist": 9,  # wrist
    "right_hand_wrist_twist": 10,  # wrist
    # Legs
    "left_upper_leg": 11,  # hip
    "right_upper_leg": 12,  # hip
    "left_lower_leg": 13,  # knee
    "right_lower_leg": 14,  # knee
    "left_foot_ankle": 15,  # ankle
    "right_foot_ankle": 16,  # ankle
}


class QuestHandLandmark(IntEnum):
    """Quest 3 + Oak-D hand keypoint indices following the CSV column ordering."""

    PALM = 0
    WRIST = 1
    THUMB_METACARPAL = 2
    THUMB_PROXIMAL = 3
    THUMB_DISTAL = 4
    THUMB_TIP = 5
    INDEX_METACARPAL = 6
    INDEX_PROXIMAL = 7
    INDEX_INTERMEDIATE = 8
    INDEX_DISTAL = 9
    INDEX_TIP = 10
    MIDDLE_METACARPAL = 11
    MIDDLE_PROXIMAL = 12
    MIDDLE_INTERMEDIATE = 13
    MIDDLE_DISTAL = 14
    MIDDLE_TIP = 15
    RING_METACARPAL = 16
    RING_PROXIMAL = 17
    RING_INTERMEDIATE = 18
    RING_DISTAL = 19
    RING_TIP = 20
    LITTLE_METACARPAL = 21
    LITTLE_PROXIMAL = 22
    LITTLE_INTERMEDIATE = 23
    LITTLE_DISTAL = 24
    LITTLE_TIP = 25


_QUEST_HAND_LANDMARK_PREFIXES: tuple[str, ...] = (
    "palm",
    "wrist",
    "thumb_metacarpal",
    "thumb_proximal",
    "thumb_distal",
    "thumb_tip",
    "index_metacarpal",
    "index_proximal",
    "index_intermediate",
    "index_distal",
    "index_tip",
    "middle_metacarpal",
    "middle_proximal",
    "middle_intermediate",
    "middle_distal",
    "middle_tip",
    "ring_metacarpal",
    "ring_proximal",
    "ring_intermediate",
    "ring_distal",
    "ring_tip",
    "little_metacarpal",
    "little_proximal",
    "little_intermediate",
    "little_distal",
    "little_tip",
)


class QuestHandSide(IntEnum):
    """Hands available in the Quest CSV exports."""

    LEFT = 0
    RIGHT = 1

    @property
    def label(self) -> str:
        return "left" if self is QuestHandSide.LEFT else "right"

    @property
    def entity_suffix(self) -> str:
        return f"{self.label}_hand"


QUEST_HAND_LANDMARK_COUNT: int = len(QuestHandLandmark)
UME_HAND_LANDMARK_COUNT: int = len(LANDMARK)
LANDMARK_TO_QUEST_INDEX: Int64[ndarray, "n_ume_kpts"] = np.array(
    [
        QuestHandLandmark.THUMB_TIP.value,
        QuestHandLandmark.INDEX_TIP.value,
        QuestHandLandmark.MIDDLE_TIP.value,
        QuestHandLandmark.RING_TIP.value,
        QuestHandLandmark.LITTLE_TIP.value,
        QuestHandLandmark.WRIST.value,
        QuestHandLandmark.THUMB_PROXIMAL.value,
        QuestHandLandmark.THUMB_DISTAL.value,
        QuestHandLandmark.INDEX_PROXIMAL.value,
        QuestHandLandmark.INDEX_INTERMEDIATE.value,
        QuestHandLandmark.INDEX_DISTAL.value,
        QuestHandLandmark.MIDDLE_PROXIMAL.value,
        QuestHandLandmark.MIDDLE_INTERMEDIATE.value,
        QuestHandLandmark.MIDDLE_DISTAL.value,
        QuestHandLandmark.RING_PROXIMAL.value,
        QuestHandLandmark.RING_INTERMEDIATE.value,
        QuestHandLandmark.RING_DISTAL.value,
        QuestHandLandmark.LITTLE_PROXIMAL.value,
        QuestHandLandmark.LITTLE_INTERMEDIATE.value,
        QuestHandLandmark.LITTLE_DISTAL.value,
        QuestHandLandmark.PALM.value,
    ],
    dtype=np.int64,
)


@serde(type_check=coerce)
class QuestHandDictRow:
    """Raw CSV row for Quest 3 + Oak-D hand poses."""

    ts_ns: int = serde_field(rename="ts_ns")
    """Capture timestamp measured in nanoseconds from recording start."""

    palm_x: float
    """Palm center X coordinate in meters."""
    palm_y: float
    """Palm center Y coordinate in meters."""
    palm_z: float
    """Palm center Z coordinate in meters."""

    wrist_x: float
    """Wrist joint X coordinate in meters."""
    wrist_y: float
    """Wrist joint Y coordinate in meters."""
    wrist_z: float
    """Wrist joint Z coordinate in meters."""

    thumb_metacarpal_x: float
    """Thumb metacarpal joint X coordinate in meters."""
    thumb_metacarpal_y: float
    """Thumb metacarpal joint Y coordinate in meters."""
    thumb_metacarpal_z: float
    """Thumb metacarpal joint Z coordinate in meters."""

    thumb_proximal_x: float
    """Thumb proximal joint X coordinate in meters."""
    thumb_proximal_y: float
    """Thumb proximal joint Y coordinate in meters."""
    thumb_proximal_z: float
    """Thumb proximal joint Z coordinate in meters."""

    thumb_distal_x: float
    """Thumb distal joint X coordinate in meters."""
    thumb_distal_y: float
    """Thumb distal joint Y coordinate in meters."""
    thumb_distal_z: float
    """Thumb distal joint Z coordinate in meters."""

    thumb_tip_x: float
    """Thumb fingertip X coordinate in meters."""
    thumb_tip_y: float
    """Thumb fingertip Y coordinate in meters."""
    thumb_tip_z: float
    """Thumb fingertip Z coordinate in meters."""

    index_metacarpal_x: float
    """Index metacarpal joint X coordinate in meters."""
    index_metacarpal_y: float
    """Index metacarpal joint Y coordinate in meters."""
    index_metacarpal_z: float
    """Index metacarpal joint Z coordinate in meters."""

    index_proximal_x: float
    """Index proximal joint X coordinate in meters."""
    index_proximal_y: float
    """Index proximal joint Y coordinate in meters."""
    index_proximal_z: float
    """Index proximal joint Z coordinate in meters."""

    index_intermediate_x: float
    """Index intermediate joint X coordinate in meters."""
    index_intermediate_y: float
    """Index intermediate joint Y coordinate in meters."""
    index_intermediate_z: float
    """Index intermediate joint Z coordinate in meters."""

    index_distal_x: float
    """Index distal joint X coordinate in meters."""
    index_distal_y: float
    """Index distal joint Y coordinate in meters."""
    index_distal_z: float
    """Index distal joint Z coordinate in meters."""

    index_tip_x: float
    """Index fingertip X coordinate in meters."""
    index_tip_y: float
    """Index fingertip Y coordinate in meters."""
    index_tip_z: float
    """Index fingertip Z coordinate in meters."""

    middle_metacarpal_x: float
    """Middle metacarpal joint X coordinate in meters."""
    middle_metacarpal_y: float
    """Middle metacarpal joint Y coordinate in meters."""
    middle_metacarpal_z: float
    """Middle metacarpal joint Z coordinate in meters."""

    middle_proximal_x: float
    """Middle proximal joint X coordinate in meters."""
    middle_proximal_y: float
    """Middle proximal joint Y coordinate in meters."""
    middle_proximal_z: float
    """Middle proximal joint Z coordinate in meters."""

    middle_intermediate_x: float
    """Middle intermediate joint X coordinate in meters."""
    middle_intermediate_y: float
    """Middle intermediate joint Y coordinate in meters."""
    middle_intermediate_z: float
    """Middle intermediate joint Z coordinate in meters."""

    middle_distal_x: float
    """Middle distal joint X coordinate in meters."""
    middle_distal_y: float
    """Middle distal joint Y coordinate in meters."""
    middle_distal_z: float
    """Middle distal joint Z coordinate in meters."""

    middle_tip_x: float
    """Middle fingertip X coordinate in meters."""
    middle_tip_y: float
    """Middle fingertip Y coordinate in meters."""
    middle_tip_z: float
    """Middle fingertip Z coordinate in meters."""

    ring_metacarpal_x: float
    """Ring metacarpal joint X coordinate in meters."""
    ring_metacarpal_y: float
    """Ring metacarpal joint Y coordinate in meters."""
    ring_metacarpal_z: float
    """Ring metacarpal joint Z coordinate in meters."""

    ring_proximal_x: float
    """Ring proximal joint X coordinate in meters."""
    ring_proximal_y: float
    """Ring proximal joint Y coordinate in meters."""
    ring_proximal_z: float
    """Ring proximal joint Z coordinate in meters."""

    ring_intermediate_x: float
    """Ring intermediate joint X coordinate in meters."""
    ring_intermediate_y: float
    """Ring intermediate joint Y coordinate in meters."""
    ring_intermediate_z: float
    """Ring intermediate joint Z coordinate in meters."""

    ring_distal_x: float
    """Ring distal joint X coordinate in meters."""
    ring_distal_y: float
    """Ring distal joint Y coordinate in meters."""
    ring_distal_z: float
    """Ring distal joint Z coordinate in meters."""

    ring_tip_x: float
    """Ring fingertip X coordinate in meters."""
    ring_tip_y: float
    """Ring fingertip Y coordinate in meters."""
    ring_tip_z: float
    """Ring fingertip Z coordinate in meters."""

    little_metacarpal_x: float
    """Little metacarpal joint X coordinate in meters."""
    little_metacarpal_y: float
    """Little metacarpal joint Y coordinate in meters."""
    little_metacarpal_z: float
    """Little metacarpal joint Z coordinate in meters."""

    little_proximal_x: float
    """Little proximal joint X coordinate in meters."""
    little_proximal_y: float
    """Little proximal joint Y coordinate in meters."""
    little_proximal_z: float
    """Little proximal joint Z coordinate in meters."""

    little_intermediate_x: float
    """Little intermediate joint X coordinate in meters."""
    little_intermediate_y: float
    """Little intermediate joint Y coordinate in meters."""
    little_intermediate_z: float
    """Little intermediate joint Z coordinate in meters."""

    little_distal_x: float
    """Little distal joint X coordinate in meters."""
    little_distal_y: float
    """Little distal joint Y coordinate in meters."""
    little_distal_z: float
    """Little distal joint Z coordinate in meters."""

    little_tip_x: float
    """Little fingertip X coordinate in meters."""
    little_tip_y: float
    """Little fingertip Y coordinate in meters."""
    little_tip_z: float
    """Little fingertip Z coordinate in meters."""


@serde(type_check=coerce)
class QuestBodyDictRow:
    """Raw CSV row for Quest 3 + Oak-D full-body poses (32 joints)."""

    ts_ns: int = serde_field(rename="ts_ns")
    """Capture timestamp measured in nanoseconds from recording start."""

    # Spine / torso
    root_x: float
    root_y: float
    root_z: float
    root_qx: float
    root_qy: float
    root_qz: float
    root_qw: float

    hips_x: float
    hips_y: float
    hips_z: float
    hips_qx: float
    hips_qy: float
    hips_qz: float
    hips_qw: float

    spine_lower_x: float
    spine_lower_y: float
    spine_lower_z: float
    spine_lower_qx: float
    spine_lower_qy: float
    spine_lower_qz: float
    spine_lower_qw: float

    spine_middle_x: float
    spine_middle_y: float
    spine_middle_z: float
    spine_middle_qx: float
    spine_middle_qy: float
    spine_middle_qz: float
    spine_middle_qw: float

    spine_upper_x: float
    spine_upper_y: float
    spine_upper_z: float
    spine_upper_qx: float
    spine_upper_qy: float
    spine_upper_qz: float
    spine_upper_qw: float

    chest_x: float
    chest_y: float
    chest_z: float
    chest_qx: float
    chest_qy: float
    chest_qz: float
    chest_qw: float

    neck_x: float
    neck_y: float
    neck_z: float
    neck_qx: float
    neck_qy: float
    neck_qz: float
    neck_qw: float

    head_x: float
    head_y: float
    head_z: float
    head_qx: float
    head_qy: float
    head_qz: float
    head_qw: float

    # Left arm chain
    left_shoulder_x: float
    left_shoulder_y: float
    left_shoulder_z: float
    left_shoulder_qx: float
    left_shoulder_qy: float
    left_shoulder_qz: float
    left_shoulder_qw: float

    left_scapula_x: float
    left_scapula_y: float
    left_scapula_z: float
    left_scapula_qx: float
    left_scapula_qy: float
    left_scapula_qz: float
    left_scapula_qw: float

    left_arm_upper_x: float
    left_arm_upper_y: float
    left_arm_upper_z: float
    left_arm_upper_qx: float
    left_arm_upper_qy: float
    left_arm_upper_qz: float
    left_arm_upper_qw: float

    left_arm_lower_x: float
    left_arm_lower_y: float
    left_arm_lower_z: float
    left_arm_lower_qx: float
    left_arm_lower_qy: float
    left_arm_lower_qz: float
    left_arm_lower_qw: float

    left_hand_wrist_twist_x: float
    left_hand_wrist_twist_y: float
    left_hand_wrist_twist_z: float
    left_hand_wrist_twist_qx: float
    left_hand_wrist_twist_qy: float
    left_hand_wrist_twist_qz: float
    left_hand_wrist_twist_qw: float

    # Right arm chain
    right_shoulder_x: float
    right_shoulder_y: float
    right_shoulder_z: float
    right_shoulder_qx: float
    right_shoulder_qy: float
    right_shoulder_qz: float
    right_shoulder_qw: float

    right_scapula_x: float
    right_scapula_y: float
    right_scapula_z: float
    right_scapula_qx: float
    right_scapula_qy: float
    right_scapula_qz: float
    right_scapula_qw: float

    right_arm_upper_x: float
    right_arm_upper_y: float
    right_arm_upper_z: float
    right_arm_upper_qx: float
    right_arm_upper_qy: float
    right_arm_upper_qz: float
    right_arm_upper_qw: float

    right_arm_lower_x: float
    right_arm_lower_y: float
    right_arm_lower_z: float
    right_arm_lower_qx: float
    right_arm_lower_qy: float
    right_arm_lower_qz: float
    right_arm_lower_qw: float

    right_hand_wrist_twist_x: float
    right_hand_wrist_twist_y: float
    right_hand_wrist_twist_z: float
    right_hand_wrist_twist_qx: float
    right_hand_wrist_twist_qy: float
    right_hand_wrist_twist_qz: float
    right_hand_wrist_twist_qw: float

    # Left leg chain
    left_upper_leg_x: float
    left_upper_leg_y: float
    left_upper_leg_z: float
    left_upper_leg_qx: float
    left_upper_leg_qy: float
    left_upper_leg_qz: float
    left_upper_leg_qw: float

    left_lower_leg_x: float
    left_lower_leg_y: float
    left_lower_leg_z: float
    left_lower_leg_qx: float
    left_lower_leg_qy: float
    left_lower_leg_qz: float
    left_lower_leg_qw: float

    left_foot_ankle_twist_x: float
    left_foot_ankle_twist_y: float
    left_foot_ankle_twist_z: float
    left_foot_ankle_twist_qx: float
    left_foot_ankle_twist_qy: float
    left_foot_ankle_twist_qz: float
    left_foot_ankle_twist_qw: float

    left_foot_ankle_x: float
    left_foot_ankle_y: float
    left_foot_ankle_z: float
    left_foot_ankle_qx: float
    left_foot_ankle_qy: float
    left_foot_ankle_qz: float
    left_foot_ankle_qw: float

    left_foot_subtalar_x: float
    left_foot_subtalar_y: float
    left_foot_subtalar_z: float
    left_foot_subtalar_qx: float
    left_foot_subtalar_qy: float
    left_foot_subtalar_qz: float
    left_foot_subtalar_qw: float

    left_foot_transverse_x: float
    left_foot_transverse_y: float
    left_foot_transverse_z: float
    left_foot_transverse_qx: float
    left_foot_transverse_qy: float
    left_foot_transverse_qz: float
    left_foot_transverse_qw: float

    left_foot_ball_x: float
    left_foot_ball_y: float
    left_foot_ball_z: float
    left_foot_ball_qx: float
    left_foot_ball_qy: float
    left_foot_ball_qz: float
    left_foot_ball_qw: float

    # Right leg chain
    right_upper_leg_x: float
    right_upper_leg_y: float
    right_upper_leg_z: float
    right_upper_leg_qx: float
    right_upper_leg_qy: float
    right_upper_leg_qz: float
    right_upper_leg_qw: float

    right_lower_leg_x: float
    right_lower_leg_y: float
    right_lower_leg_z: float
    right_lower_leg_qx: float
    right_lower_leg_qy: float
    right_lower_leg_qz: float
    right_lower_leg_qw: float

    right_foot_ankle_twist_x: float
    right_foot_ankle_twist_y: float
    right_foot_ankle_twist_z: float
    right_foot_ankle_twist_qx: float
    right_foot_ankle_twist_qy: float
    right_foot_ankle_twist_qz: float
    right_foot_ankle_twist_qw: float

    right_foot_ankle_x: float
    right_foot_ankle_y: float
    right_foot_ankle_z: float
    right_foot_ankle_qx: float
    right_foot_ankle_qy: float
    right_foot_ankle_qz: float
    right_foot_ankle_qw: float

    right_foot_subtalar_x: float
    right_foot_subtalar_y: float
    right_foot_subtalar_z: float
    right_foot_subtalar_qx: float
    right_foot_subtalar_qy: float
    right_foot_subtalar_qz: float
    right_foot_subtalar_qw: float

    right_foot_transverse_x: float
    right_foot_transverse_y: float
    right_foot_transverse_z: float
    right_foot_transverse_qx: float
    right_foot_transverse_qy: float
    right_foot_transverse_qz: float
    right_foot_transverse_qw: float

    right_foot_ball_x: float
    right_foot_ball_y: float
    right_foot_ball_z: float
    right_foot_ball_qx: float
    right_foot_ball_qy: float
    right_foot_ball_qz: float
    right_foot_ball_qw: float

@dataclass
class QuestHandPoseSample:
    """Single Quest hand pose sample containing timestamp and 3D landmarks."""

    timestamp_ns: int
    """Relative timestamp, in nanoseconds from the recording start."""
    keypoints_m: Float32[ndarray, "n_quest_kpts=26 3"]
    """3D keypoints expressed in meters within the Quest coordinate frame."""


@dataclass
class QuestHandPoseSequence:
    """Sequence of Quest hand pose samples parsed from the Quest CSV export."""

    timestamps_ns: Int64[ndarray, "n_frames"]
    """Monotonic capture timestamps for each frame, measured in nanoseconds."""
    keypoints_m: Float32[ndarray, "n_frames n_quest_kpts=26 3"]
    """Per-frame 3D keypoint coordinates in meters."""

    def __len__(self) -> int:
        return int(self.timestamps_ns.shape[0])

    def __iter__(self) -> Iterator[QuestHandPoseSample]:
        for frame_idx in range(len(self)):
            keypoints_frame: Float32[ndarray, "n_quest_kpts=26 3"] = self.keypoints_m[frame_idx]
            yield QuestHandPoseSample(
                timestamp_ns=int(self.timestamps_ns[frame_idx]),
                keypoints_m=keypoints_frame,
            )


@dataclass
class QuestBodyPoseSample:
    """Single Quest full-body pose sample containing timestamped joint data."""

    timestamp_ns: int
    """Relative timestamp, in nanoseconds from the recording start."""
    joint_positions_m: Float32[ndarray, "n_body_joints=32 3"]
    """Per-joint world-space positions expressed in meters."""
    joint_rotations_xyzw: Float32[ndarray, "n_body_joints=32 4"]
    """Per-joint world-space orientations as ``(x, y, z, w)`` quaternions."""


@dataclass
class QuestBodyPoseSequence:
    """Sequence of Quest full-body poses parsed from the Quest CSV export."""

    timestamps_ns: Int64[ndarray, "n_frames"]
    """Monotonic capture timestamps for each frame, measured in nanoseconds."""
    joint_positions_m: Float32[ndarray, "n_frames n_body_joints=32 3"]
    """Stack of joint positions per frame, measured in meters."""
    joint_rotations_xyzw: Float32[ndarray, "n_frames n_body_joints=32 4"]
    """Stack of joint quaternions per frame, stored as ``(x, y, z, w)``."""

    def __len__(self) -> int:
        return int(self.timestamps_ns.shape[0])

    def __iter__(self) -> Iterator[QuestBodyPoseSample]:
        for frame_idx in range(len(self)):
            yield QuestBodyPoseSample(
                timestamp_ns=int(self.timestamps_ns[frame_idx]),
                joint_positions_m=self.joint_positions_m[frame_idx],
                joint_rotations_xyzw=self.joint_rotations_xyzw[frame_idx],
            )


@dataclass
class Quest3VisualizeConfig:
    """Structured configuration for running the Quest3/Oak-D visualization CLI."""

    rr_config: RerunTyroConfig
    """Command-line options for spawning and configuring the Rerun viewer."""
    data_dir: Path
    """Path to the root directory of the Quest3 Oak-D dataset."""


def _select_common_video_timestamps(
    *,
    left_timestamps: Int[ndarray, "n_left"],
    right_timestamps: Int[ndarray, "n_right"],
) -> Int64[ndarray, "n_common"]:
    """Intersect two per-frame timestamp arrays and keep the shared prefix.

    The Quest left/right eye MP4s occasionally disagree by a handful of frames.
    Because we currently downsample all tracker data to the video cadence, we
    trim to the overlapping prefix and warn the caller so the data loss is
    explicit.

    Args:
        left_timestamps: Nanosecond timestamps emitted by the left-eye MP4.
        right_timestamps: Nanosecond timestamps emitted by the right-eye MP4.

    Returns:
        Int64[np.ndarray, "n_common"]: Prefix of ``left_timestamps`` that
        overlaps with ``right_timestamps``. The length equals the minimum frame
        count among both streams.
    """
    left_ts: Int64[ndarray, "n_left"] = np.asarray(left_timestamps, dtype=np.int64)
    right_ts: Int64[ndarray, "n_right"] = np.asarray(right_timestamps, dtype=np.int64)
    if left_ts.size == 0 or right_ts.size == 0:
        raise ValueError("Quest eye videos must contain at least one frame to drive logging.")
    n_common: int = int(min(left_ts.size, right_ts.size))
    if left_ts.size != right_ts.size or not np.array_equal(left_ts[:n_common], right_ts[:n_common]):
        warnings.warn(
            "Quest eye videos have mismatched timestamps; trimming to the overlapping prefix.",
            stacklevel=2,
        )
    return left_ts[:n_common]


def _nearest_sample_indices(
    *,
    source_timestamps_ns: Int64[ndarray, "n_source"],
    target_timestamps_ns: Int64[ndarray, "n_target"],
) -> Int64[ndarray, "n_target"]:
    """Map each ``target`` timestamp to the closest source sample index.

    Args:
        source_timestamps_ns: Monotonic nanosecond timestamps belonging to the
            high-rate tracker stream.
        target_timestamps_ns: Desired nanosecond timestamps, typically the
            MP4-derived ``video_time`` array.

    Returns:
        Int64[np.ndarray, "n_target"]: Index per target timestamp referencing
        the most recent source sample (nearest neighbor to the left).
    """
    if source_timestamps_ns.size == 0:
        raise ValueError("Cannot resample an empty timeline.")
    insertion_points: Int64[ndarray, "n_target"] = (
        np.searchsorted(source_timestamps_ns, target_timestamps_ns, side="right") - 1
    ).astype(np.int64)
    insertion_points[insertion_points < 0] = 0
    insertion_points[insertion_points >= source_timestamps_ns.size] = source_timestamps_ns.size - 1
    return insertion_points


def _resample_head_extrinsics(
    *,
    samples: Sequence[QuestHeadExtrinsicsSample],
    target_timestamps_ns: Int64[ndarray, "n_target"],
) -> list[QuestHeadExtrinsicsSample]:
    """Subsample head extrinsics to match the MP4 frame cadence.

    Args:
        samples: Original tracker samples emitted by Quest at a higher rate.
        target_timestamps_ns: Target nanosecond timestamps (one per video
            frame) that should drive logging.

    Returns:
        list[QuestHeadExtrinsicsSample]: Downsampled extrinsics sharing the
        ``target_timestamps_ns`` cadence so they stay in sync with video logs.
    """
    source_timestamps_ns: Int64[ndarray, "n_source"] = np.asarray(
        [sample.timestamp_ns for sample in samples],
        dtype=np.int64,
    )
    indices: Int64[ndarray, "n_target"] = _nearest_sample_indices(
        source_timestamps_ns=source_timestamps_ns,
        target_timestamps_ns=target_timestamps_ns,
    )
    resampled: list[QuestHeadExtrinsicsSample] = []
    for idx, target_timestamp_ns in zip(indices, target_timestamps_ns, strict=True):
        source_sample: QuestHeadExtrinsicsSample = samples[int(idx)]
        resampled.append(
            QuestHeadExtrinsicsSample(
                timestamp_ns=int(target_timestamp_ns),
                left_extrinsics=source_sample.left_extrinsics,
                right_extrinsics=source_sample.right_extrinsics,
            )
        )
    return resampled


def _resample_hand_sequence(
    *,
    sequence: QuestHandPoseSequence,
    target_timestamps_ns: Int64[ndarray, "n_target"],
) -> QuestHandPoseSequence:
    """Return a new hand sequence with one sample per ``target_timestamps_ns``.

    Args:
        sequence: Raw Quest hand pose sequence containing the original cadence.
        target_timestamps_ns: Target timeline shared with the MP4 logs.

    Returns:
        QuestHandPoseSequence: Copy of ``sequence`` trimmed/resampled to the
        requested timestamps.
    """
    source_timestamps_ns: Int64[ndarray, "n_source"] = sequence.timestamps_ns.astype(np.int64, copy=False)
    indices: Int64[ndarray, "n_target"] = _nearest_sample_indices(
        source_timestamps_ns=source_timestamps_ns,
        target_timestamps_ns=target_timestamps_ns,
    )
    keypoints_resampled: Float32[ndarray, "n_target n_quest_kpts=26 3"] = sequence.keypoints_m[indices]
    resampled_sequence = QuestHandPoseSequence(
        timestamps_ns=target_timestamps_ns.astype(np.int64, copy=False),
        keypoints_m=keypoints_resampled.astype(np.float32, copy=False),
    )
    return resampled_sequence


def load_hand_sequence(csv_path: Path) -> QuestHandPoseSequence:
    """Load Quest 3 hand landmarks from the CSV export."""
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open(encoding="utf-8", newline="") as file:
        reader: DictReader[str] = DictReader(file)
        samples: list[QuestHandPoseSample] = []
        for row_dict in reader:
            if not row_dict or all(value == "" for value in row_dict.values()):
                continue

            sample: QuestHandDictRow = from_dict(QuestHandDictRow, row_dict)
            # convert to numpy array
            xyz_hand_list: list[tuple[float, float, float]] = [
                (getattr(sample, f"{prefix}_x"), getattr(sample, f"{prefix}_y"), getattr(sample, f"{prefix}_z"))
                for prefix in _QUEST_HAND_LANDMARK_PREFIXES
            ]
            xyz_hand: Float32[ndarray, "n_landmarks=26 3"] = np.array(xyz_hand_list, dtype=np.float32)
            timestamp_ns: int = int(sample.ts_ns)

            samples.append(QuestHandPoseSample(timestamp_ns=timestamp_ns, keypoints_m=xyz_hand))

    if not samples:
        raise ValueError(f"CSV file {csv_path} does not contain any pose rows.")

    timestamps_ns: Int64[ndarray, "n_frames"] = np.asarray([sample.timestamp_ns for sample in samples], dtype=np.int64)
    keypoints_stack: Float32[ndarray, "n_frames n_quest_kpts=26 3"] = np.stack(
        [sample.keypoints_m for sample in samples],
        axis=0,
    )

    sequence = QuestHandPoseSequence(
        timestamps_ns=timestamps_ns,
        keypoints_m=keypoints_stack,
    )
    sequence.timestamps_ns = sequence.timestamps_ns - int(sequence.timestamps_ns[0])

    zero_mask: Bool[ndarray, "n_frames n_quest_kpts=26"] = np.asarray(
        np.isclose(sequence.keypoints_m, 0.0, atol=1e-6).all(axis=-1),
        dtype=bool,
    )
    sequence.keypoints_m[zero_mask] = np.nan
    return sequence


def load_body_sequence(csv_path: Path) -> QuestBodyPoseSequence:
    """Load Quest 3 full-body joints from CSV, zero-offset timestamps, and NaN-out zeros."""

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open(encoding="utf-8", newline="") as file:
        reader: DictReader[str] = DictReader(file)
        samples: list[QuestBodyPoseSample] = []
        for row_dict in reader:
            if not row_dict or all(value == "" for value in row_dict.values()):
                continue

            sample_row: QuestBodyDictRow = from_dict(QuestBodyDictRow, row_dict)
            timestamp_ns: int = int(sample_row.ts_ns)

            positions_list: list[tuple[float, float, float]] = []
            rotations_list: list[tuple[float, float, float, float]] = []
            for joint_name in _QUEST_BODY_JOINT_NAMES:
                pos: tuple[float, float, float] = (
                    getattr(sample_row, f"{joint_name}_x"),
                    getattr(sample_row, f"{joint_name}_y"),
                    getattr(sample_row, f"{joint_name}_z"),
                )
                rot: tuple[float, float, float, float] = (
                    getattr(sample_row, f"{joint_name}_qx"),
                    getattr(sample_row, f"{joint_name}_qy"),
                    getattr(sample_row, f"{joint_name}_qz"),
                    getattr(sample_row, f"{joint_name}_qw"),
                )
                positions_list.append(pos)
                rotations_list.append(rot)

            joint_positions_m: Float32[ndarray, "n_body_joints=32 3"] = np.array(positions_list, dtype=np.float32)
            joint_rotations_xyzw: Float32[ndarray, "n_body_joints=32 4"] = np.array(
                rotations_list,
                dtype=np.float32,
            )
            samples.append(
                QuestBodyPoseSample(
                    timestamp_ns=timestamp_ns,
                    joint_positions_m=joint_positions_m,
                    joint_rotations_xyzw=joint_rotations_xyzw,
                )
            )

    if not samples:
        raise ValueError(f"CSV file {csv_path} does not contain any pose rows.")

    timestamps_ns: Int64[ndarray, "n_frames"] = np.asarray(
        [sample.timestamp_ns for sample in samples],
        dtype=np.int64,
    )
    joint_positions_stack: Float32[ndarray, "n_frames n_body_joints=32 3"] = np.stack(
        [sample.joint_positions_m for sample in samples],
        axis=0,
    )
    joint_rotations_stack: Float32[ndarray, "n_frames n_body_joints=32 4"] = np.stack(
        [sample.joint_rotations_xyzw for sample in samples],
        axis=0,
    )

    timestamps_ns = timestamps_ns - int(timestamps_ns[0])

    zero_mask_body: Bool[ndarray, "n_frames n_body_joints=32"] = np.asarray(
        np.isclose(joint_positions_stack, 0.0, atol=1e-6).all(axis=-1),
        dtype=bool,
    )
    joint_positions_stack[zero_mask_body] = np.nan

    return QuestBodyPoseSequence(
        timestamps_ns=timestamps_ns,
        joint_positions_m=joint_positions_stack,
        joint_rotations_xyzw=joint_rotations_stack,
    )


def _resample_body_sequence(
    *,
    sequence: QuestBodyPoseSequence,
    target_timestamps_ns: Int64[ndarray, "n_target"],
) -> QuestBodyPoseSequence:
    source_timestamps_ns: Int64[ndarray, "n_source"] = sequence.timestamps_ns.astype(np.int64, copy=False)
    indices: Int64[ndarray, "n_target"] = _nearest_sample_indices(
        source_timestamps_ns=source_timestamps_ns,
        target_timestamps_ns=target_timestamps_ns,
    )

    positions_resampled: Float32[ndarray, "n_target n_body_joints=32 3"] = sequence.joint_positions_m[indices]
    rotations_resampled: Float32[ndarray, "n_target n_body_joints=32 4"] = sequence.joint_rotations_xyzw[indices]

    return QuestBodyPoseSequence(
        timestamps_ns=target_timestamps_ns.astype(np.int64, copy=False),
        joint_positions_m=positions_resampled.astype(np.float32, copy=False),
        joint_rotations_xyzw=rotations_resampled.astype(np.float32, copy=False),
    )


@dataclass(slots=True)
class QuestHeadExtrinsicsSample:
    """Quest head pose extrinsics accompanied by capture timestamp."""

    timestamp_ns: int
    """Relative timestamp, measured in nanoseconds from recording start."""
    left_extrinsics: Extrinsics
    """Camera-to-world pose describing the left-eye tracking camera."""
    right_extrinsics: Extrinsics
    """Camera-to-world pose describing the right-eye tracking camera."""


def _log_annotation_context() -> None:
    coco_description: ClassDescription = ClassDescription(
        info=AnnotationInfo(id=0, label="COCO Wholebody", color=(0, 0, 255)),
        keypoint_annotations=[AnnotationInfo(id=kpt_id, label=COCO_133_ID2NAME[kpt_id]) for kpt_id in COCO_133_IDS],
        keypoint_connections=COCO_133_LINKS,
    )

    rr.log(
        "/",
        rr.AnnotationContext(
            [
                coco_description,
            ]
        ),
        static=True,
    )


def _log_coco133_annotations(
    *,
    left_sequence: QuestHandPoseSequence,
    right_sequence: QuestHandPoseSequence,
    body_sequence: QuestBodyPoseSequence,
    head_extrinsics: Sequence[QuestHeadExtrinsicsSample],
    left_intrinsics: Intrinsics,
    right_intrinsics: Intrinsics,
    quest_left_cam_path: Path,
    quest_right_cam_path: Path,
    timeline: str,
) -> None:
    """Log COCO-133 joints by fusing Quest hands, body, and head extrinsics."""
    left_keypoints: Float32[ndarray, "n_frames_left 21 3"] = left_sequence.keypoints_m[:, LANDMARK_TO_QUEST_INDEX]
    right_keypoints: Float32[ndarray, "n_frames_right 21 3"] = right_sequence.keypoints_m[:, LANDMARK_TO_QUEST_INDEX]
    frame_count: int = min(len(head_extrinsics), left_keypoints.shape[0], right_keypoints.shape[0], len(body_sequence))
    if frame_count == 0:
        return

    coco_entity_path: Path = Path("/world/gt/coco133_xyz")
    for frame_idx in range(frame_count):
        timestamp_ns: int = int(head_extrinsics[frame_idx].timestamp_ns)

        kpts_lr: Float32[ndarray, "2 21 3"] = np.stack(
            (
                left_keypoints[frame_idx].astype(np.float32, copy=False),
                right_keypoints[frame_idx].astype(np.float32, copy=False),
            ),
            axis=0,
        )
        # Start with hands mapped into COCO-133
        coco_frame: Float32[ndarray, "133 4"] = assembly21_to_coco133(kpts_lr)

        # Overlay body joints into the same COCO frame (only if missing)
        body_positions: Float32[ndarray, "n_body_joints=32 3"] = body_sequence.joint_positions_m[frame_idx].copy()

        # Widen shoulders to roughly match hip width for better visual alignment
        left_sh_idx: int = _QUEST_BODY_NAME_TO_IDX["left_shoulder"]
        right_sh_idx: int = _QUEST_BODY_NAME_TO_IDX["right_shoulder"]
        left_sh: Float32[ndarray, "3"] = body_positions[left_sh_idx]
        right_sh: Float32[ndarray, "3"] = body_positions[right_sh_idx]

        left_hip_idx: int = _QUEST_BODY_NAME_TO_IDX["left_upper_leg"]
        right_hip_idx: int = _QUEST_BODY_NAME_TO_IDX["right_upper_leg"]
        left_hip: Float32[ndarray, "3"] = body_positions[left_hip_idx]
        right_hip: Float32[ndarray, "3"] = body_positions[right_hip_idx]

        sh_dir: Float32[ndarray, "3"] = right_sh - left_sh
        sh_width: float = float(np.linalg.norm(sh_dir))
        hip_dir: Float32[ndarray, "3"] = right_hip - left_hip
        hip_width: float = float(np.linalg.norm(hip_dir))

        if hip_width > 1e-6:
            dir_unit: Float32[ndarray, "3"] = hip_dir / hip_width if sh_width < 1e-6 else sh_dir / sh_width
            shoulder_center: Float32[ndarray, "3"] = 0.5 * (left_sh + right_sh)
            desired_half: float = 0.5 * hip_width

            new_left_sh: Float32[ndarray, "3"] = shoulder_center - dir_unit * desired_half
            new_right_sh: Float32[ndarray, "3"] = shoulder_center + dir_unit * desired_half

            delta_left: Float32[ndarray, "3"] = new_left_sh - left_sh
            delta_right: Float32[ndarray, "3"] = new_right_sh - right_sh

            body_positions[left_sh_idx] = new_left_sh
            body_positions[right_sh_idx] = new_right_sh

            # propagate shoulder adjustment down each arm chain
            left_chain = [
                _QUEST_BODY_NAME_TO_IDX["left_arm_upper"],
                _QUEST_BODY_NAME_TO_IDX["left_arm_lower"],
                _QUEST_BODY_NAME_TO_IDX["left_hand_wrist_twist"],
            ]
            right_chain = [
                _QUEST_BODY_NAME_TO_IDX["right_arm_upper"],
                _QUEST_BODY_NAME_TO_IDX["right_arm_lower"],
                _QUEST_BODY_NAME_TO_IDX["right_hand_wrist_twist"],
            ]

            for idx in left_chain:
                body_positions[idx] = body_positions[idx] + delta_left
            for idx in right_chain:
                body_positions[idx] = body_positions[idx] + delta_right

        for joint_name, coco_id in _QUEST_BODY_TO_COCO_ID.items():
            joint_idx: int = _QUEST_BODY_NAME_TO_IDX[joint_name]
            xyz: Float32[ndarray, "3"] = body_positions[joint_idx]
            if np.isnan(xyz).any():
                continue
            current_conf: float = float(coco_frame[coco_id, 3])
            if current_conf == 0.0 or np.isnan(current_conf):
                coco_frame[coco_id, :3] = xyz
                coco_frame[coco_id, 3] = np.float32(1.0)

        positions: Float32[ndarray, "133 3"] = coco_frame[:, :3]
        confidences: Float32[ndarray, "133"] = np.nan_to_num(coco_frame[:, 3], nan=0.0).astype(np.float32, copy=False)
        invalid_mask: Bool[ndarray, "133"] = np.asarray(np.isnan(positions).any(axis=1), dtype=bool)
        confidences[invalid_mask] = np.float32(0.0)

        rr.set_time(timeline, duration=np.timedelta64(timestamp_ns, "ns"))
        rr.log(
            str(coco_entity_path),
            Points3DWithConfidence(
                positions=positions,
                confidences=confidences,
                class_ids=0,
                keypoint_ids=COCO_133_IDS,
                show_labels=False,
            ),
            )

        head_sample: QuestHeadExtrinsicsSample = head_extrinsics[frame_idx]
        left_pinhole: PinholeParameters = PinholeParameters(
            name="quest_left_eye",
            extrinsics=head_sample.left_extrinsics,
            intrinsics=left_intrinsics,
        )
        right_pinhole: PinholeParameters = PinholeParameters(
            name="quest_right_eye",
            extrinsics=head_sample.right_extrinsics,
            intrinsics=right_intrinsics,
        )

        for cam_path, pinhole_param in (
            (quest_left_cam_path, left_pinhole),
            (quest_right_cam_path, right_pinhole),
        ):
            _log_projected_keypoints(
                camera_path=cam_path,
                pinhole_param=pinhole_param,
                positions=positions,
                confidences=confidences,
                keypoint_ids=COCO_133_IDS,
                class_id=0,
                entity_suffix="coco133_uv",
            )


def _log_projected_keypoints(
    *,
    camera_path: Path,
    pinhole_param: PinholeParameters,
    positions: Float32[ndarray, "n_kpts 3"],
    confidences: Float32[ndarray, "n_kpts"],
    keypoint_ids: Sequence[int],
    class_id: int,
    entity_suffix: str,
) -> None:
    """Project 3D joints into image space and stream them (with confidences) to Rerun."""

    if len(keypoint_ids) != positions.shape[0]:
        raise ValueError("Keypoint ID list must match number of positions provided.")

    uv_positions: Float32[ndarray, "n_kpts 2"] = np.full((positions.shape[0], 2), np.nan, dtype=np.float32)
    uv_confidences: Float32[ndarray, "n_kpts"] = np.zeros_like(confidences, dtype=np.float32)
    valid_mask: Bool[ndarray, "n_kpts"] = (~np.isnan(positions).any(axis=1)) & (confidences > 0.0)
    valid_indices_all: Int[ndarray, "n_valid"] = np.flatnonzero(valid_mask)
    if valid_indices_all.size > 0:
        xyz_hom_valid: Float32[ndarray, "n_valid 4"] = np.concatenate(
            [positions[valid_indices_all], np.ones((valid_indices_all.size, 1), dtype=np.float32)],
            axis=1,
        )
        xyz_hom_stack: Float32[ndarray, "1 n_valid 4"] = xyz_hom_valid[np.newaxis, ...]
        Pall: Float64[ndarray, "1 3 4"] = pinhole_param.projection_matrix[np.newaxis, ...].astype(np.float64)
        uv_raw: Float32[ndarray, "n_valid 2"] = proj_3d_vectorized(xyz_hom=xyz_hom_stack, P=Pall)[0, 0].astype(
            np.float32,
            copy=False,
        )

        cam_T_world: Float32[ndarray, "4 4"] = pinhole_param.extrinsics.cam_T_world.astype(np.float32)
        xyz_cam: Float32[ndarray, "n_valid 4"] = (cam_T_world @ xyz_hom_valid.T).T
        depth: Float32[ndarray, "n_valid"] = xyz_cam[:, 2]

        intrinsics = pinhole_param.intrinsics
        width: float = float(intrinsics.width if intrinsics.width is not None else 2.0 * intrinsics.cx)
        height: float = float(intrinsics.height if intrinsics.height is not None else 2.0 * intrinsics.cy)
        bounds_mask: Bool[ndarray, "n_valid"] = (
            (uv_raw[:, 0] >= 0.0) & (uv_raw[:, 0] <= width) & (uv_raw[:, 1] >= 0.0) & (uv_raw[:, 1] <= height)
        )
        positive_depth: Bool[ndarray, "n_valid"] = depth > 0.0
        final_mask: Bool[ndarray, "n_valid"] = bounds_mask & positive_depth
        if np.any(final_mask):
            final_indices: Int[ndarray, "k"] = valid_indices_all[final_mask]
            uv_positions[final_indices] = uv_raw[final_mask]
            uv_confidences[final_indices] = confidences[final_indices]

    rr.log(
        str(camera_path / "pinhole" / entity_suffix),
        Points2DWithConfidence(
            positions=uv_positions,
            confidences=uv_confidences,
            class_ids=class_id,
            keypoint_ids=list(keypoint_ids),
            show_labels=False,
        ),
    )



@serde(type_check=coerce)
class QuestHeadPoseDictRow:
    """Raw CSV row containing Quest head pose information for both controllers."""

    ts_ns: int = serde_field(rename="ts_ns")
    """Capture timestamp of the sample, measured in nanoseconds."""

    left_pos_x: float
    """Left tracker X coordinate in meters."""
    left_pos_y: float
    """Left tracker Y coordinate in meters."""
    left_pos_z: float
    """Left tracker Z coordinate in meters."""
    left_quat_x: float
    """Left tracker quaternion X component."""
    left_quat_y: float
    """Left tracker quaternion Y component."""
    left_quat_z: float
    """Left tracker quaternion Z component."""
    left_quat_w: float
    """Left tracker quaternion W component."""

    right_pos_x: float
    """Right tracker X coordinate in meters."""
    right_pos_y: float
    """Right tracker Y coordinate in meters."""
    right_pos_z: float
    """Right tracker Z coordinate in meters."""
    right_quat_x: float
    """Right tracker quaternion X component."""
    right_quat_y: float
    """Right tracker quaternion Y component."""
    right_quat_z: float
    """Right tracker quaternion Z component."""
    right_quat_w: float
    """Right tracker quaternion W component."""


def _quaternion_xyzw_to_rotation_matrix(quaternion_xyzw: Float32[ndarray, "4"]) -> Float32[ndarray, "3 3"]:
    """Convert an (x, y, z, w) quaternion into a world-from-camera rotation matrix."""
    quat_f64: Float64[ndarray, "4"] = quaternion_xyzw.astype(np.float64)
    norm: float = float(np.linalg.norm(quat_f64))
    if norm == 0.0:
        raise ValueError("Encountered zero-norm quaternion while building Quest head pose extrinsics.")
    quat_unit: Float64[ndarray, "4"] = quat_f64 / norm
    x: float = float(quat_unit[0])
    y: float = float(quat_unit[1])
    z: float = float(quat_unit[2])
    w: float = float(quat_unit[3])

    xx: float = x * x
    yy: float = y * y
    zz: float = z * z
    xy: float = x * y
    xz: float = x * z
    yz: float = y * z
    wx: float = w * x
    wy: float = w * y
    wz: float = w * z

    rotation_matrix: Float32[ndarray, "3 3"] = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    return rotation_matrix


def load_head_sequence(head_csv_path: Path) -> list[QuestHeadExtrinsicsSample]:
    """Parse Quest head pose CSV rows into timestamped camera extrinsics."""
    with head_csv_path.open(encoding="utf-8", newline="") as file:
        reader: DictReader[str] = DictReader(file)
        samples: list[QuestHeadExtrinsicsSample] = []
        for row_dict in reader:
            if not row_dict or all(value == "" for value in row_dict.values()):
                continue
            quest_row: QuestHeadPoseDictRow = from_dict(QuestHeadPoseDictRow, row_dict)
            position_m: Float32[ndarray, "3"] = np.array(
                [quest_row.left_pos_x, quest_row.left_pos_y, quest_row.left_pos_z],
                dtype=np.float32,
            )
            rotation_xyzw: Float32[ndarray, "4"] = np.array(
                [quest_row.left_quat_x, quest_row.left_quat_y, quest_row.left_quat_z, quest_row.left_quat_w],
                dtype=np.float32,
            )
            world_R_cam: Float32[ndarray, "3 3"] = _quaternion_xyzw_to_rotation_matrix(rotation_xyzw)
            world_t_cam: Float32[ndarray, "3"] = position_m
            left_extrinsics: Extrinsics = Extrinsics(world_R_cam=world_R_cam, world_t_cam=world_t_cam)

            right_position_m: Float32[ndarray, "3"] = np.array(
                [quest_row.right_pos_x, quest_row.right_pos_y, quest_row.right_pos_z],
                dtype=np.float32,
            )
            right_rotation_xyzw: Float32[ndarray, "4"] = np.array(
                [quest_row.right_quat_x, quest_row.right_quat_y, quest_row.right_quat_z, quest_row.right_quat_w],
                dtype=np.float32,
            )
            right_world_R_cam: Float32[ndarray, "3 3"] = _quaternion_xyzw_to_rotation_matrix(right_rotation_xyzw)
            right_world_t_cam: Float32[ndarray, "3"] = right_position_m
            right_extrinsics: Extrinsics = Extrinsics(world_R_cam=right_world_R_cam, world_t_cam=right_world_t_cam)

            timestamp_ns: int = int(quest_row.ts_ns)
            sample: QuestHeadExtrinsicsSample = QuestHeadExtrinsicsSample(
                timestamp_ns=timestamp_ns,
                left_extrinsics=left_extrinsics,
                right_extrinsics=right_extrinsics,
            )
            samples.append(sample)

    if not samples:
        raise ValueError(f"CSV file {head_csv_path} does not contain any pose rows.")

    return samples


@serde
class QuestLensIntrinsics:
    """Pinhole camera intrinsics exported by the Quest device."""

    focal_length_x: float
    """Focal length along the X axis expressed in pixels."""
    focal_length_y: float
    """Focal length along the Y axis expressed in pixels."""
    principal_point_x: float
    """Principal point horizontal offset in pixels."""
    principal_point_y: float
    """Principal point vertical offset in pixels."""
    skew: int
    """Skew coefficient coupling the X and Y axes."""
    available: bool
    """Flag indicating whether calibrated intrinsics are available."""


@serde(type_check=coerce)
class QuestCaptureResolution:
    """Image resolution used for Quest capture."""

    width: int
    """Capture width in pixels."""
    height: int
    """Capture height in pixels."""


@serde(type_check=coerce)
class QuestCameraIntrinsicsDocument:
    """Quest camera intrinsics JSON document. Keep only important bits."""

    lens_intrinsics: QuestLensIntrinsics
    """Calibrated lens intrinsics expressed in pixel units."""
    capture_resolution: QuestCaptureResolution
    """Image resolution active during capture."""
    positional_layout: str | None = None
    """Optional tag describing whether the camera is layed out on the left or right eye."""


@serde(type_check=coerce)
class QuestCombinedCalibrationDocument:
    """Quest calibration document containing intrinsics for multiple sensors."""

    intrinsics: list[QuestCameraIntrinsicsDocument]
    """Per-camera intrinsics blocks present in the calibration JSON."""


def load_camera_intrinsics(intrinsics_path: Path, *, positional_layout: str | None = None) -> Intrinsics:
    """Load Quest camera intrinsics metadata from a JSON file."""
    if not intrinsics_path.exists():
        raise FileNotFoundError(intrinsics_path)

    with intrinsics_path.open(encoding="utf-8") as file:
        raw_intrinsics: dict[str, object] = json.load(file)

    def _build_intrinsics(doc: QuestCameraIntrinsicsDocument) -> Intrinsics:
        return Intrinsics(
            camera_conventions="RDF",
            fl_x=doc.lens_intrinsics.focal_length_x,
            fl_y=doc.lens_intrinsics.focal_length_y,
            cx=doc.lens_intrinsics.principal_point_x,
            cy=doc.lens_intrinsics.principal_point_y,
            width=doc.capture_resolution.width,
            height=doc.capture_resolution.height,
        )

    if "intrinsics" in raw_intrinsics:
        calibration_doc: QuestCombinedCalibrationDocument = from_dict(QuestCombinedCalibrationDocument, raw_intrinsics)
        if positional_layout is None:
            raise ValueError(
                "Combined calibration JSON detected. Please provide positional_layout (e.g. 'left' or 'right')."
            )

        matching_docs: list[QuestCameraIntrinsicsDocument] = [
            doc for doc in calibration_doc.intrinsics if doc.positional_layout == positional_layout
        ]
        if not matching_docs:
            available_layouts: set[str] = {
                doc.positional_layout for doc in calibration_doc.intrinsics if doc.positional_layout is not None
            }
            raise ValueError(
                f"Could not find intrinsics for layout '{positional_layout}' in {intrinsics_path}. "
                f"Available layouts: {sorted(available_layouts)}"
            )
        return _build_intrinsics(matching_docs[0])

    intrinsics_doc: QuestCameraIntrinsicsDocument = from_dict(QuestCameraIntrinsicsDocument, raw_intrinsics)
    if positional_layout is not None and intrinsics_doc.positional_layout not in (None, positional_layout):
        raise ValueError(
            f"Requested layout '{positional_layout}' mismatches JSON layout '{intrinsics_doc.positional_layout}'."
        )
    return _build_intrinsics(intrinsics_doc)


def _log_head_cameras(
    samples: Sequence[QuestHeadExtrinsicsSample],
    *,
    left_intrinsics: Intrinsics,
    right_intrinsics: Intrinsics,
    left_cam_path: Path,
    right_cam_path: Path,
    timeline: str = "video_time",
) -> list[PinholeParameters]:
    """Log Quest head cameras over time using the provided intrinsics and extrinsics."""

    left_pinhole_list: list[PinholeParameters] = []
    for sample in samples:
        rr.set_time(timeline, duration=np.timedelta64(sample.timestamp_ns, "ns"))

        # Left eye camera
        left_camera_params: PinholeParameters = PinholeParameters(
            name="quest_left_eye",
            extrinsics=sample.left_extrinsics,
            intrinsics=left_intrinsics,
        )
        log_pinhole(
            camera=left_camera_params,
            cam_log_path=left_cam_path,
            static=False,
            image_plane_distance=0.05,
        )
        left_pinhole_list.append(left_camera_params)

        # Right eye camera
        right_camera_params: PinholeParameters = PinholeParameters(
            name="quest_right_eye",
            extrinsics=sample.right_extrinsics,
            intrinsics=right_intrinsics,
        )
        log_pinhole(
            camera=right_camera_params,
            cam_log_path=right_cam_path,
            static=False,
            image_plane_distance=0.05,
        )
    return left_pinhole_list


def load_and_log_quest_data(
    config: Quest3VisualizeConfig, *, timeline: str = "video_time"
) -> tuple[list[Path], list[PinholeParameters], Int64[ndarray, "n_frames"]]:
    """Ingest Quest3+OAK data, log resampled tracks, and return pinhole roots.

    Args:
        config: Filesystem + viewer configuration describing where Quest data
            lives and how to initialize Rerun.
        timeline: Logical timeline used for all emitted logs. Defaults to
            ``"video_time"`` which matches our MP4 timestamps.

    Returns:
        list[Path]: ``/world/ego/quest3_left/right`` pinhole entity roots so
        callers can embed them inside a blueprint.
        list[PinholeParameters]: Logged pinhole parameters for the left head cameras.
        Int64[ndarray, "n_frames"]: Timestamps for each video frame in nanoseconds.
    """
    data_root: Path = config.data_dir
    if not data_root.exists():
        raise FileNotFoundError(data_root)

    rr.log("/", rr.ViewCoordinates.RUB, static=True)

    left_csv: Path = data_root / "quest" / "left_hand_poses.csv"
    right_csv: Path = data_root / "quest" / "right_hand_poses.csv"
    head_csv: Path = data_root / "quest" / "head_pose.csv"
    body_csv: Path = data_root / "quest" / "body_poses.csv"
    calibration_json: Path = data_root / "quest" / "calibration.json"
    left_video_path: Path = data_root / "quest" / "left.mp4"
    right_video_path: Path = data_root / "quest" / "right.mp4"
    if not left_csv.exists():
        raise FileNotFoundError(left_csv)
    if not right_csv.exists():
        raise FileNotFoundError(right_csv)
    if not head_csv.exists():
        raise FileNotFoundError(head_csv)
    if not calibration_json.exists():
        raise FileNotFoundError(calibration_json)
    if not body_csv.exists():
        raise FileNotFoundError(body_csv)
    if not left_video_path.exists():
        raise FileNotFoundError(left_video_path)
    if not right_video_path.exists():
        raise FileNotFoundError(right_video_path)

    head_extrinsics: list[QuestHeadExtrinsicsSample] = load_head_sequence(head_csv)
    body_sequence_raw: QuestBodyPoseSequence = load_body_sequence(body_csv)
    left_intrinsics: Intrinsics = load_camera_intrinsics(calibration_json, positional_layout="left")
    right_intrinsics: Intrinsics = load_camera_intrinsics(calibration_json, positional_layout="right")

    quest_left_cam_path: Path = Path("/world/ego/quest3_left")
    quest_right_cam_path: Path = Path("/world/ego/quest3_right")

    _log_annotation_context()

    _left_video_timestamps_ns: Int[ndarray, "num_frames"] = log_video(
        video_source=left_video_path,
        video_log_path=quest_left_cam_path / "pinhole" / "video",
        timeline=timeline,
    )
    _right_video_timestamps_ns: Int[ndarray, "num_frames"] = log_video(
        video_source=right_video_path,
        video_log_path=quest_right_cam_path / "pinhole" / "video",
        timeline=timeline,
    )

    quest_video_timestamps_ns: Int64[ndarray, "n_frames"] = _select_common_video_timestamps(
        left_timestamps=_left_video_timestamps_ns,
        right_timestamps=_right_video_timestamps_ns,
    )

    # TODO(#exoego-timelines): revisit once we support multi-rate logging instead of
    # clamping everything to video_time. For now we intentionally drop the Quest
    # tracker samples to guarantee one keypoint/extrinsic per video frame.
    resampled_head_extrinsics: list[QuestHeadExtrinsicsSample] = _resample_head_extrinsics(
        samples=head_extrinsics,
        target_timestamps_ns=quest_video_timestamps_ns,
    )

    body_sequence: QuestBodyPoseSequence = _resample_body_sequence(
        sequence=body_sequence_raw,
        target_timestamps_ns=quest_video_timestamps_ns,
    )

    sequence_map: dict[QuestHandSide, QuestHandPoseSequence] = {
        side: load_hand_sequence(csv_path)
        for side, csv_path in (
            (QuestHandSide.LEFT, left_csv),
            (QuestHandSide.RIGHT, right_csv),
        )
    }
    if QuestHandSide.LEFT not in sequence_map or QuestHandSide.RIGHT not in sequence_map:
        raise ValueError("Both left and right hand pose sequences are required for COCO-133 logging.")

    sequence_map[QuestHandSide.LEFT] = _resample_hand_sequence(
        sequence=sequence_map[QuestHandSide.LEFT],
        target_timestamps_ns=quest_video_timestamps_ns,
    )
    sequence_map[QuestHandSide.RIGHT] = _resample_hand_sequence(
        sequence=sequence_map[QuestHandSide.RIGHT],
        target_timestamps_ns=quest_video_timestamps_ns,
    )

    left_pinhole_list: list[PinholeParameters] = _log_head_cameras(
        resampled_head_extrinsics,
        left_intrinsics=left_intrinsics,
        right_intrinsics=right_intrinsics,
        left_cam_path=quest_left_cam_path,
        right_cam_path=quest_right_cam_path,
        timeline=timeline,
    )

    _log_coco133_annotations(
        left_sequence=sequence_map[QuestHandSide.LEFT],
        right_sequence=sequence_map[QuestHandSide.RIGHT],
        body_sequence=body_sequence,
        head_extrinsics=resampled_head_extrinsics,
        left_intrinsics=left_intrinsics,
        right_intrinsics=right_intrinsics,
        quest_left_cam_path=quest_left_cam_path,
        quest_right_cam_path=quest_right_cam_path,
        timeline=timeline,
    )

    quest_pinhole_paths: list[Path] = [
        quest_left_cam_path / "pinhole",
        quest_right_cam_path / "pinhole",
    ]
    return quest_pinhole_paths, left_pinhole_list, quest_video_timestamps_ns


def main(config: Quest3VisualizeConfig) -> None:
    load_and_log_quest_data(config)
