import json
from collections.abc import Iterator, Sequence
from csv import DictReader
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
import rerun as rr
from jaxtyping import Bool, Float32, Float64, Int64, UInt16
from numpy import ndarray
from rerun import AnnotationInfo, ClassDescription
from serde import coerce, from_dict, serde
from serde import field as serde_field

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.ops import conventions
from simplecv.ops.triangulate import proj_3d_vectorized
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole, log_video
from simplecv.umetrack_temp.generic_hand_model_numpy import LANDMARK, UME_HAND_CONNECTIONS


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

UME_HAND_KEYPOINT_IDS: UInt16[ndarray, "n_ume_kpts"] = np.array(
    [landmark.value for landmark in LANDMARK],
    dtype=np.uint16,
)

QUEST_HAND_CLASS_IDS_BY_SIDE: dict[QuestHandSide, UInt16[ndarray, "n_ume_kpts"]] = {
    side: np.full(UME_HAND_LANDMARK_COUNT, int(side), dtype=np.uint16) for side in QuestHandSide
}


@serde(type_check=coerce)
class QuestHandDictRow:
    """Raw CSV row for Quest 3 + Oak-D hand poses."""

    ts_seconds: float = serde_field(rename="timestamp")
    """Original timestamp in seconds."""

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
class Quest3OakDVisualizeConfig:
    """Structured configuration for running the Quest3/Oak-D visualization CLI."""

    rr_config: RerunTyroConfig
    """Command-line options for spawning and configuring the Rerun viewer."""
    data_dir: Path
    """Path to the root directory of the Quest3 Oak-D dataset."""


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
            timestamp_ns: int = int(np.floor(sample.ts_seconds * 1_000_000_000.0))

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


def _log_annotation_context(log_path: str, *, sides: Sequence[QuestHandSide]) -> None:
    rr.log(log_path, rr.ViewCoordinates.RUB, static=True)
    class_descriptions = [
        ClassDescription(
            info=AnnotationInfo(label=f"Quest3 {side.label} hand", id=int(side)),
            keypoint_annotations=[AnnotationInfo(id=landmark.value) for landmark in LANDMARK],
            keypoint_connections=list(UME_HAND_CONNECTIONS),
        )
        for side in sides
    ]
    rr.log(log_path, rr.AnnotationContext(class_descriptions), static=True)


@dataclass(slots=True)
class QuestHeadExtrinsicsSample:
    """Quest head pose extrinsics accompanied by capture timestamp."""

    timestamp_ns: int
    """Relative timestamp, measured in nanoseconds from recording start."""
    left_extrinsics: Extrinsics
    """Camera-to-world pose describing the left-eye tracking camera."""
    right_extrinsics: Extrinsics
    """Camera-to-world pose describing the right-eye tracking camera."""


def log_hand_sequence(
    sequence: QuestHandPoseSequence,
    head_extrinsics: list[QuestHeadExtrinsicsSample],
    left_intrinsics: Intrinsics,
    right_intrinsics: Intrinsics,
    *,
    side: QuestHandSide,
    log_path: str,
    timeline: str = "quest_time",
) -> None:
    """Log a Quest hand sequence into the active Rerun recording."""
    entity_path: str = f"{log_path}/{side.entity_suffix}"
    class_ids: UInt16[ndarray, "n_ume_kpts=21"] = QUEST_HAND_CLASS_IDS_BY_SIDE[side]
    head_extrinsic: QuestHeadExtrinsicsSample
    for sample, head_extrinsic in zip(sequence, head_extrinsics, strict=False):
        rr.set_time(timeline, duration=np.timedelta64(sample.timestamp_ns, "ns"))
        mapped_keypoints: Float32[ndarray, "n_ume_kpts=21 3"] = sample.keypoints_m[LANDMARK_TO_QUEST_INDEX]
        rr.log(
            f"{entity_path}/landmarks",
            rr.Points3D(
                mapped_keypoints,
                keypoint_ids=UME_HAND_KEYPOINT_IDS,
                class_ids=class_ids,
                show_labels=False,
            ),
        )

        left_pinhole: PinholeParameters = PinholeParameters(
            name="quest_left_eye",
            extrinsics=head_extrinsic.left_extrinsics,
            intrinsics=left_intrinsics,
        )
        right_pinhole: PinholeParameters = PinholeParameters(
            name="quest_right_eye",
            extrinsics=head_extrinsic.right_extrinsics,
            intrinsics=right_intrinsics,
        )
        # project into each eye camera
        for camera_name, pinhole_param in [("left", left_pinhole), ("right", right_pinhole)]:
            xyz_hom_stack: Float32[ndarray, "n_frames=1 n_ume_kpts=21 4"] = np.concatenate(
                [mapped_keypoints, np.ones_like(mapped_keypoints[..., :1])], axis=-1
            )[np.newaxis, ...]
            Pall_exo: Float64[ndarray, "n_frames=1 3 4"] = pinhole_param.projection_matrix[np.newaxis, ...]
            uv_raw_stack: Float64[ndarray, "n_frames=1 n_views n_ume_kpts=21 2"] = proj_3d_vectorized(
                xyz_hom=xyz_hom_stack, P=Pall_exo
            )
            uv_frame: Float32[ndarray, "n_ume_kpts=21 2"] = uv_raw_stack[0, 0].astype(np.float32)

            xyz_world_hom: Float32[ndarray, "n_ume_kpts=21 4"] = xyz_hom_stack[0]
            world_T_cam: Float32[ndarray, "4 4"] = pinhole_param.extrinsics.world_T_cam.astype(np.float32)
            xyz_cam: Float32[ndarray, "n_ume_kpts=21 4"] = (world_T_cam @ xyz_world_hom.T).T
            depth_cam: Float32[ndarray, "n_ume_kpts=21"] = xyz_cam[:, 2]
            depth_mask: Bool[ndarray, "n_ume_kpts=21"] = depth_cam > 0.0

            intrinsics = pinhole_param.intrinsics
            width: float = float(intrinsics.width if intrinsics.width is not None else 2.0 * intrinsics.cx)
            height: float = float(intrinsics.height if intrinsics.height is not None else 2.0 * intrinsics.cy)
            bounds_mask: Bool[ndarray, "n_ume_kpts=21"] = (
                (uv_frame[:, 0] >= 0.0)
                & (uv_frame[:, 0] <= width)
                & (uv_frame[:, 1] >= 0.0)
                & (uv_frame[:, 1] <= height)
            )

            valid_mask: Bool[ndarray, "n_ume_kpts=21"] = depth_mask & bounds_mask
            uv_frame = np.where(valid_mask[:, None], uv_frame, np.nan)

            rr.log(
                f"/world/ego/quest3/{camera_name}/pinhole/video/uv_{side.label}",
                rr.Points2D(
                    uv_frame,
                    keypoint_ids=UME_HAND_KEYPOINT_IDS,
                    class_ids=class_ids,
                    show_labels=False,
                ),
            )


@serde(type_check=coerce)
class QuestHeadPoseDictRow:
    """Raw CSV row containing Quest head pose information for both controllers."""

    ts_seconds: float = serde_field(rename="timestamp")
    """Original timestamp of the sample, measured in seconds."""

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
            # Convert from OpenGL (RUB) to OpenCV (RDF) convention
            world_T_cam_gl: Float32[np.ndarray, "4 4"] = left_extrinsics.world_T_cam.astype(np.float32)
            world_T_cam_cv: Float32[np.ndarray, "4 4"] = conventions.convert_pose(
                world_T_cam_gl,
                src_convention=conventions.CC.GL,
                dst_convention=conventions.CC.CV,
            )
            left_translation_cv: Float32[ndarray, "3"] = world_T_cam_cv[:3, 3].astype(np.float32)
            left_extrinsics: Extrinsics = Extrinsics(
                world_R_cam=world_T_cam_cv[:3, :3],
                world_t_cam=left_translation_cv,
            )

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

            # Convert from OpenGL (RUB) to OpenCV (RDF) convention
            right_world_T_cam_gl: Float32[np.ndarray, "4 4"] = right_extrinsics.world_T_cam.astype(np.float32)
            world_T_cam_cv: Float32[np.ndarray, "4 4"] = conventions.convert_pose(
                right_world_T_cam_gl,
                src_convention=conventions.CC.GL,
                dst_convention=conventions.CC.CV,
            )
            right_translation_cv: Float32[ndarray, "3"] = world_T_cam_cv[:3, 3].astype(np.float32)
            right_extrinsics: Extrinsics = Extrinsics(
                world_R_cam=world_T_cam_cv[:3, :3],
                world_t_cam=right_translation_cv,
            )

            timestamp_ns: int = int(np.floor(quest_row.ts_seconds * 1_000_000_000.0))
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
    """Quest camera intrinsics JSON document. Keep only important bits"""

    lens_intrinsics: QuestLensIntrinsics
    """Calibrated lens intrinsics expressed in pixel units."""
    capture_resolution: QuestCaptureResolution
    """Image resolution active during capture."""


def load_camera_intrinsics(intrinsics_path: Path) -> Intrinsics:
    """Load Quest camera intrinsics metadata from a JSON file."""
    if not intrinsics_path.exists():
        raise FileNotFoundError(intrinsics_path)

    with intrinsics_path.open(encoding="utf-8") as file:
        raw_intrinsics: dict[str, object] = json.load(file)

    intrinsics_doc: QuestCameraIntrinsicsDocument = from_dict(QuestCameraIntrinsicsDocument, raw_intrinsics)
    intrinsics: Intrinsics = Intrinsics(
        camera_conventions="RDF",
        fl_x=intrinsics_doc.lens_intrinsics.focal_length_x,
        fl_y=intrinsics_doc.lens_intrinsics.focal_length_y,
        cx=intrinsics_doc.lens_intrinsics.principal_point_x,
        cy=intrinsics_doc.lens_intrinsics.principal_point_y,
        width=intrinsics_doc.capture_resolution.width,
        height=intrinsics_doc.capture_resolution.height,
    )
    return intrinsics


def _log_head_cameras(
    samples: Sequence[QuestHeadExtrinsicsSample],
    *,
    left_intrinsics: Intrinsics,
    right_intrinsics: Intrinsics,
    log_path: str,
    timeline: str = "quest_time",
) -> None:
    """Log Quest head cameras over time using the provided intrinsics and extrinsics."""

    left_cam_path: Path = Path(f"{log_path}/left")
    right_cam_path: Path = Path(f"{log_path}/right")

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


def main(config: Quest3OakDVisualizeConfig) -> None:
    data_root: Path = config.data_dir
    if not data_root.exists():
        raise FileNotFoundError(data_root)

    left_csv: Path = data_root / "quest" / "left_hand_poses.csv"
    right_csv: Path = data_root / "quest" / "right_hand_poses.csv"
    head_csv: Path = data_root / "quest" / "head_pose.csv"
    intrinsics_left_json: Path = data_root / "quest" / "intrinsics_left.json"
    intrinsics_right_json: Path = data_root / "quest" / "intrinsics_right.json"
    left_video_path: Path = data_root / "quest" / "left.mp4"
    right_video_path: Path = data_root / "quest" / "right.mp4"
    if not left_csv.exists():
        raise FileNotFoundError(left_csv)
    if not right_csv.exists():
        raise FileNotFoundError(right_csv)
    if not head_csv.exists():
        raise FileNotFoundError(head_csv)
    if not intrinsics_left_json.exists():
        raise FileNotFoundError(intrinsics_left_json)
    if not intrinsics_right_json.exists():
        raise FileNotFoundError(intrinsics_right_json)
    if not left_video_path.exists():
        raise FileNotFoundError(left_video_path)
    if not right_video_path.exists():
        raise FileNotFoundError(right_video_path)

    head_extrinsics: list[QuestHeadExtrinsicsSample] = load_head_sequence(head_csv)
    left_intrinsics: Intrinsics = load_camera_intrinsics(intrinsics_left_json)
    right_intrinsics: Intrinsics = load_camera_intrinsics(intrinsics_right_json)

    _log_head_cameras(
        head_extrinsics,
        left_intrinsics=left_intrinsics,
        right_intrinsics=right_intrinsics,
        log_path="/world/ego/quest3",
    )

    _log_annotation_context(
        "/world/ego/quest3",
        sides=[QuestHandSide.LEFT, QuestHandSide.RIGHT],
    )

    _left_video_timestamps_ns = log_video(
        video_path=left_video_path,
        video_log_path=Path("/world/ego/quest3/left/pinhole/video"),
        timeline="quest_time",
    )
    _right_video_timestamps_ns = log_video(
        video_path=right_video_path,
        video_log_path=Path("/world/ego/quest3/right/pinhole/video"),
        timeline="quest_time",
    )

    sequences: list[tuple[QuestHandSide, QuestHandPoseSequence]] = [
        (QuestHandSide.LEFT, load_hand_sequence(left_csv)),
        (QuestHandSide.RIGHT, load_hand_sequence(right_csv)),
    ]

    log_path = "quest3_oakd"
    _log_annotation_context("/", sides=[side for side, _ in sequences])
    for side, sequence in sequences:
        log_hand_sequence(
            sequence,
            head_extrinsics,
            left_intrinsics=left_intrinsics,
            right_intrinsics=right_intrinsics,
            side=side,
            log_path=log_path,
        )
