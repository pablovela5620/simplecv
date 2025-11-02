import json
from collections.abc import Iterator, Sequence
from csv import DictReader
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
import rerun as rr
from jaxtyping import Bool, Float32, Int64, UInt16
from numpy import ndarray
from rerun import AnnotationInfo, ClassDescription
from serde import coerce, from_dict, serde
from serde import field as serde_field

from simplecv.camera_parameters import Intrinsics
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.umetrack_temp.generic_hand_model import LANDMARK, UME_HAND_CONNECTIONS


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

_CSV_AXES: tuple[str, ...] = ("x", "y", "z")


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

QUEST_HAND_CSV_COLUMNS: tuple[str, ...] = ("timestamp",) + tuple(
    f"{prefix}_{axis}" for prefix in _QUEST_HAND_LANDMARK_PREFIXES for axis in _CSV_AXES
)

QUEST_HEAD_CSV_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "left_pos_x",
    "left_pos_y",
    "left_pos_z",
    "left_quat_x",
    "left_quat_y",
    "left_quat_z",
    "left_quat_w",
    "right_pos_x",
    "right_pos_y",
    "right_pos_z",
    "right_quat_x",
    "right_quat_y",
    "right_quat_z",
    "right_quat_w",
)


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
        fieldnames: Sequence[str] | None = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"CSV file {csv_path} is missing a header row.")
        normalized_header = tuple(field.strip() for field in fieldnames)
        if normalized_header != QUEST_HAND_CSV_COLUMNS:
            msg: str = (
                f"Unexpected CSV header in {csv_path}. Expected {QUEST_HAND_CSV_COLUMNS} but found {normalized_header}."
            )
            raise ValueError(msg)

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


def log_hand_sequence(
    sequence: QuestHandPoseSequence, *, side: QuestHandSide, log_path: str, timeline: str = "quest_time"
) -> None:
    """Log a Quest hand sequence into the active Rerun recording."""
    entity_path = f"{log_path}/{side.entity_suffix}"
    class_ids: UInt16[ndarray, "n_ume_kpts=21"] = QUEST_HAND_CLASS_IDS_BY_SIDE[side]
    for sample in sequence:
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


@dataclass
class QuestHeadPoseSample:
    """Quest head pose sample containing left-eye pose represented as position and quaternion."""

    timestamp_ns: int
    """Relative timestamp, in nanoseconds from the recording start."""
    position_m: Float32[ndarray, "3"]
    """Left-eye position expressed in meters within the Quest coordinate frame."""
    rotation_xyzw: Float32[ndarray, "4"]
    """Left-eye orientation as a quaternion (x, y, z, w)."""


def load_head_sequence(head_csv_path: Path) -> list[QuestHeadPoseSample]:
    """Parse Quest head pose CSV rows into structured records."""
    with head_csv_path.open(encoding="utf-8", newline="") as file:
        reader: DictReader[str] = DictReader(file)
        fieldnames: Sequence[str] | None = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"CSV file {head_csv_path} is missing a header row.")
        normalized_header = tuple(field.strip() for field in fieldnames)
        if normalized_header != QUEST_HEAD_CSV_COLUMNS:
            msg: str = f"Unexpected CSV header in {head_csv_path}. Expected {QUEST_HEAD_CSV_COLUMNS} but found {normalized_header}."
            raise ValueError(msg)

        samples: list[QuestHeadPoseSample] = []
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
            sample = QuestHeadPoseSample(
                timestamp_ns=int(np.floor(quest_row.ts_seconds * 1_000_000_000.0)),
                position_m=position_m,
                rotation_xyzw=rotation_xyzw,
            )
            samples.append(sample)

    if not samples:
        raise ValueError(f"CSV file {head_csv_path} does not contain any pose rows.")

    return samples


def load_camera_intrinsics(intrinsics_path: Path) -> Intrinsics:
    """Load Quest camera intrinsics metadata from a JSON file."""
    if not intrinsics_path.exists():
        raise FileNotFoundError(intrinsics_path)

    with intrinsics_path.open(encoding="utf-8") as file:
        raw_intrinsics: dict[str, object] = json.load(file)

    lens_intrinsics_section: dict[str, object] = raw_intrinsics.get("lens_intrinsics", {})
    capture_resolution_section: dict[str, object] = raw_intrinsics.get("capture_resolution", {})

    focal_length_x: float = float(lens_intrinsics_section.get("focal_length_x", 0.0))
    focal_length_y: float = float(lens_intrinsics_section.get("focal_length_y", 0.0))
    principal_point_x: float = float(lens_intrinsics_section.get("principal_point_x", 0.0))
    principal_point_y: float = float(lens_intrinsics_section.get("principal_point_y", 0.0))

    width_value: float | int | None = capture_resolution_section.get("width")
    height_value: float | int | None = capture_resolution_section.get("height")
    image_width: int | None = int(width_value) if width_value is not None else None
    image_height: int | None = int(height_value) if height_value is not None else None

    quest_intrinsics = Intrinsics(
        camera_conventions="RDF",
        fl_x=focal_length_x,
        fl_y=focal_length_y,
        cx=principal_point_x,
        cy=principal_point_y,
        width=image_width,
        height=image_height,
    )
    return quest_intrinsics


def main(config: Quest3OakDVisualizeConfig) -> None:
    data_root: Path = config.data_dir
    if not data_root.exists():
        raise FileNotFoundError(data_root)

    left_csv: Path = data_root / "quest" / "left_hand_poses.csv"
    right_csv: Path = data_root / "quest" / "right_hand_poses.csv"
    head_csv: Path = data_root / "quest" / "head_pose.csv"
    intrinsics_left_json: Path = data_root / "quest" / "intrinsics_left.json"
    if not left_csv.exists():
        raise FileNotFoundError(left_csv)
    if not right_csv.exists():
        raise FileNotFoundError(right_csv)
    if not head_csv.exists():
        raise FileNotFoundError(head_csv)
    if not intrinsics_left_json.exists():
        raise FileNotFoundError(intrinsics_left_json)

    head_samples: list[QuestHeadPoseSample] = load_head_sequence(head_csv)
    _left_intrinsics: Intrinsics = load_camera_intrinsics(intrinsics_left_json)

    sequences: list[tuple[QuestHandSide, QuestHandPoseSequence]] = [
        (QuestHandSide.LEFT, load_hand_sequence(left_csv)),
        (QuestHandSide.RIGHT, load_hand_sequence(right_csv)),
    ]

    log_path = "quest3_oakd"
    _log_annotation_context(log_path, sides=[side for side, _ in sequences])
    for side, sequence in sequences:
        log_hand_sequence(sequence, side=side, log_path=log_path)
