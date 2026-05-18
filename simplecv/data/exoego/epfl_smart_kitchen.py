from __future__ import annotations

import ast
import csv
import json
import warnings
from collections.abc import Generator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from jaxtyping import Float32, Int
from natsort import natsorted
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates

from simplecv.camera_parameters import BrownConradyDistortion, Extrinsics, Intrinsics, PinholeParameters
from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence, ManoStack
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels, ExoEgoSample
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig
from simplecv.data.exoego.sequence_identity import SequenceIdentity
from simplecv.data.skeleton.coco_133 import LEFT_HAND_IDX, RIGHT_HAND_IDX

EpflExoCameraName = Literal[
    "output0",
    "Aoutput0",
    "Aoutput1",
    "Aoutput2",
    "Aoutput3",
    "Boutput0",
    "Boutput1",
    "Boutput2",
    "Boutput3",
]

EPFL_EXO_CAMERA_NAMES: tuple[EpflExoCameraName, ...] = (
    "output0",
    "Aoutput0",
    "Aoutput1",
    "Aoutput2",
    "Aoutput3",
    "Boutput0",
    "Boutput1",
    "Boutput2",
    "Boutput3",
)
EPFL_EGO_CAMERA_NAME: str = "hololens"


@dataclass
class EpflSmartKitchenConfig(BaseExoEgoDatasetConfig):
    """EPFL-Smart-Kitchen dataset config."""

    _target: type = field(default_factory=lambda: EpflSmartKitchenSequence)
    """Target class to instantiate."""
    root_directory: Path = Path("/mnt/8tb/data/epfl-smart-kitchen-av1")
    """Directory containing the AV1-transcoded EPFL public release layout."""
    split: Literal["train", "test"] = "train"
    """Dataset split containing the selected session."""
    participant_id: str = "YH2002"
    """EPFL participant identifier, preserved exactly from the public release."""
    session_name: str = "2023_12_04_10_15_23"
    """EPFL session directory name, preserved exactly from the public release."""
    exo_camera_names: tuple[EpflExoCameraName, ...] = EPFL_EXO_CAMERA_NAMES
    """Exocentric RGB camera names to load."""


def video_session_dir(cfg: EpflSmartKitchenConfig) -> Path:
    """Return the EPFL public-video session directory for ``cfg``."""
    return cfg.root_directory / "Public_release_videos" / cfg.split / cfg.participant_id / cfg.session_name


def pose_session_dir(cfg: EpflSmartKitchenConfig) -> Path:
    """Return the EPFL public-pose session directory for ``cfg``."""
    return cfg.root_directory / "Public_release_pose" / cfg.split / cfg.participant_id / cfg.session_name


def body_pose_path(cfg: EpflSmartKitchenConfig) -> Path:
    """Return the EPFL SMPL-derived body keypoint CSV path."""
    return pose_session_dir(cfg) / "pose_3d" / "pose3d_smpl.csv"


def hand_pose_path(cfg: EpflSmartKitchenConfig) -> Path:
    """Return the EPFL MANO-derived hand keypoint CSV path."""
    return pose_session_dir(cfg) / "pose_3d" / "pose3d_mano.csv"


def camera_matrix_path(cfg: EpflSmartKitchenConfig) -> Path:
    """Return the per-session camera calibration JSON path."""
    return video_session_dir(cfg) / "meta_data" / "camera_matrix.json"


def holo_pose_path(cfg: EpflSmartKitchenConfig) -> Path:
    """Return the per-session HoloLens metadata CSV path."""
    return video_session_dir(cfg) / "meta_data" / "holo_data_wpose.csv"


def rgb_timestamps_path(cfg: EpflSmartKitchenConfig) -> Path:
    """Return the per-session RGB timestamp metadata path."""
    return video_session_dir(cfg) / "meta_data" / "timestamps.txt"


def video_path_for_camera(cfg: EpflSmartKitchenConfig, camera_name: str) -> Path:
    """Return the RGB MP4 path for an EPFL camera."""
    video_path: Path = video_session_dir(cfg) / "videos" / f"{camera_name}.mp4"
    return video_path


def load_camera_matrix(cfg: EpflSmartKitchenConfig) -> dict[str, dict]:
    """Load EPFL ``camera_matrix.json``."""
    path: Path = camera_matrix_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"EPFL camera calibration not found: {path}")
    data: dict[str, dict] = json.loads(path.read_text())
    return data


def _matrix4(value: object, *, field_name: str) -> ndarray:
    matrix: ndarray = np.asarray(value, dtype=np.float32)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.size == 16:
        return matrix.reshape(4, 4)
    raise ValueError(f"{field_name} must be a 4x4 transform, got shape {matrix.shape}")


def _matrix3(value: object, *, field_name: str) -> ndarray:
    matrix: ndarray = np.asarray(value, dtype=np.float32)
    if matrix.shape == (3, 3):
        return matrix
    if matrix.size == 9:
        return matrix.reshape(3, 3)
    raise ValueError(f"{field_name} must be a 3x3 matrix, got shape {matrix.shape}")


def _video_size(video_path: Path) -> tuple[int, int]:
    capture: cv2.VideoCapture = cv2.VideoCapture(str(video_path))
    try:
        width: int = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height: int = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Could not read video dimensions from {video_path}")
    return width, height


def _camera_size(entry: dict, video_path: Path) -> tuple[int, int]:
    width_raw: object | None = entry.get("width")
    height_raw: object | None = entry.get("height")
    if width_raw is not None and height_raw is not None:
        return int(width_raw), int(height_raw)
    return _video_size(video_path)


def _distortion_from_entry(camera_name: str, entry: dict) -> BrownConradyDistortion | None:
    coeffs_raw: object | None = entry.get("dist")
    if coeffs_raw is None:
        return None
    coeffs: list[float] = [float(value) for value in np.asarray(coeffs_raw, dtype=np.float32).reshape(-1)]
    if len(coeffs) == 0:
        return None
    if len(coeffs) not in {4, 5, 8, 12, 14}:
        warnings.warn(
            f"Unsupported EPFL distortion coefficient count for {camera_name}: {len(coeffs)}; using pinhole fallback.",
            stacklevel=2,
        )
        return None
    padded: list[float] = coeffs + [0.0] * (14 - len(coeffs))
    return BrownConradyDistortion(
        k1=padded[0],
        k2=padded[1],
        p1=padded[2],
        p2=padded[3],
        k3=padded[4],
        k4=padded[5],
        k5=padded[6],
        k6=padded[7],
        s1=padded[8],
        s2=padded[9],
        s3=padded[10],
        s4=padded[11],
        tau_x=padded[12],
        tau_y=padded[13],
    )


def _pinhole_from_entry(
    *,
    camera_name: str,
    entry: dict,
    video_path: Path,
    cam_T_world: ndarray,
) -> PinholeParameters:
    k_matrix: ndarray = _matrix3(entry["K"], field_name=f"{camera_name}.K")
    width, height = _camera_size(entry, video_path)
    intrinsics: Intrinsics = Intrinsics.from_k_matrix(
        camera_conventions="RDF",
        k_matrix=k_matrix,
        width=width,
        height=height,
    )
    extrinsics: Extrinsics = Extrinsics(
        cam_R_world=cam_T_world[:3, :3],
        cam_t_world=cam_T_world[:3, 3],
    )
    return PinholeParameters(
        name=camera_name,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        distortion=_distortion_from_entry(camera_name, entry),
    )


def load_hololens_world_to_camera_poses(cfg: EpflSmartKitchenConfig) -> list[ndarray]:
    """Load per-frame HoloLens world-to-camera transforms from ``holo_data_wpose.csv``."""
    path: Path = holo_pose_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"EPFL HoloLens pose CSV not found: {path}")

    transforms_raw: list[Float32[ndarray, "4 4"] | None] = []
    missing_count: int = 0
    with path.open(newline="") as file:
        reader: csv.DictReader = csv.DictReader(file)
        if reader.fieldnames is None or "world2holo" not in reader.fieldnames:
            raise ValueError(f"EPFL HoloLens pose CSV must contain a world2holo column: {path}")
        for row in reader:
            text: str = row["world2holo"].strip()
            value: object = json.loads(text)
            if value == []:
                transforms_raw.append(None)
                missing_count += 1
                continue
            transforms_raw.append(_matrix4(value, field_name="world2holo"))
    if not transforms_raw:
        raise ValueError(f"EPFL HoloLens pose CSV contains no poses: {path}")

    first_valid: Float32[ndarray, "4 4"] | None = next(
        (transform for transform in transforms_raw if transform is not None),
        None,
    )
    if first_valid is None:
        raise ValueError(f"EPFL HoloLens pose CSV contains no valid world2holo transforms: {path}")
    if missing_count > 0:
        warnings.warn(
            f"EPFL HoloLens pose CSV has missing {missing_count} HoloLens world2holo transforms; "
            "carrying the previous valid pose when available and the first valid pose for leading gaps.",
            stacklevel=2,
        )

    transforms: list[ndarray] = []
    last_valid: Float32[ndarray, "4 4"] | None = None
    for transform in transforms_raw:
        if transform is None:
            fallback: Float32[ndarray, "4 4"] = last_valid if last_valid is not None else first_valid
            transforms.append(fallback.copy())
        else:
            transforms.append(transform)
            last_valid = transform
    return transforms


def load_rgb_timestamps_ns(cfg: EpflSmartKitchenConfig) -> Int[ndarray, "n_frames"]:
    """Load EPFL RGB frame timestamps from microseconds and normalize to nanoseconds."""
    path: Path = rgb_timestamps_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"EPFL RGB timestamp metadata not found: {path}")
    timestamps_us: Int[ndarray, "n_frames"] = np.loadtxt(path, dtype=np.int64, ndmin=1)
    if timestamps_us.size == 0:
        raise ValueError(f"EPFL RGB timestamp metadata contains no rows: {path}")
    timestamps_us = timestamps_us.reshape(-1)
    timestamps_ns: Int[ndarray, "n_frames"] = ((timestamps_us - timestamps_us[0]) * np.int64(1000)).astype(np.int64)
    return timestamps_ns


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"EPFL pose CSV not found: {path}")
    with path.open(newline="") as file:
        reader: csv.DictReader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"EPFL pose CSV is missing a header row: {path}")
        rows: list[dict[str, str]] = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"EPFL pose CSV contains no rows: {path}")
    return rows


class _NonFiniteLiteralTransformer(ast.NodeTransformer):
    """Allow EPFL CSV cells to contain bare NaN/Infinity tokens."""

    _CONSTANTS: dict[str, float] = {
        "NaN": float("nan"),
        "nan": float("nan"),
        "Infinity": float("inf"),
        "inf": float("inf"),
    }

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id not in self._CONSTANTS:
            raise ValueError(f"Unsupported numeric token in EPFL CSV cell: {node.id}")
        return ast.copy_location(ast.Constant(value=self._CONSTANTS[node.id]), node)


def _parse_numeric_cell(value: str) -> ndarray:
    text: str = value.strip()
    expression: ast.Expression = ast.parse(text, mode="eval")
    transformed_expression: ast.Expression = _NonFiniteLiteralTransformer().visit(expression)
    ast.fix_missing_locations(transformed_expression)
    parsed: object = ast.literal_eval(transformed_expression)
    array: ndarray = np.asarray(parsed, dtype=np.float32)
    return array


def _parse_vector_cell(value: str, *, expected_len: int | None = None, field_name: str) -> ndarray:
    vector: ndarray = _parse_numeric_cell(value).reshape(-1).astype(np.float32, copy=False)
    if expected_len is not None and vector.shape[0] != expected_len:
        raise ValueError(f"{field_name} must contain {expected_len} values, got {vector.shape[0]}")
    return vector


def _parse_keypoints(rows: list[dict[str, str]], *, num_keypoints: int, path: Path) -> ndarray:
    keypoints: list[ndarray] = []
    for row_idx, row in enumerate(rows):
        if "kp3ds" not in row:
            raise KeyError(f"EPFL pose CSV is missing kp3ds column: {path}")
        frame_keypoints: ndarray = _parse_numeric_cell(row["kp3ds"]).astype(np.float32, copy=False)
        if frame_keypoints.shape == (1, num_keypoints, 3):
            frame_keypoints = frame_keypoints[0]
        elif frame_keypoints.shape != (num_keypoints, 3):
            if frame_keypoints.size != num_keypoints * 3:
                raise ValueError(f"EPFL kp3ds row {row_idx} in {path} cannot reshape to ({num_keypoints}, 3); got {frame_keypoints.shape}")
            frame_keypoints = frame_keypoints.reshape(num_keypoints, 3)
        keypoints.append(frame_keypoints)
    return np.stack(keypoints).astype(np.float32, copy=False)


def _parse_scalar(row: dict[str, str], key: str, *, default: float | None = None) -> float:
    value: str | None = row.get(key)
    if value is None or value == "":
        if default is None:
            raise KeyError(f"Missing scalar column {key}")
        return default
    return float(value)


def _parse_confidences(
    rows: list[dict[str, str]],
    *,
    num_keypoints: int,
    mode: Literal["body", "hand"],
    finite_mask: ndarray,
) -> ndarray:
    if "kp3ds_conf" not in rows[0]:
        return finite_mask.astype(np.float32)

    conf_list: list[ndarray] = [_parse_vector_cell(row["kp3ds_conf"], expected_len=num_keypoints, field_name="kp3ds_conf") for row in rows]
    conf: ndarray = np.stack(conf_list).astype(np.float32, copy=False)
    if mode == "body" and "l2_dist" in rows[0]:
        empty_body_count: int = sum(1 for row in rows if row.get("l2_dist") in {None, ""})
        if empty_body_count:
            warnings.warn(
                f"Found {empty_body_count} empty EPFL body l2 distance values; treating them as accepted confidence.",
                stacklevel=2,
            )
        body_l2: ndarray = np.array(
            [_parse_scalar(row, "l2_dist", default=0.0) for row in rows],
            dtype=np.float32,
        )
        conf = conf * (body_l2 < np.float32(0.09))[:, np.newaxis]
    if mode == "hand" and "l2_dist_left" in rows[0] and "l2_dist_right" in rows[0]:
        empty_left_count: int = sum(1 for row in rows if row.get("l2_dist_left") in {None, ""})
        empty_right_count: int = sum(1 for row in rows if row.get("l2_dist_right") in {None, ""})
        empty_count: int = empty_left_count + empty_right_count
        if empty_count:
            warnings.warn(
                f"Found {empty_count} empty EPFL hand l2 distance values; treating them as accepted confidence.",
                stacklevel=2,
            )
        left_l2: ndarray = np.array([_parse_scalar(row, "l2_dist_left", default=0.0) for row in rows], dtype=np.float32)
        right_l2: ndarray = np.array(
            [_parse_scalar(row, "l2_dist_right", default=0.0) for row in rows],
            dtype=np.float32,
        )
        conf[:, :21] = conf[:, :21] * (left_l2 < np.float32(0.06))[:, np.newaxis]
        conf[:, 21:] = conf[:, 21:] * (right_l2 < np.float32(0.06))[:, np.newaxis]
    conf = np.where(finite_mask, conf, np.float32(0.0)).astype(np.float32)
    return conf


def _apply_confidence_to_xyz(
    xyz: Float32[ndarray, "num_frames num_kpts 3"],
    conf: Float32[ndarray, "num_frames num_kpts"],
) -> tuple[Float32[ndarray, "num_frames num_kpts 3"], Float32[ndarray, "num_frames num_kpts"]]:
    xyz_out: ndarray = xyz.astype(np.float32, copy=True)
    conf_out: ndarray = conf.astype(np.float32, copy=True)
    finite_mask: ndarray = np.isfinite(xyz_out).all(axis=-1)
    conf_out = np.where(finite_mask, conf_out, np.float32(0.0)).astype(np.float32)
    xyz_out[conf_out <= np.float32(0.0)] = np.nan
    return xyz_out, conf_out


def _cell_from_aliases(row: dict[str, str], keys: tuple[str, ...], *, field_name: str) -> str:
    for key in keys:
        value: str | None = row.get(key)
        if value is not None and value != "":
            return value
    raise KeyError(f"Missing EPFL column {field_name}; accepted aliases: {keys}")


def _root_axis_angle(row: dict[str, str], keys: tuple[str, ...], *, field_name: str) -> ndarray:
    root_raw: ndarray = _parse_numeric_cell(_cell_from_aliases(row, keys, field_name=field_name)).astype(
        np.float32,
        copy=False,
    )
    if root_raw.shape == (3,):
        return root_raw
    if root_raw.shape == (3, 3):
        from scipy.spatial.transform import Rotation

        return Rotation.from_matrix(root_raw).as_rotvec().astype(np.float32)
    if root_raw.size == 9:
        from scipy.spatial.transform import Rotation

        return Rotation.from_matrix(root_raw.reshape(3, 3)).as_rotvec().astype(np.float32)
    raise ValueError(f"{field_name} must be axis-angle length 3 or rotation matrix 3x3, got {root_raw.shape}")


def _mano_pose_from_row(row: dict[str, str], *, side: Literal["left", "right"]) -> ndarray:
    """Build the MANO 48-vector for one hand from a CSV row.

    EPFL packs ``{side}_poses`` as a 48-vector whose first three entries are zero
    (placeholder root) and stores the true root axis-angle separately in
    ``{side}_Rh``. The 45-vector path supports older slices that drop the
    placeholder. In both cases we always overlay ``Rh`` onto the root.
    """
    pose_key: str = f"{side}_poses"
    pose: ndarray = _parse_vector_cell(row[pose_key], field_name=pose_key)
    root: ndarray = _root_axis_angle(row, (f"{side}_Rh", f"{side}_RH"), field_name=f"{side}_Rh")
    if pose.shape[0] == 48:
        full: ndarray = pose.astype(np.float32, copy=True)
        full[:3] = root.astype(np.float32, copy=False)
        return full
    if pose.shape[0] == 45:
        return np.concatenate([root, pose]).astype(np.float32, copy=False)
    if pose.shape[0] < 45:
        raise ValueError(
            f"EPFL {pose_key} has {pose.shape[0]} coefficients, which looks like PCA coefficients. "
            "SimpleCV does not yet validate EPFL MANO PCA expansion, so refusing to pad them as axis-angle."
        )
    raise ValueError(f"EPFL {pose_key} cannot be represented as MANO 48-vector; got {pose.shape[0]} values")


def _load_mano_stack(hand_rows: list[dict[str, str]]) -> ManoStack:
    """Build the per-sequence ``ManoStack`` from EPFL ``pose3d_mano.csv`` rows.

    Note: EPFL packs translations in the standard SMPL/MANO convention where the
    root rotation also rotates the template wrist
    (``wrist_world = R_root @ root_j_template + Th``). SimpleCV's
    ``MANOLayerNP`` keeps the root joint at the template position, so to make
    the layer reproduce the released keypoints we must store
    ``trans = Th + (R_root - I) @ root_j_template`` per hand and frame. The
    template wrist depends on per-hand betas, so this loader constructs a
    short-lived ``MANOLayerNP`` for each hand to read ``root_trans``. As a
    consequence, loading EPFL labels currently requires the MANO model assets
    (``MANO_LEFT.pkl`` / ``MANO_RIGHT.pkl``); they are auto-downloaded on first
    use and cached under ``simplecv/data/``. Other ExoEgo loaders (HoCap,
    Hot3D) still load labels asset-free.
    """
    required_column_aliases: dict[str, tuple[str, ...]] = {
        "left_poses": ("left_poses",),
        "right_poses": ("right_poses",),
        "left_Rh": ("left_Rh", "left_RH"),
        "right_Rh": ("right_Rh", "right_RH"),
        "left_Th": ("left_Th", "left_TH"),
        "right_Th": ("right_Th", "right_TH"),
        "left_shapes": ("left_shapes",),
        "right_shapes": ("right_shapes",),
    }
    missing_columns: list[str] = [
        field_name for field_name, aliases in required_column_aliases.items() if not any(alias in hand_rows[0] for alias in aliases)
    ]
    if missing_columns:
        raise KeyError(f"EPFL MANO parameter columns missing from pose3d_mano.csv: {missing_columns}")

    left_shapes: ndarray = np.stack([_parse_vector_cell(row["left_shapes"], expected_len=10, field_name="left_shapes") for row in hand_rows]).astype(
        np.float32, copy=False
    )
    right_shapes: ndarray = np.stack(
        [_parse_vector_cell(row["right_shapes"], expected_len=10, field_name="right_shapes") for row in hand_rows]
    ).astype(np.float32, copy=False)
    left_delta: float = float(np.nanmax(np.abs(left_shapes - left_shapes[0])))
    right_delta: float = float(np.nanmax(np.abs(right_shapes - right_shapes[0])))
    if left_delta > 1e-3 or right_delta > 1e-3:
        warnings.warn(
            f"EPFL MANO shape parameters drift over time (left max={left_delta:.6f} right max={right_delta:.6f}); using first-frame values.",
            stacklevel=2,
        )

    num_frames: int = len(hand_rows)
    so3: ndarray = np.zeros((num_frames, 2, 48), dtype=np.float32)
    trans: ndarray = np.zeros((num_frames, 2, 3), dtype=np.float32)
    for frame_idx, row in enumerate(hand_rows):
        so3[frame_idx, 0] = _mano_pose_from_row(row, side="right")
        so3[frame_idx, 1] = _mano_pose_from_row(row, side="left")
        trans[frame_idx, 0] = _parse_vector_cell(
            _cell_from_aliases(row, ("right_Th", "right_TH"), field_name="right_Th"),
            expected_len=3,
            field_name="right_Th",
        )
        trans[frame_idx, 1] = _parse_vector_cell(
            _cell_from_aliases(row, ("left_Th", "left_TH"), field_name="left_Th"),
            expected_len=3,
            field_name="left_Th",
        )

    # ManoStack convention: index 0 = right, index 1 = left.
    betas: ndarray = np.stack(
        [right_shapes[0], left_shapes[0]],
        axis=0,
    ).astype(np.float32, copy=False)

    # EPFL packs translations using the standard MANO convention where the root
    # rotation also rotates the template wrist (wrist_world = R_root @ root_j + Th).
    # SimpleCV's MANOLayerNP adds `trans` after FK without rotating root_j, so we
    # must convert: store `Th + (R_root - I) @ root_j_template` per frame/hand. The
    # template wrist `root_j` depends on the per-hand betas, hence the per-hand
    # MANOLayerNP construction below.
    from scipy.spatial.transform import Rotation

    from simplecv.ops.mano.mano_np import MANOLayerNP

    root_template_per_hand: list[Float32[ndarray, "3"]] = [
        MANOLayerNP(side="right", betas=betas[0], use_pca=False).root_trans.reshape(3).astype(np.float32),
        MANOLayerNP(side="left", betas=betas[1], use_pca=False).root_trans.reshape(3).astype(np.float32),
    ]
    eye3: Float32[ndarray, "3 3"] = np.eye(3, dtype=np.float32)
    for hand_idx in range(2):
        rotvec_seq: Float32[ndarray, "n_frames 3"] = so3[:, hand_idx, :3]
        R_seq: Float32[ndarray, "n_frames 3 3"] = Rotation.from_rotvec(rotvec_seq).as_matrix().astype(np.float32)
        root_j: Float32[ndarray, "3"] = root_template_per_hand[hand_idx]
        delta: Float32[ndarray, "n_frames 3"] = np.einsum("bij,j->bi", R_seq - eye3[None], root_j).astype(np.float32)
        trans[:, hand_idx] = trans[:, hand_idx] + delta

    return ManoStack(betas=betas, so3=so3, trans=trans, use_pca=False)


class EpflSmartKitchenSequence(BaseExoEgoSequence[EpflSmartKitchenConfig]):
    """EPFL-Smart-Kitchen ExoEgo dataset adapter."""

    _SPLITS: tuple[Literal["train", "test"], ...] = ("train", "test")

    @classmethod
    def sequence_identity_for_config(cls, cfg: EpflSmartKitchenConfig) -> SequenceIdentity:
        return SequenceIdentity(
            dataset="epfl-smart-kitchen",
            parts=(cfg.split, cfg.participant_id, cfg.session_name),
        )

    def __getitem__(self, idx: int | None = None, ts_nano: np.timedelta64 | None = None) -> ExoEgoSample:
        canonical_idx, ts_ns = self._resolve_canonical(idx=idx, ts_nano=ts_nano)
        ego_cam_params_list, ego_bgr_list = self._sample_ego(ts_ns)
        exo_cam_params_list, exo_bgr_list = self._sample_exo(ts_ns)
        labels: ExoEgoLabels | None = self._sample_labels(canonical_idx, ts_ns)
        return ExoEgoSample(
            canonical_index=canonical_idx,
            canonical_timestamp_ns=ts_ns,
            ego_cam_params_list=ego_cam_params_list,
            ego_bgr_list=ego_bgr_list,
            exo_cam_params_list=exo_cam_params_list,
            exo_bgr_list=exo_bgr_list,
            labels=labels,
        )

    def _build_ego(self) -> BaseEgoSequence[EpflSmartKitchenConfig] | None:
        from simplecv.data.ego.epfl_smart_kitchen_ego import EpflSmartKitchenEgoSequence

        return EpflSmartKitchenEgoSequence(cfg=self.config)

    def _build_exo(self) -> BaseExoSequence[EpflSmartKitchenConfig] | None:
        from simplecv.data.exo.epfl_smart_kitchen_exo import EpflSmartKitchenExoSequence

        return EpflSmartKitchenExoSequence(cfg=self.config)

    def load_stream_timestamps_ns(self) -> dict[str, Int[ndarray, "n_frames"]]:
        stream_ts: dict[str, Int[ndarray, "n_frames"]] = {}
        self._ego_stream_names.clear()
        self._exo_stream_names.clear()
        rgb_timestamps_ns: Int[ndarray, "n_frames"] = load_rgb_timestamps_ns(self.config)

        if self.ego_sequence is not None:
            for name in self.ego_sequence.ego_video_names:
                stream_name: str = f"ego/{name}"
                stream_ts[stream_name] = rgb_timestamps_ns
                self._ego_stream_names.append(stream_name)

        if self.exo_sequence is not None:
            for name in self.exo_sequence.exo_video_names:
                stream_name = f"exo/{name}"
                stream_ts[stream_name] = rgb_timestamps_ns
                self._exo_stream_names.append(stream_name)

        return stream_ts

    def load_labels(self) -> ExoEgoLabels | None:
        body_path: Path = body_pose_path(self.config)
        hand_path: Path = hand_pose_path(self.config)
        body_rows: list[dict[str, str]] = _read_csv_rows(body_path)
        hand_rows: list[dict[str, str]] = _read_csv_rows(hand_path)
        if len(body_rows) != len(hand_rows):
            raise ValueError(f"EPFL body and hand label frame counts must match: {body_path} has {len(body_rows)}, {hand_path} has {len(hand_rows)}")

        body_xyz_raw: Float32[ndarray, "num_frames body_kpts=17 3"] = _parse_keypoints(
            body_rows,
            num_keypoints=17,
            path=body_path,
        )
        hand_xyz_raw: Float32[ndarray, "num_frames hand_kpts=42 3"] = _parse_keypoints(
            hand_rows,
            num_keypoints=42,
            path=hand_path,
        )
        body_finite: ndarray = np.isfinite(body_xyz_raw).all(axis=-1)
        hand_finite: ndarray = np.isfinite(hand_xyz_raw).all(axis=-1)
        body_conf_raw: Float32[ndarray, "num_frames body_kpts=17"] = _parse_confidences(
            body_rows,
            num_keypoints=17,
            mode="body",
            finite_mask=body_finite,
        )
        hand_conf_raw: Float32[ndarray, "num_frames hand_kpts=42"] = _parse_confidences(
            hand_rows,
            num_keypoints=42,
            mode="hand",
            finite_mask=hand_finite,
        )
        body_xyz: Float32[ndarray, "num_frames body_kpts=17 3"]
        body_conf: Float32[ndarray, "num_frames body_kpts=17"]
        body_confidence_result: tuple[
            Float32[ndarray, "num_frames body_kpts=17 3"],
            Float32[ndarray, "num_frames body_kpts=17"],
        ] = _apply_confidence_to_xyz(body_xyz_raw, body_conf_raw)
        body_xyz = body_confidence_result[0]
        body_conf = body_confidence_result[1]
        hand_xyz: Float32[ndarray, "num_frames hand_kpts=42 3"]
        hand_conf: Float32[ndarray, "num_frames hand_kpts=42"]
        hand_confidence_result: tuple[
            Float32[ndarray, "num_frames hand_kpts=42 3"],
            Float32[ndarray, "num_frames hand_kpts=42"],
        ] = _apply_confidence_to_xyz(hand_xyz_raw, hand_conf_raw)
        hand_xyz = hand_confidence_result[0]
        hand_conf = hand_confidence_result[1]

        num_frames: int = int(body_xyz.shape[0])
        label_timestamps_ns: Int[ndarray, "num_frames"] = load_rgb_timestamps_ns(self.config)
        if label_timestamps_ns.shape[0] != num_frames:
            raise ValueError(
                "EPFL label frame count must match RGB timestamp count: "
                f"labels have {num_frames} frames, timestamps have {label_timestamps_ns.shape[0]} frames."
            )
        xyzc_stack: Float32[ndarray, "num_frames coco_kpts=133 4"] = np.full(
            (num_frames, 133, 4),
            np.nan,
            dtype=np.float32,
        )
        xyzc_stack[:, :, 3] = np.float32(0.0)
        xyzc_stack[:, :17, :3] = body_xyz
        xyzc_stack[:, :17, 3] = body_conf
        xyzc_stack[:, LEFT_HAND_IDX, :3] = hand_xyz[:, :21]
        xyzc_stack[:, LEFT_HAND_IDX, 3] = hand_conf[:, :21]
        xyzc_stack[:, RIGHT_HAND_IDX, :3] = hand_xyz[:, 21:]
        xyzc_stack[:, RIGHT_HAND_IDX, 3] = hand_conf[:, 21:]

        mano_stack: ManoStack = _load_mano_stack(hand_rows)
        return ExoEgoLabels(xyzc_stack=xyzc_stack, timestamps_ns=label_timestamps_ns, mano_stack=mano_stack)

    @classmethod
    def iter_episode_sequences(cls, cfg: EpflSmartKitchenConfig) -> Generator["EpflSmartKitchenSequence", None, None]:
        for split, participant_id, session_name in cls._episode_specs(cfg):
            episode_cfg: EpflSmartKitchenConfig = replace(
                cfg,
                split=split,
                participant_id=participant_id,
                session_name=session_name,
            )
            yield cls(episode_cfg)

    @classmethod
    def num_sequences_for_config(cls, cfg: EpflSmartKitchenConfig) -> int:
        return len(cls._episode_specs(cfg))

    @classmethod
    def _episode_specs(cls, cfg: EpflSmartKitchenConfig) -> list[tuple[Literal["train", "test"], str, str]]:
        pose_root: Path = cfg.root_directory / "Public_release_pose"
        video_root: Path = cfg.root_directory / "Public_release_videos"
        specs: list[tuple[Literal["train", "test"], str, str]] = []
        for split in cls._SPLITS:
            split_dir: Path = pose_root / split
            if not split_dir.exists():
                continue
            participant_dirs: list[Path] = natsorted([path for path in split_dir.iterdir() if path.is_dir()])
            for participant_dir in participant_dirs:
                session_dirs: list[Path] = natsorted([path for path in participant_dir.iterdir() if path.is_dir()])
                for session_dir in session_dirs:
                    video_dir: Path = video_root / split / participant_dir.name / session_dir.name
                    if not video_dir.exists():
                        continue
                    specs.append((split, participant_dir.name, session_dir.name))
        return specs

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        return ViewCoordinates.RIGHT_HAND_Z_UP
