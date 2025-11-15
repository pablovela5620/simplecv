import json
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any, Literal, cast

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import tyro
from jaxtyping import Float
from natsort import natsorted
from numpy import ndarray
from rerun.blueprint import ContainerLike
from serde import field as serde_field
from serde import serde
from serde.compat import SerdeError
from serde.json import from_json
from tqdm.auto import tqdm

from simplecv.apis.quest3_oakd_loader import Quest3OakDVisualizeConfig, load_and_log_quest_data
from simplecv.camera_parameters import BrownConradyDistortion, Extrinsics, Intrinsics, PinholeParameters
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole, log_video
from simplecv.video_utils import Resolution, reencode_video_optimal


@dataclass
class IngestConfig:
    """Structured configuration for running the exo/ego visualization CLI."""

    rr_config: RerunTyroConfig
    """Command-line options for spawning and configuring the Rerun viewer."""
    exoego_dir: Path
    """Path to the directory containing 'exo' and/or 'ego' subdirectories with video files."""
    verbose: bool = False
    """Enable verbose console logging during ingestion."""
    reencode_to_av1: bool = True
    """Force AV1 MP4 re-encoding (with 720p ceiling) before logging videos."""


def validate_exoego_dir(exoego_dir: Path) -> tuple[Path | None, Path | None]:
    if not exoego_dir.exists():
        raise ValueError(f"The provided directory does not exist: {exoego_dir}")
    if not exoego_dir.is_dir():
        raise ValueError(f"The provided path is not a directory: {exoego_dir}")

    # make sure that either "exo" or "ego" subdirectory exists
    if not (exoego_dir / "exo").exists() and not (exoego_dir / "ego").exists():
        raise ValueError(f"The provided directory does not contain 'exo' or 'ego' subdirectory: {exoego_dir}")

    return (exoego_dir / "exo") if (exoego_dir / "exo").exists() else None, (exoego_dir / "ego") if (
        exoego_dir / "ego"
    ).exists() else None


@dataclass(frozen=True, slots=True)
class VideoProbeResult:
    """Metadata extracted via ffprobe for the primary video stream."""

    codec_name: str
    width: int
    height: int
    format_name: str


@dataclass(frozen=True, slots=True)
class PrepareVideoForLoggingResult:
    """Prepared video asset details suitable for Rerun logging."""

    prepared_path: Path
    """Filesystem path that will be logged (original or temporary)."""
    metadata: VideoProbeResult
    """ffprobe metadata gathered after any conversions."""
    should_cleanup: bool
    """Flag indicating whether ``prepared_path`` is a temporary file to delete."""


@dataclass(frozen=True, slots=True)
class VideoIngestEntry:
    """Structured container coupling a source video with its camera/video entity paths."""

    source_path: Path
    """Filesystem path of the input video on disk."""
    camera_log_path: Path
    """Root Rerun entity path for this camera (e.g. '/world/exo/<camera_id>')."""
    video_log_path: Path
    """Rerun entity path for the video stream (e.g. '/world/exo/<camera_id>/pinhole/video')."""

    @property
    def pinhole_log_path(self) -> Path:
        """Rerun entity path for the camera's pinhole node."""

        return self.camera_log_path / "pinhole"


@serde
class OakIntrinsics:
    """Intrinsics payload emitted by the OAK capture rig."""

    socket: Literal["CAM_A", "CAM_B", "CAM_C"]
    """Hardware connector identifier reported by DepthAI (`CAM_A`, `CAM_B`, or `CAM_C`)."""
    width: int
    """Captured frame width in pixels for the associated stream."""
    height: int
    """Captured frame height in pixels for the associated stream."""
    intrinsics: Float[ndarray, "3 3"]
    """Row-major 3×3 pinhole intrinsics matrix (fx, fy, cx, cy)."""
    distortion: Float[ndarray, "14"]
    """Full 14-coefficient distortion vector provided by the OAK calibration tool."""
    fov_deg: float
    """Horizontal field of view in degrees reported for the sensor."""


@serde
class OakExtrinsics:
    R: Float[ndarray, "3 3"]
    """Rotation matrix describing the rigid transform."""
    T: Float[ndarray, "3"]
    """Translation vector describing the rigid transform."""
    matrix_4x4: Float[ndarray, "4 4"]
    """Homogeneous 4x4 transformation matrix."""


@serde
class OakCalibration:
    """Calibration payload emitted by the OAK capture rig."""

    device_mxid: str
    """Unique Matrix/DepthAI device identifier encoded in the `calibration.json`."""
    right: OakIntrinsics
    """Right mono camera intrinsics/extrinsics block forwarded verbatim from the capture tool."""
    left: OakIntrinsics
    """Left mono camera intrinsics/extrinsics block used as the stereo reference."""
    rgb: OakIntrinsics | None
    """Optional RGB sensor calibration payload (``None`` for depth-only recordings)."""
    extrinsics_left_to_right: OakExtrinsics
    """Rigid transform describing ``left``→``right`` alignment in the calibration file's units."""
    extrinsics_left_to_rgb: OakExtrinsics | None
    """Rigid transform describing ``left``→``rgb`` alignment, omitted when RGB is absent."""


_OAK_DIST_COEFF_COUNT: int = 14
_DEPTHAI_CM_TO_MM: float = 10.0


@serde
class DepthAIResolution:
    width: int | None = None
    height: int | None = None
    left: int | None = None
    top: int | None = None
    right: int | None = None
    bottom: int | None = None


@serde
class DepthAILensIntrinsics:
    focal_length_x: float
    focal_length_y: float
    principal_point_x: float
    principal_point_y: float
    skew: float | None = None
    available: bool | None = None


@serde
class DepthAIDistortion:
    coefficients: list[float] | None = None
    available: bool | None = None


@serde
class DepthAILensInfo:
    focal_length_mm: list[float] | float | None = None
    fov_deg: float | None = None


@serde
class DepthAISensorInfo:
    physical_size_mm: DepthAIResolution | None = None
    active_array_size: DepthAIResolution | None = None
    pixel_array_size: DepthAIResolution | None = None


@serde
class DepthAIIntrinsicsEntry:
    camera_id: str
    lens_intrinsics: DepthAILensIntrinsics
    positional_layout: str | None = None
    capture_resolution: DepthAIResolution | None = None
    sensor_info: DepthAISensorInfo | None = None
    distortion: DepthAIDistortion | None = None
    lens_info: DepthAILensInfo | None = None


@serde
class DepthAIExtrinsicsEntry:
    R: list[list[float]]
    T: list[float]
    type: str | None = None
    from_: str | None = serde_field(default=None, rename="from")
    to: str | None = None
    matrix_4x4: list[list[float]] | None = None


@serde
class DepthAICalibration:
    intrinsics: list[DepthAIIntrinsicsEntry]
    extrinsics: list[DepthAIExtrinsicsEntry]
    device_id: str | None = None
    device_mxid: str | None = None


def _load_oak_calibration(calibration_json_path: Path) -> OakCalibration:
    """Load an OAK calibration JSON emitted either by the legacy or the new DepthAI exporter."""

    payload_text: str = calibration_json_path.read_text()
    try:
        return from_json(OakCalibration, payload_text)
    except SerdeError:
        depthai_payload: DepthAICalibration = from_json(DepthAICalibration, payload_text)
        return _oak_calibration_from_depthai_payload(depthai_payload)


def _oak_calibration_from_depthai_payload(payload: DepthAICalibration) -> OakCalibration:
    """Convert the DepthAI calibration schema (intrinsics/extrinsics lists) into ``OakCalibration``."""

    intrinsics_by_layout: dict[str, OakIntrinsics] = {}
    intrinsics_by_id: dict[str, OakIntrinsics] = {}
    for entry in payload.intrinsics:
        intr_obj: OakIntrinsics = _oak_intrinsics_from_depthai_doc(entry)
        if entry.positional_layout:
            intrinsics_by_layout[entry.positional_layout.lower()] = intr_obj
        if entry.camera_id:
            intrinsics_by_id[entry.camera_id.lower()] = intr_obj

    left_intr: OakIntrinsics | None = intrinsics_by_layout.get("left") or intrinsics_by_id.get("cam_c")
    right_intr: OakIntrinsics | None = intrinsics_by_layout.get("right") or intrinsics_by_id.get("cam_b")
    rgb_intr: OakIntrinsics | None = (
        intrinsics_by_layout.get("center") or intrinsics_by_layout.get("rgb") or intrinsics_by_id.get("cam_a")
    )

    if left_intr is None or right_intr is None:
        raise ValueError("Unable to locate left/right camera intrinsics in the DepthAI calibration JSON.")

    left_id: str = left_intr.socket or "left"
    right_id: str = right_intr.socket or "right"
    rgb_id: str | None = rgb_intr.socket if rgb_intr and rgb_intr.socket else None

    extr_lr_doc: DepthAIExtrinsicsEntry | None = _find_depthai_extrinsics_doc(
        payload.extrinsics,
        expected_type="left_to_right",
        from_id=left_id,
        to_id=right_id,
    )
    if extr_lr_doc is None:
        raise ValueError("DepthAI calibration JSON is missing the left-to-right extrinsics block.")
    extr_lr: OakExtrinsics = _oak_extrinsics_from_depthai_doc(extr_lr_doc)

    extr_rgb_doc: DepthAIExtrinsicsEntry | None = None
    if rgb_id is not None:
        extr_rgb_doc = _find_depthai_extrinsics_doc(
            payload.extrinsics,
            expected_type="left_to_rgb",
            from_id=left_id,
            to_id=rgb_id,
        )
    extr_rgb: OakExtrinsics | None = _oak_extrinsics_from_depthai_doc(extr_rgb_doc) if extr_rgb_doc else None

    device_mxid: str = payload.device_mxid or payload.device_id or ""

    return OakCalibration(
        device_mxid=device_mxid,
        right=right_intr,
        left=left_intr,
        rgb=rgb_intr,
        extrinsics_left_to_right=extr_lr,
        extrinsics_left_to_rgb=extr_rgb,
    )


def _oak_intrinsics_from_depthai_doc(doc: DepthAIIntrinsicsEntry) -> OakIntrinsics:
    """Construct ``OakIntrinsics`` from a DepthAI calibration intrinsics entry."""

    fx: float = float(doc.lens_intrinsics.focal_length_x)
    fy: float = float(doc.lens_intrinsics.focal_length_y)
    cx: float = float(doc.lens_intrinsics.principal_point_x)
    cy: float = float(doc.lens_intrinsics.principal_point_y)

    capture_block: DepthAIResolution | None = doc.capture_resolution
    if capture_block is None and doc.sensor_info is not None:
        capture_block = doc.sensor_info.pixel_array_size
    if capture_block is None or capture_block.width is None or capture_block.height is None:
        raise ValueError("DepthAI calibration intrinsics entry is missing capture resolution info.")
    width: int = int(capture_block.width)
    height: int = int(capture_block.height)

    k_matrix: Float[np.ndarray, "3 3"] = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    distortion_block: DepthAIDistortion | None = doc.distortion
    coeffs: Float[np.ndarray, "14"] = _coerce_distortion_coeffs(
        distortion_block.coefficients if distortion_block else None
    )

    lens_info_block: DepthAILensInfo | None = doc.lens_info
    if lens_info_block is not None and isinstance(lens_info_block.fov_deg, (int, float)):
        fov_deg: float = float(lens_info_block.fov_deg)
    else:
        fov_deg = _fov_from_fx(width, fx)

    socket: str = doc.camera_id

    return OakIntrinsics(
        socket=socket,
        width=width,
        height=height,
        intrinsics=k_matrix,
        distortion=coeffs,
        fov_deg=fov_deg,
    )


def _find_depthai_extrinsics_doc(
    entries: list[DepthAIExtrinsicsEntry],
    *,
    expected_type: str,
    from_id: str,
    to_id: str | None,
) -> DepthAIExtrinsicsEntry | None:
    """Return the extrinsics entry matching ``expected_type`` or the provided endpoints."""

    expected_type_lower: str = expected_type.lower()
    from_id_lower: str = from_id.lower()
    to_id_lower: str | None = to_id.lower() if to_id is not None else None

    for entry in entries:
        if entry.type and entry.type.lower() == expected_type_lower:
            return entry

    for entry in entries:
        from_lower: str | None = entry.from_.lower() if isinstance(entry.from_, str) else None
        to_lower: str | None = entry.to.lower() if isinstance(entry.to, str) else None
        if from_lower == from_id_lower and (to_id_lower is None or to_lower == to_id_lower):
            return entry

    return None


def _oak_extrinsics_from_depthai_doc(entry: DepthAIExtrinsicsEntry) -> OakExtrinsics:
    """Convert a DepthAI extrinsics block (centimeters) into ``OakExtrinsics`` with millimeter storage."""

    R: Float[np.ndarray, "3 3"] = np.array(entry.R, dtype=np.float32)
    if R.shape != (3, 3):
        raise ValueError("DepthAI extrinsics rotation matrix must be 3x3.")
    translation_cm: Float[np.ndarray, "3"] = np.array(entry.T, dtype=np.float32).reshape(3)
    translation_mm: Float[np.ndarray, "3"] = translation_cm * float(_DEPTHAI_CM_TO_MM)

    if entry.matrix_4x4 is not None:
        matrix_raw: Float[np.ndarray, "4 4"] = np.array(entry.matrix_4x4, dtype=np.float32)
        if matrix_raw.shape != (4, 4):
            raise ValueError("DepthAI extrinsics matrix must be 4x4 when provided.")
    else:
        matrix_raw = np.eye(4, dtype=np.float32)
        matrix_raw[:3, :3] = R
    matrix_raw = matrix_raw.copy()
    matrix_raw[:3, 3] = translation_mm

    return OakExtrinsics(
        R=R,
        T=translation_mm,
        matrix_4x4=matrix_raw,
    )


def _coerce_distortion_coeffs(coefficients_obj: Any) -> Float[np.ndarray, "14"]:
    """Pad/trim the provided distortion list to the 14-term Brown–Conrady vector."""

    if coefficients_obj is None:
        coeffs: Float[np.ndarray, "14"] = np.zeros(_OAK_DIST_COEFF_COUNT, dtype=np.float32)
        return coeffs

    coeffs_array: Float[np.ndarray, "_"] = np.array(coefficients_obj, dtype=np.float32).reshape(-1)
    if coeffs_array.size < _OAK_DIST_COEFF_COUNT:
        coeffs_padded: Float[np.ndarray, "14"] = np.pad(
            coeffs_array,
            (0, _OAK_DIST_COEFF_COUNT - coeffs_array.size),
            constant_values=0.0,
        ).astype(np.float32)
        return coeffs_padded
    if coeffs_array.size > _OAK_DIST_COEFF_COUNT:
        coeffs_trimmed: Float[np.ndarray, "14"] = coeffs_array[:_OAK_DIST_COEFF_COUNT].astype(np.float32)
        return coeffs_trimmed
    return coeffs_array.astype(np.float32)


def _make_intrinsics(oak_i: OakIntrinsics) -> Intrinsics:
    K: Float[np.ndarray, "3 3"] = oak_i.intrinsics.astype(np.float32)
    fx: float = float(K[0, 0])
    fy: float = float(K[1, 1])
    cx: float = float(K[0, 2])
    cy: float = float(K[1, 2])
    width: int = int(oak_i.width)
    height: int = int(oak_i.height)

    intr: Intrinsics = Intrinsics(
        camera_conventions="RDF",
        fl_x=fx,
        fl_y=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
    )
    return intr


def _make_distortion(oak_i: OakIntrinsics) -> BrownConradyDistortion:
    d: Float[np.ndarray, "14"] = oak_i.distortion.astype(np.float32)
    # OpenCV 14-term order: [k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,tau_x,tau_y]
    k1: float = float(d[0])
    k2: float = float(d[1])
    p1: float = float(d[2])
    p2: float = float(d[3])
    k3: float = float(d[4])
    k4: float = float(d[5])
    k5: float = float(d[6])
    k6: float = float(d[7])
    s1: float = float(d[8])
    s2: float = float(d[9])
    s3: float = float(d[10])
    s4: float = float(d[11])
    tau_x: float = float(d[12])
    tau_y: float = float(d[13])

    dist: BrownConradyDistortion = BrownConradyDistortion(
        k1=k1,
        k2=k2,
        p1=p1,
        p2=p2,
        k3=k3,
        k4=k4,
        k5=k5,
        k6=k6,
        s1=s1,
        s2=s2,
        s3=s3,
        s4=s4,
        tau_x=tau_x,
        tau_y=tau_y,
    )
    return dist


def _make_extrinsics(H_left_to_cam: Float[np.ndarray, "4 4"]) -> Extrinsics:
    R: Float[np.ndarray, "3 3"] = H_left_to_cam[:3, :3].astype(np.float32)
    t: Float[np.ndarray, "3"] = H_left_to_cam[:3, 3].astype(np.float32) * 1e-3  # mm -> meters
    extr: Extrinsics = Extrinsics(world_R_cam=R, world_t_cam=t)
    return extr


def _build_pinhole(name: str, oak_i: OakIntrinsics, H_left_to_cam: Float[np.ndarray, "4 4"]) -> PinholeParameters:
    intr: Intrinsics = _make_intrinsics(oak_i)
    dist: BrownConradyDistortion = _make_distortion(oak_i)
    extr: Extrinsics = _make_extrinsics(H_left_to_cam)
    pinhole: PinholeParameters = PinholeParameters(name=name, intrinsics=intr, extrinsics=extr, distortion=dist)
    return pinhole


def oak_calib_to_pinhole(oak_calib: OakCalibration) -> list[PinholeParameters]:
    """Convert an OAK calibration payload into a list of SimpleCV PinholeParameters (DRY)."""
    pinholes: list[PinholeParameters] = []

    # Left is the world frame (identity 4×4)
    I_left: Float[np.ndarray, "4 4"] = np.eye(4, dtype=np.float32)
    pinholes.append(_build_pinhole("left", oak_calib.left, I_left))

    # Right (requires left->right 4×4)
    H_lr: Float[np.ndarray, "4 4"] = oak_calib.extrinsics_left_to_right.matrix_4x4.astype(np.float32)
    pinholes.append(_build_pinhole("right", oak_calib.right, H_lr))

    # RGB if present (requires both rgb intrinsics and left->rgb 4×4)
    if oak_calib.rgb is not None and oak_calib.extrinsics_left_to_rgb is not None:
        H_lrgb: Float[np.ndarray, "4 4"] = oak_calib.extrinsics_left_to_rgb.matrix_4x4.astype(np.float32)
        pinholes.append(_build_pinhole("rgb", oak_calib.rgb, H_lrgb))

    return pinholes


# ---- minimal SO(3) utilities ----


def _skew(v: Float[ndarray, "3"]) -> Float[ndarray, "3 3"]:
    K: Float[ndarray, "3 3"] = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float32)
    return K


def _so3_exp(omega: Float[ndarray, "3"]) -> Float[ndarray, "3 3"]:
    theta: float = float(np.linalg.norm(omega))
    I3: Float[ndarray, "3 3"] = np.eye(3, dtype=np.float32)
    if theta < 1e-9:
        R: Float[ndarray, "3 3"] = I3.copy()
        return R
    k: Float[ndarray, "3"] = (omega / theta).astype(np.float32)
    K: Float[ndarray, "3 3"] = _skew(k)
    s: float = float(np.sin(theta))
    c: float = float(np.cos(theta))
    R: Float[ndarray, "3 3"] = (I3 + s * K + (1.0 - c) * (K @ K)).astype(np.float32)
    return R


def _so3_log(R: Float[ndarray, "3 3"]) -> Float[ndarray, "3"]:
    tr: float = float(np.trace(R))
    cos_theta: float = max(min((tr - 1.0) * 0.5, 1.0), -1.0)
    theta: float = float(np.arccos(cos_theta))
    if theta < 1e-9:
        zero3: Float[ndarray, "3"] = np.zeros(3, dtype=np.float32)
        return zero3
    denom: float = 2.0 * float(np.sin(theta))
    W: Float[ndarray, "3 3"] = ((R - R.T) / denom).astype(np.float32)
    # vee(W) gives the rotation axis; multiply by theta to get omega
    omega: Float[ndarray, "3"] = np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=np.float32) * theta
    return omega


def _fov_from_fx(width: int, fx: float) -> float:
    half: float = float(np.arctan((0.5 * width) / fx))
    fov: float = float(np.degrees(2.0 * half))
    return fov


# ---- main helper (2-cam only, no magic scale) ----


def fill_missing_rgb(cal: OakCalibration) -> OakCalibration:
    """
    Populate `cal.rgb` and `cal.extrinsics_left_to_rgb` using only the 2-cam JSON.
    Intrinsics prior: clone mono focal (mean of left/right), center principal point, zero distortion.
    Extrinsics: SE(3) square-root of left->right, i.e., (I + R_h) T_h = T_lr and R_h = exp(0.5 * log R_lr).
    """

    # --- Intrinsics (RGB) -------------------------------------------------------
    if cal.rgb is None:
        K_l: Float[ndarray, "3 3"] = np.asarray(cal.left.intrinsics, dtype=np.float64)
        K_r: Float[ndarray, "3 3"] = np.asarray(cal.right.intrinsics, dtype=np.float64)
        w: int = int(cal.left.width)
        h: int = int(cal.left.height)

        fx: float = float(0.5 * (K_l[0, 0] + K_r[0, 0]))  # clone mono focal (no external scale)
        fy: float = float(0.5 * (K_l[1, 1] + K_r[1, 1]))
        cx: float = float(w * 0.5)  # center PP to avoid carrying mono offsets
        cy: float = float(h * 0.5)

        K_rgb64: Float[ndarray, "3 3"] = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        d_rgb64: Float[ndarray, "14"] = np.zeros(14, dtype=np.float64)
        fov_deg: float = _fov_from_fx(w, fx)

        K_rgb: Float[ndarray, "3 3"] = K_rgb64.astype(np.float32)
        d_rgb: Float[ndarray, "14"] = d_rgb64.astype(np.float32)

        cal.rgb = OakIntrinsics(
            socket="CAM_A",
            width=w,
            height=h,
            intrinsics=K_rgb,
            distortion=d_rgb,
            fov_deg=fov_deg,  # reported for completeness; derived from fx
        )

    # --- Extrinsics (left -> rgb) ----------------------------------------------
    if cal.extrinsics_left_to_rgb is None:
        R_lr64: Float[ndarray, "3 3"] = np.asarray(cal.extrinsics_left_to_right.R, dtype=np.float64)
        T_lr64: Float[ndarray, "3"] = np.asarray(cal.extrinsics_left_to_right.T, dtype=np.float64)

        w_lr: Float[ndarray, "3"] = _so3_log(R_lr64)
        R_h64: Float[ndarray, "3 3"] = _so3_exp(0.5 * w_lr)

        I3: Float[ndarray, "3 3"] = np.eye(3, dtype=np.float64)
        A: Float[ndarray, "3 3"] = I3 + R_h64
        # Guard against near-singularity only if rotations were ~pi (not your case)
        T_h64: Float[ndarray, "3"] = np.linalg.solve(A, T_lr64)

        H64: Float[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
        H64[:3, :3] = R_h64
        H64[:3, 3] = T_h64

        cal.extrinsics_left_to_rgb = OakExtrinsics(
            R=R_h64.astype(np.float32),
            T=T_h64.astype(np.float32),
            matrix_4x4=H64.astype(np.float32),
        )

    return cal


def probe_video_stream(video_path: Path) -> VideoProbeResult:
    """
    Inspect ``video_path`` using ffprobe and return the first video stream metadata.

    Raises:
        RuntimeError: If ffprobe fails during inspection.
        ValueError: If no valid video stream metadata is available.
    """

    ffprobe_cmd: list[str] = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height",
        "-show_entries",
        "format=format_name",
        "-of",
        "json",
        str(video_path),
    ]
    process: CompletedProcess[str] = subprocess.run(
        ffprobe_cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        stderr_output: str = process.stderr.strip()
        raise RuntimeError(f"ffprobe failed when inspecting {video_path}: {stderr_output}")

    probe_data: dict[str, object] = json.loads(process.stdout)
    streams_raw: object = probe_data.get("streams", [])
    if not isinstance(streams_raw, list) or not streams_raw:
        raise ValueError(f"No video streams found in {video_path}")

    stream_info_obj: object = streams_raw[0]
    if not isinstance(stream_info_obj, dict):
        raise ValueError(f"Malformed ffprobe stream metadata for {video_path}")

    codec_name_obj: object | None = stream_info_obj.get("codec_name")
    codec_name: str | None = codec_name_obj if isinstance(codec_name_obj, str) else None
    width_obj: object | None = stream_info_obj.get("width")
    width: int | None = _coerce_int(width_obj)
    height_obj: object | None = stream_info_obj.get("height")
    height: int | None = _coerce_int(height_obj)
    format_obj: object | None = probe_data.get("format", {})
    format_name_obj: object | None = format_obj.get("format_name") if isinstance(format_obj, dict) else None
    format_name: str | None = format_name_obj if isinstance(format_name_obj, str) else None

    if codec_name is None or width is None or height is None or format_name is None:
        raise ValueError(f"Missing stream metadata for {video_path}")

    return VideoProbeResult(
        codec_name=codec_name,
        width=width,
        height=height,
        format_name=format_name,
    )


def _format_tokens(format_name: str) -> set[str]:
    """Normalize the comma-separated ffprobe format name list."""

    return {token.strip().lower() for token in format_name.split(",") if token}


def _coerce_int(value: object) -> int | None:
    """Attempt to convert ffprobe numeric fields to integers."""

    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def prepare_video_for_logging(
    video_path: Path,
    *,
    verbose: bool = False,
    reencode_to_av1: bool = False,
) -> PrepareVideoForLoggingResult:
    """
    Optionally ensure ``video_path`` is AV1 encoded, stored as MP4, and respects the 1280x720 ceiling.

    Returns:
        A ``PrepareVideoForLoggingResult`` capturing the prepared path, metadata, and
        whether the prepared asset is temporary.

    Args:
        video_path: Path to the source video on disk.
        verbose: Whether to emit detailed logging from the underlying encoding utilities.
        reencode_to_av1: When true, convert non-AV1 content into AV1 MP4 constrained to 720p.
    """

    probe_result: VideoProbeResult = probe_video_stream(video_path)
    if not reencode_to_av1:
        return PrepareVideoForLoggingResult(
            prepared_path=video_path,
            metadata=probe_result,
            should_cleanup=False,
        )

    needs_reencode: bool = False
    resize_resolution: Resolution | None = None

    if "mp4" not in _format_tokens(probe_result.format_name):
        needs_reencode = True
    if probe_result.codec_name != "av1":
        needs_reencode = True
    if probe_result.width > 1280 or probe_result.height > 720:
        resize_resolution = "720p"
        needs_reencode = True

    if not needs_reencode:
        if probe_result.width > 1280 or probe_result.height > 720:
            raise ValueError(
                f"{video_path} exceeds the maximum resolution (found {probe_result.width}x{probe_result.height})."
            )
        return PrepareVideoForLoggingResult(
            prepared_path=video_path,
            metadata=probe_result,
            should_cleanup=False,
        )

    prepared_video_path: Path = reencode_video_optimal(
        input_video_path=video_path,
        resize=resize_resolution,
        verbose=verbose,
    )
    prepared_probe: VideoProbeResult = probe_video_stream(prepared_video_path)

    if "mp4" not in _format_tokens(prepared_probe.format_name):
        prepared_video_path.unlink(missing_ok=True)
        raise RuntimeError(f"Re-encoded video is not MP4: {prepared_video_path}")
    if prepared_probe.codec_name != "av1":
        prepared_video_path.unlink(missing_ok=True)
        raise RuntimeError(f"Re-encoded video is not AV1: {prepared_video_path}")
    if prepared_probe.width > 1280 or prepared_probe.height > 720:
        prepared_video_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Re-encoded video exceeds the maximum resolution: {prepared_probe.width}x{prepared_probe.height}"
        )

    return PrepareVideoForLoggingResult(
        prepared_path=prepared_video_path,
        metadata=prepared_probe,
        should_cleanup=True,
    )


def collect_video_entries(
    video_dir: Path,
    *,
    log_root: Path,
) -> list[VideoIngestEntry]:
    """Gather source/log path pairs for every MP4 inside ``video_dir``."""

    all_video_paths: list[Path] = natsorted(video_dir.glob("*.mp4"))
    assert all_video_paths, f"No .mp4 files found in directory: {video_dir}"

    video_entries: list[VideoIngestEntry] = [
        VideoIngestEntry(
            source_path=video_path,
            camera_log_path=log_root / video_path.stem,
            video_log_path=(log_root / video_path.stem / "pinhole" / "video"),
        )
        for video_path in all_video_paths
    ]
    return video_entries


def ingest_video_directory(
    video_entries: list[VideoIngestEntry],
    *,
    timeline: str,
    verbose: bool,
    progress_label: str,
    reencode_to_av1: bool,
) -> list[Path]:
    """Ingest the provided videos ensuring uniform encoding and resolution constraints.

    Returns:
        list[Path]: Entity paths where the prepared videos were logged.
    """

    assert video_entries, "No video entries provided for ingestion."

    # 1. Start by loading the calibration file from the first video's parent directory and logging it ONLY if ego
    perspective: str = video_entries[0].source_path.parent.name
    assert perspective in {"exo", "ego"}, f"Unexpected perspective directory name: {perspective}"

    if perspective == "ego":
        calibration_json_path: Path = video_entries[0].source_path.parent / "calibration.json"
        assert calibration_json_path.exists(), f"Calibration file not found at {calibration_json_path}"
        calibration: OakCalibration = _load_oak_calibration(calibration_json_path)
        calibration: OakCalibration = fill_missing_rgb(
            calibration,
        )

        # convert from OakCalibration -> simplecv PinholeCamera
        pinhole_param_list: list[PinholeParameters] = oak_calib_to_pinhole(calibration)
        # sort pinholes to match the video entries order
        pinhole_param_list = sorted(
            pinhole_param_list,
            key=lambda param: (
                next(
                    (
                        idx
                        for idx, entry in enumerate(video_entries)
                        if param.name.lower() in entry.camera_log_path.name.lower()
                    ),
                    len(video_entries),
                ),
                param.name.lower(),
            ),
        )
    else:
        pinhole_param_list = []
    expected_resolution: tuple[int, int] | None = None
    logged_video_entities: list[Path] = []
    iterable_entries: Iterable[VideoIngestEntry] = cast(
        Iterable[VideoIngestEntry],
        tqdm(video_entries, desc=progress_label, leave=False),
    )
    for idx, entry in enumerate(iterable_entries):
        if video_entries[0].source_path.parent.name == "ego":
            pinhole: PinholeParameters = pinhole_param_list[idx]
            assert pinhole.name.lower() in entry.camera_log_path.name.lower(), (
                f"Camera name mismatch: pinhole '{pinhole.name}' vs. entry '{entry.camera_log_path.name}'"
            )
            log_pinhole(
                camera=pinhole,
                cam_log_path=entry.camera_log_path,
                static=True,
            )
        prepared_video_result: PrepareVideoForLoggingResult = prepare_video_for_logging(
            video_path=entry.source_path,
            verbose=verbose,
            reencode_to_av1=reencode_to_av1,
        )
        prepared_path: Path = prepared_video_result.prepared_path
        metadata: VideoProbeResult = prepared_video_result.metadata
        should_cleanup: bool = prepared_video_result.should_cleanup
        actual_resolution: tuple[int, int] = (metadata.width, metadata.height)

        if reencode_to_av1:
            if expected_resolution is None:
                expected_resolution = actual_resolution
            elif actual_resolution != expected_resolution:
                if should_cleanup:
                    prepared_path.unlink(missing_ok=True)
                raise ValueError(
                    f"Video {entry.source_path} has resolution {actual_resolution} which does not match "
                    f"the expected resolution {expected_resolution}."
                )

        log_video(
            video_path=prepared_path,
            video_log_path=entry.video_log_path,
            timeline=timeline,
        )
        logged_entity: Path = entry.video_log_path
        logged_video_entities.append(logged_entity)

        if should_cleanup:
            prepared_path.unlink(missing_ok=True)

    return logged_video_entities


def create_ingest_view(
    *,
    exo_video_log_paths: list[Path] | None,
    ego_video_log_paths: list[Path] | None,
    quest_video_log_paths: list[Path] | None,
) -> ContainerLike:
    """
    Assemble a Rerun container/view showing exo videos along the bottom row and ego videos on the right column.

    Paths should point to the `.../pinhole` entities that contain the actual video nodes.
    """

    main_view = rrb.Spatial3DView(origin="/")

    combined_ego_paths: list[Path] = []
    if ego_video_log_paths:
        combined_ego_paths.extend(ego_video_log_paths)
    if quest_video_log_paths:
        combined_ego_paths.extend(quest_video_log_paths)

    if combined_ego_paths:
        ego_views = [
            rrb.Tabs(
                rrb.Spatial2DView(origin=str(pinhole_log_path)),
            )
            for pinhole_log_path in combined_ego_paths
        ]
        main_view = rrb.Horizontal(
            contents=[
                main_view,
                rrb.Vertical(contents=ego_views),
            ],
            column_shares=[4, 1],
        )

    if exo_video_log_paths:
        exo_views = [
            rrb.Tabs(
                rrb.Spatial2DView(origin=str(pinhole_log_path)),
            )
            for pinhole_log_path in exo_video_log_paths
        ]
        main_view = rrb.Vertical(
            contents=[
                main_view,
                rrb.Horizontal(contents=exo_views),
            ],
            row_shares=[4, 1],
        )

    return main_view


def main(config: IngestConfig) -> None:
    validate_exoego_dir(config.exoego_dir)
    print(f"Ingesting data from {config.exoego_dir} to RRD at {config.exoego_dir}")

    parent_log_path: Path = Path("/world")
    timeline: str = "video_time"
    # set the coordinate system for the entire world
    rr.log("/", rr.ViewCoordinates.RDF, static=True)
    # set time to 0 at the start of the timeline
    rr.set_time(timeline=timeline, duration=0)
    dir_tuple: tuple[Path | None, Path | None] = validate_exoego_dir(config.exoego_dir)
    exo_dir: Path | None = dir_tuple[0]
    ego_dir: Path | None = dir_tuple[1]

    exo_entries: list[VideoIngestEntry] = (
        collect_video_entries(
            video_dir=exo_dir,
            log_root=parent_log_path / "exo",
        )
        if exo_dir is not None
        else []
    )
    ego_entries: list[VideoIngestEntry] = (
        collect_video_entries(
            video_dir=ego_dir,
            log_root=parent_log_path / "ego",
        )
        if ego_dir is not None
        else []
    )

    quest_pinhole_paths: list[Path] | None = None
    quest_dir: Path = config.exoego_dir / "quest"
    if quest_dir.exists():
        quest_config = Quest3OakDVisualizeConfig(rr_config=config.rr_config, data_dir=config.exoego_dir)
        quest_pinhole_paths = load_and_log_quest_data(quest_config, timeline=timeline)

    ingest_view: ContainerLike = create_ingest_view(
        exo_video_log_paths=[entry.pinhole_log_path for entry in exo_entries] or None,
        ego_video_log_paths=[entry.pinhole_log_path for entry in ego_entries] or None,
        quest_video_log_paths=quest_pinhole_paths,
    )
    rr.send_blueprint(rrb.Blueprint(ingest_view, collapse_panels=True))

    if exo_entries:
        ingest_video_directory(
            video_entries=exo_entries,
            timeline=timeline,
            verbose=config.verbose,
            progress_label="Ingesting exo videos",
            reencode_to_av1=config.reencode_to_av1,
        )

    if ego_entries:
        # TODO make sure that the ego calibration is in meters not millimeters
        ingest_video_directory(
            video_entries=ego_entries,
            timeline=timeline,
            verbose=config.verbose,
            progress_label="Ingesting ego videos",
            reencode_to_av1=config.reencode_to_av1,
        )


def entrypoint() -> None:
    """Entrypoint leveraging Tyro to expose the ingest workflow via CLI."""

    tyro.extras.set_accent_color("bright_cyan")
    config: IngestConfig = tyro.cli(
        IngestConfig,
        description="Given a directory with ego/exo recordings, save them to RRD and visualize them with Rerun.",
    )
    main(config=config)


if __name__ == "__main__":
    entrypoint()
