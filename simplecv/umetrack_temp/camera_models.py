from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from jaxtyping import Float


@dataclass(slots=True)
class FisheyeCameraParameter:
    """Fisheye camera intrinsics and extrinsics for UmeTrack recordings."""

    name: str
    """Identifier for logging and debugging."""
    width: int = 0
    """Image width in pixels."""
    height: int = 0
    """Image height in pixels."""
    fx: float = 0.0
    """Horizontal focal length in pixels."""
    fy: float = 0.0
    """Vertical focal length in pixels."""
    cx: float = 0.0
    """Principal point x-coordinate in pixels."""
    cy: float = 0.0
    """Principal point y-coordinate in pixels."""
    extrinsic_r: Float[np.ndarray, "3 3"] = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    """Camera rotation matrix (cam_T_world[:3, :3])."""
    extrinsic_t: Float[np.ndarray, "3"] = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    """Camera translation vector (cam_T_world[:3, 3])."""
    _dist_coeff_k: Float[np.ndarray, "6"] = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    """Radial distortion coefficients k1-k6."""
    _dist_coeff_p: Float[np.ndarray, "2"] = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    """Tangential distortion coefficients p1-p2."""

    def set_intrinsic(
        self,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> None:
        """Update the intrinsic parameters for the camera."""
        self.width = int(width)
        self.height = int(height)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)

    def intrinsic33(self) -> Float[np.ndarray, "3 3"]:
        """Return the 3x3 intrinsic calibration matrix."""
        intrinsic: Float[np.ndarray, "3 3"] = np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return intrinsic

    def set_dist_coeff(
        self,
        dist_coeff_k: Iterable[float],
        dist_coeff_p: Iterable[float],
    ) -> None:
        """Store radial (k*) and tangential (p*) distortion coefficients."""
        padded_k: Float[np.ndarray, "6"] = np.zeros(6, dtype=np.float32)
        padded_p: Float[np.ndarray, "2"] = np.zeros(2, dtype=np.float32)

        for idx, value in enumerate(dist_coeff_k):
            if idx >= 6:
                break
            padded_k[idx] = float(value)

        for idx, value in enumerate(dist_coeff_p):
            if idx >= 2:
                break
            padded_p[idx] = float(value)

        self._dist_coeff_k = padded_k
        self._dist_coeff_p = padded_p

    def get_dist_coeff(self) -> Float[np.ndarray, "8"]:
        """Return distortion coefficients ordered for radial+tangential mixing."""
        coeffs: Float[np.ndarray, "8"] = np.array(
            [
                self._dist_coeff_k[0],
                self._dist_coeff_k[1],
                self._dist_coeff_p[0],
                self._dist_coeff_p[1],
                self._dist_coeff_k[2],
                self._dist_coeff_k[3],
                self._dist_coeff_k[4],
                self._dist_coeff_k[5],
            ],
            dtype=np.float32,
        )
        return coeffs

    def set_KRT(
        self,
        K: Float[np.ndarray, "3 3"] | None,
        R: Float[np.ndarray, "3 3"] | None,
        T: Float[np.ndarray, "3"] | None,
    ) -> None:
        """Update calibration matrix and/or extrinsics from optional inputs."""
        if K is not None:
            self.fx = float(K[0, 0])
            self.fy = float(K[1, 1])
            self.cx = float(K[0, 2])
            self.cy = float(K[1, 2])

        if R is not None:
            self.extrinsic_r = np.array(R, dtype=np.float32)

        if T is not None:
            self.extrinsic_t = np.array(T, dtype=np.float32)

    def save_intrinsics(self, output_path: Path) -> None:
        """Persist intrinsic parameters for debugging purposes."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        intrinsic: Float[np.ndarray, "3 3"] = self.intrinsic33()
        np.save(output_path, intrinsic)


@dataclass(slots=True)
class PinholeCameraParameter:
    """Minimal pinhole camera parameter representation for crop generation."""

    K: Float[np.ndarray, "3 3"]
    """Camera intrinsic matrix."""
    R: Float[np.ndarray, "3 3"]
    """Rotation matrix representing cam_T_world orientation."""
    T: Float[np.ndarray, "3"]
    """Translation vector representing cam_T_world position."""
    height: int
    """Output image height in pixels."""
    width: int
    """Output image width in pixels."""
    world2cam: bool = True
    """Whether the extrinsics are stored as world_T_cam (mirrors XRPrimer API)."""
    convention: str = "opencv"
    """Extrinsic convention tag (kept for compatibility)."""
    _dist_coeff: Float[np.ndarray, "8"] = field(default_factory=lambda: np.zeros(8, dtype=np.float32))
    """Radial+tangential distortion coefficients (zeros for crops)."""

    def __post_init__(self) -> None:
        self.K = np.array(self.K, dtype=np.float32)
        self.R = np.array(self.R, dtype=np.float32)
        self.T = np.array(self.T, dtype=np.float32)
        self.height = int(self.height)
        self.width = int(self.width)

    def intrinsic33(self) -> Float[np.ndarray, "3 3"]:
        """Return the 3x3 intrinsic calibration matrix."""
        return self.K

    def get_dist_coeff(self) -> Float[np.ndarray, "8"]:
        """Return distortion coefficients (all zeros for crops)."""
        return self._dist_coeff

    def get_extrinsic_r(self) -> Float[np.ndarray, "3 3"]:
        """Return the rotation component of the extrinsic transform."""
        return self.R

    def get_extrinsic_t(self) -> Float[np.ndarray, "3"]:
        """Return the translation component of the extrinsic transform."""
        return self.T

    @property
    def extrinsic_r(self) -> Float[np.ndarray, "3 3"]:
        """Alias rotation matrix for compatibility with logging helpers."""
        return self.R

    @property
    def extrinsic_t(self) -> Float[np.ndarray, "3"]:
        """Alias translation vector for compatibility with logging helpers."""
        return self.T

    def set_KRT(
        self,
        K: Float[np.ndarray, "3 3"] | None,
        R: Float[np.ndarray, "3 3"] | None,
        T: Float[np.ndarray, "3"] | None,
    ) -> None:
        """Update intrinsic or extrinsic matrices, mirroring XRPrimer's API."""
        if K is not None:
            self.K = np.array(K, dtype=np.float32)
        if R is not None:
            self.R = np.array(R, dtype=np.float32)
        if T is not None:
            self.T = np.array(T, dtype=np.float32)
