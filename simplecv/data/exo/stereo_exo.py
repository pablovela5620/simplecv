from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Float32
from numpy import ndarray
from serde import serde
from serde.json import from_json

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters, rescale_intri
from simplecv.data.exo.base_exo import BaseExoSequence

ExoCamName = Literal["p1", "p2", "p3"]


@serde
class _IntrinsicsRec:
    camera_matrix: Float32[ndarray, "3 3"]
    dist_coeffs: Float32[ndarray, "..."]
    image_size: list[int]
    reproj_rms: float


@serde
class _ExtrinsicsRec:
    rvec: Float32[ndarray, "3"]
    tvec: Float32[ndarray, "3"]
    R: Float32[ndarray, "3 3"]
    confidence: float


@serde
class ExoCalib:
    camera_name: ExoCamName
    intrinsics: _IntrinsicsRec
    extrinsics: _ExtrinsicsRec


class StereoExoSequence(BaseExoSequence):
    """Stationary exo cameras (p1, p2, p3) with fixed extrinsics."""

    def __getitem__(self, idx: int) -> None:
        return None

    def load_video_paths(self) -> list[Path]:
        exo_dir: Path = self.config.root_directory / self.config.sequence_name / "exo"
        assert exo_dir.exists(), f"Directory {exo_dir} does not exist"

        video_paths: list[Path] = []
        for cam in sorted([d for d in exo_dir.iterdir() if d.is_dir()]):
            mov_path: Path = cam / f"{cam.name}.mov"
            mp4_path: Path = cam / f"{cam.name}.mp4"
            # Clean up broken symlink if present
            if mp4_path.is_symlink() and not mp4_path.exists():
                try:
                    mp4_path.unlink()
                except OSError:
                    pass
            if mp4_path.exists():
                video_paths.append(mp4_path)
                continue
            if mov_path.exists():
                try:
                    os.symlink(str(mov_path.resolve()), mp4_path)
                except FileExistsError:
                    pass
                except OSError:
                    pass
                if mp4_path.exists():
                    video_paths.append(mp4_path)
                else:
                    video_paths.append(mov_path)
            else:
                raise FileNotFoundError(f"No video found for exo cam at {cam}")

        return video_paths

    def load_exo_cams(self) -> list[PinholeParameters]:
        exo_dir: Path = self.config.root_directory / self.config.sequence_name / "exo"
        assert exo_dir.exists(), f"Directory {exo_dir} does not exist"

        exo_cam_list: list[PinholeParameters] = []
        for cam_dir in sorted([d for d in exo_dir.iterdir() if d.is_dir()]):
            calib_path: Path = cam_dir / f"{cam_dir.name}_calibration.json"
            assert calib_path.exists(), f"Missing calibration: {calib_path}"
            calib: ExoCalib = from_json(ExoCalib, calib_path.read_text())

            K = calib.intrinsics.camera_matrix.astype(np.float32)
            width, height = int(calib.intrinsics.image_size[0]), int(calib.intrinsics.image_size[1])
            intri = Intrinsics(
                camera_conventions="RDF",
                fl_x=float(K[0, 0]),
                fl_y=float(K[1, 1]),
                cx=float(K[0, 2]),
                cy=float(K[1, 2]),
                width=width,
                height=height,
            )

            # Match intrinsics to actual video resolution if different (use OpenCV capture)
            video_path: Path | None = None
            mp4_path = cam_dir / f"{cam_dir.name}.mp4"
            mov_path = cam_dir / f"{cam_dir.name}.mov"
            if mp4_path.exists():
                video_path = mp4_path
            elif mov_path.exists():
                video_path = mov_path
            if video_path is not None:
                try:
                    import cv2

                    cap = cv2.VideoCapture(str(video_path))
                    if cap.isOpened():
                        vwidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        vheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap.release()
                        if vwidth > 0 and vheight > 0 and (vwidth != intri.width or vheight != intri.height):
                            intri = rescale_intri(intri, target_width=vwidth, target_height=vheight)
                except Exception:
                    # If OpenCV unavailable or fails, keep original calibration size
                    pass

            # OpenCV convention typically provides world->cam (R, t). Use that directly.
            R_wc: Float32[ndarray, "3 3"] = calib.extrinsics.R.astype(np.float32)
            t_wc: Float32[ndarray, "3"] = calib.extrinsics.tvec.astype(np.float32)
            extri = Extrinsics(world_R_cam=R_wc, world_t_cam=t_wc)

            exo_cam_list.append(PinholeParameters(name=calib.camera_name, intrinsics=intri, extrinsics=extri))

        return exo_cam_list

    @property
    def image_plane_distance(self) -> int | float:
        return 0.1
