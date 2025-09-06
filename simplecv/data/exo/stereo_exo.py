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
from simplecv.video_io import VideoReader
import contextlib

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
                with contextlib.suppress(OSError):
                    mp4_path.unlink()
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
            # Data format changed: prefer <cam>_calibration.json, fallback to calibration.json
            calib_path: Path = cam_dir / f"{cam_dir.name}_calibration.json"
            if not calib_path.exists():
                alt_path = cam_dir / "calibration.json"
                assert alt_path.exists(), (
                    f"Missing calibration JSON next to {cam_dir}. Tried: {calib_path.name} and {alt_path.name}"
                )
                calib_path = alt_path
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

            # Match intrinsics to actual video resolution using project helpers
            video_path: Path | None = None
            mp4_path = cam_dir / f"{cam_dir.name}.mp4"
            mov_path = cam_dir / f"{cam_dir.name}.mov"
            if mp4_path.exists():
                video_path = mp4_path
            elif mov_path.exists():
                video_path = mov_path
            if video_path is not None:
                # Map calibration K (defined for intrinsics.image_size) to the actual
                # video resolution. Same aspect ratio → use rescale_intri.
                # Different aspect ratio (letterbox/crop) → uniform scale + pad offsets
                # to preserve geometry and principal point.
                try:
                    vr = VideoReader(video_path)
                    vwidth = int(vr.width)
                    vheight = int(vr.height)
                    # If resolutions differ, adjust intrinsics.
                    if (intri.width is not None and intri.height is not None) and (
                        vwidth != intri.width or vheight != intri.height
                    ):
                        # If aspect ratio matches, simple anisotropic rescale is correct.
                        ow, oh = float(intri.width), float(intri.height)
                        if abs((vwidth / vheight) - (ow / oh)) < 1e-6:
                            intri = rescale_intri(intri, target_width=vwidth, target_height=vheight)
                        else:
                            # Letterbox/pad case: uniform scale with offsets to maintain principal point alignment
                            sw = float(vwidth) / ow
                            sh = float(vheight) / oh
                            s = min(sw, sh)
                            content_w = ow * s
                            content_h = oh * s
                            pad_x = (float(vwidth) - content_w) * 0.5
                            pad_y = (float(vheight) - content_h) * 0.5

                            intri = Intrinsics(
                                camera_conventions=intri.camera_conventions,
                                fl_x=float(intri.fl_x * s),
                                fl_y=float(intri.fl_y * s),
                                cx=float(intri.cx * s + pad_x),
                                cy=float(intri.cy * s + pad_y),
                                width=int(vwidth),
                                height=int(vheight),
                            )
                except Exception:
                    # If probe fails, keep original calibration size
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
