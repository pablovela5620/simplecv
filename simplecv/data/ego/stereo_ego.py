import atexit
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Float32
from numpy import ndarray
from rerun_bindings import Recording

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.data.ego.base_ego import BaseEgoSequence, EgoData
from simplecv.rerun_log_utils import mux_h264_to_mp4, read_h264_samples_from_rrd
from simplecv.video_io import VideoReader

if TYPE_CHECKING:
    from simplecv.data.exoego.stereo import StereoConfig

CameraName = Literal["left", "right"]

# Fixed Rerun entity/timeline names as produced by t265_slam.py
TIMELINE: str = "video_time"
LEFT_VIDEO_ENTITY: str = "t265/left/pinhole/video_stream"
RIGHT_VIDEO_ENTITY: str = "t265/right/pinhole/video_stream"
LEFT_CAM_ENTITY: str = "t265/left"
RIGHT_CAM_ENTITY: str = "t265/right"


def _to_numpy_list_of_arrays(arr: pa.ChunkedArray) -> list[np.ndarray]:
    # Handles nested types (e.g., [[9 floats]] or [[3 floats]])
    py_list = arr.combine_chunks().to_pylist()
    out: list[np.ndarray] = []
    for v in py_list:
        if isinstance(v, list) and len(v) == 1 and isinstance(v[0], list):
            v = v[0]
        out.append(np.array(v, dtype=np.float32))
    return out


class StereoEgoSequence(BaseEgoSequence):
    """
    Ego stereo sequence loader for left/right cameras.

    Input is an .rrd recording produced by t265_slam.py.
    We remux the left/right H.264 streams to temporary mp4 files and
    reconstruct per-frame PinholeParameters by querying Transform3D & Pinhole
    components from the recording via the Rerun DataFrame API.
    """

    config: "StereoConfig"

    def load_video_paths(self) -> list[Path]:
        cfg: StereoConfig = self.config
        assert cfg.rrd_path is not None, "stereo.rrd_path must be provided"
        rrd_path: Path = cfg.rrd_path

        # Create a temporary directory to hold remuxed videos; cleaned on exit
        self._remux_tmpdir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(prefix="stereo_ego_remux_")
        left_mp4: Path = Path(self._remux_tmpdir.name) / "left.mp4"
        right_mp4: Path = Path(self._remux_tmpdir.name) / "right.mp4"
        atexit.register(self._remux_tmpdir.cleanup)

        # Validate the expected timeline and entities exist in the recording
        self._validate_rrd_entities(rrd_path)

        # Remux to temp files
        times, samples = read_h264_samples_from_rrd(str(rrd_path), LEFT_VIDEO_ENTITY, TIMELINE)
        mux_h264_to_mp4(times, samples, str(left_mp4))
        times, samples = read_h264_samples_from_rrd(str(rrd_path), RIGHT_VIDEO_ENTITY, TIMELINE)
        mux_h264_to_mp4(times, samples, str(right_mp4))

        assert left_mp4.exists(), f"Expected remuxed file at {left_mp4}"
        assert right_mp4.exists(), f"Expected remuxed file at {right_mp4}"
        return [left_mp4, right_mp4]

    def load_ego_cams(self) -> dict[CameraName, list[PinholeParameters]]:
        cfg: StereoConfig = self.config
        assert cfg.rrd_path is not None, "stereo.rrd_path must be provided"

        # Load recording once
        rec: Recording = rr.dataframe.load_recording(str(cfg.rrd_path))

        def load_intrinsics(cam_entity: str) -> Intrinsics:
            cam_entity_pinhole = f"{cam_entity}/pinhole"
            view = rec.view(index=TIMELINE, contents=cam_entity_pinhole)
            # Extract K, width, height from Pinhole
            _, k_col, w_col, h_col = view.select(
                TIMELINE,
                f"{cam_entity_pinhole}:Pinhole:image_from_camera",
                f"{cam_entity_pinhole}:Pinhole:width",
                f"{cam_entity_pinhole}:Pinhole:height",
            ).read_all()
            # Static; take first
            # image_from_camera is stored as a list of fixed_size_list[9]
            k_py = k_col.combine_chunks().to_pylist()
            assert len(k_py) > 0 and k_py[0] is not None, "Missing Pinhole:image_from_camera data"
            k_item = k_py[0]
            if isinstance(k_item, list) and len(k_item) == 1 and isinstance(k_item[0], list):
                k_item = k_item[0]
            # Stored as column-major (Mat3x3). Reconstruct accordingly.
            k_arr = np.array(k_item, dtype=np.float32).reshape(3, 3, order="F")
            K: Float32[ndarray, "3 3"] = k_arr
            # width/height columns may include nulls on static rows; pick first non-null
            w_list = w_col.combine_chunks().to_pylist()
            h_list = h_col.combine_chunks().to_pylist()
            width = next((int(v) for v in w_list if v is not None), None)
            height = next((int(v) for v in h_list if v is not None), None)
            if width is None or height is None:
                # Fallback: derive from K center if unset; may be corrected later
                cx = float(K[0, 2])
                cy = float(K[1, 2])
                width = int(max(1, round(2 * cx)))
                height = int(max(1, round(2 * cy)))
            return Intrinsics(
                camera_conventions="RDF",
                fl_x=float(K[0, 0]),
                fl_y=float(K[1, 1]),
                cx=float(K[0, 2]),
                cy=float(K[1, 2]),
                width=int(width),
                height=int(height),
            )

        def load_extrinsics_series(cam_entity: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
            view = rec.view(index=TIMELINE, contents=cam_entity)
            _, t_col, R_col = view.select(
                TIMELINE,
                f"{cam_entity}:Transform3D:translation",
                f"{cam_entity}:Transform3D:mat3x3",
            ).read_all()
            t_list = _to_numpy_list_of_arrays(t_col)  # list of (3,) arrays
            R_flat_list = _to_numpy_list_of_arrays(R_col)  # list of (9,) arrays (column-major)
            R_list: list[np.ndarray] = [r.reshape(3, 3, order="F") for r in R_flat_list]
            return t_list, R_list

        left_intri: Intrinsics = load_intrinsics(LEFT_CAM_ENTITY)
        right_intri: Intrinsics = load_intrinsics(RIGHT_CAM_ENTITY)
        left_t_list, left_R_list = load_extrinsics_series(LEFT_CAM_ENTITY)
        right_t_list, right_R_list = load_extrinsics_series(RIGHT_CAM_ENTITY)

        # Build per-sample PinholeParameters lists from series
        left_cam_list: list[PinholeParameters] = []
        right_cam_list: list[PinholeParameters] = []
        for t, R in zip(left_t_list, left_R_list, strict=False):
            left_extri = Extrinsics(cam_R_world=R.astype(np.float32), cam_t_world=t.astype(np.float32))
            left_cam_list.append(PinholeParameters(name="left", intrinsics=left_intri, extrinsics=left_extri))
        for t, R in zip(right_t_list, right_R_list, strict=False):
            right_extri = Extrinsics(cam_R_world=R.astype(np.float32), cam_t_world=t.astype(np.float32))
            right_cam_list.append(PinholeParameters(name="right", intrinsics=right_intri, extrinsics=right_extri))

        return {"left": left_cam_list, "right": right_cam_list}

    def align_cams_and_videos(
        self, video_path_list: list[Path], ego_cam_dict: dict[CameraName, list[PinholeParameters]]
    ) -> tuple[dict[CameraName, list[PinholeParameters]], dict[CameraName, Path]]:
        assert len(video_path_list) == 2, f"Expected two videos, got {len(video_path_list)}"
        assert set(ego_cam_dict.keys()) == {"left", "right"}, "Expected left/right cameras in calibration"

        # Map paths by filename hint
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

        # Trim/extend per-frame camera params to match the shortest video length
        left_len: int = len(VideoReader(mapping["left"]))
        right_len: int = len(VideoReader(mapping["right"]))
        n_frames: int = min(left_len, right_len)
        left_list: list[PinholeParameters] = ego_cam_dict["left"]
        right_list: list[PinholeParameters] = ego_cam_dict["right"]
        if len(left_list) != n_frames:
            if len(left_list) < n_frames and left_list:
                # Extend by repeating the last
                last: PinholeParameters = left_list[-1]
                left_list = left_list + [last] * (n_frames - len(left_list))
            else:
                left_list = left_list[:n_frames]

        if len(right_list) != n_frames:
            if len(right_list) < n_frames and right_list:
                last = right_list[-1]
                right_list = right_list + [last] * (n_frames - len(right_list))
            else:
                right_list = right_list[:n_frames]

        ego_cam_dict = {"left": left_list, "right": right_list}
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
        cams_len: int = len(next(iter(self._ego_cam_dict.values())))
        vids_len: int = len(self.ego_video_readers)
        return min(cams_len, vids_len)

    @property
    def image_plane_distance(self) -> int | float:
        return 0.02

    # ───────────────────────── helpers ───────────────────────── #
    def _validate_rrd_entities(self, rrd_path: Path) -> None:
        """Validate that expected timeline and entities exist in the recording.

        Raises AssertionError with a helpful message if anything is missing.
        """
        rec = rr.dataframe.load_recording(str(rrd_path))
        schema = rec.schema()
        try:
            raw_index_cols = list(schema.index_columns())
            timeline_names = [str(getattr(it, "name", it)) for it in raw_index_cols]
        except Exception:
            timeline_names = []

        if TIMELINE not in set(timeline_names):
            raise AssertionError(
                f"Timeline '{TIMELINE}' not found in RRD. Available timelines: {sorted(timeline_names)}"
            )

        # Gather component column names (dict or list depending on API version)
        try:
            comp_cols_obj = schema.component_columns()  # may be dict-like or iterable
            comp_descs = list(comp_cols_obj.keys()) if isinstance(comp_cols_obj, dict) else list(comp_cols_obj)
        except Exception:
            comp_descs = []

        def _has_component(entity: str, suffix: str) -> bool:
            target_paths: tuple[str, str] = (entity, f"/{entity}")
            for d in comp_descs:
                try:
                    entity_path = getattr(d, "entity_path", None)
                    component_name = getattr(d, "component", None)
                except Exception:
                    entity_path = None
                    component_name = None
                if entity_path in target_paths and isinstance(component_name, str) and component_name.endswith(suffix):
                    return True
            return False

        missing: list[str] = []
        if not _has_component(LEFT_VIDEO_ENTITY, "VideoStream:sample"):
            missing.append(f"{LEFT_VIDEO_ENTITY} (VideoStream)")
        if not _has_component(RIGHT_VIDEO_ENTITY, "VideoStream:sample"):
            missing.append(f"{RIGHT_VIDEO_ENTITY} (VideoStream)")
        if not _has_component(f"{LEFT_CAM_ENTITY}/pinhole", "Pinhole:image_from_camera"):
            missing.append(f"{LEFT_CAM_ENTITY}/pinhole (Pinhole)")
        if not _has_component(f"{RIGHT_CAM_ENTITY}/pinhole", "Pinhole:image_from_camera"):
            missing.append(f"{RIGHT_CAM_ENTITY}/pinhole (Pinhole)")
        if not _has_component(LEFT_CAM_ENTITY, "Transform3D:translation"):
            missing.append(f"{LEFT_CAM_ENTITY} (Transform3D)")
        if not _has_component(RIGHT_CAM_ENTITY, "Transform3D:translation"):
            missing.append(f"{RIGHT_CAM_ENTITY} (Transform3D)")

        if missing:
            raise AssertionError(
                "Missing expected entities/components in RRD: "
                + ", ".join(missing)
                + ".\nCheck the recording logged by t265_slam.py and entity base path 't265'."
            )
