from collections.abc import Generator
from pathlib import Path

import numpy as np
import rerun as rr
from jaxtyping import Float32, UInt8
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates
from serde.json import from_json

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.conversion_utils import NerfstudioData
from simplecv.data.exoego.base_exo_ego import (
    BaseExoEgoSequence,
    ExoBatchData,
    ExoData,
)
from simplecv.data.skeleton.mediapipe import (
    MEDIAPIPE_ID2NAME,
    MEDIAPIPE_IDS,
    MEDIAPIPE_LINKS,
)
from simplecv.ops.conventions import CameraConventions, convert_pose


class MulticamSequence(BaseExoEgoSequence):
    def __init__(
        self,
        data_path: Path,
        sequence_name: str,
        subject_id: str | None = None,
        load_labels: bool = False,
    ) -> None:
        super().__init__(data_path, sequence_name, subject_id, load_labels)

    def __len__(self) -> int:
        assert len(self.video_path_list) > 0, "No videos found."
        # Make sure all cameras have the same number of images
        return len(self.exo_video_readers)

    def __iter__(self) -> Generator[ExoData, None, None]:
        for idx in range(len(self)):
            # Yield the result of __getitem__ for iteration
            yield self[idx]

    def __getitem__(self, idx: int) -> ExoData:
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range for sequence of length {len(self)}")

        bgr_list: list[UInt8[ndarray, "480 640 3"]] = self.exo_video_readers[idx]
        if self.load_labels:
            xyz: Float32[ndarray, "2 21 3"] = self.exo_batch_data.xyz_stack[idx]
            uv_dict: dict[str, Float32[ndarray, "2 21 2"]] = {
                cam_name: uv_stack[idx] for cam_name, uv_stack in self.exo_batch_data.uv_stack_dict.items()
            }
        else:
            xyz = None
            uv_dict = None
        return ExoData(cam_params_list=self.exo_cam_list, bgr_list=bgr_list, xyz=xyz, uv_dict=uv_dict)

    def load_exo_batch_data(self, data_path: Path, sequence_name: str, subject_id: str | None) -> ExoBatchData:
        raise NotImplementedError(
            "load_exo_batch_data is not implemented for MulticamSequence, use --no-load-labels flag"
        )

    def load_video_paths(self, data_path: Path, sequence_name: str, subject_id: str | None = None) -> list[Path]:
        videos_dir: Path = data_path / sequence_name / "multicam-videos"
        video_path_list = list(videos_dir.glob("*.mp4"))
        return sorted(video_path_list)

    def load_exo_cameras(
        self, data_path: Path, sequence_name: str, subject_id: str | None = None
    ) -> list[PinholeParameters]:
        """Load the exocentric cameras for a sequence."""
        sequence_path: Path = data_path / sequence_name
        exo_cam_json: Path = sequence_path / "transforms.json"
        assert exo_cam_json.exists(), f"Path {exo_cam_json} does not exist."
        nerfstudio_data: NerfstudioData = from_json(
            NerfstudioData,
            exo_cam_json.read_text(),
        )
        # right now assuming all cameras have the same intri, but this is not always true
        intri = Intrinsics(
            camera_conventions="RDF",
            fl_x=nerfstudio_data.fl_x,
            fl_y=nerfstudio_data.fl_y,
            cx=nerfstudio_data.cx,
            cy=nerfstudio_data.cy,
            height=nerfstudio_data.h,
            width=nerfstudio_data.w,
        )

        exo_cam_list: list[PinholeParameters] = []
        for idx, frame in enumerate(nerfstudio_data.frames):
            # Load the camera extrinsics
            world_T_cam_gl: Float32[ndarray, "4 4"] = frame.world_T_cam_gl.astype(np.float32)
            world_T_cam_cv: Float32[ndarray, "4 4"] = convert_pose(
                world_T_cam_gl, CameraConventions.GL, CameraConventions.CV
            )
            extri = Extrinsics(
                world_R_cam=world_T_cam_cv[:3, :3],
                world_t_cam=world_T_cam_cv[:3, 3],
            )
            # Create a PinholeParameters object for each camera
            exo_cam: PinholeParameters = PinholeParameters(
                name=f"cam_{idx}",
                intrinsics=intri,
                extrinsics=extri,
                distortion=None,
            )
            exo_cam_list.append(exo_cam)

        return exo_cam_list

    @property
    def hand_links(self) -> tuple[tuple[int, int], ...]:
        return MEDIAPIPE_LINKS

    @property
    def hand_ids(self) -> list[int]:
        return MEDIAPIPE_IDS

    @property
    def hand_id2name(self) -> dict[int, str]:
        return MEDIAPIPE_ID2NAME

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        """Get mapping from joint ID to joint name."""
        return rr.ViewCoordinates.RIGHT_HAND_Z_DOWN

    @property
    def depth_paths(self) -> list[dict[str, Path]] | None:
        return None
