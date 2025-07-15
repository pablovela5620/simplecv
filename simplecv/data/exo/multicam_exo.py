from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Float32
from numpy import ndarray
from serde.json import from_json

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.conversion_utils import NerfstudioData
from simplecv.data.exo.base_exo import BaseExoSequence, ExoBatchData
from simplecv.ops.conventions import CameraConventions, convert_pose

if TYPE_CHECKING:
    from simplecv.data.exoego.multicam import MulticamConfig


class MulticamExoSequence(BaseExoSequence):
    config: "MulticamConfig"

    def __len__(self) -> int:
        assert len(self._video_path_list) > 0, "No videos found."
        # Make sure all cameras have the same number of images
        return len(self.exo_video_readers)

    def __getitem__(self, idx: int) -> None:
        # bgr_list: list[UInt8[ndarray, "H W 3"]] = self.exo_video_readers[idx]
        # if self.config.load_labels:
        #     xyz: Float32[ndarray, "2 21 3"] = self.exo_batch_data.xyz_stack[idx]
        #     uv_dict: dict[str, Float32[ndarray, "2 21 2"]] = {
        #         cam_name: uv_stack[idx] for cam_name, uv_stack in self.exo_batch_data.uv_stack_dict.items()
        #     }
        # else:
        #     xyz = None
        #     uv_dict = None
        # return ExoData(
        #     cam_params_list=self.exo_cam_list,
        #     bgr_list=bgr_list,
        #     xyz=xyz,
        #     uv_dict=uv_dict,
        # )
        return None

    def load_video_paths(self) -> list[Path]:
        """Load the paths to the video files."""
        videos_dir: Path = self.config.root_directory / self.config.sequence_name / "multicam-videos"
        video_path_list = list(videos_dir.glob("*.mp4"))
        return sorted(video_path_list)

    def load_exo_cams(self) -> list[PinholeParameters]:
        """Load the exocentric cameras for a sequence."""
        sequence_path: Path = self.config.root_directory / self.config.sequence_name
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
    def depth_paths(self) -> None:
        """Return depth paths if available; currently not implemented."""
        return None

    @property
    def image_plane_distance(self) -> int | float:
        """Get the image plane distance for the camera."""
        return 25

    # @property
    # def depth_paths(self) -> list[dict[ExoCameraIDs, Path]]:
    #     return self._depth_paths
