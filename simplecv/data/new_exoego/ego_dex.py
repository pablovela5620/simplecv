from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import rerun as rr
from jaxtyping import Float, Float32
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates
from scipy.spatial.transform import Rotation as R
from serde import serde
from serde.yaml import from_yaml

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.data.new_exoego.base_ego import BaseEgoDatasetConfig, BaseEgoSequence, EgoData
from simplecv.video_utils import reencode_video_optimal


@dataclass
class EgoDexConfig(BaseEgoDatasetConfig):
    _target: type = field(default_factory=lambda: EgoDexSequence)
    root_directory: Path = Path("/home/pablo/0Dev/data/ego-dex")
    split: Literal["train", "val", "test"] = "test"
    sequence_name: str = "add_remove_lid"
    episode: int = 0


class EgoDexSequence(BaseEgoSequence):
    config: EgoDexConfig

    def load_video_paths(self) -> list[Path]:
        sequence_path: Path = self.config.root_directory / self.config.split / self.config.sequence_name
        video_path: Path = sequence_path / f"{self.config.episode}.mp4"
        assert video_path.exists(), f"Path {video_path} does not exist."

        new_video_path: Path = reencode_video_optimal(input_video_path=video_path)

        return [new_video_path]

    def load_ego_cams(self) -> dict[str, list[PinholeParameters]]:
        sequence_path: Path = self.config.root_directory / self.config.split / self.config.sequence_name
        hdf5_path: Path = sequence_path / f"{self.config.episode}.hdf5"

        assert hdf5_path.exists(), f"Path {hdf5_path} does not exist."

        with h5py.File(f"{hdf5_path}", "r") as h5py_file:
            # contains intrinsics that are right now manually set
            # camera = h5py_file["camera"]
            transforms = h5py_file["transforms"]

            # there are some problems with the intrinsics files in hdf5, they're always the same so set to a default
            # fmt: off
            intrinsics: Float[ndarray, "3 3"] = np.array(
                [[736.6339, 0.0, 960.0],
                [0.0, 736.6339, 540.0],
                [0.0, 0.0, 1.0]]).astype(np.float32)
            # fmt: on

            fl_x: float = float(intrinsics[0, 0])
            fl_y: float = float(intrinsics[1, 1])
            cx: float = float(intrinsics[0, 2])
            cy: float = float(intrinsics[1, 2])

            world_T_camera: Float[ndarray, "n_frames 4 4"] = transforms.get("camera")[:]
            ego_cam_list: list[PinholeParameters] = []
            cam_name = "avp_camera"
            for i in range(world_T_camera.shape[0]):
                pinhole = PinholeParameters(
                    name=cam_name,
                    intrinsics=Intrinsics(fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy, camera_conventions="RDF"),
                    extrinsics=Extrinsics(world_R_cam=world_T_camera[i][:3, :3], world_t_cam=world_T_camera[i][:3, 3]),
                )
                ego_cam_list.append(pinhole)

        ego_cam_dict: dict[str, list[PinholeParameters]] = {cam_name: ego_cam_list}

        return ego_cam_dict

    def align_cams_and_videos(
        self, video_path_list: list[Path], ego_cam_dict: dict[str, list[PinholeParameters]]
    ) -> tuple[dict[str, list[PinholeParameters]], dict[str, Path]]:
        """Align cameras and videos based on the sequence."""
        assert len(video_path_list) == 1, f"Expected single video, got {len(video_path_list)}"
        assert len(ego_cam_dict) == 1, f"Expected single camera, got {len(ego_cam_dict)}"

        # Get the single camera name and video path
        cam_name = list(ego_cam_dict.keys())[0]
        video_path = video_path_list[0]

        video_to_cam_map = {cam_name: video_path}

        return ego_cam_dict, video_to_cam_map

    # def _parse_joints_(self):
    #     joints_list: list[Float32[ndarray, "n_frames 3"]] = []
    #     for joint_name in AVP_ID2NAME.values():
    #         joint_transform: Float32[ndarray, "n_frames 4 4"] = transforms.get(joint_name)[:]
    #         joint_xyz: Float32[ndarray, "n_frames 3"] = joint_transform[:, :3, 3]
    #         joints_list.append(joint_xyz)

    #     joints_xyz: Float32[ndarray, "n_frames 68 3"] = np.stack(joints_list, axis=1)

    def __getitem__(self, idx) -> EgoData:
        return EgoData()

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        """Get mapping from joint ID to joint name."""
        return rr.ViewCoordinates.RUB
