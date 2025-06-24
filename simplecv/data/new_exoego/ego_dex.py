from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import rerun as rr
from jaxtyping import Float, Float32
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.data.exoego.skeleton.avp_fullbody import AVP_ID2NAME, AVP_IDS, avp_to_coco_hands
from simplecv.data.new_exoego.base_ego import BaseEgoDatasetConfig, BaseEgoSequence, EgoData, EgoLabels
from simplecv.ops.triangulate import proj_3d_vectorized
from simplecv.video_utils import reencode_video_optimal
from einops import rearrange


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

    def load_labels(self) -> EgoLabels:
        """Load labels for the sequence, if applicable."""
        # In this case, we are not loading any labels.
        # This method can be extended in the future if needed.
        ego_labels: EgoLabels = self._parse_joints()
        return ego_labels

    def _parse_joints(self):
        sequence_path: Path = self.config.root_directory / self.config.split / self.config.sequence_name
        hdf5_path: Path = sequence_path / f"{self.config.episode}.hdf5"
        assert hdf5_path.exists(), f"Path {hdf5_path} does not exist."

        with h5py.File(f"{hdf5_path}", "r") as h5py_file:
            # contains intrinsics that are right now manually set
            # camera = h5py_file["camera"]
            transforms = h5py_file["transforms"]
            joints_list: list[Float32[ndarray, "n_frames 3"]] = []
            for joint_name in AVP_ID2NAME.values():
                joint_transform: Float32[ndarray, "n_frames 4 4"] = transforms.get(joint_name)[:]
                joint_xyz: Float32[ndarray, "n_frames 3"] = joint_transform[:, :3, 3]
                joints_list.append(joint_xyz)

            xyz_stack: Float32[ndarray, "n_frames 68 3"] = np.stack(joints_list, axis=1)

            try:
                confidences = h5py_file["confidences"]
                conf_list: list[Float32[ndarray, "n_frames 3"]] = []
                for joint_name in AVP_ID2NAME.values():
                    conf: Float32[ndarray, "n_frames"] = confidences.get(joint_name)[:]  # noqa: UP037
                    conf_list.append(conf)

                conf_stack: Float32[ndarray, "n_frames 68"] = np.stack(conf_list, axis=1)
                conf_stack: Float32[ndarray, "n_frames 68 1"] = rearrange(
                    conf_stack, "n_frames n_joints -> n_frames n_joints 1"
                )
            except KeyError:
                conf_stack: Float32[ndarray, "n_frames 68 1"] = np.ones(
                    (xyz_stack.shape[0], xyz_stack.shape[1], 1), dtype=np.float32
                )  # default confidence of 1.0 for all joints

        # convert from AVP to COCO 133
        xyz_coco_stack, conf_coco_stack = avp_to_coco_hands(xyz_avp=xyz_stack, conf_avp=conf_stack)
        xyzc_stack: Float32[ndarray, "n_frames 133 4"] = np.concatenate([xyz_coco_stack, conf_coco_stack], axis=-1)
        # homogeneous coordinates for projection
        # xyz_hom_stack: Float32[ndarray, "n_frames 68 4"] = np.concatenate(
        #     [xyz_stack, np.ones_like(xyz_stack[..., :1])], axis=-1
        # )

        # project 3D points to 2D using the camera parameters
        # ego_cam_dict = self.ego_cam_dict
        # # there should be only one camera in the dict, assert that
        # assert len(ego_cam_dict) == 1, f"Expected single camera, got {len(ego_cam_dict)}"
        # pinhole_params: PinholeParameters = next(iter(ego_cam_dict.values()))[0]
        # P: Float32[ndarray, "3 4"] = pinhole_params.projection_matrix.astype(np.float32)
        # # n_views 3 4, in this case only one view since its not a multicamera dataset
        # Pall: Float32[ndarray, "1 3 4"] = P[np.newaxis, ...]

        # uv_stack: Float32[ndarray, "n_frames 1 68 2"] = proj_3d_vectorized(xyz_hom=xyz_hom_stack, P=Pall)

        # # --- mark 2‑D points that project outside the image bounds BEFORE we
        # #     fuse them with confidences so that the NaNs propagate ---
        # uv_stack[..., 0] = np.where(
        #     (uv_stack[..., 0] < 0) | (uv_stack[..., 0] > pinhole_params.intrinsics.width),
        #     np.nan,
        #     uv_stack[..., 0],
        # )
        # uv_stack[..., 1] = np.where(
        #     (uv_stack[..., 1] < 0) | (uv_stack[..., 1] > pinhole_params.intrinsics.height),
        #     np.nan,
        #     uv_stack[..., 1],
        # )

        # uvc_stack: Float32[ndarray, "n_frames 1 68 3"] = np.concatenate(
        #     [uv_stack, conf_stack[:, np.newaxis, ...]], axis=-1
        # )

        return EgoLabels(xyzc_stack=xyzc_stack)

    def __getitem__(self, idx) -> EgoData:
        return EgoData()

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        """Get mapping from joint ID to joint name."""
        return rr.ViewCoordinates.RUB

    @property
    def image_plane_distance(self) -> int | float:
        """Get the image plane distance for the camera."""
        return 0.075
