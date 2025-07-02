from collections.abc import Generator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import rerun as rr
from einops import rearrange
from jaxtyping import Float32
from natsort import natsorted
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates

from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.ego.ego_dex import EgoDexSequence as EgoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig
from simplecv.data.skeleton.avp_fullbody import AVP_ID2NAME, avp_to_coco_hands


@dataclass
class EgoDexConfig(BaseExoEgoDatasetConfig):
    _target: type = field(default_factory=lambda: EgoDexSequence)
    root_directory: Path = Path("/home/pablo/0Dev/data/ego-dex")
    split: Literal["train", "val", "test"] = "test"
    sequence_name: str = "add_remove_lid"
    episode: int = 0


class EgoDexSequence(BaseExoEgoSequence):
    config: EgoDexConfig

    def __getitem__(self, idx):
        return None

    def _build_ego(self) -> BaseEgoSequence | None:
        return EgoSequence(cfg=self.config)

    def _build_exo(self) -> BaseEgoSequence | None:
        return None

    def load_labels(self) -> ExoEgoLabels | None:
        """Load labels for the sequence, if applicable."""
        xyzc_stack: Float32[ndarray, "n_frames 133 4"] = self._parse_joints()
        return ExoEgoLabels(xyzc_stack=xyzc_stack)

    def _parse_joints(self) -> Float32[ndarray, "n_frames 133 4"]:
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

        return xyzc_stack

    @classmethod
    def iter_episode_sequences(cls, cfg: EgoDexConfig) -> Generator["EgoDexSequence", None, None]:
        """
        Iterates over all episode sequences in the dataset specified by the given configuration.

        This class method yields `EgoDexSequence` instances for each sequence found in the dataset directory structure.
        It expects the dataset to be organized with subject directories named "subject_*", each containing sequence directories.

        Args:
            cfg (EgoDexConfig): Configuration object specifying the root directory and other parameters.

        Yields:
            EgoDexSequence: An instance for each sequence found, with configuration updated for the current subject and sequence.

        Notes:
            - Uses natural sorting for subject and sequence directories.
            - Prints subject ID and sequence name for each iteration using `icecream.ic`.
            - Pauses execution for user input after each sequence (likely for debugging).
        """
        root: Path = cfg.root_directory
        sequence_dirs: list[Path] = natsorted(
            [d for d in (root / cfg.split).iterdir() if d.is_dir()],
        )
        for sequence_dir in sequence_dirs:
            episodes_int: list[int] = natsorted([int(episode.stem) for episode in sequence_dir.glob("*.hdf5")])
            for episode in episodes_int:
                new_cfg = replace(
                    cfg,
                    sequence_name=sequence_dir.name,
                    episode=episode,
                )

                yield cls(new_cfg)

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        """Get mapping from joint ID to joint name."""
        return rr.ViewCoordinates.RUB

    @property
    def image_plane_distance(self) -> int | float:
        """Get the image plane distance for the camera."""
        return 0.075
