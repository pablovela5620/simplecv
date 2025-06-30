from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args

import numpy as np
import rerun as rr
from jaxtyping import Float32
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates
from serde.yaml import from_yaml

from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.ego.hocap_ego import ExoCameraIDs, HocapEgoSequence, HOCapExtrinsicsData
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exo.hocap_exo import HocapExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig
from simplecv.data.skeleton.coco_133 import LEFT_HAND_IDX, RIGHT_HAND_IDX


@dataclass
class HocapConfig(BaseExoEgoDatasetConfig):
    _target: type = field(default_factory=lambda: HocapSequence)
    root_directory: Path = Path("/mnt/8tb/data/hocap/datasets")
    split: Literal["train", "val", "test"] | None = None
    subject_id: str = "8"
    sequence_name: str = "20231024_180733"


class HocapSequence(BaseExoEgoSequence):
    config: HocapConfig

    def __getitem__(self, idx):
        return None

    def _build_ego(self) -> BaseEgoSequence | None:
        return HocapEgoSequence(cfg=self.config)

    def _build_exo(self) -> BaseExoSequence | None:
        return HocapExoSequence(cfg=self.config)

    def load_labels(self) -> ExoEgoLabels:
        """Load labels for the sequence, if applicable."""
        # 2D keypoints are not available for HoloLens, so we will load 3D labels from the first camera
        calibration_path: Path = self.config.root_directory / "calibration"
        extrinsics_directory: Path = calibration_path / "extrinsics"

        extrinsic_yaml: Path = extrinsics_directory / "extrinsics_20231014.yaml"
        assert extrinsic_yaml.exists(), f"Path {extrinsic_yaml} does not exist."

        extri_hocap: HOCapExtrinsicsData = from_yaml(HOCapExtrinsicsData, extrinsic_yaml.read_text())

        label_path: Path = self.config.root_directory / f"subject_{self.config.subject_id}" / self.config.sequence_name
        assert label_path.exists(), f"Path {label_path} does not exist."
        # hololens does not contain labels, so we need to load the 3d labels from any  other camera
        cam_name: ExoCameraIDs = get_args(ExoCameraIDs)[0]  # Assuming the first camera is the HoloLens
        world_T_cam: Float32[ndarray, "4 4"] = extri_hocap.world_T_cam_dict.get(cam_name)
        cam_dir: Path = label_path / cam_name  # Assuming the first camera is the HoloLens

        npz_paths: list[Path] = sorted(cam_dir.glob("*.npz"))
        assert len(npz_paths) > 0, f"No .npz files found in {cam_dir}. Expected at least one file."

        xyz_list: list[Float32[ndarray, "133 3"]] = []
        for npz_path in npz_paths:
            npz_data = np.load(npz_path)
            xyz_cam: Float32[ndarray, "2 21 3"] = npz_data["hand_joints_3d"]
            # convert to world coordinates
            # Assuming hand_joints_3d_cam.shape == (2, 21, 3) and world_T_cam.shape == (4, 4)
            # Create homogeneous coordinates by concatenating a ones column along the last axis
            ones: Float32[ndarray, "2 21 1"] = np.ones(
                (*xyz_cam.shape[:-1], 1),
                dtype=xyz_cam.dtype,
            )
            xyz_cam_homogeneous: Float32[ndarray, "2 21 4"] = np.concatenate([xyz_cam, ones], axis=-1)

            # filger out -1 (not detected) values
            xyz_cam_homogeneous = np.where(xyz_cam_homogeneous == -1, np.nan, xyz_cam_homogeneous)

            # Transform all joints at once using matrix multiplication.
            # The multiplication is broadcast over the first two dimensions.
            xyz_world_homogeneous: Float32[ndarray, "2 21 4"] = xyz_cam_homogeneous @ world_T_cam.T

            # Extract the 3D world coordinates (ignore the homogeneous component)
            xyz_world: Float32[ndarray, "2 21 3"] = xyz_world_homogeneous[..., :3]
            xyz_list.append(xyz_world)

        xyz_stack: Float32[ndarray, "num_frames 2 21 3"] = np.stack(xyz_list)
        right_xyz: Float32[ndarray, "num_frames 21 3"] = xyz_stack[:, 0, :, :]
        left_xyz: Float32[ndarray, "num_frames 21 3"] = xyz_stack[:, 1, :, :]

        coco_xyz_stack: Float32[ndarray, "num_frames 133 3"] = np.full(
            (len(npz_paths), 133, 3), np.nan, dtype=np.float32
        )
        # fill in the right and left hand joints
        coco_xyz_stack[:, RIGHT_HAND_IDX, :] = right_xyz
        coco_xyz_stack[:, LEFT_HAND_IDX, :] = left_xyz
        # generate a confidence stack with all ones if not nan otherwise 0
        conf_stack: Float32[ndarray, "num_frames 133 1"] = np.where(np.isnan(coco_xyz_stack), 0.0, 1.0).astype(
            np.float32
        )[..., 0:1]
        # create xyzc stack
        xyzc_stack: Float32[ndarray, "num_frames 133 4"] = np.concatenate([coco_xyz_stack, conf_stack], axis=-1)
        return ExoEgoLabels(xyzc_stack=xyzc_stack)

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        """Get mapping from joint ID to joint name."""
        return rr.ViewCoordinates.RIGHT_HAND_Z_UP

    @property
    def image_plane_distance(self) -> int | float:
        """Get the image plane distance for the camera."""
        return 0.1
