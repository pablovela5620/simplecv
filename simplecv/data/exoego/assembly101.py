import json
from collections.abc import Generator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
from jaxtyping import Float32
from natsort import natsorted
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates
from serde import field as serde_field
from serde import from_dict, serde
from tqdm import tqdm

from simplecv.data.ego.assembly101_ego import Assembly101EgoSequence
from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.exo.assembly101_exo import Assembly101ExoSequence
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig
from simplecv.data.skeleton.assembly_hands import assembly21_to_coco133
from simplecv.video_utils import Resolution


@serde
class Hand3DKeypoints:
    # Use the "rename" parameter to indicate that the JSON key "0" should map to serde_field "left"
    left: Float32[ndarray, "21 3"] = serde_field(rename="0")
    # And similarly for "1" -> "right"
    right: Float32[ndarray, "21 3"] = serde_field(rename="1")


@dataclass
class Assembly101Config(BaseExoEgoDatasetConfig):
    _target: type = field(default_factory=lambda: Assembly101Sequence)
    root_directory: Path = Path("/mnt/8tb/data/assembly101-original/")
    split: Literal["train", "val", "test"] | None = None
    sequence_name: str = "nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724"  # "nusar-2021_action_both_9012-c07c_9012_user_id_2021-02-01_164345"
    resize: Resolution | None = None  # Resize the video to this resolution, if None, no resizing is done.
    encoding: Literal["h264", "av1", "av1-720-new"] = "av1-720-new"  # Encoding format of the video files.


class Assembly101Sequence(BaseExoEgoSequence[Assembly101Config]):
    """Assembly101 dataset adapter with 3D annotations expressed in meters."""

    def __getitem__(self, idx):
        return None

    def _build_ego(self) -> BaseEgoSequence[Assembly101Config] | None:
        return Assembly101EgoSequence(cfg=self.config)

    def _build_exo(self) -> BaseExoSequence[Assembly101Config] | None:
        return Assembly101ExoSequence(cfg=self.config)

    def load_labels(self) -> ExoEgoLabels:
        """Load COCO-133 hand keypoints in meters for the current sequence."""
        ### Load 3D keypoints ###
        landmarks3d_dir: Path = self.config.root_directory / "assembly101_camera_and_hand_poses" / "landmarks3D"
        assert landmarks3d_dir.exists(), f"Directory {landmarks3d_dir} does not exist"
        xyz_json_path: Path = landmarks3d_dir / f"{self.config.sequence_name}.json"
        assert xyz_json_path.exists(), f"File {xyz_json_path} does not exist"
        with open(xyz_json_path) as f:
            all_xyz_dict: dict[str, dict[str, list[list[float]]]] = json.loads(f.read())

        # sort all_3d_landmarks by frame number
        all_xyz_dict = dict(sorted(all_xyz_dict.items(), key=lambda item: int(item[0])))

        all_xyz_dict: dict[int, Hand3DKeypoints] = {
            int(k): from_dict(Hand3DKeypoints, v) for k, v in all_xyz_dict.items()
        }

        xyz_stack_list: list[Float32[ndarray, "2 21 3"]] = []
        for frame_number, _ in enumerate(tqdm(all_xyz_dict)):
            keypoints: Hand3DKeypoints = all_xyz_dict[frame_number]
            xyz_stack_list.append(np.stack((keypoints.left, keypoints.right), axis=0, dtype=np.float32))

        # Concatenate keypoints from all frames vertically to get a (num_frames 21, 3) array.
        xyz_stack_mm: Float32[ndarray, "num_frames 2 21 3"] = np.stack(xyz_stack_list, axis=0)
        num_frames = xyz_stack_mm.shape[0]

        # Convert millimeter coordinates provided by the dataset to meters.
        xyz_stack: Float32[ndarray, "num_frames 2 21 3"] = xyz_stack_mm * np.float32(1e-3)

        xyzc_stack: Float32[ndarray, "num_frames 133 4"] = np.full((num_frames, 133, 4), np.nan, dtype=np.float32)
        for f in range(num_frames):
            xyzc_stack[f] = assembly21_to_coco133(xyz_stack[f])

        return ExoEgoLabels(
            xyzc_stack=xyzc_stack,
        )

    @classmethod
    def iter_episode_sequences(cls, cfg: Assembly101Config) -> Generator["Assembly101Sequence", None, None]:
        """
        Iterates over all episode sequences in the dataset specified by the given configuration.

        This class method yields `Assembly101Sequence` instances for each sequence found in the dataset directory structure.
        It expects the dataset to be organized with subject directories named "subject_*", each containing sequence directories.

        Args:
            cfg (Assembly101Config): Configuration object specifying the root directory and other parameters.

        Yields:
            Assembly101Sequence: An instance for each sequence found, with configuration updated for the current subject and sequence.

        Notes:
            - Uses natural sorting for subject and sequence directories.
            - Prints subject ID and sequence name for each iteration using `icecream.ic`.
            - Pauses execution for user input after each sequence (likely for debugging).
        """
        root: Path = cfg.root_directory
        videos_dir: Path = root / "videos" / "av1"
        assert videos_dir.exists(), f"Directory {videos_dir} does not exist"

        sequence_dirs: list[Path] = natsorted(
            [d for d in videos_dir.iterdir() if d.is_dir()],
        )
        for sequence_dir in sequence_dirs:
            # print(sequence_dir.name)  # Optionally use logging here
            new_cfg = replace(
                cfg,
                sequence_name=sequence_dir.name,
            )

            try:
                seq = cls(new_cfg)  # may raise
            except Exception as e:
                print(f"[skip] {sequence_dir.name}: {e}")
                continue  # go on to the next directory
            else:
                yield seq

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        return rr.ViewCoordinates.BUL

    @property
    def image_plane_distance(self) -> int | float:
        """Get the image plane distance for the camera in meters."""
        return 0.035
