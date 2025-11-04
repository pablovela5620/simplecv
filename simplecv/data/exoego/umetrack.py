from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
from jaxtyping import Float32
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates

from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.ego.umetrack_ego import UmeTrackEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig


@dataclass
class UmeTrackConfig(BaseExoEgoDatasetConfig):
    _target: type = field(default_factory=lambda: UmeTrackSequence)
    root_directory: Path = Path("/mnt/8tb/data/umetrack-split")
    data_type: Literal["synthetic", "real"] = "real"
    split: Literal["training", "testing"] = "training"
    hand_interaction: Literal["separate_hand", "hand_hand"] = "separate_hand"
    user_name: str = "user_15"
    recording_id: int = 0


class UmeTrackSequence(BaseExoEgoSequence[UmeTrackConfig]):
    """Assembly101 dataset adapter with 3D annotations expressed in meters."""

    def __getitem__(self, idx) -> None:
        return None

    def _build_ego(self) -> BaseEgoSequence[UmeTrackConfig] | None:
        return UmeTrackEgoSequence(cfg=self.config)

    def _build_exo(self) -> BaseExoSequence[UmeTrackConfig] | None:
        return None

    def load_labels(self) -> ExoEgoLabels:
        """Load COCO-133 hand keypoints in meters for the current sequence."""

        xyzc_stack: Float32[ndarray, "num_frames 133 4"] = np.full((1, 133, 4), np.nan, dtype=np.float32)

        return ExoEgoLabels(
            xyzc_stack=xyzc_stack,
        )

    @classmethod
    def iter_episode_sequences(cls, cfg: UmeTrackConfig) -> Generator["UmeTrackSequence", None, None]:
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
        raise NotImplementedError("UmeTrack dataset does not support iterating over multiple sequences.")

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        return rr.ViewCoordinates.BUL

    @property
    def image_plane_distance(self) -> int | float:
        """Get the image plane distance for the camera in meters."""
        return 0.035
