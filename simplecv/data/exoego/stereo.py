from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

import rerun as rr
from rerun.components.view_coordinates import ViewCoordinates

from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.ego.stereo_ego import StereoEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exo.stereo_exo import StereoExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig


@dataclass
class StereoConfig(BaseExoEgoDatasetConfig):
    _target: type = field(default_factory=lambda: StereoSequence)
    root_directory: Path = Path("data/stereo-sample")
    sequence_name: str = "ds8"


class StereoSequence(BaseExoEgoSequence):
    config: StereoConfig

    def __getitem__(self, idx: int) -> None:
        return None

    def _build_ego(self) -> BaseEgoSequence | None:
        return StereoEgoSequence(cfg=self.config)

    def _build_exo(self) -> BaseExoSequence | None:
        # return StereoExoSequence(cfg=self.config)
        return None

    def load_labels(self) -> ExoEgoLabels | None:
        """No labels available; return an empty COCO-133 buffer to satisfy pipelines."""
        import numpy as np
        from jaxtyping import Float32
        from numpy import ndarray

        # Provide a single empty frame to satisfy downstream batching without real labels
        xyz = np.full((1, 133, 3), np.nan, dtype=np.float32)
        conf = np.zeros((1, 133, 1), dtype=np.float32)
        xyzc_stack: Float32[ndarray, "1 133 4"] = np.concatenate([xyz, conf], axis=-1)
        return ExoEgoLabels(xyzc_stack=xyzc_stack)

    @classmethod
    def iter_episode_sequences(cls, cfg: StereoConfig) -> Generator["StereoSequence", None, None]:
        """
        Iterates over all episode sequences in the dataset specified by the given configuration.

        This class method yields `MulticamSequence` instances for each sequence found in the dataset directory structure.
        It expects the dataset to be organized with subject directories named "subject_*", each containing sequence directories.

        Args:
            cfg (MulticamConfig): Configuration object specifying the root directory and other parameters.

        Yields:
            MulticamSequence: An instance for each sequence found, with configuration updated for the current subject and sequence.

        Notes:
            - Uses natural sorting for subject and sequence directories.
            - Prints subject ID and sequence name for each iteration using `icecream.ic`.
            - Pauses execution for user input after each sequence (likely for debugging).
        """
        # root: Path = cfg.root_directory

        # subject_dirs: list[Path] = natsorted([d for d in root.glob("*") if d.is_dir()])

        # # iterate through subject directories and get each sequence
        # for subj_dir in subject_dirs:
        #     seq_dirs: list[Path] = natsorted([d for d in subj_dir.iterdir() if d.is_dir()])
        #     subject_id: str = subj_dir.name.split("_")[-1]  # e.g., "8" from "subject_8"
        #     for seq_dir in seq_dirs:
        #         new_cfg = replace(
        #             cfg,
        #             subject_id=subject_id,
        #             sequence_name=seq_dir.name,
        #         )
        #         yield cls(new_cfg)
        raise NotImplementedError("MulticamSequence.iter_episode_sequences is not implemented.")

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        """Get mapping from joint ID to joint name."""
        return rr.ViewCoordinates.RIGHT_HAND_Z_DOWN

    @property
    def image_plane_distance(self) -> int | float:
        """Get the image plane distance for the camera."""
        return 0.1
