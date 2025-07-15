from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

import rerun as rr
from rerun.components.view_coordinates import ViewCoordinates

from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exo.multicam_exo import MulticamExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig


@dataclass
class MulticamConfig(BaseExoEgoDatasetConfig):
    _target: type = field(default_factory=lambda: MulticamSequence)
    root_directory: Path = Path("data/multicam-sample")
    sequence_name: str = "card-shuffle1"


class MulticamSequence(BaseExoEgoSequence):
    config: MulticamConfig

    def __getitem__(self, idx: int) -> None:
        return None

    def _build_ego(self) -> None:
        return None

    def _build_exo(self) -> BaseExoSequence | None:
        return MulticamExoSequence(cfg=self.config)

    def load_labels(self) -> ExoEgoLabels | None:
        """Load labels for the sequence, if applicable."""
        return None

    @classmethod
    def iter_episode_sequences(cls, cfg: MulticamConfig) -> Generator["MulticamSequence", None, None]:
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
