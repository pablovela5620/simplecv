from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import rerun as rr
from rerun.components.view_coordinates import ViewCoordinates

from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.ego.ego_dex import EgoDexSequence as EgoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig


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

    def load_labels(self):
        """Load labels for the sequence, if applicable."""
        pass

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        """Get mapping from joint ID to joint name."""
        return rr.ViewCoordinates.RUB

    @property
    def image_plane_distance(self) -> int | float:
        """Get the image plane distance for the camera."""
        return 0.075
