from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import rerun as rr
from rerun.components.view_coordinates import ViewCoordinates

from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.ego.hocap_ego import HocapEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exo.hocap_exo import HocapExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig


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

    def load_labels(self):
        """Load labels for the sequence, if applicable."""
        pass

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        """Get mapping from joint ID to joint name."""
        return rr.ViewCoordinates.RIGHT_HAND_Z_UP

    @property
    def image_plane_distance(self) -> int | float:
        """Get the image plane distance for the camera."""
        return 0.1
