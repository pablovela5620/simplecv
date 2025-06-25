from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import rerun as rr
from rerun.components.view_coordinates import ViewCoordinates

from simplecv.data.ego.assembly101_ego import Assembly101EgoSequence
from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.exo.assembly101_exo import Assembly101ExoSequence
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig


@dataclass
class Assembly101Config(BaseExoEgoDatasetConfig):
    _target: type = field(default_factory=lambda: Assembly101Sequence)
    root_directory: Path = Path("/mnt/8tb/data/assembly101-original/")
    split: Literal["train", "val", "test"] | None = None
    subject_id: str | None = None
    sequence_name: str = "nusar-2021_action_both_9081-a30_9081_user_id_2021-02-12_155525"  # "nusar-2021_action_both_9012-c07c_9012_user_id_2021-02-01_164345"


class Assembly101Sequence(BaseExoEgoSequence):
    config: Assembly101Config

    def __getitem__(self, idx):
        return None

    def _build_ego(self) -> BaseEgoSequence | None:
        return Assembly101EgoSequence(cfg=self.config)

    def _build_exo(self) -> BaseExoSequence | None:
        return Assembly101ExoSequence(cfg=self.config)

    def load_labels(self):
        """Load labels for the sequence, if applicable."""
        pass

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        return rr.ViewCoordinates.BUL

    @property
    def image_plane_distance(self) -> int | float:
        """Get the image plane distance for the camera."""
        return 35
