from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

import rerun as rr
from rerun.components.view_coordinates import ViewCoordinates

from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exo.rrd_exo import RRDExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig


@dataclass
class RRDExoEgoConfig(BaseExoEgoDatasetConfig):
    _target: type = field(default_factory=lambda: RRDSequence)
    rrd_path: Path = Path("/path/to/rrd/file.rrd")
    load_labels: bool = False
    # Required: .rrd file produced by tools/t265_slam.py


class RRDSequence(BaseExoEgoSequence[RRDExoEgoConfig]):
    def __getitem__(self, idx: int) -> None:
        return None

    def _build_ego(self) -> BaseEgoSequence[RRDExoEgoConfig] | None:
        return None

    def _build_exo(self) -> BaseExoSequence[RRDExoEgoConfig] | None:
        return RRDExoSequence(self.config)

    def load_labels(self) -> ExoEgoLabels | None:
        """Return an empty COCO-133 buffer with NaNs, sized to ego length."""
        raise NotImplementedError("RRD exo-ego labels not implemented yet")

    @classmethod
    def iter_episode_sequences(cls, cfg: RRDExoEgoConfig) -> Generator["RRDSequence", None, None]:
        raise NotImplementedError("RRDSequence.iter_episode_sequences is not implemented.")

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        """Get mapping from joint ID to joint name."""
        return rr.ViewCoordinates.RUB

    @property
    def image_plane_distance(self) -> int | float:
        """Get the image plane distance for the camera."""
        return 0.1
