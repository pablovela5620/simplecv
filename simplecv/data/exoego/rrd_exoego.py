from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rerun as rr
from jaxtyping import Float32
from numpy import ndarray
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
    load_labels: bool = True
    # Required: .rrd file produced by tools/t265_slam.py


class RRDSequence(BaseExoEgoSequence[RRDExoEgoConfig]):
    def __getitem__(self, idx: int) -> None:
        return None

    def __len__(self) -> int:  # type: ignore[override]
        if self.exo_sequence is not None:
            return len(self.exo_sequence.exo_video_readers)
        return 0

    def _build_ego(self) -> BaseEgoSequence[RRDExoEgoConfig] | None:
        return None

    def _build_exo(self) -> BaseExoSequence[RRDExoEgoConfig] | None:
        return RRDExoSequence(self.config)

    def load_labels(self) -> ExoEgoLabels | None:
        """Return an empty COCO-133 buffer with NaNs, sized to ego length."""
        if self.exo_sequence is None:
            return None
        n_frames: int = len(self.exo_sequence.exo_video_readers)
        if n_frames <= 0:
            return None
        xyzc_stack: Float32[ndarray, "n_frames 133 4"] = np.full((n_frames, 133, 4), np.nan, dtype=np.float32)
        return ExoEgoLabels(xyzc_stack=xyzc_stack)

    @classmethod
    def iter_episode_sequences(cls, cfg: RRDExoEgoConfig) -> Generator["RRDSequence", None, None]:
        raise NotImplementedError("RRDSequence.iter_episode_sequences is not implemented.")

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        """Get mapping from joint ID to joint name."""
        return rr.ViewCoordinates.RFU

    @property
    def image_plane_distance(self) -> int | float:
        """Get the image plane distance for the camera."""
        if self.exo_sequence is not None:
            return self.exo_sequence.image_plane_distance
        return 0.1
