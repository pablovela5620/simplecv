from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rerun as rr
from jaxtyping import Float32, Int
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates

from simplecv.data.ego.base_ego import BaseEgoSequence, EgoData
from simplecv.data.ego.ego100k_ego import Ego100KEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig


@dataclass
class Egocentric100KConfig(BaseExoEgoDatasetConfig):
    _target: type = field(default_factory=lambda: Ego100KSequence)
    repo_id: str = "builddotai/Egocentric-100K"
    factory_id: str = "factory001"
    worker_id: str = "worker001"
    download_root: Path | None = None
    tmp_dir: Path | None = None
    load_labels: bool = True


class Ego100KSequence(BaseExoEgoSequence[Egocentric100KConfig]):
    """Egocentric-100K adapter (ego-only, no labels, no exo)."""

    def __getitem__(self, idx: int) -> EgoData:
        assert self.ego_sequence is not None, "Ego sequence not initialized."
        return self.ego_sequence[idx]

    def __len__(self) -> int:
        if self.ego_sequence is None:
            return 0
        return len(self.ego_sequence)

    def _build_ego(self) -> BaseEgoSequence[Egocentric100KConfig] | None:
        return Ego100KEgoSequence(cfg=self.config)

    def _build_exo(self) -> BaseExoSequence[Egocentric100KConfig] | None:
        return None

    def load_labels(self) -> ExoEgoLabels | None:
        """Return NaN-filled COCO-133 stack to satisfy viewer expectations."""
        if self.ego_sequence is None:
            return None
        num_frames: int = len(self.ego_sequence)
        xyzc_stack: Float32[ndarray, "num_frames 133 4"] = np.full(
            (num_frames, 133, 4),
            np.nan,
            dtype=np.float32,
        )
        xyzc_stack[..., 3] = np.float32(0.0)
        timestamps_ns: Int[ndarray, "num_frames"] | None = None
        return ExoEgoLabels(
            xyzc_stack=xyzc_stack,
            timestamps_ns=timestamps_ns,
        )

    @classmethod
    def iter_episode_sequences(cls, cfg: Egocentric100KConfig) -> Generator["Ego100KSequence", None, None]:
        """Yield exactly one sequence for the configured factory/worker."""
        yield cls(cfg)

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        return rr.ViewCoordinates.RDF

    @property
    def image_plane_distance(self) -> int | float:
        """Get the image plane distance for the camera."""
        if self.ego_sequence is not None:
            return self.ego_sequence.image_plane_distance
        return 0.05
