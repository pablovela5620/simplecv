from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass

from jaxtyping import Float
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates

from simplecv.camera_parameters import PinholeParameters
from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig
from simplecv.image_types import BGRList


@dataclass
class EgoData:
    cam_params_list: list[PinholeParameters]
    bgr_list: BGRList


@dataclass
class EgoLabels:
    xyzc_stack: Float[ndarray, "num_frames 133 4"]


class BaseExoEgoSequence(ABC):
    config: BaseExoEgoDatasetConfig

    def __init__(
        self,
        cfg: BaseExoEgoDatasetConfig,
    ) -> None:
        self.config: BaseExoEgoDatasetConfig = cfg
        self.ego_sequence: BaseEgoSequence | None = self._build_ego()
        self.exo_sequence: BaseExoSequence | None = self._build_exo()
        # if self.config.load_labels:
        #     self._ego_labels: EgoLabels = self.load_labels()

    def __len__(self) -> int:
        # Return the length based on the first camera's pinhole parameters list
        return 0

    def __iter__(self) -> Generator[EgoData, None, None]:
        for idx in range(len(self)):
            # Yield the result of __getitem__ for iteration
            yield self[idx]

    @abstractmethod
    def _build_ego(self) -> BaseEgoSequence | None:
        """Build the ego sequence based on the configuration."""

    @abstractmethod
    def _build_exo(self) -> BaseExoSequence | None:
        """Build the exo sequence based on the configuration."""

    @abstractmethod
    def __getitem__(self, idx: int) -> EgoData:
        """Get the EgoData for a specific index."""

    @abstractmethod
    def load_labels(self):
        """Load labels for the sequence, if applicable."""

    @property
    @abstractmethod
    def world_coordinate_system(self) -> ViewCoordinates:
        """Return the world coordinate system for the sequence."""
