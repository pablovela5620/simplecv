from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass

from jaxtyping import Float
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates

from simplecv.camera_parameters import PinholeParameters
from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence, ManoStack
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig
from simplecv.image_types import BGRList


@dataclass
class EgoData:
    cam_params_list: list[PinholeParameters]
    bgr_list: BGRList


@dataclass
class ExoEgoLabels:
    xyzc_stack: Float[ndarray, "num_frames 133 4"]
    mano_stack: ManoStack | None = None


class BaseExoEgoSequence(ABC):
    config: BaseExoEgoDatasetConfig

    def __init__(
        self,
        cfg: BaseExoEgoDatasetConfig,
    ) -> None:
        self.config: BaseExoEgoDatasetConfig = cfg
        self.ego_sequence: BaseEgoSequence | None = self._build_ego()
        self.exo_sequence: BaseExoSequence | None = self._build_exo()
        if self.config.load_labels:
            self._exoego_labels: ExoEgoLabels | None = self.load_labels()

    def __len__(self) -> int:
        # Return the length based on the first camera's pinhole parameters list
        return 0

    def __iter__(self) -> Generator[EgoData, None, None]:
        for idx in range(len(self)):
            # Yield the result of __getitem__ for iteration
            yield self[idx]

    def iter_dataset(self):
        """Sugar so you can call this on an *instance*."""
        yield from self.__class__.iter_episode_sequences(self.config)

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
    def load_labels(self) -> ExoEgoLabels | None:
        """Load labels for the sequence, if applicable."""

    @classmethod
    @abstractmethod
    def iter_episode_sequences(cls, cfg: BaseExoEgoDatasetConfig) -> Generator["BaseExoEgoSequence", None, None]: ...

    # def project_xyz(self):
    # xyz_hom: Float32[ndarray, "21 4"] = np.hstack((xyz, np.ones((21, 1)))).astype(np.float32)
    # P_ego: Float32[ndarray, "3 4"] = current_ego_cam.projection_matrix.astype(np.float32)
    # P_ego: Float32[ndarray, "1 3 4"] = rearrange(P_ego, "n m -> 1 n m")
    # uvc_ego: Float32[ndarray, "1 21 3"] = projectN3(xyz_hom, P_ego).astype(np.float32)
    # uv_ego: Float32[ndarray, "21 2"] = uvc_ego[0, :, :2]
    # uv_ego[uv_ego == -1] = np.nan

    @property
    @abstractmethod
    def world_coordinate_system(self) -> ViewCoordinates:
        """Return the world coordinate system for the sequence."""

    @property
    def exoego_labels(self) -> ExoEgoLabels | None:
        """Return the labels for the sequence, if available."""
        return getattr(self, "_exoego_labels", None)
