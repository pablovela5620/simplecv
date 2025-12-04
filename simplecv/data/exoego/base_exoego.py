from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from typing import Generic, Self, TypeVar

from jaxtyping import Float, Float32, Int, UInt8
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates

from simplecv.camera_parameters import PinholeParameters
from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence, ManoStack
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig
from simplecv.image_types import BGRList

ConfigT = TypeVar("ConfigT", bound=BaseExoEgoDatasetConfig)


@dataclass
class EgoData:
    cam_params_list: list[PinholeParameters]
    bgr_list: BGRList


@dataclass
class ExoEgoLabels:
    """3D COCO-133 annotations plus optional MANO and timing metadata.

    Attributes
    ----------
    xyzc_stack:
        3D keypoints in meters with per-joint confidence ``[x, y, z, c]``.
    timestamps_ns:
        Nanosecond timestamps aligned with ``xyzc_stack``; when provided they are
        used as-is for logging to keep original recording cadence. When ``None``
        the visualization falls back to the video timeline, matching the prior
        behaviour but risking drift when label and video frame rates differ.
    mano_stack:
        Optional MANO parameters associated with the same frames.
    """

    xyzc_stack: Float[ndarray, "num_frames 133 4"]
    timestamps_ns: Int[ndarray, "num_frames"] | None = None
    mano_stack: ManoStack | None = None


@dataclass
class EnvironmentMesh:
    vertex_positions: Float32[ndarray, "num_vertices 3"]
    triangle_indices: Int[ndarray, "num_faces 3"]
    vertex_normals: Float32[ndarray, "num_vertices 3"] | None = None
    vertex_colors: UInt8[ndarray, "num_vertices 4"] | None = None


class BaseExoEgoSequence(ABC, Generic[ConfigT]):
    config: ConfigT

    def __init__(
        self,
        cfg: ConfigT,
    ) -> None:
        self.config: ConfigT = cfg
        self.ego_sequence: BaseEgoSequence[ConfigT] | None = self._build_ego()
        self.exo_sequence: BaseExoSequence[ConfigT] | None = self._build_exo()
        if self.config.load_labels:
            self._exoego_labels: ExoEgoLabels | None = self.load_labels()
        self._environment_mesh: EnvironmentMesh | None = self.load_environment_mesh()

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
    def _build_ego(self) -> BaseEgoSequence[ConfigT] | None:
        """Build the ego sequence based on the configuration."""

    @abstractmethod
    def _build_exo(self) -> BaseExoSequence[ConfigT] | None:
        """Build the exo sequence based on the configuration."""

    @abstractmethod
    def __getitem__(self, idx: int) -> EgoData:
        """Get the EgoData for a specific index."""

    @abstractmethod
    def load_labels(self) -> ExoEgoLabels | None:
        """Load labels for the sequence, if applicable."""

    def load_environment_mesh(self) -> EnvironmentMesh | None:
        """Optional hook for loading a static environment mesh."""
        return None

    @classmethod
    @abstractmethod
    def iter_episode_sequences(cls: type[Self], cfg: ConfigT) -> Generator[Self, None, None]: ...

    @property
    @abstractmethod
    def world_coordinate_system(self) -> ViewCoordinates:
        """Return the world coordinate system for the sequence."""

    @property
    def exoego_labels(self) -> ExoEgoLabels | None:
        """Return the labels for the sequence, if available."""
        return getattr(self, "_exoego_labels", None)

    @property
    def environment_mesh(self) -> EnvironmentMesh | None:
        """Return the static environment mesh, if available."""
        return getattr(self, "_environment_mesh", None)
