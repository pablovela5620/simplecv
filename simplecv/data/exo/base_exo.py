from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from jaxtyping import Float32
from numpy import ndarray

from simplecv.camera_parameters import PinholeParameters
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig
from simplecv.image_types import BGRList
from simplecv.video_io import MultiVideoReader

CamNameType = TypeVar("CamNameType", bound=str)


@dataclass
class ManoStack:
    # https://github.com/rerun-io/pi0-lerobot/blob/a2f7f415f48379349bb05c88893c99f07c5209e3/src/pi0_lerobot/apis/visualize_exo_ego.py
    betas: Float32[ndarray, "10"]  # only a single set for all frames and hands
    # 0 for right hand, 1 for left hand, p 0:48, t 48:51
    poses: Float32[ndarray, "num_frames 2 51"]  # 2 hands 51 angles (3*17)


@dataclass
class ExoData:
    cam_params_list: list[PinholeParameters]
    bgr_list: BGRList
    # assumes left | right hand
    xyz: Float32[ndarray, "2 21 3"] | None
    uv_dict: dict[str, Float32[ndarray, "2 21 2"]] | None


@dataclass
class ExoBatchData:
    uv_stack_dict: dict[str, Float32[ndarray, "num_frames 2 21 2"]]
    xyz_stack: Float32[ndarray, "num_frames 2 21 3"]
    mano_stack: ManoStack | None = None


class BaseExoSequence(ABC):
    config: BaseExoEgoDatasetConfig

    def __init__(
        self,
        cfg: BaseExoEgoDatasetConfig,
    ) -> None:
        self.config: BaseExoEgoDatasetConfig = cfg
        self._video_path_list: list[Path] = self.load_video_paths()
        self._exo_cam_list: list[PinholeParameters] = self.load_exo_cams()
        self.exo_video_readers: MultiVideoReader = MultiVideoReader(
            video_paths=[video_path for video_path in self._video_path_list]
        )

    def __len__(self) -> int:
        # Return the length based on the first camera's pinhole parameters list
        if self._exo_cam_list:
            return len(next(iter(self._exo_cam_list.values())))
        return 0

    def __iter__(self) -> Generator[ExoData, None, None]:
        for idx in range(len(self)):
            # Yield the result of __getitem__ for iteration
            yield self[idx]

    @abstractmethod
    def __getitem__(self, idx: int) -> ExoData:
        pass

    @abstractmethod
    def load_video_paths(self) -> list[Path]:
        """Load the paths to the video files."""
        pass

    @abstractmethod
    def load_exo_cams(self) -> list[PinholeParameters]:
        pass

    @property
    def exo_cam_list(self) -> list[PinholeParameters]:
        """Get the dictionary of egocentric cameras."""
        return self._exo_cam_list

    @property
    @abstractmethod
    def image_plane_distance(self) -> int | float:
        pass
