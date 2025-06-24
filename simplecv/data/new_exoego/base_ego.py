from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from jaxtyping import Float
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates

from simplecv.camera_parameters import PinholeParameters
from simplecv.configs.base_config import InstantiateConfig
from simplecv.image_types import BGRList
from simplecv.video_io import MultiVideoReader

CamNameType = TypeVar("CamNameType", bound=str)


@dataclass
class EgoData:
    cam_params_list: list[PinholeParameters]
    bgr_list: BGRList


@dataclass
class EgoLabels:
    xyzc_stack: Float[ndarray, "num_frames 133 4"]
    # uv_stack_dict: dict[str, Float[ndarray, "..."]]
    # uvc_stack: Float[ndarray, "n_frames n_views 68 3"] | None = None  # 2D landmarks for each view and frame


@dataclass
class BaseEgoDatasetConfig(InstantiateConfig):
    root_directory: Path = Path()
    load_labels: bool = True


class BaseEgoSequence(ABC):
    config: BaseEgoDatasetConfig

    def __init__(
        self,
        cfg: BaseEgoDatasetConfig,
    ) -> None:
        self.config: BaseEgoDatasetConfig = cfg
        self._ego_cam_dict: dict[CamNameType, list[PinholeParameters]] = self.load_ego_cams()
        self._video_path_list: list[Path] = self.load_video_paths()
        # Sort the cameras and videos based on the sequence to make sure they align correctly
        sorted_cams_and_videos: tuple[dict[CamNameType, list[PinholeParameters]], dict[CamNameType, Path]] = (
            self.align_cams_and_videos(video_path_list=self._video_path_list, ego_cam_dict=self._ego_cam_dict)
        )
        # validate that the number of cameras matches the number of videos and names are aligned
        assert len(sorted_cams_and_videos[0]) == len(sorted_cams_and_videos[1]), (
            f"Number of cameras ({len(sorted_cams_and_videos[0])}) does not match number of videos "
            f"({len(sorted_cams_and_videos[1])})."
        )
        assert set(sorted_cams_and_videos[0].keys()) == set(sorted_cams_and_videos[1].keys()), (
            f"Camera names {set(sorted_cams_and_videos[0].keys())} do not match video names "
            f"{set(sorted_cams_and_videos[1].keys())}."
        )
        # extract the ego camera dictionary and video paths
        self._ego_cam_dict = sorted_cams_and_videos[0]
        self._video_path_list = list(sorted_cams_and_videos[1].values())

        self.ego_video_readers: MultiVideoReader = MultiVideoReader(
            video_paths=[video_path for video_path in self._video_path_list]
        )
        if self.config.load_labels:
            self._ego_labels: EgoLabels = self.load_labels()

    def __len__(self) -> int:
        # Return the length based on the first camera's pinhole parameters list
        if self._ego_cam_dict:
            return len(next(iter(self._ego_cam_dict.values())))
        return 0

    def __iter__(self) -> Generator[EgoData, None, None]:
        for idx in range(len(self)):
            # Yield the result of __getitem__ for iteration
            yield self[idx]

    @abstractmethod
    def __getitem__(self, idx: int) -> EgoData:
        pass

    @abstractmethod
    def load_video_paths(self) -> list[Path]:
        """Load the paths to the video files."""
        pass

    @abstractmethod
    def load_ego_cams(self) -> dict[str, list[PinholeParameters]]:
        pass

    @abstractmethod
    def load_labels(self) -> EgoLabels:
        """Load labels for the sequence, if applicable."""
        pass

    @abstractmethod
    def align_cams_and_videos(
        self, video_path_list: list[Path], ego_cam_dict: dict[CamNameType, list[PinholeParameters]]
    ) -> tuple[dict[CamNameType, list[PinholeParameters]], dict[CamNameType, Path]]:
        """Align cameras and videos based on the sequence."""
        pass

    @property
    def ego_cam_dict(self) -> dict[str, list[PinholeParameters]]:
        """Get the dictionary of egocentric cameras."""
        return self._ego_cam_dict

    @property
    def ego_labels(self) -> EgoLabels:
        """Get the dictionary of egocentric cameras."""
        return self._ego_labels

    @property
    @abstractmethod
    def world_coordinate_system(self) -> ViewCoordinates:
        pass

    @property
    @abstractmethod
    def image_plane_distance(self) -> int | float:
        pass
