from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from simplecv.camera_parameters import PinholeParameters
from simplecv.image_types import BGRList


@dataclass
class ExoData:
    cam_params_list: list[PinholeParameters]
    bgr_list: BGRList


@dataclass
class ExoDataBatch:
    cam_params_list: list[PinholeParameters]
    bgr_list: BGRList


class BaseExoEgoSequence(ABC):
    def __init__(
        self, data_path: Path, sequence_name: str, user_id: str | None = None
    ) -> None:
        self._exo_cam_list: list[PinholeParameters] = self.load_exo_cameras(
            data_path, sequence_name, user_id
        )

    @abstractmethod
    def __iter__(self) -> ExoData:
        pass

    @abstractmethod
    def load_video_paths(self) -> list[Path]:
        """Load the paths to the video files."""
        pass

    @abstractmethod
    def load_exo_cameras(
        self, data_path: Path, sequence_name: str, user_id: str | None = None
    ) -> list[PinholeParameters]:
        pass

    @property
    def exo_cam_list(self) -> list[PinholeParameters]:
        """Get the list of exocentric cameras."""
        return self._exo_cam_list
