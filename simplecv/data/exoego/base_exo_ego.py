from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import rerun as rr
from jaxtyping import Float32
from numpy import ndarray

from simplecv.camera_parameters import PinholeParameters
from simplecv.image_types import BGRList
from simplecv.video_io import MultiVideoReader


@dataclass
class ManoStack:
    betas: Float32[ndarray, "10"]  # only a single set for all frames and hands
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


class BaseExoEgoSequence(ABC):
    def __init__(
        self,
        data_path: Path,
        sequence_name: str,
        subject_id: str | None = None,
        load_labels: bool = False,
    ) -> None:
        self._exo_cam_list: list[PinholeParameters] = self.load_exo_cameras(data_path, sequence_name, subject_id)
        self.video_path_list: list[Path] = self.load_video_paths(
            data_path=data_path, sequence_name=sequence_name, subject_id=subject_id
        )
        self.exo_video_readers: MultiVideoReader = MultiVideoReader(
            video_paths=[video_path for video_path in self.video_path_list]
        )
        self.load_labels = load_labels
        if self.load_labels:
            self.exo_batch_data: ExoBatchData = self.load_exo_batch_data(data_path, sequence_name, subject_id)

    def __iter__(self) -> Generator[ExoData, None, None]:
        for idx in range(len(self)):
            # Yield the result of __getitem__ for iteration
            yield self[idx]

    @abstractmethod
    def __getitem__(self, idx: int) -> ExoData:
        pass

    @abstractmethod
    def load_video_paths(self, data_path: Path, sequence_name: str, subject_id: str | None = None) -> list[Path]:
        """Load the paths to the video files."""
        pass

    @abstractmethod
    def load_exo_cameras(
        self, data_path: Path, sequence_name: str, subject_id: str | None = None
    ) -> list[PinholeParameters]:
        pass

    @abstractmethod
    def load_exo_batch_data(self, data_path: Path, sequence_name: str, subject_id: str | None = None) -> ExoBatchData:
        """Load the exocentric data for a sequence."""
        pass

    @property
    def exo_cam_list(self) -> list[PinholeParameters]:
        """Get the list of exocentric cameras."""
        return self._exo_cam_list

    @property
    @abstractmethod
    def hand_links(self) -> tuple[tuple[int, int], ...]:
        """Get the links between hand joints."""
        pass

    @property
    @abstractmethod
    def hand_ids(self) -> list[int]:
        """Get the IDs of hand joints."""
        pass

    @property
    @abstractmethod
    def hand_id2name(self) -> dict[int, str]:
        """Get mapping from joint ID to joint name."""
        pass

    @property
    @abstractmethod
    def world_coordinate_system(self) -> rr.ViewCoordinates:
        """Get mapping from joint ID to joint name."""
        pass

    @property
    @abstractmethod
    def depth_paths(self) -> list[dict[str, Path]] | None:
        """Get mapping from joint ID to joint name."""
        pass
