from __future__ import annotations

import atexit
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from simplecv.camera_parameters import PinholeParameters
from simplecv.data.exo.base_exo import BaseExoSequence

if TYPE_CHECKING:
    from simplecv.data.exoego.rrd_exoego import RRDExoEgoConfig
else:  # pragma: no cover - runtime alias to avoid circular import
    from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig as RRDExoEgoConfig
ExoCamName = Literal["p1", "p2", "p3"]


class RRDExoSequence(BaseExoSequence[RRDExoEgoConfig]):

    def __getitem__(self, idx: int) -> None:
        return None

    def load_video_paths(self) -> list[Path]:
        # first lets check that the video files exist in the correct paths
        rrd_path: Path = self.config.rrd_path

        # Create a temporary directory to hold remuxed videos; cleaned on exit
        self._remux_tmpdir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(prefix="stereo_ego_remux_")
        left_mp4: Path = Path(self._remux_tmpdir.name) / "left.mp4"
        right_mp4: Path = Path(self._remux_tmpdir.name) / "right.mp4"
        atexit.register(self._remux_tmpdir.cleanup)
        raise NotImplementedError("RRD exo videos not implemented yet")

    def load_exo_cams(self) -> list[PinholeParameters]:
        raise NotImplementedError("RRD exo cameras not implemented yet")

    @property
    def image_plane_distance(self) -> int | float:
        return 0.1
