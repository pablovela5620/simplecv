"""Aria Gen2 Pilot dataset adapter for the ExoEgo visualization pipeline.

Ego-only (Aria device, no exo cameras). Uses preprocessed AV1 MP4 videos
(from VRS), MPS SLAM trajectories for camera poses. MPS hand tracking
provides wrist+palm positions only (not full finger keypoints), so labels
are not mapped to COCO-133.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import rerun as rr
from jaxtyping import Int
from natsort import natsorted
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates

from simplecv.data.ego.aria_gen2_pilot_ego import AriaGen2PilotEgoSequence
from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels, ExoEgoSample
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig


@dataclass
class AriaGen2PilotConfig(BaseExoEgoDatasetConfig):
    """Configuration for Aria Gen2 Pilot sequences."""

    _target: type = field(default_factory=lambda: AriaGen2PilotSequence)
    base_directory: Path = Path("/mnt/8tb/data/aria-gen2-pilot")
    """Base directory containing sequence subdirectories."""
    sequence_name: str = "walk_1"
    """Sequence folder name (e.g. 'walk_1', 'cook_0', 'eat_0')."""


class AriaGen2PilotSequence(BaseExoEgoSequence[AriaGen2PilotConfig]):
    """Aria Gen2 Pilot dataset adapter (ego-only, Aria device)."""

    def __init__(self, cfg: AriaGen2PilotConfig) -> None:
        self._ego_stream_names: list[str] = []
        self._exo_stream_names: list[str] = []
        super().__init__(cfg)

    def _sequence_dir(self) -> Path:
        return Path(self.config.base_directory) / self.config.sequence_name

    def __getitem__(self, idx: int | None = None, ts_nano: np.timedelta64 | None = None) -> ExoEgoSample:
        canonical_idx, ts_ns = self._resolve_canonical(idx=idx, ts_nano=ts_nano)
        ego_cam_params_list, ego_bgr_list = self._sample_ego(ts_ns)
        labels: ExoEgoLabels | None = self._sample_labels(canonical_idx, ts_ns)

        return ExoEgoSample(
            canonical_index=canonical_idx,
            canonical_timestamp_ns=ts_ns,
            ego_cam_params_list=ego_cam_params_list,
            ego_bgr_list=ego_bgr_list,
            exo_cam_params_list=None,
            exo_bgr_list=None,
            labels=labels,
        )

    def _build_ego(self) -> BaseEgoSequence[AriaGen2PilotConfig] | None:
        return AriaGen2PilotEgoSequence(cfg=self.config)

    def _build_exo(self) -> BaseExoSequence[AriaGen2PilotConfig] | None:
        return None  # ego-only

    def load_stream_timestamps_ns(self) -> dict[str, Int[ndarray, "n_frames"]]:
        """Return per-stream timestamps for ego videos."""
        stream_ts: dict[str, Int[ndarray, "n_frames"]] = {}
        self._ego_stream_names.clear()

        if self.ego_sequence is not None:
            for name, video_path in zip(
                self.ego_sequence.ego_video_names,
                self.ego_sequence.ego_video_paths,
                strict=True,
            ):
                stream_name: str = f"ego/{name}"
                timestamps: Int[ndarray, "n_frames"] = rr.AssetVideo(path=video_path).read_frame_timestamps_nanos()
                stream_ts[stream_name] = timestamps
                self._ego_stream_names.append(stream_name)

        return stream_ts

    def load_labels(self) -> ExoEgoLabels | None:
        """MPS hand tracking provides wrist+palm only, not COCO-133 keypoints.

        Return None since the data doesn't map to the expected 133-keypoint format.
        """
        return None

    @classmethod
    def iter_episode_sequences(cls, cfg: AriaGen2PilotConfig) -> Generator["AriaGen2PilotSequence", None, None]:
        """Iterate over all sequences in the dataset directory.

        Yields one ``AriaGen2PilotSequence`` per folder with preprocessed MP4s.
        """
        root: Path = cfg.base_directory
        assert root.exists(), f"Aria Gen2 Pilot root directory {root} does not exist."

        def _has_preprocessed_streams(d: Path) -> bool:
            simplecv_dir: Path = d / "_simplecv"
            return simplecv_dir.is_dir() and any(simplecv_dir.glob("*.mp4"))

        seq_dirs: list[Path] = natsorted([
            d for d in root.iterdir()
            if d.is_dir() and _has_preprocessed_streams(d)
        ])

        for seq_dir in seq_dirs:
            episode_cfg: AriaGen2PilotConfig = replace(
                cfg,
                sequence_name=seq_dir.name,
            )
            try:
                yield cls(episode_cfg)
            except Exception as exc:  # pragma: no cover
                print(f"[skip] {seq_dir.name}: {exc}")

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        """Aria: gravity = [0,0,-9.81] → +Z is up."""
        return rr.ViewCoordinates.RIGHT_HAND_Z_UP
