from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
from jaxtyping import Float32, Int
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates
from rerun_bindings import Recording

from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.ego.rrd_ego import RRDEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exo.rrd_exo import RRDExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig


@dataclass
class RRDExoEgoConfig(BaseExoEgoDatasetConfig):
    _target: type = field(default_factory=lambda: RRDSequence)
    rrd_path: Path = Path("/path/to/rrd/file.rrd")
    load_labels: bool = True
    # Required: .rrd file produced by tools/t265_slam.py


class RRDSequence(BaseExoEgoSequence[RRDExoEgoConfig]):
    def __getitem__(self, idx: int) -> None:
        return None

    def __len__(self) -> int:  # type: ignore[override]
        sequence_lengths: list[int] = []
        if self.exo_sequence is not None:
            sequence_lengths.append(len(self.exo_sequence.exo_video_readers))
        if self.ego_sequence is not None:
            sequence_lengths.append(len(self.ego_sequence.ego_video_readers))
        if sequence_lengths:
            return min(sequence_lengths)
        return 0

    def _build_ego(self) -> BaseEgoSequence[RRDExoEgoConfig] | None:
        try:
            return RRDEgoSequence(self.config)
        except AssertionError as exc:
            if "No ego camera streams" in str(exc):
                return None
            raise

    def _build_exo(self) -> BaseExoSequence[RRDExoEgoConfig] | None:
        try:
            return RRDExoSequence(self.config)
        except AssertionError as exc:
            if "No exo camera streams" in str(exc):
                return None
            raise

    def load_labels(self) -> ExoEgoLabels | None:
        """Load COCO-133 3D keypoints and confidences from the RRD recording."""
        sequence_lengths: list[int] = []
        if self.exo_sequence is not None:
            sequence_lengths.append(len(self.exo_sequence.exo_video_readers))
        if self.ego_sequence is not None:
            sequence_lengths.append(len(self.ego_sequence.ego_video_readers))
        n_frames: int | None = min(sequence_lengths) if sequence_lengths else None
        if n_frames is None or n_frames <= 0:
            return None

        rrd_path: Path = self.config.rrd_path
        assert rrd_path.exists(), f"RRD path {rrd_path} does not exist"

        # Reuse the exo-side recording cache so we don't reopen the RRD.
        recording: Recording = getattr(
            self.exo_sequence, "_recording", rr.dataframe.load_recording(str(rrd_path))
        )
        schema: Any = recording.schema()
        timeline: str = getattr(
            self.exo_sequence,
            "_video_timeline",
            self._select_timeline(schema),
        )

        entity_path: str = "world/gt/coco133_xyz"
        view: Any = recording.view(index=timeline, contents=entity_path)
        # Pull both the positions and confidences so we can keep their timestamp alignment.
        table: Any = view.select(
            timeline,
            f"{entity_path}:Points3D:positions",
            f"{entity_path}:simplecv.KeypointConfidence3D:confidences",
        ).read_all()
        time_col, positions_col, confidences_col = table

        keypoint_times_ns: Int[ndarray, "n_samples"] = time_col.combine_chunks().to_numpy().astype(np.int64)
        positions_py: list[list[list[float]] | list[Any] | None] = positions_col.to_pylist()
        confidences_py: list[list[float] | None] = confidences_col.to_pylist()

        sample_count: int = min(len(keypoint_times_ns), len(positions_py), len(confidences_py))
        if sample_count == 0:
            xyzc_stack: Float32[ndarray, "n_frames 133 4"] = np.full(
                (n_frames, 133, 4), np.nan, dtype=np.float32
            )
            return ExoEgoLabels(xyzc_stack=xyzc_stack)

        frame_limit: int = min(n_frames, sample_count)

        effective_times: Int[ndarray, "frame_limit"] = keypoint_times_ns[:frame_limit]
        if frame_limit > 1:
            # Estimate the native logging period so we can map to frame indices.
            frame_period_ns: int = int(round(np.median(np.diff(effective_times))))
            frame_period_ns = max(frame_period_ns, 1)
        else:
            frame_period_ns = 1

        # Convert timestamps back into absolute video frame indices (respecting offsets).
        start_frame_idx: int = int(round(effective_times[0] / frame_period_ns)) if frame_period_ns else 0
        relative_frames: Int[ndarray, "frame_limit"] = np.round(
            (effective_times - effective_times[0]) / frame_period_ns
        ).astype(int)
        frame_indices: Int[ndarray, "frame_limit"] = start_frame_idx + relative_frames

        xyz_stack: Float32[ndarray, "n_frames 133 3"] = np.full((n_frames, 133, 3), np.nan, dtype=np.float32)
        conf_stack: Float32[ndarray, "n_frames 133"] = np.full((n_frames, 133), np.nan, dtype=np.float32)

        for sample_idx in range(frame_limit):
            frame_idx: int = int(frame_indices[sample_idx])
            if frame_idx < 0 or frame_idx >= n_frames:
                continue

            positions_entry: list[list[float]] | list[Any] | None = positions_py[sample_idx]
            confidences_entry: list[float] | None = confidences_py[sample_idx]

            if positions_entry:
                # Accept both flattened and nested representations from the Arrow table.
                positions_arr_raw: Float32[ndarray, "n 3"] | Float32[ndarray, "1 n 3"] = np.asarray(
                    positions_entry, dtype=np.float32
                )
                if positions_arr_raw.ndim == 3 and positions_arr_raw.shape[0] == 1:
                    positions_arr_raw = positions_arr_raw[0]
                if positions_arr_raw.ndim == 2 and positions_arr_raw.shape[1] == 3:
                    positions_arr: Float32[ndarray, "n 3"] = positions_arr_raw
                    keypoint_count: int = min(positions_arr.shape[0], 133)
                    xyz_stack[frame_idx, :keypoint_count, :] = positions_arr[:keypoint_count, :]

            if confidences_entry:
                # Confidence arrays mirror the positions layout but are 1-D per sample.
                confidences_arr_raw: Float32[ndarray, "n"] | Float32[ndarray, "1 n"] = np.asarray(
                    confidences_entry, dtype=np.float32
                )
                if confidences_arr_raw.ndim == 2 and confidences_arr_raw.shape[0] == 1:
                    confidences_arr_raw = confidences_arr_raw[0]
                if confidences_arr_raw.ndim == 1:
                    confidences_arr: Float32[ndarray, "n"] = confidences_arr_raw
                    keypoint_conf_count: int = min(confidences_arr.shape[0], 133)
                    conf_stack[frame_idx, :keypoint_conf_count] = confidences_arr[:keypoint_conf_count]

        xyzc_stack: Float32[ndarray, "n_frames 133 4"] = np.concatenate(
            [xyz_stack, conf_stack[..., np.newaxis]], axis=-1
        )
        return ExoEgoLabels(xyzc_stack=xyzc_stack)

    def _select_timeline(self, schema: Any) -> str:
        timeline_names: list[str] = []
        try:
            for index_col in schema.index_columns():
                timeline_name = getattr(index_col, "name", None)
                if timeline_name is None:
                    timeline_name = str(index_col)
                timeline_names.append(str(timeline_name))
        except Exception:
            pass

        preferred_order: tuple[str, ...] = ("video_time", "time", "timestamp", "frame_time")
        for candidate in preferred_order:
            if candidate in timeline_names:
                return candidate
        if timeline_names:
            return timeline_names[0]
        raise AssertionError("No timeline columns found in recording schema")

    @classmethod
    def iter_episode_sequences(cls, cfg: RRDExoEgoConfig) -> Generator["RRDSequence", None, None]:
        raise NotImplementedError("RRDSequence.iter_episode_sequences is not implemented.")

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        """Get mapping from joint ID to joint name."""
        return rr.ViewCoordinates.RFU

    @property
    def image_plane_distance(self) -> int | float:
        """Get the image plane distance for the camera."""
        if self.exo_sequence is not None:
            return self.exo_sequence.image_plane_distance
        return 0.1
