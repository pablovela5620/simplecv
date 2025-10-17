from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
from jaxtyping import Float32, Int, UInt8
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates
from rerun_bindings import Recording

from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.ego.rrd_ego import RRDEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exo.rrd_exo import RRDExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, EnvironmentMesh, ExoEgoLabels
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
        recording_cached: Recording | None = None
        if self.exo_sequence is not None:
            recording_cached = getattr(self.exo_sequence, "_recording", None)
        if recording_cached is None:
            recording_cached = rr.dataframe.load_recording(str(rrd_path))
        recording: Recording = recording_cached
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

    def load_environment_mesh(self) -> EnvironmentMesh | None:
        """Load the static environment mesh from the recording, if any."""
        rrd_path: Path = self.config.rrd_path
        if not rrd_path.exists():
            return None

        recording_cached: Recording | None = None
        if self.exo_sequence is not None:
            recording_cached = getattr(self.exo_sequence, "_recording", None)
        if recording_cached is None:
            recording_cached = rr.dataframe.load_recording(str(rrd_path))
        recording: Recording = recording_cached
        schema: Any = recording.schema()
        entity_path: str = "world/gt/env_mesh"

        available_components: set[str] = self._available_mesh_components(schema, entity_path)
        if "Mesh3D:vertex_positions" not in available_components or "Mesh3D:triangle_indices" not in available_components:
            return None

        selectors: list[str] = [
            f"{entity_path}:Mesh3D:vertex_positions",
            f"{entity_path}:Mesh3D:triangle_indices",
        ]
        include_normals: bool = "Mesh3D:vertex_normals" in available_components
        include_colors: bool = "Mesh3D:vertex_colors" in available_components
        if include_normals:
            selectors.append(f"{entity_path}:Mesh3D:vertex_normals")
        if include_colors:
            selectors.append(f"{entity_path}:Mesh3D:vertex_colors")

        candidate_timelines: list[str | None] = []
        if self.exo_sequence is not None:
            candidate_timelines.append(getattr(self.exo_sequence, "_video_timeline", None))
        candidate_timelines.extend(["video_time", "log_time", "log_tick"])

        examined: set[str | None] = set()
        for timeline in candidate_timelines:
            if timeline is None or timeline in examined:
                continue
            examined.add(timeline)
            try:
                view = recording.view(index=timeline, contents=entity_path)
            except ValueError:
                continue

            samples: list[dict[str, Any]] = self._read_mesh_samples_from_view(
                view=view,
                timeline=timeline,
                selectors=selectors,
            )

            for sample in samples:
                positions = self._parse_vertex_positions(sample.get(f"{entity_path}:Mesh3D:vertex_positions"))
                triangles = self._parse_triangle_indices(sample.get(f"{entity_path}:Mesh3D:triangle_indices"))
                if positions is None or triangles is None:
                    continue

                normals = self._parse_vertex_normals(
                    sample.get(f"{entity_path}:Mesh3D:vertex_normals"),
                    expected_vertices=len(positions),
                )
                colors = self._parse_vertex_colors(
                    sample.get(f"{entity_path}:Mesh3D:vertex_colors"),
                    expected_vertices=len(positions),
                )

                return EnvironmentMesh(
                    vertex_positions=positions,
                    triangle_indices=triangles,
                    vertex_normals=normals,
                    vertex_colors=colors,
                )
        return None

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

    @staticmethod
    def _available_mesh_components(schema: Any, entity_path: str) -> set[str]:
        components: set[str] = set()
        for descriptor in schema.component_columns():
            entity = getattr(descriptor, "entity_path", None)
            component = getattr(descriptor, "component", None)
            if entity is None or component is None:
                continue
            entity_str = str(entity).lstrip("/")
            if entity_str == entity_path:
                components.add(str(component))
        return components

    @staticmethod
    def _parse_vertex_positions(entry: Any) -> Float32[ndarray, "num_vertices 3"] | None:
        if entry is None:
            return None
        positions = np.asarray(entry, dtype=np.float32)
        positions = np.squeeze(positions)
        if positions.ndim != 2 or positions.shape[1] != 3:
            return None
        return np.ascontiguousarray(positions.astype(np.float32), dtype=np.float32)

    @staticmethod
    def _parse_triangle_indices(entry: Any) -> Int[ndarray, "num_faces 3"] | None:
        if entry is None:
            return None
        triangles = np.asarray(entry, dtype=np.int32)
        triangles = np.squeeze(triangles)
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            return None
        return np.ascontiguousarray(triangles.astype(np.int32), dtype=np.int32)

    @staticmethod
    def _parse_vertex_normals(
        entry: Any,
        *,
        expected_vertices: int,
    ) -> Float32[ndarray, "num_vertices 3"] | None:
        if entry is None:
            return None
        normals = np.asarray(entry, dtype=np.float32)
        normals = np.squeeze(normals)
        if normals.ndim != 2 or normals.shape[1] != 3:
            return None
        normals = normals[:expected_vertices]
        return np.ascontiguousarray(normals.astype(np.float32), dtype=np.float32)

    @staticmethod
    def _parse_vertex_colors(
        entry: Any,
        *,
        expected_vertices: int,
    ) -> UInt8[ndarray, "num_vertices 4"] | None:
        if entry is None:
            return None
        colors_np = np.asarray(entry)
        if colors_np.size == 0:
            return None

        colors_np = np.squeeze(colors_np)

        if colors_np.ndim == 1:
            colors_uint32 = colors_np.astype(np.uint32, copy=False)
            colors = np.empty((colors_uint32.shape[0], 4), dtype=np.uint8)
            colors[:, 0] = (colors_uint32 >> 24) & 0xFF
            colors[:, 1] = (colors_uint32 >> 16) & 0xFF
            colors[:, 2] = (colors_uint32 >> 8) & 0xFF
            colors[:, 3] = colors_uint32 & 0xFF
        elif colors_np.ndim == 2 and colors_np.shape[1] in (3, 4):
            if np.issubdtype(colors_np.dtype, np.floating):
                try:
                    max_value = float(np.nanmax(colors_np))
                except ValueError:
                    max_value = 1.0
                if max_value <= 1.0:
                    colors_np = np.nan_to_num(colors_np, nan=0.0)
                    colors_np = np.clip(colors_np, 0.0, 1.0) * 255.0
            colors_np = colors_np.astype(np.uint8, copy=False)
            if colors_np.shape[1] == 3:
                alpha = np.full((colors_np.shape[0], 1), 255, dtype=np.uint8)
                colors = np.concatenate([colors_np, alpha], axis=1)
            else:
                colors = colors_np
        else:
            return None

        if colors.shape[0] > expected_vertices:
            colors = colors[:expected_vertices]
        return np.ascontiguousarray(colors.astype(np.uint8), dtype=np.uint8)

    @staticmethod
    def _read_mesh_samples_from_view(
        *,
        view: Any,
        timeline: str,
        selectors: list[str],
    ) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []

        try:
            static_reader = view.select_static(*selectors)
        except ValueError:
            static_reader = None

        if static_reader is not None:
            table_static: Any = static_reader.read_all()
            if table_static is not None and table_static.num_rows > 0:
                column_data = {
                    selector: table_static.column(idx).combine_chunks().to_pylist()
                    for idx, selector in enumerate(selectors)
                }
                for row_idx in range(table_static.num_rows):
                    samples.append({selector: column_data[selector][row_idx] for selector in selectors})
                return samples

        try:
            table_dynamic: Any = view.select(timeline, *selectors).read_all()
        except ValueError:
            table_dynamic = None

        if table_dynamic is None or table_dynamic.num_rows == 0:
            return samples

        column_data = {
            selector: column.combine_chunks().to_pylist()
            for selector, column in zip(selectors, table_dynamic.columns[1:], strict=True)
        }
        for row_idx in range(table_dynamic.num_rows):
            samples.append({selector: column_data[selector][row_idx] for selector in selectors})
        return samples

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
