from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rerun as rr
from jaxtyping import Float32, Int, UInt8
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates
from rerun_bindings import Recording, RecordingView

from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.ego.rrd_ego import RRDEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exo.rrd_exo import RRDExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, EnvironmentMesh, ExoEgoLabels, ExoEgoSample
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig


@dataclass
class RRDExoEgoConfig(BaseExoEgoDatasetConfig):
    _target: type = field(default_factory=lambda: RRDSequence)
    rrd_path: Path = Path("/path/to/rrd/file.rrd")
    load_labels: bool = True
    # Required: .rrd file produced by tools/t265_slam.py


class RRDSequence(BaseExoEgoSequence[RRDExoEgoConfig]):
    _recording: Recording | None = None

    def __init__(self, cfg: RRDExoEgoConfig) -> None:
        # Load once and share with ego/exo/labels.
        self._recording = rr.dataframe.load_recording(str(cfg.rrd_path))
        super().__init__(cfg)

    def __getitem__(
        self,
        idx: int | None = None,
        ts_nano: np.timedelta64 | None = None,
    ) -> ExoEgoSample:
        """
        Fetch a time-synchronised ego/exo sample aligned to the canonical timeline.
        """
        canonical_idx, ts_ns = self._resolve_canonical(idx=idx, ts_nano=ts_nano)
        ego_cam_params_list, ego_bgr_list = self._sample_ego(ts_ns)
        exo_cam_params_list, exo_bgr_list = self._sample_exo(ts_ns)
        ego_depth_list = self._sample_ego_depths(ts_ns)
        exo_depth_list = self._sample_exo_depths(ts_ns)
        labels: ExoEgoLabels | None = self._sample_labels(canonical_idx, ts_ns)

        return ExoEgoSample(
            canonical_index=canonical_idx,
            canonical_timestamp_ns=ts_ns,
            ego_cam_params_list=ego_cam_params_list,
            ego_bgr_list=ego_bgr_list,
            ego_depth_list=ego_depth_list,
            exo_cam_params_list=exo_cam_params_list,
            exo_bgr_list=exo_bgr_list,
            exo_depth_list=exo_depth_list,
            labels=labels,
        )

    def __len__(self) -> int:  # type: ignore[override]
        return int(self.canonical_timestamps_ns.shape[0])

    def _build_ego(self) -> BaseEgoSequence[RRDExoEgoConfig] | None:
        try:
            ego_seq = RRDEgoSequence(self.config, recording=self._recording)
            return ego_seq
        except AssertionError as exc:
            if "No ego camera streams" in str(exc):
                return None
            raise

    def _build_exo(self) -> BaseExoSequence[RRDExoEgoConfig] | None:
        try:
            exo_seq = RRDExoSequence(self.config, recording=self._recording)
            return exo_seq
        except AssertionError as exc:
            if "No exo camera streams" in str(exc):
                return None
            raise

    def load_stream_timestamps_ns(self) -> dict[str, Int[ndarray, "n_frames"]]:
        """Return per-stream timestamps for ego/exo videos (and labels if present)."""

        stream_ts: dict[str, Int[ndarray, "n_frames"]] = {}
        self._ego_stream_names.clear()
        self._exo_stream_names.clear()

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

        if self.exo_sequence is not None:
            for name, video_path in zip(
                self.exo_sequence.exo_video_names,
                self.exo_sequence.exo_video_paths,
                strict=True,
            ):
                stream_name = f"exo/{name}"
                timestamps = rr.AssetVideo(path=video_path).read_frame_timestamps_nanos()
                stream_ts[stream_name] = timestamps
                self._exo_stream_names.append(stream_name)

        labels: ExoEgoLabels | None = self.exoego_labels
        if labels is not None and labels.timestamps_ns is not None:
            stream_ts["labels"] = labels.timestamps_ns

        return stream_ts

    def load_labels(self) -> ExoEgoLabels | None:
        """Load COCO-133 3D keypoints and confidences from the RRD recording."""
        rrd_path: Path = self.config.rrd_path
        assert rrd_path.exists(), f"RRD path {rrd_path} does not exist"

        if self._recording is None:
            self._recording = rr.dataframe.load_recording(str(rrd_path))
        recording: Recording = self._recording

        timeline: str = "video_time"
        entity_path: str = "world/gt/coco133_xyz"
        view: RecordingView = recording.view(index=timeline, contents=entity_path)
        # Pull both the positions and confidences so we can keep their timestamp alignment.
        df: pd.DataFrame = view.select(
            timeline,
            f"{entity_path}:Points3D:positions",
            f"{entity_path}:simplecv.KeypointConfidence3D:confidences",
        ).read_pandas()

        positions_series: pd.DataFrame | pd.Series | None = df[f"/{entity_path}:Points3D:positions"]
        confidences_series: pd.DataFrame | pd.Series | None = df[
            f"/{entity_path}:simplecv.KeypointConfidence3D:confidences"
        ]
        timestamps_series: pd.Series | None = df.get(timeline)

        if positions_series is None or confidences_series is None:
            return None

        positions_arrays: list[Float32[ndarray, "133 3"]] = [
            np.stack(entry, axis=0).astype(np.float32, copy=False) for entry in positions_series.to_numpy()
        ]
        xyz_stack: Float32[ndarray, "num_frames 133 3"] = np.stack(positions_arrays, axis=0)

        confidences_arrays: list[Float32[ndarray, "133"]] = [
            np.asarray(entry, dtype=np.float32) for entry in confidences_series.to_numpy()
        ]
        conf_stack: Float32[ndarray, "num_frames 133"] = np.stack(confidences_arrays, axis=0)

        timestamps_ns: Int[ndarray, "num_frames"] | None = None
        if timestamps_series is not None:
            timestamps_ns = np.asarray(timestamps_series.to_numpy(), dtype=np.int64)

        xyzc_stack: Float32[ndarray, "num_frames 133 4"] = np.concatenate(
            [xyz_stack, conf_stack[..., np.newaxis]],
            axis=-1,
        )
        return ExoEgoLabels(
            xyzc_stack=xyzc_stack,
            timestamps_ns=timestamps_ns,
        )

    def load_environment_mesh(self) -> EnvironmentMesh | None:
        """Load the static environment mesh from the recording, if any."""
        rrd_path: Path = self.config.rrd_path
        if not rrd_path.exists():
            return None

        recording: Recording | None = self._recording
        assert recording is not None, f"RRD recording at {rrd_path} could not be loaded."
        schema: Any = recording.schema()
        entity_path: str = "world/gt/env_mesh"

        available_components: set[str] = self._available_mesh_components(schema, entity_path)
        if (
            "Mesh3D:vertex_positions" not in available_components
            or "Mesh3D:triangle_indices" not in available_components
        ):
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
        return rr.ViewCoordinates.RUF

    @property
    def image_plane_distance(self) -> int | float:
        """Get the image plane distance for the camera."""
        if self.exo_sequence is not None:
            return self.exo_sequence.image_plane_distance
        return 0.1
