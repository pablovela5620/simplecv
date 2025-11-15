from __future__ import annotations

import atexit
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Float32
from numpy import ndarray
from rerun_bindings import Recording

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.rerun_log_utils import (
    get_video_cache,
    mux_h264_to_mp4,
    read_h264_samples_from_rrd,
    write_asset_video_blob,
)

if TYPE_CHECKING:
    from simplecv.data.exoego.rrd_exoego import RRDExoEgoConfig
else:  # pragma: no cover - runtime alias to avoid circular import
    from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig as RRDExoEgoConfig


@dataclass(slots=True)
class _RRDCameraStream:
    """Metadata describing an exo camera stream discovered in an RRD file."""

    name: str
    video_entity: str
    pinhole_entity: str
    transform_entity: str
    data_kind: Literal["video_stream", "asset_video"]


class RRDExoSequence(BaseExoSequence[RRDExoEgoConfig]):
    """RRD-backed exo sequence that remuxes recorded H.264 streams into mp4 assets."""

    def __getitem__(self, idx: int) -> None:
        return None

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.exo_video_readers)

    @property
    def exo_video_names(self) -> list[str]:  # type: ignore[override]
        camera_streams: list[_RRDCameraStream] | None = getattr(self, "_camera_streams", None)
        if camera_streams:
            stream_names: list[str] = [stream.name for stream in camera_streams]
            return stream_names
        return super().exo_video_names

    def load_video_paths(self) -> list[Path]:
        rrd_path: Path = self.config.rrd_path
        assert rrd_path.exists(), f"RRD path {rrd_path} does not exist"

        self._remux_tmpdir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(prefix="rrd_exo_remux_")
        atexit.register(self._remux_tmpdir.cleanup)

        self._recording: Recording = rr.dataframe.load_recording(str(rrd_path))
        schema = self._recording.schema()
        self._video_timeline: str = self._select_timeline(schema)
        self._camera_streams: list[_RRDCameraStream] = self._discover_camera_streams(schema)
        assert self._camera_streams, "No exo camera streams found in recording"

        video_cache = get_video_cache()
        # Store remuxed MP4s on disk so subsequent visualizations reuse them instantly.
        # TODO(pablo): Replace cache reuse with an RRD-backed video reader once MultiVideoReader can operate on blobs.

        video_paths: list[Path] = []
        for camera_stream in self._camera_streams:
            mp4_path: Path = Path(self._remux_tmpdir.name) / f"{camera_stream.name}.mp4"
            if video_cache is not None:
                cached_path = video_cache.get(rrd_path=rrd_path, camera_name=camera_stream.name)
                if cached_path is not None:
                    shutil.copy2(cached_path, mp4_path)
                    video_paths.append(mp4_path)
                    continue
            match camera_stream.data_kind:
                case "video_stream":
                    times, samples = read_h264_samples_from_rrd(
                        str(rrd_path), camera_stream.video_entity, self._video_timeline
                    )
                    mux_h264_to_mp4(times, samples, str(mp4_path))
                case "asset_video":
                    write_asset_video_blob(
                        self._recording,
                        timeline=self._video_timeline,
                        video_entity=camera_stream.video_entity,
                        output_path=mp4_path,
                    )
                case _:
                    raise ValueError(f"Unsupported data kind for RRD camera stream: {camera_stream.data_kind}")

            assert mp4_path.exists(), f"Expected remuxed video at {mp4_path}"
            if video_cache is not None:
                video_cache.store(rrd_path=rrd_path, camera_name=camera_stream.name, source_path=mp4_path)
            video_paths.append(mp4_path)

        return video_paths

    def load_exo_cams(self) -> list[PinholeParameters]:
        recording: Recording = getattr(self, "_recording", rr.dataframe.load_recording(str(self.config.rrd_path)))
        schema = recording.schema()
        timeline: str = getattr(self, "_video_timeline", self._select_timeline(schema))
        camera_streams: list[_RRDCameraStream] = getattr(
            self,
            "_camera_streams",
            self._discover_camera_streams(schema),
        )
        assert camera_streams, "No exo camera streams found in recording"

        exo_cams: list[PinholeParameters] = []
        for camera_stream in camera_streams:
            try:
                intrinsics = self._load_intrinsics(recording, camera_stream.pinhole_entity, timeline)
                extrinsics = self._load_extrinsics(recording, camera_stream.transform_entity, timeline)
            except ValueError as exc:
                warnings.warn(
                    f"Skipping camera '{camera_stream.name}' due to missing metadata: {exc}",
                    stacklevel=2,
                )
                continue
            exo_cams.append(PinholeParameters(name=camera_stream.name, intrinsics=intrinsics, extrinsics=extrinsics))
        return exo_cams

    def _discover_camera_streams(self, schema: Any) -> list[_RRDCameraStream]:
        component_columns = schema.component_columns()
        descriptors: list[Any] = (
            list(component_columns.keys()) if isinstance(component_columns, dict) else list(component_columns)
        )

        stream_map: dict[str, _RRDCameraStream] = {}
        for descriptor in descriptors:
            entity_path = getattr(descriptor, "entity_path", None)
            component_name = getattr(descriptor, "component", None)
            if not isinstance(component_name, str) or entity_path is None:
                continue

            entity_str = str(entity_path).lstrip("/")
            if not entity_str.startswith("world/exo"):
                continue

            data_kind: Literal["video_stream", "asset_video"] | None = None
            if component_name.endswith("VideoStream:sample"):
                data_kind = "video_stream"
            elif component_name.endswith("AssetVideo:blob"):
                data_kind = "asset_video"

            if data_kind is None:
                continue

            video_entity = entity_str
            pinhole_entity = str(PurePosixPath(video_entity).parent)
            transform_entity = str(PurePosixPath(pinhole_entity).parent)
            camera_name = PurePosixPath(transform_entity).name

            stream = stream_map.get(video_entity)
            if stream is None:
                stream_map[video_entity] = _RRDCameraStream(
                    name=camera_name,
                    video_entity=video_entity,
                    pinhole_entity=pinhole_entity,
                    transform_entity=transform_entity,
                    data_kind=data_kind,
                )
            else:
                # Prefer video streams if both exist; otherwise keep detected kind.
                if stream.data_kind != data_kind and data_kind == "video_stream":
                    stream_map[video_entity] = _RRDCameraStream(
                        name=camera_name,
                        video_entity=video_entity,
                        pinhole_entity=pinhole_entity,
                        transform_entity=transform_entity,
                        data_kind=data_kind,
                    )

        camera_streams: list[_RRDCameraStream] = sorted(stream_map.values(), key=lambda stream: stream.name)
        return camera_streams

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

    def _load_intrinsics(self, recording: Recording, pinhole_entity: str, timeline: str) -> Intrinsics:
        view = recording.view(index=timeline, contents=pinhole_entity)
        try:
            reader = view.select_static(
                f"{pinhole_entity}:Pinhole:image_from_camera",
                f"{pinhole_entity}:Pinhole:camera_xyz",
                f"{pinhole_entity}:Pinhole:resolution",
            )
        except ValueError:
            reader = None

        k_value: Any | None = None
        camera_xyz_value: Any | None = None
        resolution_value: Any | None = None

        if reader is not None:
            batch = reader.read_next_batch()
            if batch is not None and batch.num_rows > 0:
                k_value = self._first_valid_value(
                    batch.column(0),
                    component_name=f"{pinhole_entity}:Pinhole:image_from_camera",
                )
                camera_xyz_value = self._first_valid_value(
                    batch.column(1),
                    allow_none=True,
                    component_name=f"{pinhole_entity}:Pinhole:camera_xyz",
                )
                resolution_value = self._first_valid_value(
                    batch.column(2),
                    allow_none=True,
                    component_name=f"{pinhole_entity}:Pinhole:resolution",
                )

        if k_value is None:
            _, k_col_dyn, camera_xyz_col_dyn, resolution_col_dyn = view.select(
                timeline,
                f"{pinhole_entity}:Pinhole:image_from_camera",
                f"{pinhole_entity}:Pinhole:camera_xyz",
                f"{pinhole_entity}:Pinhole:resolution",
            ).read_all()
            k_value = self._first_valid_value(k_col_dyn, component_name=f"{pinhole_entity}:Pinhole:image_from_camera")
            camera_xyz_value = self._first_valid_value(
                camera_xyz_col_dyn,
                allow_none=True,
                component_name=f"{pinhole_entity}:Pinhole:camera_xyz",
            )
            resolution_value = self._first_valid_value(
                resolution_col_dyn,
                allow_none=True,
                component_name=f"{pinhole_entity}:Pinhole:resolution",
            )

        if k_value is None:
            raise ValueError(f"Missing image_from_camera for {pinhole_entity}")
        if isinstance(k_value, list) and len(k_value) == 1 and isinstance(k_value[0], list):
            k_value = k_value[0]
        k_matrix: Float32[ndarray, "3 3"] = np.array(k_value, dtype=np.float32).reshape(3, 3, order="F")

        camera_conventions = "RDF"
        if isinstance(camera_xyz_value, list) and len(camera_xyz_value) == 3:
            axis_tuple = tuple(int(v) for v in camera_xyz_value)
            if axis_tuple == (3, 5, 2):
                camera_conventions = "RUB"

        width: int | None = None
        height: int | None = None
        if isinstance(resolution_value, list) and len(resolution_value) == 2:
            width = int(round(resolution_value[0]))
            height = int(round(resolution_value[1]))

        if width is None:
            width = int(round(2 * float(k_matrix[0, 2])))
        if height is None:
            height = int(round(2 * float(k_matrix[1, 2])))

        return Intrinsics(
            camera_conventions=camera_conventions,
            fl_x=float(k_matrix[0, 0]),
            fl_y=float(k_matrix[1, 1]),
            cx=float(k_matrix[0, 2]),
            cy=float(k_matrix[1, 2]),
            width=width,
            height=height,
        )

    def _load_extrinsics(self, recording: Recording, entity: str, timeline: str) -> Extrinsics:
        view = recording.view(index=timeline, contents=entity)
        translation_value: list[float] | None = None
        rotation_value: list[float] | None = None

        try:
            static_reader = view.select_static(
                f"{entity}:Transform3D:translation",
                f"{entity}:Transform3D:mat3x3",
            )
        except ValueError:
            static_reader = None

        if static_reader is not None:
            batch = static_reader.read_next_batch()
            if batch is not None:
                t_col_static = batch.column(0)
                R_col_static = batch.column(1)
                if t_col_static.null_count != len(t_col_static):
                    translation_value = t_col_static[0].as_py()
                if R_col_static.null_count != len(R_col_static):
                    rotation_scalar = R_col_static[0].as_py()
                    rotation_value = rotation_scalar if isinstance(rotation_scalar, list) else None

        if translation_value is None or rotation_value is None:
            _, t_col, R_col = view.select(
                timeline,
                f"{entity}:Transform3D:translation",
                f"{entity}:Transform3D:mat3x3",
            ).read_all()
            translation_value = translation_value or self._first_valid_value(
                t_col,
                component_name=f"{entity}:Transform3D:translation",
            )
            rotation_value = rotation_value or self._first_valid_value(
                R_col,
                component_name=f"{entity}:Transform3D:mat3x3",
            )

        translation_arr = np.array(translation_value, dtype=np.float32)
        if translation_arr.ndim > 1:
            translation_arr = translation_arr.reshape(-1)
        translation: Float32[ndarray, "3"] = translation_arr.astype(np.float32)

        rotation_arr = np.array(rotation_value, dtype=np.float32)
        if rotation_arr.ndim > 1:
            rotation_arr = rotation_arr.reshape(-1)
        rotation: Float32[ndarray, "3 3"] = rotation_arr.reshape(3, 3, order="F")
        return Extrinsics(cam_R_world=rotation, cam_t_world=translation)

    def _first_valid_value(
        self,
        column: pa.ChunkedArray,
        *,
        allow_none: bool = False,
        component_name: str | None = None,
    ) -> Any:
        for value in column.combine_chunks().to_pylist():
            if value is None and not allow_none:
                continue
            if value is not None or allow_none:
                return value
        if allow_none:
            return None
        column_name = component_name or "(unknown component)"
        raise ValueError(f"Expected at least one non-null value in column '{column_name}'")

    @property
    def image_plane_distance(self) -> int | float:
        return 0.1
