from __future__ import annotations

import atexit
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Float32
from numpy import ndarray
from rerun_bindings import Recording

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.data.ego.base_ego import BaseEgoSequence, CamNameType, EgoData
from simplecv.rerun_log_utils import (
    get_video_cache,
    mux_h264_to_mp4,
    read_h264_samples_from_rrd,
    write_asset_video_blob,
)
from simplecv.video_io import VideoReader

if TYPE_CHECKING:
    from simplecv.data.exoego.rrd_exoego import RRDExoEgoConfig
else:  # pragma: no cover - runtime alias to avoid circular import
    from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig as RRDExoEgoConfig


@dataclass(slots=True)
class _RRDEgoCameraStream:
    """Metadata describing an ego camera stream discovered in an RRD file."""

    name: str
    video_entity: str
    pinhole_entity: str
    transform_entity: str
    data_kind: Literal["video_stream", "asset_video"]


class RRDEgoSequence(BaseEgoSequence[RRDExoEgoConfig]):
    """RRD-backed ego sequence that remuxes recorded H.264 streams into mp4 assets."""

    def load_video_paths(self) -> list[Path]:
        recording: Recording = self._ensure_recording()
        streams: list[_RRDEgoCameraStream] = self._camera_streams
        assert streams, "No ego camera streams found in recording"

        remux_tmpdir: tempfile.TemporaryDirectory[str] | None = getattr(self, "_remux_tmpdir", None)
        if remux_tmpdir is None:
            self._remux_tmpdir = tempfile.TemporaryDirectory(prefix="rrd_ego_remux_")
            atexit.register(self._remux_tmpdir.cleanup)
        remux_tmpdir = self._remux_tmpdir

        rrd_path: Path = self.config.rrd_path
        assert rrd_path.exists(), f"RRD path {rrd_path} does not exist"

        video_cache = get_video_cache()
        # Cache remuxed MP4s on disk so repeat runs avoid the expensive AssetVideo extraction.
        # TODO(pablo): Once MultiVideoReader understands RRD blobs directly, drop the cache in favor of in-memory readers.

        video_path_map: dict[str, Path] = {}
        for stream in streams:
            mp4_path: Path = Path(remux_tmpdir.name) / f"{stream.name}.mp4"
            if video_cache is not None:
                cached_path = video_cache.get(rrd_path=rrd_path, camera_name=stream.name)
                if cached_path is not None:
                    shutil.copy2(cached_path, mp4_path)
                    video_path_map[stream.name] = mp4_path
                    continue
            match stream.data_kind:
                case "video_stream":
                    times, samples = read_h264_samples_from_rrd(
                        str(rrd_path),
                        stream.video_entity,
                        self._video_timeline,
                    )
                    mux_h264_to_mp4(times, samples, str(mp4_path))
                case "asset_video":
                    write_asset_video_blob(
                        recording,
                        timeline=self._video_timeline,
                        video_entity=stream.video_entity,
                        output_path=mp4_path,
                    )
                case _:
                    raise ValueError(f"Unsupported data kind for RRD camera stream: {stream.data_kind}")
            assert mp4_path.exists(), f"Expected remuxed ego video at {mp4_path}"
            if video_cache is not None:
                video_cache.store(rrd_path=rrd_path, camera_name=stream.name, source_path=mp4_path)
            video_path_map[stream.name] = mp4_path

        self._video_path_map: dict[str, Path] = video_path_map
        ordered_paths: list[Path] = [video_path_map[stream.name] for stream in streams]
        return ordered_paths

    def load_ego_cams(self) -> dict[CamNameType, list[PinholeParameters]]:
        recording: Recording = self._ensure_recording()
        streams: list[_RRDEgoCameraStream] = self._camera_streams
        assert streams, "No ego camera streams found in recording"

        ego_cam_dict: dict[str, list[PinholeParameters]] = {}
        calibrated_streams: list[_RRDEgoCameraStream] = []
        for stream in streams:
            try:
                intrinsics: Intrinsics = self._load_intrinsics(recording, stream.pinhole_entity, self._video_timeline)
            except ValueError as exc:
                warnings.warn(
                    (
                        f"\033[33mSkipping ego camera '{stream.name}' due to missing metadata: {exc}. "
                        "Video frames are still remuxed but remain unlogged until intrinsics are available.\033[0m"
                    ),
                    stacklevel=2,
                )
                continue
            translations, rotations = self._load_extrinsics_series(
                recording,
                stream.transform_entity,
                self._video_timeline,
            )
            min_len: int = min(len(translations), len(rotations))
            if min_len == 0:
                translation_default: Float32[ndarray, "3"] = np.zeros(3, dtype=np.float32)
                rotation_default: Float32[ndarray, "3 3"] = np.eye(3, dtype=np.float32)
                extrinsics_default = Extrinsics(cam_R_world=rotation_default, cam_t_world=translation_default)
                ego_cam_dict[stream.name] = [
                    PinholeParameters(name=stream.name, intrinsics=intrinsics, extrinsics=extrinsics_default)
                ]
                calibrated_streams.append(stream)
                continue

            cam_params: list[PinholeParameters] = []
            for idx in range(min_len):
                translation_vec: Float32[ndarray, "3"] = translations[idx]
                rotation_mat: Float32[ndarray, "3 3"] = rotations[idx]
                extrinsics = Extrinsics(cam_R_world=rotation_mat, cam_t_world=translation_vec)
                cam_params.append(PinholeParameters(name=stream.name, intrinsics=intrinsics, extrinsics=extrinsics))
            if cam_params:
                ego_cam_dict[stream.name] = cam_params
                calibrated_streams.append(stream)

        if calibrated_streams:
            self._camera_streams = calibrated_streams
        return cast(dict[CamNameType, list[PinholeParameters]], ego_cam_dict)

    def align_cams_and_videos(
        self,
        video_path_list: list[Path],
        ego_cam_dict: dict[CamNameType, list[PinholeParameters]],
    ) -> tuple[dict[CamNameType, list[PinholeParameters]], dict[CamNameType, Path]]:
        video_by_name: dict[str, Path] = {path.stem: path for path in video_path_list}
        assert video_by_name, "No remuxed ego videos were produced"

        aligned_cam_dict: dict[str, list[PinholeParameters]] = {}
        aligned_video_map: dict[str, Path] = {}

        for cam_name, cam_params in ego_cam_dict.items():
            video_path = video_by_name.get(cam_name)
            if video_path is None:
                continue

            reader = VideoReader(video_path)
            video_len: int = len(reader)
            if not cam_params:
                continue

            if len(cam_params) < video_len:
                last_param: PinholeParameters = cam_params[-1]
                cam_params = cam_params + [last_param] * (video_len - len(cam_params))
            elif len(cam_params) > video_len:
                cam_params = cam_params[:video_len]

            aligned_cam_dict[cam_name] = cam_params
            aligned_video_map[cam_name] = video_path

        assert aligned_cam_dict, "No ego cameras aligned with the recorded videos"

        ordered_names: list[str] = sorted(aligned_video_map.keys())
        ordered_cam_dict: dict[str, list[PinholeParameters]] = {
            name: aligned_cam_dict[name] for name in ordered_names
        }
        ordered_video_map: dict[str, Path] = {name: aligned_video_map[name] for name in ordered_names}
        return (
            cast(dict[CamNameType, list[PinholeParameters]], ordered_cam_dict),
            cast(dict[CamNameType, Path], ordered_video_map),
        )

    def __getitem__(self, idx: int) -> EgoData:
        cam_params_list: list[PinholeParameters] = [cam_list[idx] for cam_list in self._ego_cam_dict.values()]
        return EgoData(
            cam_params_list=cam_params_list,
            bgr_list=self.ego_video_readers[idx],
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.ego_video_readers)

    @property
    def ego_video_names(self) -> list[str]:  # type: ignore[override]
        streams: list[_RRDEgoCameraStream] | None = getattr(self, "_camera_streams", None)
        if streams:
            stream_names: list[str] = [stream.name for stream in streams]
            return stream_names
        return super().ego_video_names

    @property
    def image_plane_distance(self) -> int | float:
        return 0.02

    # ───────────────────────── helpers ───────────────────────── #
    def _ensure_recording(self) -> Recording:
        recording: Recording | None = getattr(self, "_recording", None)
        if recording is None:
            rrd_path: Path = self.config.rrd_path
            assert rrd_path.exists(), f"RRD path {rrd_path} does not exist"

            recording = rr.dataframe.load_recording(str(rrd_path))
            schema = recording.schema()
            self._video_timeline: str = self._select_timeline(schema)
            self._camera_streams: list[_RRDEgoCameraStream] = self._discover_camera_streams(schema)
            self._recording = recording
        return recording

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
            k_value = self._first_valid_value(
                k_col_dyn,
                component_name=f"{pinhole_entity}:Pinhole:image_from_camera",
            )
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

    def _load_extrinsics_series(
        self,
        recording: Recording,
        entity: str,
        timeline: str,
    ) -> tuple[list[Float32[ndarray, "3"]], list[Float32[ndarray, "3 3"]]]:
        view = recording.view(index=timeline, contents=entity)
        translation_list: list[Float32[ndarray, "3"]] = []
        rotation_flat_list: list[Float32[ndarray, "9"]] = []

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
                translation_list = self._column_to_vec3_list(batch.column(0))
                rotation_flat_list = self._column_to_vec9_list(batch.column(1))

        if not translation_list or not rotation_flat_list:
            try:
                _, t_col, R_col = view.select(
                    timeline,
                    f"{entity}:Transform3D:translation",
                    f"{entity}:Transform3D:mat3x3",
                ).read_all()
                if not translation_list:
                    translation_list = self._column_to_vec3_list(t_col)
                if not rotation_flat_list:
                    rotation_flat_list = self._column_to_vec9_list(R_col)
            except ValueError:
                pass

        if not translation_list:
            zero_translation: Float32[ndarray, "3"] = np.zeros(3, dtype=np.float32)
            translation_list = [zero_translation]
        if not rotation_flat_list:
            identity_flat: Float32[ndarray, "9"] = np.eye(3, dtype=np.float32).reshape(9)
            rotation_flat_list = [identity_flat]

        rotation_list: list[Float32[ndarray, "3 3"]] = [
            rotation.reshape(3, 3, order="F") for rotation in rotation_flat_list
        ]
        return translation_list, rotation_list

    def _column_to_vec3_list(
        self,
        column: pa.Array | pa.ChunkedArray | None,
    ) -> list[Float32[ndarray, "3"]]:
        if column is None:
            return []
        py_values = (
            column.combine_chunks().to_pylist() if isinstance(column, pa.ChunkedArray) else column.to_pylist()
        )
        vectors: list[Float32[ndarray, "3"]] = []
        for value in py_values:
            if value is None:
                continue
            if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
                value = value[0]
            vector_np = np.array(value, dtype=np.float32).reshape(-1)
            if vector_np.size < 3:
                continue
            vector: Float32[ndarray, "3"] = vector_np[:3]
            vectors.append(vector)
        return vectors

    def _column_to_vec9_list(
        self,
        column: pa.Array | pa.ChunkedArray | None,
    ) -> list[Float32[ndarray, "9"]]:
        if column is None:
            return []
        py_values = (
            column.combine_chunks().to_pylist() if isinstance(column, pa.ChunkedArray) else column.to_pylist()
        )
        vectors: list[Float32[ndarray, "9"]] = []
        for value in py_values:
            if value is None:
                continue
            if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
                value = value[0]
            vector_np = np.array(value, dtype=np.float32).reshape(-1)
            if vector_np.size < 9:
                continue
            vector: Float32[ndarray, "9"] = vector_np[:9]
            vectors.append(vector)
        return vectors

    def _discover_camera_streams(self, schema: Any) -> list[_RRDEgoCameraStream]:
        component_columns = schema.component_columns()
        descriptors: list[Any] = (
            list(component_columns.keys()) if isinstance(component_columns, dict) else list(component_columns)
        )

        stream_map: dict[str, _RRDEgoCameraStream] = {}
        for descriptor in descriptors:
            entity_path = getattr(descriptor, "entity_path", None)
            component_name = getattr(descriptor, "component", None)
            if not isinstance(component_name, str) or entity_path is None:
                continue

            entity_str = str(entity_path).lstrip("/")
            if not entity_str.startswith("world/ego"):
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
            camera_name: str = PurePosixPath(transform_entity).name

            stream = stream_map.get(video_entity)
            if stream is None:
                stream_map[video_entity] = _RRDEgoCameraStream(
                    name=camera_name,
                    video_entity=video_entity,
                    pinhole_entity=pinhole_entity,
                    transform_entity=transform_entity,
                    data_kind=data_kind,
                )
            else:
                if stream.data_kind != data_kind and data_kind == "video_stream":
                    stream_map[video_entity] = _RRDEgoCameraStream(
                        name=camera_name,
                        video_entity=video_entity,
                        pinhole_entity=pinhole_entity,
                        transform_entity=transform_entity,
                        data_kind=data_kind,
                    )

        camera_streams: list[_RRDEgoCameraStream] = sorted(stream_map.values(), key=lambda stream: stream.name)
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

    def _first_valid_value(
        self,
        column: pa.Array | pa.ChunkedArray,
        *,
        allow_none: bool = False,
        component_name: str | None = None,
    ) -> Any:
        py_values = (
            column.combine_chunks().to_pylist() if isinstance(column, pa.ChunkedArray) else column.to_pylist()
        )
        for value in py_values:
            if value is None and not allow_none:
                continue
            if value is not None or allow_none:
                return value
        if allow_none:
            return None
        column_name = component_name or "(unknown component)"
        raise ValueError(f"Expected at least one non-null value in column '{column_name}'")
