import atexit
import io
import tempfile
import warnings
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from timeit import default_timer as timer
from typing import Literal, cast

import av
import numpy as np
import pyarrow as pa
import rerun as rr
import tqdm
import tyro
from av.codec.context import CodecContext, ThreadType
from av.container.input import InputContainer
from av.packet import Packet
from av.video.frame import VideoFrame
from av.video.stream import VideoStream
from jaxtyping import UInt8
from numpy import ndarray
from pyarrow import ChunkedArray
from rerun_bindings import ComponentColumnDescriptor, Recording, RecordingView, Schema

from simplecv.image_types import BGRList, ImageBGR
from simplecv.rerun_log_utils import (
    mux_h264_to_mp4,
    read_h264_samples_from_rrd,
    write_asset_video_blob,
)
from simplecv.video_io import MultiVideoReader


@dataclass(slots=True)
class _RRDCameraStream:
    """Metadata describing an exo camera stream discovered in an RRD file."""

    name: str
    video_entity: str
    pinhole_entity: str
    transform_entity: str
    data_kind: Literal["video_stream", "asset_video"]


def _discover_camera_streams(schema: Schema, camera_sub_path: Literal["exo", "ego"]) -> list[_RRDCameraStream]:
    component_columns: list[ComponentColumnDescriptor] = schema.component_columns()
    descriptors = list(component_columns.keys()) if isinstance(component_columns, dict) else list(component_columns)

    stream_map: dict[str, _RRDCameraStream] = {}
    for descriptor in descriptors:
        entity_path = getattr(descriptor, "entity_path", None)
        component_name = getattr(descriptor, "component", None)
        if not isinstance(component_name, str) or entity_path is None:
            continue

        entity_str = str(entity_path).lstrip("/")
        if not entity_str.startswith(f"world/{camera_sub_path}"):
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


class RRDVideoReader:
    """Decode a single camera video stored inside an RRD recording."""

    def __init__(self, rrd_path: Path, recording: Recording, stream: _RRDCameraStream, timeline: str) -> None:
        self._rrd_path: Path = rrd_path
        self._recording: Recording = recording
        self._stream: _RRDCameraStream = stream
        self._timeline: str = timeline

        match_result: tuple[bytes, str | None, int]
        match self._stream.data_kind:
            case "asset_video":
                video_bytes: bytes = self._load_asset_video_bytes()
                codec_hint: str | None = None
                frame_cnt: int = self._count_asset_frames()
                match_result = (video_bytes, codec_hint, frame_cnt)
            case "video_stream":
                stream_bytes, frame_cnt = self._load_video_stream_bytes_and_count()
                match_result = (stream_bytes, "h264", frame_cnt)
            case _:
                raise ValueError(f"Unsupported data kind: {self._stream.data_kind}")

        self._video_bytes: bytes
        self._codec_hint: str | None
        self._frame_cnt: int
        self._video_bytes, self._codec_hint, self._frame_cnt = match_result

        if self._frame_cnt <= 0:
            raise ValueError(f"No frames discovered for stream '{self._stream.video_entity}'")

        self._frame_cache: BGRList = []
        self._frame_generator: Generator[ImageBGR, None, None] = self._iter_frames()
        first_frame: ImageBGR | None = self._ensure_frame_cached(0)
        if first_frame is None:
            raise ValueError(f"Unable to decode first frame for stream '{self._stream.video_entity}'")

        self._height: int = int(first_frame.shape[0])
        self._width: int = int(first_frame.shape[1])
        self._position: int = 0

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    def __len__(self) -> int:
        return self._frame_cnt

    def __getitem__(self, idx: int) -> ImageBGR:
        if idx < 0 or idx >= self._frame_cnt:
            raise IndexError("Index out of range")
        frame: ImageBGR | None = self._ensure_frame_cached(idx)
        if frame is None:
            raise IndexError("Frame not available")
        return frame

    def read(self) -> ImageBGR | None:
        if self._position >= self._frame_cnt:
            return None
        frame: ImageBGR | None = self._ensure_frame_cached(self._position)
        self._position += 1
        return frame

    def reset(self) -> None:
        self._position = 0

    def _ensure_frame_cached(self, idx: int) -> ImageBGR | None:
        while len(self._frame_cache) <= idx:
            try:
                frame: ImageBGR = next(self._frame_generator)
            except StopIteration:
                return self._frame_cache[idx] if idx < len(self._frame_cache) else None
            self._frame_cache.append(frame)
        return self._frame_cache[idx]

    def _iter_frames(self) -> Generator[ImageBGR, None, None]:
        buffer_io: io.BytesIO = io.BytesIO(self._video_bytes)
        container: InputContainer = av.open(buffer_io, mode="r", format=self._codec_hint)
        video_stream: VideoStream = container.streams.video[0]
        try:
            codec_ctx: CodecContext = video_stream.codec_context
            codec_ctx.thread_type = ThreadType.AUTO
            codec_ctx.thread_count = 0  # Let FFmpeg decide optimal threading.
        except Exception:
            pass
        try:
            for packet in container.demux(video_stream):
                packet_typed: Packet = packet
                for frame in packet_typed.decode():
                    video_frame: VideoFrame = cast(VideoFrame, frame)
                    np_frame: UInt8[ndarray, "h w 3"] = np.ascontiguousarray(video_frame.to_ndarray(format="bgr24"))
                    yield np_frame
        finally:
            container.close()

    def _load_asset_video_bytes(self) -> bytes:
        view: RecordingView = self._recording.view(index=self._timeline, contents=self._stream.video_entity)
        reader = view.select(f"{self._stream.video_entity}:AssetVideo:blob")
        batch: pa.RecordBatch | None = reader.read_next_batch()
        while batch is not None:
            column: pa.Array = batch.column(0)
            for row_idx in range(batch.num_rows):
                value: pa.Scalar | None = column[row_idx]
                if value is None:
                    continue
                if hasattr(value, "values"):
                    flattened: pa.Array = value.values.flatten()
                    buffers: list[pa.Buffer | None] = flattened.buffers()
                    if len(buffers) >= 2 and buffers[1] is not None:
                        payload_buffer: pa.Buffer = buffers[1]
                        return bytes(payload_buffer)
                data_list = value.as_py()
                if isinstance(data_list, list) and len(data_list) == 1 and isinstance(data_list[0], list):
                    data_list = data_list[0]
                return bytes(data_list)
            batch = reader.read_next_batch()
        raise ValueError(f"No AssetVideo data found for entity {self._stream.video_entity}")

    def _load_video_stream_bytes_and_count(self) -> tuple[bytes, int]:
        times_chunk: ChunkedArray
        samples_chunk: ChunkedArray
        times_chunk, samples_chunk = read_h264_samples_from_rrd(
            str(self._rrd_path), self._stream.video_entity, self._timeline
        )
        sample_array: pa.Array = samples_chunk.combine_chunks().flatten(recursive=True)
        buffers: list[pa.Buffer | None] = sample_array.buffers()
        if len(buffers) < 2:
            raise ValueError(f"Unexpected buffer layout for video stream '{self._stream.video_entity}'")
        payload_buffer_stream: pa.Buffer | None = buffers[1]
        if payload_buffer_stream is None:
            raise ValueError(f"Missing payload buffer for video stream '{self._stream.video_entity}'")
        video_bytes: bytes = bytes(payload_buffer_stream)
        frame_count: int = len(times_chunk)
        return video_bytes, frame_count

    def _count_asset_frames(self) -> int:
        view: RecordingView = self._recording.view(index=self._timeline, contents=self._stream.video_entity)
        reader = view.select(self._timeline, f"{self._stream.video_entity}:VideoFrameReference:timestamp")
        table: pa.Table | None = reader.read_all()
        return table.num_rows if table is not None else 0


class MVReaderReaderRRD:
    def __init__(
        self,
        rrd_path: Path,
        video_paths: list[Path],
        camera_sub_path: Literal["exo", "ego"],
        timeline: str = "video_time",
    ) -> None:
        self.rrd_path: Path = rrd_path
        self.timeline: str = timeline
        self._recording: Recording = rr.dataframe.load_recording(str(self.rrd_path))
        self._schema: Schema = self._recording.schema()
        self._exo_streams: list[_RRDCameraStream] = _discover_camera_streams(self._schema, camera_sub_path)

        if not self._exo_streams:
            raise ValueError("No exo camera streams discovered in RRD recording")

        name_to_stream: dict[str, _RRDCameraStream] = {stream.name.lower(): stream for stream in self._exo_streams}
        self._video_names: list[str] = []
        self.video_readers: list[RRDVideoReader] = []

        missing_paths: list[Path] = []
        for video_path in video_paths:
            candidate_names: set[str] = {video_path.stem.lower(), video_path.name.lower()}
            stream: _RRDCameraStream | None = None
            for candidate in candidate_names:
                stream = name_to_stream.get(candidate)
                if stream is not None:
                    break
            if stream is None:
                missing_paths.append(video_path)
                continue
            reader: RRDVideoReader = RRDVideoReader(self.rrd_path, self._recording, stream, self.timeline)
            self.video_readers.append(reader)
            self._video_names.append(stream.name)

        if missing_paths:
            missing_str: str = ", ".join(str(path) for path in missing_paths)
            warnings.warn(f"No matching RRD streams found for: {missing_str}", stacklevel=2)
        if not self.video_readers:
            raise ValueError("No matching RRD video readers were initialised")

        reference_reader: RRDVideoReader = self.video_readers[0]
        if any(
            reader.height != reference_reader.height or reader.width != reference_reader.width
            for reader in self.video_readers
        ):
            raise ValueError("All RRD video streams must share the same resolution for multi-view iteration")

    @property
    def height(self) -> int:
        return self.video_readers[0].height

    @property
    def width(self) -> int:
        return self.video_readers[0].width

    def __len__(self) -> int:
        return min(len(reader) for reader in self.video_readers)

    def __iter__(self) -> Generator[BGRList | None, None, None]:
        while True:
            bgr_list: BGRList = []
            for reader in self.video_readers:
                bgr_image: ImageBGR | None = reader.read()
                if bgr_image is None:
                    return
                bgr_list.append(bgr_image)
            yield bgr_list

    def __getitem__(self, idx: int) -> BGRList:
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        return [reader[idx] for reader in self.video_readers]

    @property
    def camera_names(self) -> list[str]:
        return list(self._video_names)


@dataclass
class MVReaderCompareConfig:
    rrd_path: Path
    videos_dir: Path


def main(config: MVReaderCompareConfig) -> None:
    mp4_video_paths: list[Path] = list(config.videos_dir.glob("*.mp4"))
    assert mp4_video_paths, f"No videos found in {config.videos_dir}"
    print(f"Found {len(mp4_video_paths)} videos in {config.videos_dir}: {mp4_video_paths}")
    # 1. Multi-video reader using just directly mp4 files
    start_mp4: float = timer()
    mv_reader_naive = MultiVideoReader(video_paths=mp4_video_paths)
    for bgr_list in cast(Iterable[BGRList], tqdm.tqdm(mv_reader_naive)):
        for i, bgr in enumerate(bgr_list):
            assert bgr is not None, f"Video {mp4_video_paths[i]} ended prematurely"

    print(f"Mp4 Reader: {timer() - start_mp4:.2f} seconds")
    # 2. Naive Multi-video reader using RRD files with remuxing
    start_naive: float = timer()
    timeline = "video_time"

    recording: Recording = rr.dataframe.load_recording(str(config.rrd_path))
    schema: Schema = recording.schema()
    exo_streams: list[_RRDCameraStream] = _discover_camera_streams(schema, "exo")

    _remux_tmpdir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(prefix="rrd_exo_remux_")
    atexit.register(_remux_tmpdir.cleanup)
    remuxed_video_paths: list[Path] = []
    print(_remux_tmpdir.name)
    for camera_stream in cast(
        Iterable[_RRDCameraStream],
        tqdm.tqdm(exo_streams, desc="Remuxing RRD videos"),
    ):
        match camera_stream.data_kind:
            case "video_stream":
                times, samples = read_h264_samples_from_rrd(str(config.rrd_path), camera_stream.video_entity, timeline)
                mp4_path: Path = Path(_remux_tmpdir.name) / f"{camera_stream.name}.mp4"
                mux_h264_to_mp4(times, samples, str(mp4_path))
            case "asset_video":
                mp4_path = Path(_remux_tmpdir.name) / f"{camera_stream.name}.mp4"
                write_asset_video_blob(
                    recording,
                    timeline=timeline,
                    video_entity=camera_stream.video_entity,
                    output_path=mp4_path,
                )
            case _:
                raise ValueError(f"Unsupported data kind for RRD camera stream: {camera_stream.data_kind}")

        assert mp4_path.exists(), f"Expected remuxed video at {mp4_path}"
        remuxed_video_paths.append(mp4_path)
    assert remuxed_video_paths, f"No remuxed videos found in {_remux_tmpdir.name}"
    print(f"Naive: {timer() - start_naive:.2f} seconds to remux {len(remuxed_video_paths)} videos")

    # 3. rrd loading using custom multi-video reader
    start_rrd: float = timer()
    mv_reader_rrd = MVReaderReaderRRD(
        rrd_path=config.rrd_path, video_paths=mp4_video_paths, camera_sub_path="exo", timeline=timeline
    )
    for bgr_list in cast(Iterable[BGRList], tqdm.tqdm(mv_reader_rrd)):
        for i, bgr in enumerate(bgr_list):
            assert bgr is not None, f"RRD stream {mv_reader_rrd.camera_names[i]} ended prematurely"

    print(f"RRD Reader: {timer() - start_rrd:.2f} seconds")


def entrypoint() -> None:
    """Entrypoint leveraging Tyro to expose the ingest workflow via CLI."""

    tyro.extras.set_accent_color("bright_cyan")
    config = tyro.cli(
        MVReaderCompareConfig,
        description="Given a directory with ego/exo recordings, save them to RRD and visualize them with Rerun.",
    )
    main(config=config)


if __name__ == "__main__":
    entrypoint()
