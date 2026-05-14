"""Equivalence tests for ``log_video`` AssetVideo and VideoStream methods.

Both methods must:

1. Return the same set of frame timestamps for the same source MP4
   (``test_log_video_timestamps_match``).
2. Round-trip pixels bit-identically through the ``.rrd`` storage layer
   (``test_log_video_rrd_roundtrip_pixels_match_source``) — the encoded
   bytes rerun stores must decode back to the same pixels as the source.

The tests use a synthetic H.264 MP4 generated per-session, so they're
self-contained and exercise the H.264 mp4-to-Annex-B bitstream filter
path inside ``_log_video_stream``.
"""

from __future__ import annotations

import io
from pathlib import Path

import av
import numpy as np
import pytest
import rerun as rr
from jaxtyping import UInt8
from numpy import ndarray

from simplecv.rerun_log_utils import (
    extract_asset_video_blob_fast,
    log_video,
    read_video_stream_from_rrd,
)


_FRAME_COUNT: int = 12
_WIDTH: int = 64
_HEIGHT: int = 64
_FPS: int = 30


def _frame_pixels(i: int) -> UInt8[ndarray, "h w 3"]:
    """Deterministic per-frame test pattern (RGB)."""
    r: int = (i * 23) % 256
    g: int = (i * 71 + 40) % 256
    b: int = (i * 137 + 80) % 256
    frame: UInt8[ndarray, "h w 3"] = np.empty((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
    frame[..., 0] = r
    frame[..., 1] = g
    frame[..., 2] = b
    return frame


@pytest.fixture(scope="session")
def synthetic_h264_mp4(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a deterministic H.264 MP4 with no B-frames for round-trip tests."""
    out_path: Path = tmp_path_factory.mktemp("log_video") / "synthetic.mp4"
    container: av.container.OutputContainer = av.open(str(out_path), mode="w")
    stream: av.video.stream.VideoStream = container.add_stream("libx264", rate=_FPS)
    stream.width = _WIDTH
    stream.height = _HEIGHT
    stream.pix_fmt = "yuv420p"
    stream.max_b_frames = 0
    # CRF=0 + ultrafast = lossless. Required so the round-trip decoded pixels
    # are bit-identical to the source pixels we encode here.
    stream.options = {"preset": "ultrafast", "crf": "0"}

    for i in range(_FRAME_COUNT):
        frame: av.VideoFrame = av.VideoFrame.from_ndarray(_frame_pixels(i), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return out_path


def _decode_mp4(source: Path | bytes) -> list[UInt8[ndarray, "h w 3"]]:
    """Decode every video frame in an MP4 to RGB pixels."""
    handle: io.BytesIO | str = io.BytesIO(source) if isinstance(source, bytes) else str(source)
    container: av.container.InputContainer = av.open(handle, mode="r")
    frames: list[UInt8[ndarray, "h w 3"]] = []
    try:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    finally:
        container.close()
    return frames


def _decode_annexb_samples(
    samples: list[bytes], codec_name: str
) -> list[UInt8[ndarray, "h w 3"]]:
    """Decode Annex B / OBU encoded samples back to RGB pixels via a fresh decoder."""
    decoder: av.VideoCodecContext = av.CodecContext.create(codec_name, "r")
    frames: list[UInt8[ndarray, "h w 3"]] = []
    for sample in samples:
        packet: av.Packet = av.Packet(sample)
        for frame in decoder.decode(packet):
            frames.append(frame.to_ndarray(format="rgb24"))
    for frame in decoder.decode(None):
        frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def _samples_chunked_to_bytes(samples_chunked) -> list[bytes]:
    """Pull a pyarrow ChunkedArray of list<uint8> samples into Python ``bytes``."""
    out: list[bytes] = []
    for chunk in samples_chunked.iterchunks():
        for row in chunk.to_pylist():
            # Row may be nested as list[list[int]] (rerun's archetype shape).
            while isinstance(row, list) and len(row) > 0 and isinstance(row[0], list):
                row = row[0]
            if row is None:
                continue
            out.append(bytes(row))
    return out


def _log_to_tmp_rrd(
    mp4: Path,
    method: str,
    tmp_path: Path,
    *,
    entity: str,
    timeline: str,
) -> Path:
    """Log ``mp4`` to a fresh recording stream and save it to a tmp ``.rrd``."""
    rrd_path: Path = tmp_path / f"{method}.rrd"
    rec: rr.RecordingStream = rr.RecordingStream(
        application_id=f"test-log-video-{method}",
        recording_id=f"test-{method}",
    )
    rec.save(str(rrd_path))
    log_video(mp4, Path(entity), timeline=timeline, method=method, recording=rec)  # type: ignore[arg-type]
    # Flush by dropping the reference (RecordingStream finalizes on close/drop).
    del rec
    return rrd_path


def test_log_video_timestamps_match(synthetic_h264_mp4: Path, tmp_path: Path) -> None:
    """Both methods describe the same set of frame timestamps."""
    rec_asset: rr.RecordingStream = rr.RecordingStream(
        application_id="test-log-video-asset", recording_id="ts-asset",
    )
    rec_asset.save(str(tmp_path / "asset.rrd"))
    rec_stream: rr.RecordingStream = rr.RecordingStream(
        application_id="test-log-video-stream", recording_id="ts-stream",
    )
    rec_stream.save(str(tmp_path / "stream.rrd"))

    entity: Path = Path("/video")
    ts_asset: ndarray = log_video(
        synthetic_h264_mp4, entity, method="asset_video", recording=rec_asset
    )
    ts_stream: ndarray = log_video(
        synthetic_h264_mp4, entity, method="video_stream", recording=rec_stream
    )

    assert len(ts_asset) == _FRAME_COUNT, f"AssetVideo returned {len(ts_asset)} timestamps"
    assert len(ts_stream) == _FRAME_COUNT, f"VideoStream returned {len(ts_stream)} timestamps"
    np.testing.assert_array_equal(np.sort(ts_asset), np.sort(ts_stream))


def test_log_video_rrd_roundtrip_pixels_match_source(
    synthetic_h264_mp4: Path, tmp_path: Path
) -> None:
    """Bytes stored in the RRD decode to the same pixels as the source MP4.

    Verifies the full storage pipeline for both archetypes:
    source MP4 -> log_video -> .rrd -> query back -> PyAV decode -> pixels.
    """
    src_frames: list[UInt8[ndarray, "h w 3"]] = _decode_mp4(synthetic_h264_mp4)
    assert len(src_frames) == _FRAME_COUNT

    entity: str = "/video"
    timeline: str = "video_time"

    # AssetVideo round-trip: blob bytes are the original MP4.
    asset_rrd: Path = _log_to_tmp_rrd(
        synthetic_h264_mp4, "asset_video", tmp_path, entity=entity, timeline=timeline,
    )
    asset_blob: bytes = extract_asset_video_blob_fast(
        entity.lstrip("/"), timeline=timeline, rrd_path=asset_rrd,
    )
    asset_frames: list[UInt8[ndarray, "h w 3"]] = _decode_mp4(asset_blob)

    # VideoStream round-trip: Annex B samples + codec are pulled back from the RRD.
    stream_rrd: Path = _log_to_tmp_rrd(
        synthetic_h264_mp4, "video_stream", tmp_path, entity=entity, timeline=timeline,
    )
    codec, _times, samples_chunked = read_video_stream_from_rrd(
        str(stream_rrd), entity.lstrip("/"), timeline,
    )
    assert codec == rr.VideoCodec.H264
    sample_bytes: list[bytes] = _samples_chunked_to_bytes(samples_chunked)
    assert len(sample_bytes) == _FRAME_COUNT, (
        f"Expected {_FRAME_COUNT} samples, got {len(sample_bytes)}"
    )
    stream_frames: list[UInt8[ndarray, "h w 3"]] = _decode_annexb_samples(sample_bytes, "h264")

    assert len(asset_frames) == _FRAME_COUNT
    assert len(stream_frames) == _FRAME_COUNT

    # Compare first / middle / last frames. Any decode-path bug propagates from
    # a single packet onward and will trip at least one of these checkpoints.
    for i in (0, _FRAME_COUNT // 2, _FRAME_COUNT - 1):
        np.testing.assert_array_equal(
            src_frames[i], asset_frames[i],
            err_msg=f"AssetVideo round-trip pixel mismatch at frame {i}",
        )
        np.testing.assert_array_equal(
            src_frames[i], stream_frames[i],
            err_msg=f"VideoStream round-trip pixel mismatch at frame {i}",
        )
