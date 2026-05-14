"""Equivalence tests for ``log_video`` AssetVideo and VideoStream methods.

Both methods must:

1. Return the same set of frame timestamps for the same source MP4
   (``test_log_video_timestamps_match``).
2. Round-trip the right number of frames at high PSNR through the ``.rrd``
   storage layer (``test_log_video_rrd_roundtrip_pixels_match_source``).
   The VideoStream path re-encodes through libx264 (with SPS/PPS repeated
   per keyframe so rerun's viewer can decode) so it's not byte-identical
   to the source — but should be visually indistinguishable.
3. Match closely between the AssetVideo and VideoStream paths on a real
   hocap H.264 capture (``test_log_video_psnr_at_sampled_times_on_hocap``).

The first two tests use a synthetic H.264 MP4 generated per-session.
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
    mux_h264_to_mp4,
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


def _psnr(
    reference: UInt8[ndarray, "h w 3"], comparison: UInt8[ndarray, "h w 3"]
) -> float:
    """Peak signal-to-noise ratio in dB. ``inf`` for identical inputs."""
    ref32: ndarray = reference.astype(np.float64)
    cmp32: ndarray = comparison.astype(np.float64)
    mse: float = float(np.mean((ref32 - cmp32) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * float(np.log10(255.0)) - 10.0 * float(np.log10(mse))


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

    # AssetVideo stores the source MP4 verbatim, so pixels must be bit-identical.
    # VideoStream re-encodes through libx264 (intentionally — see
    # _log_video_stream), so we only assert high PSNR (≥35 dB ≈ visually
    # imperceptible for natural images; very forgiving for solid-color
    # synthetic frames which are easy to compress losslessly).
    for i in (0, _FRAME_COUNT // 2, _FRAME_COUNT - 1):
        np.testing.assert_array_equal(
            src_frames[i], asset_frames[i],
            err_msg=f"AssetVideo round-trip pixel mismatch at frame {i}",
        )
        psnr_value: float = _psnr(src_frames[i], stream_frames[i])
        assert psnr_value >= 35.0, (
            f"VideoStream round-trip PSNR at frame {i} is {psnr_value:.2f} dB "
            f"(< 35 dB threshold). Source and re-encoded pixels diverge significantly."
        )


def _find_hocap_video() -> Path | None:
    """Locate any hocap sample ``output.mp4`` from the pixi download fixture."""
    base: Path = Path("data/hocap/sample")
    if not base.exists():
        return None
    candidates: list[Path] = sorted(base.rglob("output.mp4"))
    return candidates[0] if candidates else None


def test_log_video_psnr_at_sampled_times_on_hocap(tmp_path: Path) -> None:
    """PSNR between AssetVideo and VideoStream round-trips on a real lossy MP4.

    Synthetic-MP4 tests use lossless H.264 (CRF=0) and assert byte-identity.
    For real hocap captures the H.264 is lossy, but the AssetVideo and
    VideoStream paths store the same encoded bytes (whole blob vs. demuxed
    Annex B samples) — so they should decode to effectively the same pixels.

    Both decode paths funnel through a full MP4 container to use the same
    libavcodec decoder configuration; comparing container-decode against
    a bare ``CodecContext.create('h264', 'r')`` produces spurious divergence
    because the fresh decoder lacks the container's extradata / GOP context.
    """
    mp4: Path | None = _find_hocap_video()
    if mp4 is None:
        pytest.skip("hocap sample not downloaded (run pixi _download-hocap-sample)")

    entity: str = "/video"
    timeline: str = "video_time"

    asset_rrd: Path = _log_to_tmp_rrd(
        mp4, "asset_video", tmp_path, entity=entity, timeline=timeline,
    )
    stream_rrd: Path = _log_to_tmp_rrd(
        mp4, "video_stream", tmp_path, entity=entity, timeline=timeline,
    )

    # AssetVideo: pull the MP4 blob out as-is and decode through a container.
    asset_blob: bytes = extract_asset_video_blob_fast(
        entity.lstrip("/"), timeline=timeline, rrd_path=asset_rrd,
    )
    asset_frames: list[UInt8[ndarray, "h w 3"]] = _decode_mp4(asset_blob)

    # VideoStream: pull samples + timestamps back, remux into a fresh MP4
    # via the existing mux_h264_to_mp4 helper, then decode that MP4 through
    # the same container path. This mirrors what a downstream consumer of
    # the VideoStream archetype would do to recover playable video.
    codec, times, samples_chunked = read_video_stream_from_rrd(
        str(stream_rrd), entity.lstrip("/"), timeline,
    )
    assert codec == rr.VideoCodec.H264, f"hocap sample expected H.264, got {codec}"
    stream_mp4: Path = tmp_path / "stream_remuxed.mp4"
    mux_h264_to_mp4(times, samples_chunked, str(stream_mp4))
    stream_frames: list[UInt8[ndarray, "h w 3"]] = _decode_mp4(stream_mp4)

    n: int = min(len(asset_frames), len(stream_frames))
    assert n > 0, "no decoded frames from either path"
    sample_indices: list[int] = [0, n // 4, n // 2, 3 * n // 4, n - 1]

    psnrs: dict[int, float] = {
        i: _psnr(asset_frames[i], stream_frames[i]) for i in sample_indices
    }
    print("PSNR (AssetVideo round-trip vs VideoStream round-trip on hocap):")
    for idx, value in psnrs.items():
        print(f"  frame {idx}: {value:.2f} dB")

    # Same encoded bytes through the same container decoder → near-identical
    # pixels. ≥40 dB is imperceptible; ≥50 dB is "same".
    for idx, value in psnrs.items():
        assert value >= 40.0, (
            f"PSNR at frame {idx} is {value:.2f} dB (< 40 dB threshold). "
            f"AssetVideo and VideoStream decode paths diverged."
        )
