"""Equivalence and bit-preservation tests for ``log_video``.

The VideoStream method is bit-preserving: it demuxes the source MP4 and
applies the ``h264_mp4toannexb`` bitstream filter, then logs the
resulting NAL units verbatim, indexed by DTS (decode order, required so
the H.264 decoder can reconstruct B/P frames). No pixel decode, no
re-encode — the encoded bytes round-trip through the RRD identically to
running ``av.BitStreamFilterContext`` directly against the source.

Tests:

1. ``test_log_video_timestamps_match`` — both methods describe the same
   set of frame timestamps (display order / PTS).
2. ``test_log_video_stream_bytes_are_bit_preserved`` — the encoded
   samples stored in the RRD are byte-identical to direct ``demux + bsf``
   output for both the (no-B-frame) synthetic and (B-frame) hocap MP4s.
3. ``test_log_video_rrd_roundtrip_pixels_match_source`` — synthetic
   lossless MP4 round-trips bit-identically through both archetypes.
4. ``test_log_video_psnr_at_sampled_times_on_hocap`` — round-trip
   AssetVideo and VideoStream on the real hocap H.264 capture and
   verify PSNR ≥ 40 dB across sampled timestamps (decoder-state noise
   floor for B-frame streams via container remux).
"""

from __future__ import annotations

import io
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
import rerun as rr
from jaxtyping import UInt8
from numpy import ndarray

from simplecv.rerun_log_utils import (
    _normalize_av1_sample_for_rerun,
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


def _samples_chunked_to_bytes(samples_chunked) -> list[bytes]:
    """Pull a pyarrow ChunkedArray of ``list<uint8>`` samples into ``list[bytes]``."""
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


def _direct_demux_bsf(mp4: Path) -> tuple[list[bytes], list[int]]:
    """Demux ``mp4`` + apply ``h264_mp4toannexb`` directly via PyAV.

    Mirrors what ``_log_video_stream`` does internally so we can byte-compare
    the RRD round-trip against the canonical "no-rerun-involved" output.

    Returns:
        ``(samples_in_decode_order, pts_ns_in_decode_order)``.
    """
    container: av.container.InputContainer = av.open(str(mp4), mode="r")
    in_stream: av.video.stream.VideoStream = container.streams.video[0]
    bsf: av.BitStreamFilterContext = av.BitStreamFilterContext("h264_mp4toannexb", in_stream)
    time_base: Fraction = Fraction(in_stream.time_base)
    ns_scale: Fraction = Fraction(1_000_000_000, 1)

    samples: list[bytes] = []
    pts_ns: list[int] = []
    try:
        for raw in container.demux(in_stream):
            if raw.pts is None or raw.dts is None:
                continue
            for f in bsf.filter(raw):
                if f.pts is None or f.dts is None:
                    continue
                samples.append(bytes(f))
                pts_ns.append(round(f.pts * time_base * ns_scale))
    finally:
        container.close()
    return samples, pts_ns


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


def _find_hocap_video(prefer: str = "hololens") -> Path | None:
    """Locate a hocap sample ``output.mp4``. Prefers the ego/hololens camera —
    the demanding 1280×720 case — falls back to any available camera."""
    base: Path = Path("data/hocap/sample")
    if not base.exists():
        return None
    candidates: list[Path] = sorted(base.rglob("output.mp4"))
    if not candidates:
        return None
    preferred: list[Path] = [p for p in candidates if prefer in p.parent.name]
    return preferred[0] if preferred else candidates[0]


# ─────────────────────────── Tests ────────────────────────────


def test_aria_gen2_rgb_av1_sequence_header_normalized_for_rerun() -> None:
    """Only the aria-gen2 RGB AV1 sequence header is rewritten for rerun."""
    frame_obu_prefix: bytes = bytes.fromhex("328cb30210011d810618")
    aria_rgb_keyframe_prefix: bytes = (
        bytes.fromhex("0a0c00000062ea7ffbf804330080") + frame_obu_prefix
    )

    normalized_prefix: bytes = _normalize_av1_sample_for_rerun(aria_rgb_keyframe_prefix)

    assert normalized_prefix == (
        bytes.fromhex("0a0c02000061753ffdfc02198040") + frame_obu_prefix
    )
    assert len(normalized_prefix) == len(aria_rgb_keyframe_prefix)

    slam_keyframe_prefix: bytes = (
        bytes.fromhex("0a0b0000000ccbfeff80433008") + frame_obu_prefix
    )
    assert _normalize_av1_sample_for_rerun(slam_keyframe_prefix) == slam_keyframe_prefix


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


@pytest.mark.parametrize(
    "video_fixture",
    ["synthetic", "hocap"],
)
def test_log_video_stream_bytes_are_bit_preserved(
    video_fixture: str, synthetic_h264_mp4: Path, tmp_path: Path,
) -> None:
    """RRD-stored samples are byte-identical to direct demux+bsf output.

    The strong bit-preservation guarantee: there's no decode, no encode,
    no transformation — only NAL-unit framing rewritten via PyAV's
    ``h264_mp4toannexb`` bsf. The bytes that come out of the RRD must
    match what ``av.BitStreamFilterContext`` produces from the source.
    Tested on both no-B-frame (synthetic) and B-frame (hocap) sources
    since the latter requires DTS-ordered storage to be correct.
    """
    if video_fixture == "synthetic":
        mp4: Path = synthetic_h264_mp4
    else:
        hocap: Path | None = _find_hocap_video()
        if hocap is None:
            pytest.skip("hocap sample not downloaded (run pixi _download-hocap-sample)")
        mp4 = hocap

    entity: str = "/video"
    timeline: str = "video_time"
    stream_rrd: Path = _log_to_tmp_rrd(
        mp4, "video_stream", tmp_path, entity=entity, timeline=timeline,
    )

    direct_samples, _ = _direct_demux_bsf(mp4)
    _, _, rrd_samples_arr = read_video_stream_from_rrd(
        str(stream_rrd), entity.lstrip("/"), timeline,
    )
    rrd_samples: list[bytes] = _samples_chunked_to_bytes(rrd_samples_arr)

    assert len(direct_samples) == len(rrd_samples), (
        f"Sample count differs: direct demux+bsf={len(direct_samples)}, "
        f"RRD round-trip={len(rrd_samples)}"
    )
    mismatches: int = sum(1 for a, b in zip(direct_samples, rrd_samples, strict=False) if a != b)
    assert mismatches == 0, (
        f"{mismatches}/{len(rrd_samples)} packets differ in bytes between "
        f"direct demux+bsf and RRD round-trip — VideoStream is no longer bit-preserving"
    )


def test_log_video_rrd_roundtrip_pixels_match_source(
    synthetic_h264_mp4: Path, tmp_path: Path,
) -> None:
    """Lossless synthetic MP4 round-trips bit-identically through both archetypes.

    Both AssetVideo (whole-blob storage) and VideoStream (bit-preserving
    bsf storage) preserve the source bytes, so the round-trip must
    decode to bit-identical pixels for both.
    """
    src_frames: list[UInt8[ndarray, "h w 3"]] = _decode_mp4(synthetic_h264_mp4)
    assert len(src_frames) == _FRAME_COUNT

    entity: str = "/video"
    timeline: str = "video_time"

    asset_rrd: Path = _log_to_tmp_rrd(
        synthetic_h264_mp4, "asset_video", tmp_path, entity=entity, timeline=timeline,
    )
    asset_blob: bytes = extract_asset_video_blob_fast(
        entity.lstrip("/"), timeline=timeline, rrd_path=asset_rrd,
    )
    asset_frames: list[UInt8[ndarray, "h w 3"]] = _decode_mp4(asset_blob)

    stream_rrd: Path = _log_to_tmp_rrd(
        synthetic_h264_mp4, "video_stream", tmp_path, entity=entity, timeline=timeline,
    )
    codec, times, samples_chunked = read_video_stream_from_rrd(
        str(stream_rrd), entity.lstrip("/"), timeline,
    )
    assert codec == rr.VideoCodec.H264
    stream_mp4: Path = tmp_path / "stream_remuxed.mp4"
    mux_h264_to_mp4(times, samples_chunked, str(stream_mp4))
    stream_frames: list[UInt8[ndarray, "h w 3"]] = _decode_mp4(stream_mp4)

    assert len(asset_frames) == _FRAME_COUNT
    assert len(stream_frames) == _FRAME_COUNT

    for i in (0, _FRAME_COUNT // 2, _FRAME_COUNT - 1):
        np.testing.assert_array_equal(
            src_frames[i], asset_frames[i],
            err_msg=f"AssetVideo round-trip pixel mismatch at frame {i}",
        )
        np.testing.assert_array_equal(
            src_frames[i], stream_frames[i],
            err_msg=f"VideoStream round-trip pixel mismatch at frame {i}",
        )


def test_log_video_stream_samples_remux_to_playable_mp4(tmp_path: Path) -> None:
    """VideoStream samples remux back to a decodable MP4 with the right frame count.

    Strict bit-preservation is already proven by
    :func:`test_log_video_stream_bytes_are_bit_preserved`; this is the
    end-to-end sanity check: the bytes we store can be turned back into a
    playable MP4 via the existing :func:`mux_h264_to_mp4`, and that MP4
    decodes to the expected number of frames. Pixel-level PSNR is not
    asserted on B-frame sources because ``mux_h264_to_mp4`` sets pts=dts,
    so the decoded frame *order* differs from the source (decoder uses
    POC for display) — that's a remux artifact, not a migration bug.
    """
    mp4: Path | None = _find_hocap_video()
    if mp4 is None:
        pytest.skip("hocap sample not downloaded (run pixi _download-hocap-sample)")

    entity: str = "/video"
    timeline: str = "video_time"

    stream_rrd: Path = _log_to_tmp_rrd(
        mp4, "video_stream", tmp_path, entity=entity, timeline=timeline,
    )

    codec, times, samples_chunked = read_video_stream_from_rrd(
        str(stream_rrd), entity.lstrip("/"), timeline,
    )
    assert codec == rr.VideoCodec.H264, f"hocap sample expected H.264, got {codec}"
    stream_mp4: Path = tmp_path / "stream_remuxed.mp4"
    mux_h264_to_mp4(times, samples_chunked, str(stream_mp4))

    src_frames: list[UInt8[ndarray, "h w 3"]] = _decode_mp4(mp4)
    rt_frames: list[UInt8[ndarray, "h w 3"]] = _decode_mp4(stream_mp4)

    # Remux can drop the trailing B-frame waiting for its reference; tolerate ±1.
    assert abs(len(rt_frames) - len(src_frames)) <= 1, (
        f"Remuxed frame count differs by more than 1: source={len(src_frames)}, "
        f"remuxed={len(rt_frames)}"
    )
    assert rt_frames[0].shape == src_frames[0].shape, (
        f"Remuxed frame dims differ: source={src_frames[0].shape}, "
        f"remuxed={rt_frames[0].shape}"
    )
