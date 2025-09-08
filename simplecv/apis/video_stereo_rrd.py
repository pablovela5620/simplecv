import contextlib
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyrealsense2 as rs  # type: ignore
import rerun as rr
from av.container.output import OutputContainer
from jaxtyping import UInt8
from numpy import ndarray

from simplecv.rerun_log_utils import RerunTyroConfig


@dataclass
class VideoStereoConfig:
    """Stream T265 left fisheye to Rerun as H.264.

    Keeps things minimal and close to the Rerun camera_video_stream example,
    but sources frames from the T265 left (non-rectified) fisheye via pyrealsense2.
    """

    rr_config: RerunTyroConfig
    """Rerun setup (spawn/connect/save)."""

    parent_log_path: Path = Path("t265")
    """Parent path to log under."""

    serial: str | None = None
    """Optional device serial if multiple T265 are connected."""

    run_seconds: float | None = None
    """How many seconds to record for. None runs until Ctrl+C."""

    timeout_ms: int = 1000
    """Wait timeout (ms) for each frameset."""

    rrd_save_dir: Path | None = Path("data/rrd-debug-artifacts")
    """If set, also save a .rrd file while viewing via MultiSink."""


def _setup_t265_input(serial: str | None, timeout_ms: int):
    """Start a T265 pipeline configured for left/right fisheye streams.

    Returns (pipeline, rs, width, height)
    """

    pipeline = rs.pipeline()
    rs_cfg = rs.config()
    if serial:
        rs_cfg.enable_device(serial)
    # Fisheye 1 = left, 2 = right; format is Y8, typically 848x800 @ 30Hz
    rs_cfg.enable_stream(rs.stream.fisheye, 1, 848, 800, rs.format.y8, 30)
    # We enable right as well so framesets are delivered consistently, but we only use left.
    rs_cfg.enable_stream(rs.stream.fisheye, 2, 848, 800, rs.format.y8, 30)

    pipeline.start(rs_cfg)

    # Query actual active profile to get resolution
    profile = pipeline.get_active_profile()
    left_prof = profile.get_stream(rs.stream.fisheye, 1).as_video_stream_profile()
    width = int(left_prof.width())
    height = int(left_prof.height())
    return pipeline, width, height


def _setup_output_stream(width: int, height: int, fps: int) -> av.VideoStream:
    """Create an H.264 encoder stream using PyAV, AnnexB bitstream.

    Keep it close to the Rerun example: minimal options, low latency, no B-frames.
    Returns an av.video.VideoStream instance.
    """

    output_container: OutputContainer = av.open("/dev/null", "w", format="h264")  # Use AnnexB H.264 stream.
    output_stream: av.VideoStream = output_container.add_stream("libx264")
    output_stream.width = int(width)
    output_stream.height = int(height)
    output_stream.time_base = Fraction(1, fps)

    # Configure for low latency.
    output_stream.codec_context.options = {
        "tune": "zerolatency",
        "preset": "veryfast",
    }
    output_stream.max_b_frames = 0  # Avoid b-frames for lower latency.

    return output_stream


class _VideoStreamEncoder:
    """Threaded H.264 encoder + Rerun logger for a single grayscale stream.

    Non-blocking: frames are enqueued from the acquisition thread and encoded/logged
    on a separate worker to avoid stalling RealSense frame delivery.
    """

    def __init__(
        self,
        entity_path: str,
        timeline: str,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        self.entity_path: str = entity_path
        self.timeline: str = timeline
        self.width: int = int(width)
        self.height: int = int(height)
        self.fps: int = int(fps)

        self.q: queue.Queue[tuple[np.ndarray, int]] = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self.output_stream: av.VideoStream = _setup_output_stream(width, height, fps=fps)

        # Log stream metadata once
        rr.log(self.entity_path, rr.VideoStream(codec=rr.VideoCodec.H264), static=True)

        self._thr = threading.Thread(target=self._run, name=f"Encoder[{entity_path}]", daemon=True)
        self._thr.start()

    def enqueue(self, frame_gray: UInt8[ndarray, "h w"]) -> None:
        # Drop oldest if full to keep acquisition non-blocking
        if self.q.full():
            with contextlib.suppress(Exception):
                self.q.get_nowait()
        self.q.put((frame_gray,))

    def stop(self) -> None:
        self._stop.set()
        self._thr.join(timeout=2.0)
        # Flush remaining packets
        with contextlib.suppress(Exception):
            for packet in self.output_stream.encode(None):
                if packet.pts is None:
                    continue
                rr.set_time(self.timeline, duration=float(packet.pts * packet.time_base))
                rr.log(self.entity_path, rr.VideoStream.from_fields(sample=bytes(packet)))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                queue_tuple: tuple[np.ndarray] = self.q.get(timeout=0.1)
                frame_np: UInt8[ndarray, "h=800 w=848"] = queue_tuple[0]
            except queue.Empty:
                continue
            frame_av: av.VideoFrame = av.VideoFrame.from_ndarray(frame_np, format="gray8")

            # Let the encoder pick I/P frames
            frame_av.pict_type = av.video.frame.PictureType.NONE

            # Encode and stream to Rerun
            for packet in self.output_stream.encode(frame_av):
                if packet.pts is None:
                    continue

                rr.set_time(self.timeline, duration=float(packet.pts * packet.time_base))
                rr.log(self.entity_path, rr.VideoStream.from_fields(sample=bytes(packet)))


def main(config: VideoStereoConfig) -> int:
    """Minimal left-fisheye T265 → H.264 → Rerun VideoStream.

    Runs until Ctrl+C by default, or for --run-seconds if provided.
    """
    # If saving is requested, configure MultiSink to both view and save FIRST
    # so that subsequent static metadata (codec) is recorded to all sinks.
    if config.rrd_save_dir is not None:
        config.rrd_save_dir.mkdir(parents=True, exist_ok=True)
        rr_ver: str = getattr(rr, "__version__", "unknown")
        ts: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        rrd_file: Path = config.rrd_save_dir / f"{ts}_video_stream_rrd_{rr_ver}.rrd"
        rr.set_sinks(*[rr.GrpcSink(), rr.FileSink(str(rrd_file))])
        print(f"Saving Rerun recording to: {rrd_file}")

    left_path: Path = config.parent_log_path / "left"
    right_path: Path = config.parent_log_path / "right"

    left_stream = _VideoStreamEncoder(
        entity_path=str(left_path / "video_stream"),
        timeline="time",
        width=848,
        height=800,
        fps=30,
    )

    right_stream = _VideoStreamEncoder(
        entity_path=str(right_path / "video_stream"),
        timeline="time",
        width=848,
        height=800,
        fps=30,
    )

    # Start camera input and H.264 encoder
    pipeline, width, height = _setup_t265_input(config.serial, config.timeout_ms)
    try:
        # Run indefinitely unless a duration is provided
        start_time: float = time.time()
        while True:
            if config.run_seconds is not None and (time.time() - start_time) >= float(config.run_seconds):
                break
            frames = pipeline.wait_for_frames(config.timeout_ms)

            # Left fisheye (index 1)
            left: rs.video_frame = frames.get_fisheye_frame(1)
            right: rs.video_frame = frames.get_fisheye_frame(2)

            left_np: UInt8[ndarray, "h=800 w=848"] = np.asanyarray(left.get_data())  # type: ignore[assignment]
            right_np: UInt8[ndarray, "h=800 w=848"] = np.asanyarray(right.get_data())  # type: ignore[assignment]

            left_stream.enqueue(left_np)
            right_stream.enqueue(right_np)
    finally:
        pipeline.stop()
    return 0
