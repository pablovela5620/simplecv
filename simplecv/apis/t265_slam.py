import contextlib
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float, UInt8
from numpy import ndarray

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole


@dataclass
class T265Config:
    """Log T265 left/right fisheye images to Rerun."""

    rr_config: RerunTyroConfig
    # Note: runs until Ctrl+C; 'frames' is deprecated and ignored.

    serial: str | None = None
    """Optional device serial if multiple T265 are connected."""

    timeout_ms: int = 1000
    """Wait timeout (ms) for each frame."""

    jpeg_quality: int = 75
    """JPEG quality for image compression (lower = smaller & faster). Used only for debug image logging."""

    base_path: Path = Path("t265")
    """Base entity path for logging (e.g., 't265')."""

    rect_fov_deg: float = 110.0
    """Horizontal FOV (degrees) for rectified pinhole images. Typical: 80–110."""

    log_fisheye: bool = False
    """If True, also log raw fisheye images under '<base>/left|right/fisheye/image'."""

    target_fps: int = 30
    """Target FPS for encoder time base and PTS. T265 fisheye is typically 30 FPS."""

    run_seconds: float | None = None
    """Optional total run time in seconds. None runs until Ctrl+C."""

    rrd_save_dir: Path | None = Path("data/rrd-save-files")
    """Directory to save a .rrd recording (MultiSink). If None, don't save.

    The saved file name will be formatted as:
      {YYYYmmdd_HHMMSS}_t265_slam_rrd_{rerun_version}.rrd
    """


class _VideoStreamEncoder:
    """Threaded H.264 encoder + Rerun logger for a single grayscale stream.

    Non-blocking: frames are enqueued from the acquisition thread and encoded/logged
    on a separate worker to avoid stalling RealSense frame delivery.
    """

    def __init__(
        self,
        entity_path: str,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        import av  # type: ignore # imported lazily to keep import-time light

        self.entity_path: str = entity_path
        self.width: int = int(width)
        self.height: int = int(height)
        self.fps: int = int(fps)
        self.q: "queue.Queue[tuple[np.ndarray, int]]" = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._av = av

        # Log stream metadata once
        rr.log(self.entity_path, rr.VideoStream(codec=rr.VideoCodec.H264), static=True)

        # Configure raw codec context (no container). We let the encoder assign PTS based on our frame.pts.
        codec = av.CodecContext.create("h264", "w")
        codec.width = self.width
        codec.height = self.height
        codec.pix_fmt = "yuv420p"
        codec.framerate = Fraction(self.fps, 1)
        codec.time_base = Fraction(1, self.fps)
        # Low-latency options if available (best-effort)
        with contextlib.suppress(Exception):
            codec.options = {
                "preset": "ultrafast",
                "tune": "zerolatency",
                # Ensure SPS/PPS are present in-stream so remuxing works reliably
                "x264-params": "repeat-headers=1:keyint=60:scenecut=0",
            }
        with contextlib.suppress(Exception):
            codec.open()
        # Some builds don't require/allow explicit open()
        self._codec = codec
        self._last_pts_ns: int = 0

        self._thr = threading.Thread(target=self._run, name=f"Encoder[{entity_path}]", daemon=True)
        self._thr.start()
        self._err_printed = False
        self._log_count = 0
        self._seq = 0

    def enqueue(self, frame_gray: UInt8[ndarray, "h w"], pts_ns: int) -> None:
        pts_ns = int(pts_ns)
        # Drop oldest if full to keep acquisition non-blocking
        if self.q.full():
            with contextlib.suppress(Exception):
                self.q.get_nowait()
        self.q.put((frame_gray, pts_ns))

    def stop(self) -> None:
        self._stop.set()
        self._thr.join(timeout=2.0)
        # Flush remaining packets
        with contextlib.suppress(Exception):
            for packet in self._codec.encode(None):
                rr.set_time("video_time", duration=np.timedelta64(self._last_pts_ns, "ns"))
                rr.log(self.entity_path, rr.VideoStream.from_fields(sample=bytes(packet)))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame_np, pts_ns = self.q.get(timeout=0.05)
            except Exception:
                continue
            try:
                # Construct AVFrame from grayscale and convert to encoder pix_fmt
                frame = self._av.VideoFrame.from_ndarray(frame_np, format="gray")
                frame = frame.reformat(width=self.width, height=self.height, format=self._codec.pix_fmt)
                # Maintain a monotonic encoder PTS consistent with fps
                pts_s: float = float(pts_ns) / 1_000_000_000.0
                frame.pts = int(round(pts_s * float(self.fps)))
                frame.time_base = self._codec.time_base

                for packet in self._codec.encode(frame):
                    # Log video samples on a single seconds-based timeline
                    rr.set_time("video_time", duration=np.timedelta64(pts_ns, "ns"))
                    rr.log(self.entity_path, rr.VideoStream.from_fields(sample=bytes(packet)))
                    if self._log_count < 3:
                        with contextlib.suppress(Exception):
                            print(
                                f"{self.entity_path}: logged {len(bytes(packet))} bytes at {pts_ns} ns (~{pts_s:.3f}s)"
                            )
                        self._log_count += 1
                self._last_pts_ns = int(pts_ns)
                self._seq += 1
            except Exception as e:  # noqa: BLE001
                if not self._err_printed:
                    print(f"Encoder error on {self.entity_path}: {e}")
                    self._err_printed = True
                continue


def main(config: T265Config) -> int:
    """Stream T265 fisheye images to Rerun (left/right) as compressed images."""
    try:
        import pyrealsense2 as rs
    except Exception as e:  # noqa: BLE001
        print(f"Failed to import pyrealsense2: {e}")
        return 1

    # Minimal setup: assume device is connected; no extra validation
    # Rerun is initialized via RerunTyroConfig.__post_init__
    # Enable MultiSink file saving if requested.
    try:
        if config.rrd_save_dir is not None:
            config.rrd_save_dir.mkdir(parents=True, exist_ok=True)
            rr_ver = getattr(rr, "__version__", "unknown")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            rrd_file = config.rrd_save_dir / f"{ts}_t265_slam_rrd_{rr_ver}.rrd"
            # Set MultiSink sinks explicitly (file sink; add grpc sink if desired)
            sinks: list[rr.Sink] = [rr.GrpcSink(), rr.FileSink(str(rrd_file))]
            # If you want to also stream to a local viewer, uncomment this:
            # sinks.append(rr.GrpcSink())
            rr.set_sinks(*sinks)
            print(f"Saving Rerun recording to: {rrd_file}")
    except Exception as e:  # noqa: BLE001
        print(f"Warning: failed to enable rrd saving: {e}")

    pipeline = rs.pipeline()
    rs_cfg = rs.config()
    if config.serial:
        rs_cfg.enable_device(config.serial)

    # Enable both fisheye streams (848x800 @ 30Hz, Y8)
    rs_cfg.enable_stream(rs.stream.fisheye, 1, 848, 800, rs.format.y8, 30)
    rs_cfg.enable_stream(rs.stream.fisheye, 2, 848, 800, rs.format.y8, 30)
    # Also enable pose for SLAM trajectory
    rs_cfg.enable_stream(rs.stream.pose)

    try:
        pipeline.start(rs_cfg)
    except Exception as e:  # noqa: BLE001
        print(f"Failed to start T265 fisheye streams: {e}")
        return 2

    # Set world coordinates and send a 3D+2D blueprint
    rr.log("/", rr.ViewCoordinates.RUB, static=True)

    right_panel_contents = [
        rrb.Horizontal(
            rrb.Spatial2DView(origin=str(config.base_path / "left" / "pinhole" / "video_stream")),
            rrb.Spatial2DView(origin=str(config.base_path / "right" / "pinhole" / "video_stream")),
        )
    ]
    if config.log_fisheye:
        right_panel_contents.append(
            rrb.Horizontal(
                rrb.Spatial2DView(origin=str(config.base_path / "left" / "fisheye" / "image")),
                rrb.Spatial2DView(origin=str(config.base_path / "right" / "fisheye" / "image")),
            )
        )

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="/"),
                rrb.Vertical(*right_panel_contents),
                column_shares=[3, 2],
            ),
            collapse_panels=True,
        )
    )

    left_path = config.base_path / "left"
    right_path = config.base_path / "right"
    pose_path = config.base_path / "mid"

    def _quat_to_rot_m33(x: float, y: float, z: float, w: float) -> Float[ndarray, "3 3"]:
        """Convert quaternion (x,y,z,w) to 3x3 rotation matrix."""
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z
        R: Float[ndarray, "3 3"] = np.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
            ],
            dtype=np.float32,
        )
        return R

    # Query active profile and prepare rectification from fisheye to pinhole
    profile = pipeline.get_active_profile()
    left_stream = profile.get_stream(rs.stream.fisheye, 1).as_video_stream_profile()
    right_stream = profile.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()
    pose_stream = profile.get_stream(rs.stream.pose)

    li = left_stream.get_intrinsics()
    ri = right_stream.get_intrinsics()
    width: int = int(li.width)
    height: int = int(li.height)

    K1: Float[ndarray, "3 3"] = np.array(
        [[li.fx, 0.0, li.ppx], [0.0, li.fy, li.ppy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    D1: Float[ndarray, "4"] = np.array(li.coeffs[:4], dtype=np.float64)
    K2: Float[ndarray, "3 3"] = np.array(
        [[ri.fx, 0.0, ri.ppx], [0.0, ri.fy, ri.ppy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    D2: Float[ndarray, "4"] = np.array(ri.coeffs[:4], dtype=np.float64)

    # Left->Right extrinsics
    lr_ex = left_stream.get_extrinsics_to(right_stream)
    R_lr_raw: Float[ndarray, "3 3"] = np.reshape(np.array(lr_ex.rotation, dtype=np.float64), (3, 3))
    R_lr: Float[ndarray, "3 3"] = R_lr_raw.T
    t_lr: Float[ndarray, "3"] = np.array(lr_ex.translation, dtype=np.float64)

    # Build rectification like Intel example (manual FOV & projection)
    rect_height_px: int = height
    rect_fov_deg: float = float(config.rect_fov_deg)
    rect_fov_rad: float = rect_fov_deg * (np.pi / 180.0)
    rect_fx: float = rect_height_px / 2.0 / np.tan(rect_fov_rad / 2.0)
    rect_fy: float = rect_fx
    rect_width_px: int = rect_height_px
    cx: float = (rect_width_px - 1) / 2.0
    cy: float = (rect_height_px - 1) / 2.0

    R1: Float[ndarray, "3 3"] = np.eye(3, dtype=np.float64)
    R2: Float[ndarray, "3 3"] = R_lr.astype(np.float64)
    P1: Float[ndarray, "3 4"] = np.array(
        [[rect_fx, 0.0, cx, 0.0], [0.0, rect_fy, cy, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64
    )
    P2: Float[ndarray, "3 4"] = P1.copy()
    P2[0, 3] = t_lr[0] * rect_fx

    new_size = (rect_width_px, rect_height_px)
    lm1, lm2 = cv2.fisheye.initUndistortRectifyMap(K1, D1, R1, P1[:, :3], new_size, cv2.CV_32FC1)
    rm1, rm2 = cv2.fisheye.initUndistortRectifyMap(K2, D2, R2, P2[:, :3], new_size, cv2.CV_32FC1)

    # Precompute rectified intrinsics
    K1_rect: Float[ndarray, "3 3"] = P1[:, :3].astype(np.float32)
    K2_rect: Float[ndarray, "3 3"] = P2[:, :3].astype(np.float32)
    left_intri = Intrinsics(
        camera_conventions="RDF",
        fl_x=float(K1_rect[0, 0]),
        fl_y=float(K1_rect[1, 1]),
        cx=float(K1_rect[0, 2]),
        cy=float(K1_rect[1, 2]),
        width=new_size[0],
        height=new_size[1],
    )
    right_intri = Intrinsics(
        camera_conventions="RDF",
        fl_x=float(K2_rect[0, 0]),
        fl_y=float(K2_rect[1, 1]),
        cx=float(K2_rect[0, 2]),
        cy=float(K2_rect[1, 2]),
        width=new_size[0],
        height=new_size[1],
    )

    # Midpoint transforms (cam -> mid)
    mid_ex_left = left_stream.get_extrinsics_to(pose_stream)
    mid_R_left: Float[ndarray, "3 3"] = np.reshape(np.array(mid_ex_left.rotation, dtype=np.float64), (3, 3))
    mid_t_left: Float[ndarray, "3"] = np.array(mid_ex_left.translation, dtype=np.float64)
    mid_T_left: Float[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
    mid_T_left[:3, :3] = mid_R_left
    mid_T_left[:3, 3] = mid_t_left

    mid_ex_right = right_stream.get_extrinsics_to(pose_stream)
    mid_R_right: Float[ndarray, "3 3"] = np.reshape(np.array(mid_ex_right.rotation, dtype=np.float64), (3, 3))
    mid_t_right: Float[ndarray, "3"] = np.array(mid_ex_right.translation, dtype=np.float64)
    mid_T_right: Float[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
    mid_T_right[:3, :3] = mid_R_right
    mid_T_right[:3, 3] = mid_t_right

    # Optional: threaded H.264 video stream encoders for rectified images
    left_vs = right_vs = None
    left_vs = _VideoStreamEncoder(
        entity_path=str(left_path / "pinhole" / "video_stream"),
        width=new_size[0],
        height=new_size[1],
        fps=config.target_fps,
    )
    right_vs = _VideoStreamEncoder(
        entity_path=str(right_path / "pinhole" / "video_stream"),
        width=new_size[0],
        height=new_size[1],
        fps=config.target_fps,
    )

    try:
        i = 0
        start_ts = time.time()
        # Track a relative origin for the video timeline
        video_ts0_ms: float | None = None
        video_ts0_ns: int | None = None

        while True:
            if config.run_seconds is not None and (time.time() - start_ts) >= float(config.run_seconds):
                break
            frames = pipeline.wait_for_frames(config.timeout_ms)

            # Retrieve left/right fisheye frames
            left = None
            right = None
            try:
                left = frames.get_fisheye_frame(1)
                right = frames.get_fisheye_frame(2)
            except Exception:
                try:
                    left = frames.get_frame(rs.stream.fisheye, 1)
                    right = frames.get_frame(rs.stream.fisheye, 2)
                except Exception:
                    pass

            if left is None or right is None:
                # Robust fallback: iterate frames in the set and pick by stream index
                for f in frames:
                    try:
                        p = f.get_profile()
                        if p.stream_type() == rs.stream.fisheye:
                            if p.stream_index() == 1:
                                left = f
                            elif p.stream_index() == 2:
                                right = f
                    except Exception:
                        pass

            if left is None or right is None:
                # Skip if missing either stream in this frameset
                continue

            # Convert to numpy views (zero-copy where possible)
            left_np: UInt8[ndarray, "h w"] = np.asanyarray(left.get_data())  # type: ignore[assignment]
            right_np: UInt8[ndarray, "h w"] = np.asanyarray(right.get_data())  # type: ignore[assignment]

            # Rectify to pinhole
            left_rect: UInt8[ndarray, "h w"] = cv2.remap(left_np, lm1, lm2, interpolation=cv2.INTER_LINEAR)
            right_rect: UInt8[ndarray, "h w"] = cv2.remap(right_np, rm1, rm2, interpolation=cv2.INTER_LINEAR)

            # Use device timestamp as nanoseconds for the video timeline (relative to first frame)
            ts_left_ns: int
            try:
                ts_left_ms = float(left.get_timestamp())
                if video_ts0_ms is None:
                    video_ts0_ms = ts_left_ms
                    video_ts0_ns = int(round(ts_left_ms * 1_000_000.0))
                assert video_ts0_ns is not None
                ts_left_ns = int(round(ts_left_ms * 1_000_000.0)) - video_ts0_ns
            except Exception:
                # Fallback to monotonic wall-clock
                ts_left_ns = int(round((time.time() - start_ts) * 1_000_000_000.0))

            # Log rectified pinhole streams (preferred) or fallback to JPEG images
            left_vs.enqueue(left_rect, pts_ns=ts_left_ns)
            right_vs.enqueue(right_rect, pts_ns=ts_left_ns)

            # Also log original fisheye images for comparison (optional)
            if config.log_fisheye:
                rr.log(
                    str(left_path / "fisheye" / "image"),
                    rr.Image(left_np).compress(jpeg_quality=config.jpeg_quality),
                )
                rr.log(
                    str(right_path / "fisheye" / "image"),
                    rr.Image(right_np).compress(jpeg_quality=config.jpeg_quality),
                )

            # Log pose transform (midpoint between fisheyes) in 3D
            pose = frames.get_pose_frame()
            if pose:
                data = pose.get_pose_data()
                t: Float[ndarray, "3"] = np.array(
                    [data.translation.x, data.translation.y, data.translation.z], dtype=np.float32
                )
                R_mid_world: Float[ndarray, "3 3"] = _quat_to_rot_m33(
                    data.rotation.x, data.rotation.y, data.rotation.z, data.rotation.w
                )
                # Ensure pose/camera logs align with the video timeline (duration in nanoseconds)
                rr.set_time("video_time", duration=np.timedelta64(ts_left_ns, "ns"))
                # World_T_mid
                world_T_mid: Float[ndarray, "4 4"] = np.eye(4, dtype=np.float32)
                world_T_mid[:3, :3] = R_mid_world
                world_T_mid[:3, 3] = t

                # Compose to eye frames, then invert to get cam_T_world
                world_T_left: Float[ndarray, "4 4"] = (world_T_mid @ mid_T_left).astype(np.float32)
                world_T_right: Float[ndarray, "4 4"] = (world_T_mid @ mid_T_right).astype(np.float32)

                left_cam_T_world: Float[ndarray, "4 4"] = np.linalg.inv(world_T_left).astype(np.float32)
                right_cam_T_world: Float[ndarray, "4 4"] = np.linalg.inv(world_T_right).astype(np.float32)

                # Apply rectification rotation to camera orientation (virtual pinholes)
                # New camera coords = R_rect * old_cam; so R_cam_world_new = R_cam_world_old @ R_rect^T
                left_cam_R_world: Float[ndarray, "3 3"] = (left_cam_T_world[:3, :3] @ R1.astype(np.float32).T).astype(
                    np.float32
                )
                right_cam_R_world: Float[ndarray, "3 3"] = (right_cam_T_world[:3, :3] @ R2.astype(np.float32).T).astype(
                    np.float32
                )
                left_cam_t_world: Float[ndarray, "3"] = left_cam_T_world[:3, 3]
                right_cam_t_world: Float[ndarray, "3"] = right_cam_T_world[:3, 3]

                # Log mid pose
                rr.log(str(pose_path), rr.Transform3D(translation=t, mat3x3=R_mid_world, from_parent=True))

                # Log per-eye pinhole cameras via helper
                left_extri = Extrinsics(cam_R_world=left_cam_R_world, cam_t_world=left_cam_t_world)
                right_extri = Extrinsics(cam_R_world=right_cam_R_world, cam_t_world=right_cam_t_world)
                left_params = PinholeParameters(name="left", intrinsics=left_intri, extrinsics=left_extri)
                right_params = PinholeParameters(name="right", intrinsics=right_intri, extrinsics=right_extri)

                log_pinhole(camera=left_params, cam_log_path=left_path, image_plane_distance=0.01)
                log_pinhole(camera=right_params, cam_log_path=right_path, image_plane_distance=0.01)
            i += 1
    except KeyboardInterrupt:
        print("Interrupted by user. Stopping T265 stream…")
    except Exception as e:  # noqa: BLE001
        print(f"Error while streaming fisheye images: {e}")
        with contextlib.suppress(Exception):
            pipeline.stop()
        return 3
    finally:
        with contextlib.suppress(Exception):
            pipeline.stop()
        # Stop encoders
        with contextlib.suppress(Exception):
            if left_vs:
                left_vs.stop()
        with contextlib.suppress(Exception):
            if right_vs:
                right_vs.stop()

    return 0
