import atexit
import subprocess
import tempfile
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal


def create_temp_video_from_img_dir(
    image_directory: Path,
    fps: int = 30,
    quality: Literal["low", "medium", "high", "max", "optimal"] = "optimal",
    delete_on_exit: bool = True,
    image_extension: Literal["jpg", "png"] = "jpg",  # jpg or png
    save_file: bool = False,
) -> Path:
    """
    Create a temporary H.264 video file using NVIDIA GPU acceleration.

    Args:
        image_directory: Path to directory with images
        fps: Frames per second
        quality: Quality preset
        delete_on_exit: Whether to delete the file when program exits
        image_extension: Image file extension (jpg or png)

    Returns:
        Path to the temporary video file
    """
    # Map quality settings to NVENC presets and CQ values
    quality_settings = {
        "low": ("p6", "30"),  # preset, cq value
        "medium": ("p4", "23"),
        "high": ("p2", "18"),
        "max": ("p1", "12"),
    }

    preset, cq = quality_settings[quality]

    # Create a temporary file with .mp4 extension
    if not save_file:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            output_path = Path(temp_file.name)

        # If requested, register for deletion when program exits
        if delete_on_exit:
            atexit.register(lambda p: p.unlink(missing_ok=True), output_path)
    else:
        output_path: Path = image_directory / "output.mp4"

    # Build ffmpeg command base
    cmd_base: list[str] = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        f"{str(image_directory)}/*.{image_extension}",
    ]

    cmd_encoder_specific: list[str] = []

    if quality == "optimal":
        # AV1 NVENC settings for "optimal" quality
        cmd_encoder_specific.extend(
            [
                "-c:v",
                "av1_nvenc",
                "-preset",
                "p5",  # Balanced preset for AV1 NVENC (can be tuned: p1-p7, higher is slower)
                "-cq",
                "30",  # Constant Quality level (CRF equivalent)
                "-g",
                "2",  # Keyframe interval
                "-pix_fmt",
                "yuv420p",  # Standard pixel format
                # You might need to adjust or add other AV1 specific flags depending on your driver/ffmpeg version
                # e.g., -rc constqp -qp 30 for some rate control setups
            ]
        )
    else:
        # H.264 NVENC settings for other quality levels
        quality_settings = {
            "low": ("p6", "30"),  # preset, cq value
            "medium": ("p4", "23"),
            "high": ("p2", "18"),
            "max": ("p1", "12"),
        }
        preset, cq_h264 = quality_settings[quality]
        cmd_encoder_specific.extend(
            [
                "-c:v",
                "h264_nvenc",
                "-preset",
                preset,
                "-rc:v",
                "vbr_hq",  # High quality variable bitrate mode
                "-cq",
                cq_h264,  # Quality level
                "-b:v",
                "0",  # Let CQ control bitrate
                "-profile:v",
                "high",  # High profile for better compression
                "-g",
                "30",  # Keyframe interval for H.264
                "-bf",
                "3",  # Maximum 3 B-frames between reference frames
                "-pix_fmt",
                "yuv420p",  # Standard pixel format for compatibility
            ]
        )

    # Combine base command, encoder specific commands, and output path
    cmd: list[str] = cmd_base + cmd_encoder_specific + [str(output_path)]

    # Execute FFmpeg
    start_time = timer()
    process = subprocess.run(cmd, capture_output=True)
    end_time = timer()

    print(f"FFmpeg encoding completed in {end_time - start_time:.2f} seconds.")

    if process.returncode != 0:
        error_msg = process.stderr.decode()
        raise RuntimeError(f"FFmpeg encoding failed: {error_msg}")

    return output_path


def reencode_video_optimal(
    input_video_path: Path,
    delete_on_exit: bool = True,
    save_file: bool = False,
    output_directory: Path | None = None,
) -> Path:
    """
    Re-encode an existing video file to AV1 using optimal NVIDIA GPU accelerated settings.

    Args:
        input_video_path: Path to the input video file.
        delete_on_exit: Whether to delete the temporary output file when the program exits.
                        Only applicable if save_file is False.
        save_file: If True, saves the output video in the same directory as the input
                   or in output_directory if specified, with "_optimal.mp4" suffix.
                   If False, creates a temporary file.
        output_directory: Directory to save the output file if save_file is True.
                          If None, input_video_path.parent is used.

    Returns:
        Path to the re-encoded video file.
    """
    if not input_video_path.is_file():
        raise FileNotFoundError(f"Input video file not found: {input_video_path}")

    if not save_file:
        with tempfile.NamedTemporaryFile(suffix="_optimal.mp4", delete=False) as temp_file:
            output_path = Path(temp_file.name)
        if delete_on_exit:
            atexit.register(lambda p: p.unlink(missing_ok=True), output_path)
    else:
        base_name = input_video_path.stem
        out_dir = output_directory if output_directory else input_video_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{base_name}_optimal.mp4"

    # Build ffmpeg command base
    cmd_base: list[str] = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video_path),
    ]

    # AV1 NVENC settings for "optimal" quality
    cmd_encoder_specific: list[str] = [
        "-c:v",
        "av1_nvenc",
        "-preset",
        "p5",  # Balanced preset for AV1 NVENC
        "-cq",
        "30",  # Constant Quality level
        "-g",
        "2",  # Keyframe interval
        "-bf",
        "0",  # Set B-frames to 0 to satisfy GOP length constraint
        "-pix_fmt",
        "yuv420p",  # Standard pixel format
        "-c:a",
        "copy",  # Copy audio stream without re-encoding
    ]

    # Combine base command, encoder specific commands, and output path
    cmd: list[str] = cmd_base + cmd_encoder_specific + [str(output_path)]

    # Execute FFmpeg
    start_time = timer()
    process = subprocess.run(cmd, capture_output=True)
    end_time = timer()

    print(f"FFmpeg re-encoding to optimal AV1 completed in {end_time - start_time:.2f} seconds.")

    if process.returncode != 0:
        error_msg = process.stderr.decode()
        # Clean up temp file if error occurs and it was a temp file
        if not save_file and output_path.exists():
            output_path.unlink()
        raise RuntimeError(f"FFmpeg re-encoding failed: {error_msg}")

    return output_path
