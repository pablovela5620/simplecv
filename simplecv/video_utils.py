import atexit
import subprocess
import tempfile
from pathlib import Path
from typing import Literal


def create_temp_video_file(
    image_directory: Path,
    fps: int = 30,
    quality: Literal["low", "medium", "high", "max"] = "low",
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

    # Build ffmpeg command for NVIDIA hardware encoding
    cmd: list[str] = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        f"{str(image_directory)}/*.{image_extension}",
    ]

    # Add NVENC encoder settings
    cmd.extend(
        [
            "-c:v",
            "h264_nvenc",
            "-preset",
            preset,
            "-rc:v",
            "vbr_hq",  # High quality variable bitrate mode
            "-cq",
            cq,  # Quality level
            "-b:v",
            "0",  # Let CQ control bitrate
            "-profile:v",
            "high",  # High profile for better compression
            "-g",
            "30",  # Keyframe interval
            "-bf",
            "3",  # Maximum 3 B-frames between reference frames
            "-pix_fmt",
            "yuv420p",  # Standard pixel format for compatibility
            str(output_path),
        ]
    )

    # Execute FFmpeg
    process = subprocess.run(cmd, capture_output=True)

    if process.returncode != 0:
        error_msg = process.stderr.decode()
        raise RuntimeError(f"FFmpeg encoding failed: {error_msg}")

    return output_path
