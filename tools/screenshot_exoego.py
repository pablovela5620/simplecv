"""Take screenshots from the exoego viewer for visual testing.

Spawns the Rerun viewer, logs data, and captures screenshots via ViewerClient.

Usage:
    pixi run python tools/screenshot_exoego.py hot3d --output /tmp/screenshots/hot3d.png
    pixi run python tools/screenshot_exoego.py umetrack --output /tmp/screenshots/umetrack.png
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import tyro

from simplecv.apis.view_exoego import VisualizeConfig, visualize_exo_ego
from simplecv.configs.exoego_dataset_configs import AnnotatedExoEgoDatasetUnion
from simplecv.rerun_log_utils import RerunTyroConfig


@dataclass
class ScreenshotConfig:
    """Configuration for taking Rerun screenshots."""

    dataset: AnnotatedExoEgoDatasetUnion
    """Dataset to visualize."""
    output: Path = Path("/tmp/screenshot_exoego.png")
    """Output path for the screenshot."""
    wait_seconds: float = 3.0
    """Seconds to wait for viewer to render before screenshotting."""
    max_frames: int = 50
    """Maximum frames to log before taking screenshot."""


def main(config: ScreenshotConfig) -> None:
    """Take a screenshot of the exoego viewer."""
    import rerun as rr
    from rerun.experimental import ViewerClient

    # Spawn the viewer
    rr_config: RerunTyroConfig = RerunTyroConfig()

    vis_config: VisualizeConfig = VisualizeConfig(
        rr_config=rr_config,
        dataset=config.dataset,
    )

    # Run visualization (will spawn viewer)
    visualize_exo_ego(vis_config)

    # Wait for rendering
    time.sleep(config.wait_seconds)

    # Connect and screenshot
    config.output.parent.mkdir(parents=True, exist_ok=True)
    viewer: ViewerClient = ViewerClient()
    viewer.save_screenshot(str(config.output))
    print(f"Screenshot saved to {config.output}")


if __name__ == "__main__":
    cfg: ScreenshotConfig = tyro.cli(ScreenshotConfig)
    main(cfg)
