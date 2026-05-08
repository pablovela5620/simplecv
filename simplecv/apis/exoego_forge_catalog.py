from __future__ import annotations

import atexit
import tempfile
import time
import weakref
from dataclasses import dataclass
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb
from rerun.catalog import OnDuplicateSegmentLayer
from tqdm import tqdm

from simplecv.apis.view_exoego import create_container

APPLICATION_ID: str = "exoego-forge"

DEFAULT_CATALOG_DATASETS: tuple[str, ...] = (
    "aria-gen2",
    "assembly101",
    "hocap",
    "hot3d-aria",
    "hot3d-quest3",
    "umetrack",
    "ego-dex",
)

CATALOG_CAMERA_NAMES: dict[str, dict[str, tuple[str, ...]]] = {
    "aria-gen2": {
        "ego": ("camera-rgb", "slam-front-left", "slam-front-right", "slam-side-left", "slam-side-right"),
        "exo": (),
    },
    "assembly101": {
        "ego": ("e1", "e2", "e3", "e4"),
        "exo": ("C10095", "C10115", "C10118", "C10119", "C10379", "C10390", "C10395", "C10404"),
    },
    "hocap": {
        "ego": ("hololens_kv5h72",),
        "exo": (
            "037522251142",
            "043422252387",
            "046122250168",
            "105322251225",
            "105322251564",
            "108222250342",
            "115422250549",
            "117222250549",
        ),
    },
    "hot3d-aria": {
        "ego": ("camera-rgb", "camera-slam-left", "camera-slam-right"),
        "exo": (),
    },
    "hot3d-quest3": {
        "ego": ("camera-slam-left", "camera-slam-right"),
        "exo": (),
    },
    "umetrack": {
        "ego": ("BL", "BR", "TL", "TR"),
        "exo": (),
    },
    "ego-dex": {
        "ego": ("avp_camera",),
        "exo": (),
    },
}


def _video_log_paths(kind: str, camera_names: tuple[str, ...]) -> list[Path]:
    return [Path("world") / kind / camera_name / "pinhole" / "video" for camera_name in camera_names]


def build_exoego_catalog_blueprint(dataset_name: str) -> rrb.Blueprint:
    """Build the default catalog blueprint for one ExoEgo Forge Dataset."""
    camera_names: dict[str, tuple[str, ...]] = CATALOG_CAMERA_NAMES.get(dataset_name, {"ego": (), "exo": ()})
    container: rrb.ContainerLike = create_container(
        ego_video_log_paths=_video_log_paths("ego", camera_names["ego"]),
        exo_video_log_paths=_video_log_paths("exo", camera_names["exo"]),
    )
    return rrb.Blueprint(
        rrb.Horizontal(
            contents=[container],
            column_shares=[4, 1],
        ),
        collapse_panels=True,
    )


def discover_rrd_uris(
    rrd_root: Path,
    *,
    datasets: tuple[str, ...] = DEFAULT_CATALOG_DATASETS,
) -> dict[str, list[str]]:
    """Discover local RRD files grouped by first-level catalog Dataset directory."""
    root: Path = rrd_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"RRD root directory does not exist: {root}")

    dataset_dirs: list[Path] = (
        [root / dataset for dataset in datasets] if datasets else sorted(d for d in root.iterdir() if d.is_dir())
    )

    uris_by_dataset: dict[str, list[str]] = {}
    for dataset_dir in dataset_dirs:
        if not dataset_dir.is_dir():
            continue
        rrd_paths: list[Path] = sorted(dataset_dir.rglob("*.rrd"))
        if rrd_paths:
            uris_by_dataset[dataset_dir.name] = [path.resolve().as_uri() for path in rrd_paths]

    if not uris_by_dataset:
        dataset_desc: str = ", ".join(datasets) if datasets else "all first-level directories"
        raise FileNotFoundError(f"No RRD files found under {root} for datasets: {dataset_desc}")

    return uris_by_dataset


def mount_catalog(
    rrd_root: Path,
    *,
    datasets: tuple[str, ...] = DEFAULT_CATALOG_DATASETS,
    port: int | None = None,
    application_id: str = APPLICATION_ID,
    show_progress: bool = True,
) -> rr.server.Server:
    """Mount local ExoEgo Forge RRDs as one Rerun catalog Dataset per source."""
    uris_by_dataset: dict[str, list[str]] = discover_rrd_uris(rrd_root, datasets=datasets)
    dataset_names: list[str] = sorted(uris_by_dataset)
    total_files: int = sum(len(uris_by_dataset[name]) for name in dataset_names)

    print(
        f"Mounting catalog from {rrd_root.expanduser().resolve()} "
        f"({total_files} RRDs across {len(dataset_names)} datasets: {', '.join(dataset_names)})",
        flush=True,
    )

    server: rr.server.Server = rr.server.Server(datasets={name: [] for name in dataset_names}, port=port)
    client = server.client()

    iterator = tqdm(dataset_names, desc="register", unit="dataset", disable=not show_progress)
    for dataset_name in iterator:
        uris: list[str] = uris_by_dataset[dataset_name]
        iterator.set_postfix_str(f"{dataset_name} ({len(uris)} files)")
        dataset = client.get_dataset(dataset_name)
        dataset.register(uris, layer_name="base", on_duplicate=OnDuplicateSegmentLayer.ERROR).wait()  # type: ignore[attr-defined]
        blueprint: rrb.Blueprint = build_exoego_catalog_blueprint(dataset_name)
        tmp_dir = tempfile.TemporaryDirectory(prefix=f"{dataset_name}-")
        weakref.finalize(server, tmp_dir.cleanup)
        atexit.register(tmp_dir.cleanup)
        blueprint_path: Path = Path(tmp_dir.name) / f"{dataset_name}.rbl"
        blueprint.save(application_id, path=str(blueprint_path))
        dataset.register_blueprint(blueprint_path.resolve().as_uri(), set_default=True)

    return server


@dataclass
class CatalogConfig:
    rrd_root: Path = Path("data/exoego-forge-catalog")
    """Directory containing ``<dataset>/**/*.rrd`` files."""
    datasets: tuple[str, ...] = DEFAULT_CATALOG_DATASETS
    """Dataset directories to mount. Empty tuple scans all first-level directories."""
    port: int = 9988
    """gRPC port for the catalog server."""
    application_id: str = APPLICATION_ID
    """Application id used to save default dataset blueprints. Must match converted RRDs."""
    open_browser: bool = False
    """Also host a web viewer and open it."""
    web_port: int = 9091
    """Web viewer port. Only used when ``open_browser`` is true."""


def main(config: CatalogConfig) -> None:
    with mount_catalog(config.rrd_root, datasets=config.datasets, port=config.port, application_id=config.application_id) as server:
        url: str = server.url()
        print()
        print("-" * 72)
        print(f"  Catalog URL:  {url}")
        print()
        print("  In the Rerun viewer: + -> Open Data Source -> paste the URL")
        print(f"  Or from a terminal:  rerun {url}")
        print("-" * 72)

        if config.open_browser:
            rr.serve_web_viewer(web_port=config.web_port, open_browser=True, connect_to=url)
            print(f"\nWeb viewer hosted at http://127.0.0.1:{config.web_port} with the catalog loaded.")

        print("\nServer is up. Ctrl-C to stop.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("shutting down")
