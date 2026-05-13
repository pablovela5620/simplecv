"""Utilities for serving ExoEgo Forge RRD files through a Rerun catalog.

The module has two entry points:

* ``main`` mounts converted ExoEgo Forge recordings as catalog datasets.
* ``main_large_index`` hosts a lightweight Assembly101 table whose rows point
  at full-size RRD recordings, so the viewer can load one recording on demand.
"""

from __future__ import annotations

import atexit
import base64
import tempfile
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
from rerun import bindings
from rerun.catalog import OnDuplicateSegmentLayer
from rerun.recording_stream import RecordingStream
from tqdm import tqdm

from simplecv.apis.view_exoego import create_container

APPLICATION_ID: str = "exoego-forge"
ASSEMBLY101_LARGE_RRD_ROOT: Path = Path("/home/pablo/0Dev/personal/simplecv/data/exoego-forge-catalog/assembly101/all")
"""Generated full-size Assembly101 RRD root before Rerun manifest optimization."""
ASSEMBLY101_LARGE_OPTIMIZED_RRD_ROOT: Path = Path(
    "/home/pablo/0Dev/personal/simplecv/data/exoego-forge-catalog/assembly101/optimized"
)
"""Manifest-optimized full-size Assembly101 RRD root used for larger-than-RAM viewer testing."""
ASSEMBLY101_LARGE_CATALOG_DATASETS: tuple[str, ...] = ("assembly101",)
"""Dataset filter for the full-size Assembly101 catalog preset."""
ASSEMBLY101_LARGE_INDEX_TABLE_NAME: str = "assembly101_large_rrds"
"""Catalog table name for the lightweight full-size Assembly101 RRD index."""
TABLE_BLUEPRINT_METADATA_KEY: bytes = b"rerun:table_blueprint"
"""Arrow schema metadata key used by Rerun for experimental table blueprints."""
MARKER_FLAG_COLUMN: str = "marker_flag"
"""Boolean table flag column used by the Rerun table UI."""
ASSEMBLY101_CARD_PREVIEW_START_SECONDS: float = 0.45
"""Start of the absolute ``video_time`` window used by Assembly101 table-card previews."""
ASSEMBLY101_CARD_PREVIEW_END_SECONDS: float = 0.55
"""End of the absolute ``video_time`` window used by Assembly101 table-card previews."""
ASSEMBLY101_CARD_PREVIEW_VIDEO_KIND: str = "ego"
"""Assembly101 camera group used for the single 2D video table-card preview."""
ASSEMBLY101_CARD_PREVIEW_VIDEO_CAMERA: str = "e3"
"""Assembly101 camera name used for the single 2D video table-card preview."""

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
    """Build relative Rerun entity paths for video streams.

    Args:
        kind: Camera group name such as ``"ego"`` or ``"exo"``.
        camera_names: Camera stream names available for the dataset.

    Returns:
        Relative entity paths ending in ``pinhole/video``.
    """
    return [Path("world") / kind / camera_name / "pinhole" / "video" for camera_name in camera_names]


def build_exoego_catalog_blueprint(dataset_name: str) -> rrb.Blueprint:
    """Build the default catalog blueprint for one ExoEgo Forge dataset.

    Args:
        dataset_name: Dataset key used to select known ego and exo camera names.

    Returns:
        Rerun blueprint used as the default view when opening a dataset segment.
    """
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


def _register_default_dataset_blueprint(
    server: Any,
    dataset_entry: Any,
    *,
    dataset_name: str,
    application_id: str = APPLICATION_ID,
) -> Path:
    """Save and register the full per-segment default blueprint for one dataset.

    Args:
        server: Rerun server that owns the dataset and temporary blueprint lifetime.
        dataset_entry: Catalog dataset entry returned by the Rerun server client.
        dataset_name: Dataset key used to build the dataset-specific blueprint.
        application_id: Rerun application id used when serializing the blueprint.

    Returns:
        Path to the temporary ``.rbl`` blueprint file registered with the dataset.
    """
    blueprint: rrb.Blueprint = build_exoego_catalog_blueprint(dataset_name)
    tmp_dir = tempfile.TemporaryDirectory(prefix=f"{dataset_name}-")
    # Keep the temporary blueprint file alive for as long as the server object is alive.
    weakref.finalize(server, tmp_dir.cleanup)
    atexit.register(tmp_dir.cleanup)
    blueprint_path: Path = Path(tmp_dir.name) / f"{dataset_name}.rbl"
    blueprint.save(application_id, path=str(blueprint_path))
    dataset_entry.register_blueprint(blueprint_path.resolve().as_uri(), set_default=True)
    return blueprint_path


def discover_rrd_uris(
    rrd_root: Path,
    *,
    datasets: tuple[str, ...] = DEFAULT_CATALOG_DATASETS,
) -> dict[str, list[str]]:
    """Discover local RRD files grouped by first-level catalog dataset directory.

    Args:
        rrd_root: Directory containing one subdirectory per dataset.
        datasets: Dataset directory names to include. An empty tuple scans all
            first-level directories under ``rrd_root``.

    Returns:
        Mapping from dataset name to absolute ``file://`` URI strings for every
        discovered ``.rrd`` file.

    Raises:
        FileNotFoundError: If ``rrd_root`` does not exist or no requested
            datasets contain RRD files.
    """
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
    """Mount local ExoEgo Forge RRDs as one Rerun catalog dataset per source.

    Args:
        rrd_root: Directory containing local RRD files grouped by dataset.
        datasets: Dataset directory names to mount. An empty tuple scans all
            first-level directories under ``rrd_root``.
        port: gRPC port for the Rerun server. If ``None``, Rerun chooses a port.
        application_id: Rerun application id used for registered blueprints.
        show_progress: Whether to show a ``tqdm`` progress bar while registering.

    Returns:
        Running Rerun server with discovered RRD files registered as datasets.

    Raises:
        FileNotFoundError: If no matching RRD files are found.
    """
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
        _register_default_dataset_blueprint(
            server,
            dataset,
            dataset_name=dataset_name,
            application_id=application_id,
        )

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


@dataclass
class Assembly101LargeCatalogConfig(CatalogConfig):
    """Catalog config for viewing only the generated full-size Assembly101 RRD dataset."""

    rrd_root: Path = ASSEMBLY101_LARGE_RRD_ROOT
    """Directory containing the generated ``assembly101/**/*.rrd`` files."""
    datasets: tuple[str, ...] = ASSEMBLY101_LARGE_CATALOG_DATASETS
    """Dataset directories to mount. This preset intentionally mounts only Assembly101."""


@dataclass(frozen=True, slots=True)
class RRDIndexRow:
    """One lightweight index row for a registered RRD recording segment."""

    id: int
    """Stable row id after sorting by sequence key."""
    sequence_key: str
    """Human-readable sequence key under the Assembly101 dataset."""
    recording_uri: str
    """Catalog segment URL used by the Rerun table preview column."""
    path: str
    """Absolute filesystem path for the RRD recording."""
    size_bytes: int
    """RRD file size in bytes."""
    marker_flag: bool = False
    """User-editable marker flag column for table review workflows."""


@dataclass
class Assembly101LargeIndexConfig:
    """Config for a lightweight index of the generated full-size Assembly101 RRD dataset."""

    rrd_root: Path = ASSEMBLY101_LARGE_OPTIMIZED_RRD_ROOT
    """Manifest-optimized directory containing ``assembly101/**/*.rrd`` files."""
    port: int = 9988
    """gRPC port for the catalog server."""
    table_name: str = ASSEMBLY101_LARGE_INDEX_TABLE_NAME
    """Name of the lightweight RRD URL table to create."""
    open_browser: bool = False
    """Also host a web viewer and open it."""
    web_port: int = 9091
    """Web viewer port. Only used when ``open_browser`` is true."""


def _assembly101_dataset_dir(rrd_root: Path, *, dataset_name: str = "assembly101") -> Path:
    """Resolve the Assembly101 dataset directory from a catalog or direct root.

    Args:
        rrd_root: Either a catalog root containing ``assembly101/`` or the
            Assembly101 dataset root itself.
        dataset_name: Dataset directory name to look for under ``rrd_root``.

    Returns:
        Resolved dataset root. This directory usually contains ``all/`` for
        Assembly101 recordings.

    Raises:
        FileNotFoundError: If neither supported directory layout exists.
    """
    root: Path = rrd_root.expanduser().resolve()
    dataset_dir: Path = root / dataset_name
    if dataset_dir.is_dir():
        return dataset_dir

    if root.is_dir() and ((root / "all").is_dir() or any(root.glob("*.rrd"))):
        return root

    optimize_command: str = (
        "pixi run rerun rrd optimize "
        f"{ASSEMBLY101_LARGE_RRD_ROOT} -o {ASSEMBLY101_LARGE_OPTIMIZED_RRD_ROOT}"
    )
    raise FileNotFoundError(
        f"Manifest-optimized Assembly101 dataset directory does not exist: {dataset_dir}\n"
        f"Create it with: {optimize_command}"
    )


def _assembly101_registration_dir(dataset_dir: Path) -> Path:
    """Choose the directory to register with Rerun for Assembly101 segments.

    Args:
        dataset_dir: Resolved Assembly101 dataset root.

    Returns:
        The ``all/`` subdirectory when present, otherwise ``dataset_dir``.
    """
    all_dir: Path = dataset_dir / "all"
    if all_dir.is_dir():
        return all_dir
    return dataset_dir


def build_rrd_index_rows_from_paths(rrd_root: Path, *, dataset_name: str = "assembly101") -> list[RRDIndexRow]:
    """Build filesystem-only RRD index rows.

    This helper bypasses the Rerun catalog and is mainly useful for tests and
    diagnostics.

    Args:
        rrd_root: Catalog root or direct Assembly101 dataset root to scan.
        dataset_name: Dataset directory name to look for when ``rrd_root`` is a
            catalog root.

    Returns:
        Rows sorted by filesystem path with sequence keys relative to the
        resolved dataset root.

    Raises:
        FileNotFoundError: If the dataset root cannot be resolved or contains no
            RRD files.
    """
    dataset_dir: Path = _assembly101_dataset_dir(rrd_root, dataset_name=dataset_name)

    rrd_paths: list[Path] = sorted(dataset_dir.rglob("*.rrd"))
    if not rrd_paths:
        raise FileNotFoundError(f"No RRD files found under {dataset_dir}")
    rows: list[RRDIndexRow] = []
    for idx, rrd_path in enumerate(rrd_paths):
        resolved_path: Path = rrd_path.resolve()
        sequence_key: str = resolved_path.relative_to(dataset_dir).with_suffix("").as_posix()
        row: RRDIndexRow = RRDIndexRow(
            id=idx,
            sequence_key=sequence_key,
            recording_uri=str(resolved_path),
            path=str(resolved_path),
            size_bytes=resolved_path.stat().st_size,
        )
        rows.append(row)
    return rows


def _first_list_value(value: Any) -> Any:
    """Return the first element from Arrow list scalars converted to Python.

    Args:
        value: Python value from ``pyarrow.Array.to_pylist()``.

    Returns:
        The first list item for non-empty lists, ``None`` for empty lists, and
        the original value for non-list values.
    """
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _optional_segment_column_values(table: pa.Table, column_name: str) -> list[Any | None]:
    """Read optional segment metadata values from a catalog segment table.

    Args:
        table: Segment metadata table returned by the Rerun catalog client.
        column_name: Metadata column to read.

    Returns:
        One value per row. Missing columns are represented as ``None`` values.
    """
    if column_name not in table.schema.names:
        return [None] * table.num_rows
    return [_first_list_value(value) for value in table.column(column_name).to_pylist()]


def _sequence_key_from_recording_id(dataset_name: str, recording_id: str) -> str:
    """Recover a slash-separated sequence key from a catalog recording id.

    Args:
        dataset_name: Dataset prefix expected at the start of ``recording_id``.
        recording_id: Segment id using ``__`` separators.

    Returns:
        Sequence key with path separators restored.
    """
    prefix: str = f"{dataset_name}__"
    if recording_id.startswith(prefix):
        return recording_id[len(prefix) :].replace("__", "/")
    return recording_id.replace("__", "/")


def build_rrd_index_rows_from_dataset(
    dataset_entry: Any,
    *,
    dataset_dir: Path,
    dataset_name: str = "assembly101",
) -> list[RRDIndexRow]:
    """Build table rows from registered catalog segment URLs.

    Args:
        dataset_entry: Rerun catalog dataset entry containing registered
            Assembly101 segments.
        dataset_dir: Local dataset root used to resolve file paths and sizes.
        dataset_name: Dataset prefix used when deriving sequence keys from
            recording ids.

    Returns:
        RRD index rows sorted by sequence key and re-numbered with stable ids.

    Raises:
        FileNotFoundError: If the registered dataset has no segment rows.
    """
    segment_batches: list[pa.RecordBatch] = dataset_entry.segment_table().collect()
    if not segment_batches:
        raise FileNotFoundError(
            "Registered Assembly101 dataset has no segments. "
            "Pass the directory that directly contains optimized .rrd files to Rerun Server."
        )
    segment_table: pa.Table = pa.Table.from_batches(segment_batches)
    if segment_table.num_rows == 0:
        raise FileNotFoundError("Registered Assembly101 dataset has no segments.")
    recording_ids: list[str] = [
        str(recording_id) for recording_id in segment_table.column("rerun_segment_id").to_pylist()
    ]
    recording_uris: list[str] = [str(dataset_entry.segment_url(recording_id)) for recording_id in recording_ids]
    sequence_key_values: list[Any | None] = _optional_segment_column_values(segment_table, "property:info:sequence_key")

    rows: list[RRDIndexRow] = []
    for idx, (recording_id, recording_uri, sequence_key_value) in enumerate(
        zip(recording_ids, recording_uris, sequence_key_values, strict=True)
    ):
        sequence_key: str = (
            str(sequence_key_value)
            if sequence_key_value is not None
            else _sequence_key_from_recording_id(dataset_name, recording_id)
        )
        rrd_path: Path = (dataset_dir / f"{sequence_key}.rrd").resolve()
        size_bytes: int = rrd_path.stat().st_size if rrd_path.exists() else 0
        row: RRDIndexRow = RRDIndexRow(
            id=idx,
            sequence_key=sequence_key,
            recording_uri=recording_uri,
            path=str(rrd_path),
            size_bytes=size_bytes,
        )
        rows.append(row)

    rows.sort(key=lambda row: row.sequence_key)
    return [
        RRDIndexRow(idx, row.sequence_key, row.recording_uri, row.path, row.size_bytes, row.marker_flag)
        for idx, row in enumerate(rows)
    ]


def _require_table_blueprints() -> None:
    """Ensure the installed Rerun SDK supports experimental table blueprints.

    Raises:
        RuntimeError: If the current Rerun SDK does not expose the experimental
            table blueprint API used by the lightweight Assembly101 index.
    """
    if not hasattr(rrb, "experimental") or not hasattr(rrb.experimental, "TableBlueprint"):
        raise RuntimeError(
            "Experimental table blueprints require Rerun SDK 0.32 or newer. "
            "Run from the default Pixi environment."
        )


def build_assembly101_table_card_blueprint(*, timeline: str = "video_time") -> rrb.Blueprint:
    """Build the lightweight table-card blueprint for Assembly101 previews.

    Args:
        timeline: Timeline name used by the preview views.

    Returns:
        Rerun blueprint embedded into the Assembly101 index table schema.
    """
    camera_names: dict[str, tuple[str, ...]] = CATALOG_CAMERA_NAMES["assembly101"]

    # The 3D card should show poses, points, and camera frustums without trying
    # to draw every video subtree in the table preview.
    video_exclusion_queries: list[str] = []
    for kind in ("ego", "exo"):
        for camera_name in camera_names[kind]:
            exclusion_query: str = f"- /world/{kind}/{camera_name}/pinhole/video/**"
            video_exclusion_queries.append(exclusion_query)

    # Use a narrow absolute window near the start so each card has a stable,
    # cheap preview frame instead of scanning the full recording.
    preview_start: rr.datatypes.TimeRangeBoundary = rrb.TimeRangeBoundary.absolute(
        seconds=ASSEMBLY101_CARD_PREVIEW_START_SECONDS,
    )
    preview_end: rr.datatypes.TimeRangeBoundary = rrb.TimeRangeBoundary.absolute(
        seconds=ASSEMBLY101_CARD_PREVIEW_END_SECONDS,
    )
    preview_time_ranges: rrb.VisibleTimeRanges = rrb.VisibleTimeRanges(
        timeline=timeline,
        start=preview_start,
        end=preview_end,
    )

    scene_preview_view: rrb.Spatial3DView = rrb.Spatial3DView(
        origin="/",
        name="3D Preview",
        contents=["+ /**", *video_exclusion_queries],
        spatial_information=rrb.SpatialInformation.from_fields(show_axes=True),
        time_ranges=preview_time_ranges,
    )

    # The table card also includes one concrete video stream for quick visual
    # recognition of the sequence.
    video_origin: str = f"/world/{ASSEMBLY101_CARD_PREVIEW_VIDEO_KIND}/{ASSEMBLY101_CARD_PREVIEW_VIDEO_CAMERA}/pinhole"
    video_preview_view: rrb.Spatial2DView = rrb.Spatial2DView(
        origin=video_origin,
        name=f"{ASSEMBLY101_CARD_PREVIEW_VIDEO_KIND} {ASSEMBLY101_CARD_PREVIEW_VIDEO_CAMERA}",
        contents=f"{video_origin}/**",
    )

    return rrb.Blueprint(
        scene_preview_view,
        video_preview_view,
        collapse_panels=True,
    )


def build_rrd_index_table_blueprint(*, timeline: str = "video_time") -> str:
    """Build a Rerun table blueprint with on-demand recording previews.

    Args:
        timeline: Timeline name used by the preview views.

    Returns:
        Base64-encoded Rerun blueprint string suitable for Arrow schema metadata.

    Raises:
        RuntimeError: If the installed Rerun SDK does not support table blueprints.
    """
    _require_table_blueprints()

    blueprint: rrb.Blueprint = build_assembly101_table_card_blueprint(timeline=timeline)
    blueprint_stream = RecordingStream._from_native(
        bindings.new_blueprint(
            application_id="embedded",
            make_default=False,
            make_thread_default=False,
            default_enabled=True,
        )
    )
    blueprint_stream.set_time("blueprint", sequence=0)
    blueprint._log_to_stream(blueprint_stream)
    blueprint_stream.log(
        "/table",
        rrb.experimental.TableBlueprint(
            segment_preview_column="recording_uri",
            flag_column=MARKER_FLAG_COLUMN,
            grid_view_card_title="sequence_key",
            url_column="recording_uri",
        ),
    )
    rrb.TimePanel(timeline=timeline)._log_to_stream(blueprint_stream)

    # Rerun reads this base64 payload from Arrow schema metadata to configure table cards.
    rbl_bytes: bytes = blueprint_stream.memory_recording().drain_as_bytes()
    encoded_blueprint: str = base64.b64encode(rbl_bytes).decode("ascii")
    return f"base64:{encoded_blueprint}"


def build_rrd_index_table_schema(encoded_blueprint: str) -> pa.Schema:
    """Build the Arrow schema for the lightweight RRD index table.

    Args:
        encoded_blueprint: Base64 blueprint payload returned by
            ``build_rrd_index_table_blueprint``.

    Returns:
        Arrow schema with Rerun table index, flag column, and blueprint metadata.
    """
    return pa.schema(
        [
            pa.field("id", pa.int64(), metadata={rr.SORBET_IS_TABLE_INDEX: "true"}),
            pa.field("sequence_key", pa.utf8()),
            pa.field("recording_uri", pa.utf8()),
            pa.field("path", pa.utf8()),
            pa.field("size_bytes", pa.int64()),
            pa.field(MARKER_FLAG_COLUMN, pa.bool_(), metadata={"rerun:is_flag_column": "true"}),
        ],
        metadata={TABLE_BLUEPRINT_METADATA_KEY: encoded_blueprint.encode("ascii")},
    )


def create_rrd_index_table(client: Any, *, table_name: str, rows: list[RRDIndexRow]) -> Any:
    """Create or replace a lightweight RRD URL index table.

    Args:
        client: Rerun catalog client connected to the hosting server.
        table_name: Name of the catalog table to create.
        rows: Row records to append to the table.

    Returns:
        Created Rerun catalog table entry.

    Raises:
        RuntimeError: If the installed Rerun SDK does not support table blueprints.
    """
    existing_table_names: set[str] = set(client.table_names())
    if table_name in existing_table_names:
        client.get_table(table_name).delete()

    encoded_blueprint: str = build_rrd_index_table_blueprint()
    schema: pa.Schema = build_rrd_index_table_schema(encoded_blueprint)
    table = client.create_table(table_name, schema)
    table.append(
        id=[row.id for row in rows],
        sequence_key=[row.sequence_key for row in rows],
        recording_uri=[row.recording_uri for row in rows],
        path=[row.path for row in rows],
        size_bytes=[row.size_bytes for row in rows],
        marker_flag=[row.marker_flag for row in rows],
    )
    return table


def main_large_index(config: Assembly101LargeIndexConfig) -> None:
    """Host the lightweight Assembly101 RRD URL index.

    Args:
        config: Runtime configuration for the optimized Assembly101 dataset and
            catalog table server.
    """
    dataset_dir: Path = _assembly101_dataset_dir(config.rrd_root, dataset_name="assembly101")
    registration_dir: Path = _assembly101_registration_dir(dataset_dir)
    print(f"Serving manifest-backed Assembly101 RRD dataset from {registration_dir}.", flush=True)

    with rr.server.Server(datasets={"assembly101": registration_dir}, port=config.port) as server:
        client = server.client()
        dataset_entry = client.get_dataset("assembly101")
        _register_default_dataset_blueprint(
            server,
            dataset_entry,
            dataset_name="assembly101",
            application_id=APPLICATION_ID,
        )
        rows: list[RRDIndexRow] = build_rrd_index_rows_from_dataset(
            dataset_entry,
            dataset_dir=dataset_dir,
            dataset_name="assembly101",
        )
        total_size_bytes: int = sum(row.size_bytes for row in rows)
        print(f"Creating Assembly101 preview table ({len(rows)} RRDs, {total_size_bytes:,} bytes).", flush=True)
        table = create_rrd_index_table(client, table_name=config.table_name, rows=rows)
        catalog_url: str = server.url()
        table_url: str = f"{catalog_url}/entry/{table.id}"

        print()
        print("-" * 72)
        print(f"  Catalog URL:  {catalog_url}")
        print(f"  Index table:  {table_url}")
        print()
        print("  Open the table with:")
        print(f"    pixi run rerun {table_url}")
        print()
        print("  The table contains catalog segment URLs. Open one row to load just that recording.")
        print("  Enable: Settings > Experimental > Table cards and blueprints")
        print("-" * 72)

        if config.open_browser:
            rr.serve_web_viewer(web_port=config.web_port, open_browser=True, connect_to=table_url)
            print(f"\nWeb viewer hosted at http://127.0.0.1:{config.web_port} with the table loaded.")

        print("\nServer is up. Ctrl-C to stop.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("shutting down")


def main(config: CatalogConfig) -> None:
    """Host a Rerun catalog for converted ExoEgo Forge RRD files.

    Args:
        config: Runtime configuration for the catalog server.
    """
    with mount_catalog(
        config.rrd_root, datasets=config.datasets, port=config.port, application_id=config.application_id
    ) as server:
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
