from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from simplecv.apis.exoego_forge_catalog import (
    ASSEMBLY101_LARGE_CATALOG_DATASETS,
    ASSEMBLY101_LARGE_RRD_ROOT,
    DEFAULT_CATALOG_DATASETS,
    Assembly101LargeCatalogConfig,
    _assembly101_dataset_dir,
    _register_default_dataset_blueprint,
    build_assembly101_table_card_blueprint,
    build_exoego_catalog_blueprint,
    build_rrd_index_rows_from_dataset,
    build_rrd_index_rows_from_paths,
    discover_rrd_uris,
)
from simplecv.data.exoego.aria_gen2_pilot import AriaGen2PilotConfig, AriaGen2PilotSequence
from simplecv.data.exoego.assembly101 import Assembly101Config, Assembly101Sequence
from simplecv.data.exoego.ego_dex import EgoDexConfig, EgoDexSequence
from simplecv.data.exoego.hocap import HocapConfig, HocapSequence
from simplecv.data.exoego.hot3d import Hot3dConfig, Hot3dSequence
from simplecv.data.exoego.sequence_identity import SequenceIdentity
from simplecv.data.exoego.umetrack import UmeTrackConfig, UmeTrackSequence


def test_sequence_identity_paths_and_recording_id() -> None:
    identity = SequenceIdentity(
        dataset="umetrack",
        parts=("real", "hand_hand", "testing", "user_05", "recording_09"),
    )

    assert identity.sequence_key == "real/hand_hand/testing/user_05/recording_09"
    assert identity.recording_id == "umetrack__real__hand_hand__testing__user_05__recording_09"
    assert identity.rrd_path(Path("data/exoego-forge-catalog")) == Path(
        "data/exoego-forge-catalog/umetrack/real/hand_hand/testing/user_05/recording_09.rrd"
    )


@pytest.mark.parametrize(
    ("identity", "dataset", "sequence_key", "recording_id"),
    [
        (
            AriaGen2PilotSequence.sequence_identity_for_config(AriaGen2PilotConfig(sequence_name="cook_0")),
            "aria-gen2",
            "cook_0",
            "aria-gen2__cook_0",
        ),
        (
            Assembly101Sequence.sequence_identity_for_config(Assembly101Config(split=None, sequence_name="seq_01")),
            "assembly101",
            "all/seq_01",
            "assembly101__all__seq_01",
        ),
        (
            HocapSequence.sequence_identity_for_config(HocapConfig(subject_id="8", sequence_name="20231024_180733")),
            "hocap",
            "subject_8/20231024_180733",
            "hocap__subject_8__20231024_180733",
        ),
        (
            Hot3dSequence.sequence_identity_for_config(Hot3dConfig(headset="quest3", sequence_name="P0001_10a27bf7")),
            "hot3d-quest3",
            "P0001_10a27bf7",
            "hot3d-quest3__P0001_10a27bf7",
        ),
        (
            UmeTrackSequence.sequence_identity_for_config(
                UmeTrackConfig(
                    data_type="real",
                    hand_interaction="hand_hand",
                    split="testing",
                    user=5,
                    recording_id=9,
                )
            ),
            "umetrack",
            "real/hand_hand/testing/user_05/recording_09",
            "umetrack__real__hand_hand__testing__user_05__recording_09",
        ),
        (
            EgoDexSequence.sequence_identity_for_config(EgoDexConfig(split="test", sequence_name="add_remove_lid", episode=3)),
            "ego-dex",
            "test/add_remove_lid/episode_0003",
            "ego-dex__test__add_remove_lid__episode_0003",
        ),
    ],
)
def test_dataset_sequence_identity_for_config(
    identity: SequenceIdentity,
    dataset: str,
    sequence_key: str,
    recording_id: str,
) -> None:
    assert identity.dataset == dataset
    assert identity.sequence_key == sequence_key
    assert identity.recording_id == recording_id


def test_discover_rrd_uris_groups_by_dataset(tmp_path: Path) -> None:
    aria_rrd = tmp_path / "aria-gen2" / "cook_0.rrd"
    hocap_rrd = tmp_path / "hocap" / "subject_8" / "20231024_180733.rrd"
    hot3d_aria_rrd = tmp_path / "hot3d-aria" / "P0001_4bf4e21a.rrd"
    skipped_rrd = tmp_path / "ego100k" / "video.rrd"
    legacy_hot3d_rrd = tmp_path / "hot3d" / "aria" / "P0001_4bf4e21a.rrd"
    for path in (aria_rrd, hocap_rrd, hot3d_aria_rrd, skipped_rrd, legacy_hot3d_rrd):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a real rrd")

    uris = discover_rrd_uris(tmp_path, datasets=("aria-gen2", "hocap", "hot3d-aria"))

    assert set(uris) == {"aria-gen2", "hocap", "hot3d-aria"}
    assert uris["aria-gen2"] == [aria_rrd.resolve().as_uri()]
    assert uris["hocap"] == [hocap_rrd.resolve().as_uri()]
    assert uris["hot3d-aria"] == [hot3d_aria_rrd.resolve().as_uri()]


def test_assembly101_large_catalog_config_filters_to_assembly101(tmp_path: Path) -> None:
    assembly_rrd: Path = tmp_path / "assembly101" / "all" / "seq_01.rrd"
    hocap_rrd: Path = tmp_path / "hocap" / "subject_8" / "20231024_180733.rrd"
    for path in (assembly_rrd, hocap_rrd):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a real rrd")

    config: Assembly101LargeCatalogConfig = Assembly101LargeCatalogConfig()
    uris: dict[str, list[str]] = discover_rrd_uris(tmp_path, datasets=config.datasets)

    assert config.rrd_root == ASSEMBLY101_LARGE_RRD_ROOT
    assert config.datasets == ASSEMBLY101_LARGE_CATALOG_DATASETS
    assert config.datasets == ("assembly101",)
    assert set(config.datasets).issubset(DEFAULT_CATALOG_DATASETS)
    assert uris == {"assembly101": [assembly_rrd.resolve().as_uri()]}


class _FakeSegmentTable:
    def __init__(self, table: pa.Table) -> None:
        self._table: pa.Table = table

    def collect(self) -> list[pa.RecordBatch]:
        return self._table.to_batches()


class _FakeDatasetEntry:
    def __init__(self, table: pa.Table) -> None:
        self._table: pa.Table = table

    def segment_table(self) -> _FakeSegmentTable:
        return _FakeSegmentTable(self._table)

    def segment_url(self, recording_id: str) -> str:
        return f"rerun+http://127.0.0.1:9988/dataset/fake?segment_id={recording_id}"


class _FakeBlueprintDatasetEntry:
    def __init__(self) -> None:
        self.registered_blueprints: list[tuple[str, bool]] = []

    def register_blueprint(self, blueprint_uri: str, *, set_default: bool) -> None:
        self.registered_blueprints.append((blueprint_uri, set_default))


class _FakeServer:
    pass


def test_build_rrd_index_rows_from_paths(tmp_path: Path) -> None:
    first_rrd: Path = tmp_path / "assembly101" / "all" / "seq_01.rrd"
    second_rrd: Path = tmp_path / "assembly101" / "all" / "nested" / "seq_02.rrd"
    for path in (first_rrd, second_rrd):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a real rrd")

    rows = build_rrd_index_rows_from_paths(tmp_path)

    assert [row.id for row in rows] == [0, 1]
    assert [row.sequence_key for row in rows] == ["all/nested/seq_02", "all/seq_01"]
    assert rows[0].recording_uri == str(second_rrd.resolve())
    assert rows[1].path == str(first_rrd.resolve())
    assert [row.size_bytes for row in rows] == [len(b"not a real rrd"), len(b"not a real rrd")]


def test_assembly101_dataset_dir_accepts_direct_optimized_root(tmp_path: Path) -> None:
    optimized_root: Path = tmp_path / "assembly101" / "optimized"
    all_dir: Path = optimized_root / "all"
    all_dir.mkdir(parents=True)

    assert _assembly101_dataset_dir(optimized_root) == optimized_root.resolve()


def test_assembly101_dataset_dir_accepts_catalog_root(tmp_path: Path) -> None:
    dataset_dir: Path = tmp_path / "assembly101"
    all_dir: Path = dataset_dir / "all"
    all_dir.mkdir(parents=True)

    assert _assembly101_dataset_dir(tmp_path) == dataset_dir.resolve()


def test_register_default_dataset_blueprint_registers_full_segment_blueprint() -> None:
    dataset_entry = _FakeBlueprintDatasetEntry()
    server = _FakeServer()

    blueprint_path: Path = _register_default_dataset_blueprint(
        server,  # type: ignore[arg-type]
        dataset_entry,
        dataset_name="assembly101",
    )

    assert blueprint_path.is_file()
    assert dataset_entry.registered_blueprints == [(blueprint_path.resolve().as_uri(), True)]


def test_build_rrd_index_rows_from_registered_dataset_segments(tmp_path: Path) -> None:
    first_rrd: Path = tmp_path / "assembly101" / "all" / "seq_01.rrd"
    second_rrd: Path = tmp_path / "assembly101" / "all" / "nested" / "seq_02.rrd"
    for path in (first_rrd, second_rrd):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a real rrd")

    segment_table: pa.Table = pa.table(
        {
            "rerun_segment_id": [
                "assembly101__all__seq_01",
                "assembly101__all__nested__seq_02",
            ],
            "property:info:sequence_key": [
                ["all/seq_01"],
                ["all/nested/seq_02"],
            ],
        }
    )
    rows = build_rrd_index_rows_from_dataset(
        _FakeDatasetEntry(segment_table),
        dataset_dir=tmp_path / "assembly101",
    )

    assert [row.id for row in rows] == [0, 1]
    assert [row.sequence_key for row in rows] == ["all/nested/seq_02", "all/seq_01"]
    assert rows[0].recording_uri.endswith("segment_id=assembly101__all__nested__seq_02")
    assert rows[1].path == str(first_rrd.resolve())
    assert [row.size_bytes for row in rows] == [len(b"not a real rrd"), len(b"not a real rrd")]


def test_assembly101_table_card_blueprint_builds() -> None:
    blueprint = build_assembly101_table_card_blueprint(timeline="video_time")

    assert blueprint is not None


def test_catalog_blueprints_exist_for_default_datasets() -> None:
    for dataset_name in DEFAULT_CATALOG_DATASETS:
        blueprint = build_exoego_catalog_blueprint(dataset_name)
        assert blueprint is not None
