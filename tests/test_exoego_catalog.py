from pathlib import Path

import pytest

from simplecv.apis.exoego_forge_catalog import DEFAULT_CATALOG_DATASETS, build_exoego_catalog_blueprint, discover_rrd_uris
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


def test_catalog_blueprints_exist_for_default_datasets() -> None:
    for dataset_name in DEFAULT_CATALOG_DATASETS:
        blueprint = build_exoego_catalog_blueprint(dataset_name)
        assert blueprint is not None
