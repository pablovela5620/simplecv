import json
from collections.abc import Generator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
from jaxtyping import Float32, Int
from natsort import natsorted
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates
from serde import field as serde_field
from serde import from_dict, serde
from tqdm import tqdm

from simplecv.data.ego.assembly101_ego import Assembly101EgoSequence
from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.exo.assembly101_exo import Assembly101ExoSequence
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels, ExoEgoSample
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig
from simplecv.data.exoego.sequence_identity import SequenceIdentity
from simplecv.data.skeleton.assembly_hands import assembly21_to_coco133, assembly21_to_coco133_batched
from simplecv.video_utils import Resolution


@serde
class Hand3DKeypoints:
    # Use the "rename" parameter to indicate that the JSON key "0" should map to serde_field "left"
    left: Float32[ndarray, "21 3"] = serde_field(rename="0")
    # And similarly for "1" -> "right"
    right: Float32[ndarray, "21 3"] = serde_field(rename="1")


@dataclass
class Assembly101Config(BaseExoEgoDatasetConfig):
    _target: type = field(default_factory=lambda: Assembly101Sequence)
    root_directory: Path = Path("/mnt/8tb/data/assembly101-original/")
    split: Literal["train", "val", "test"] | None = None
    sequence_name: str = "nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724"  # "nusar-2021_action_both_9012-c07c_9012_user_id_2021-02-01_164345"
    resize: Resolution | None = None  # Resize the video to this resolution, if None, no resizing is done.


class Assembly101Sequence(BaseExoEgoSequence[Assembly101Config]):
    """Assembly101 dataset adapter with 3D annotations expressed in meters."""

    def __init__(self, cfg: Assembly101Config) -> None:
        self._ego_stream_names: list[str] = []
        self._exo_stream_names: list[str] = []
        super().__init__(cfg)

    @classmethod
    def sequence_identity_for_config(cls, cfg: Assembly101Config) -> SequenceIdentity:
        split: str = cfg.split or "all"
        return SequenceIdentity(dataset="assembly101", parts=(split, cfg.sequence_name))

    def __getitem__(self, idx: int | None = None, ts_nano: np.timedelta64 | None = None) -> ExoEgoSample:
        canonical_idx, ts_ns = self._resolve_canonical(idx=idx, ts_nano=ts_nano)
        ego_cam_params_list, ego_bgr_list = self._sample_ego(ts_ns)
        exo_cam_params_list, exo_bgr_list = self._sample_exo(ts_ns)
        labels: ExoEgoLabels | None = self._sample_labels(canonical_idx, ts_ns)

        return ExoEgoSample(
            canonical_index=canonical_idx,
            canonical_timestamp_ns=ts_ns,
            ego_cam_params_list=ego_cam_params_list,
            ego_bgr_list=ego_bgr_list,
            exo_cam_params_list=exo_cam_params_list,
            exo_bgr_list=exo_bgr_list,
            labels=labels,
        )

    def _build_ego(self) -> BaseEgoSequence[Assembly101Config] | None:
        return Assembly101EgoSequence(cfg=self.config)

    def _build_exo(self) -> BaseExoSequence[Assembly101Config] | None:
        return Assembly101ExoSequence(cfg=self.config)

    def load_stream_timestamps_ns(self) -> dict[str, Int[ndarray, "n_frames"]]:
        """Return per-stream timestamps for ego/exo videos (and labels if present).

        Optimization (exp-01): The vanilla loop reads each of the ~12 MP4 files
        from disk via ``rr.AssetVideo(path=...)`` and then ``log_video`` reads
        them again to embed the blob. Read each file once in parallel, build
        ``AssetVideo(contents=...)`` for the timestamp pass, and stash the
        bytes on each sub-sequence's ``_video_blobs`` dict so the downstream
        ``setup_scene`` reuses them via ``getattr`` instead of re-decoding.
        """
        from concurrent.futures import ThreadPoolExecutor

        stream_ts: dict[str, Int[ndarray, "n_frames"]] = {}
        self._ego_stream_names.clear()
        self._exo_stream_names.clear()

        jobs: list[tuple[str, str, Path]] = []
        if self.ego_sequence is not None:
            for name, video_path in zip(
                self.ego_sequence.ego_video_names,
                self.ego_sequence.ego_video_paths,
                strict=True,
            ):
                jobs.append((f"ego/{name}", name, video_path))
        if self.exo_sequence is not None:
            for name, video_path in zip(
                self.exo_sequence.exo_video_names,
                self.exo_sequence.exo_video_paths,
                strict=True,
            ):
                jobs.append((f"exo/{name}", name, video_path))

        if jobs:
            with ThreadPoolExecutor(max_workers=min(len(jobs), 16)) as pool:
                blob_bytes: list[bytes] = list(
                    pool.map(lambda p: p.read_bytes(), [p for _, _, p in jobs])
                )
        else:
            blob_bytes = []

        ego_blob_cache: dict[str, bytes] = {}
        exo_blob_cache: dict[str, bytes] = {}
        ego_asset_cache: dict[str, rr.AssetVideo] = {}
        exo_asset_cache: dict[str, rr.AssetVideo] = {}
        for (stream_name, cam_name, _path), blob in zip(jobs, blob_bytes, strict=True):
            # Build the AssetVideo once and reuse it in setup_scene's log_video
            # call rather than letting that path re-parse the MP4 header.
            asset: rr.AssetVideo = rr.AssetVideo(contents=blob, media_type="video/mp4")
            timestamps: Int[ndarray, "n_frames"] = asset.read_frame_timestamps_nanos()
            stream_ts[stream_name] = timestamps
            if stream_name.startswith("ego/"):
                self._ego_stream_names.append(stream_name)
                ego_blob_cache[cam_name] = blob
                ego_asset_cache[cam_name] = asset
            else:
                self._exo_stream_names.append(stream_name)
                exo_blob_cache[cam_name] = blob
                exo_asset_cache[cam_name] = asset

        if self.ego_sequence is not None and ego_blob_cache:
            self.ego_sequence._video_blobs = ego_blob_cache  # picked up by setup_scene
            self.ego_sequence._video_assets = ego_asset_cache
        if self.exo_sequence is not None and exo_blob_cache:
            self.exo_sequence._video_blobs = exo_blob_cache
            self.exo_sequence._video_assets = exo_asset_cache

        labels: ExoEgoLabels | None = self.exoego_labels
        if labels is not None and labels.timestamps_ns is not None:
            stream_ts["labels"] = labels.timestamps_ns

        return stream_ts

    def load_labels(self) -> ExoEgoLabels:
        """Load COCO-133 hand keypoints in meters for the current sequence.

        Optimized version that skips pyserde and the per-frame Python loop:
        parses the keypoint JSON directly into a pre-allocated
        ``(n_frames, 2, 21, 3)`` numpy array and converts to COCO-133 in
        one vectorized pass via ``assembly21_to_coco133_batched``.
        """
        landmarks3d_dir: Path = self.config.root_directory / "assembly101_camera_and_hand_poses" / "landmarks3D"
        assert landmarks3d_dir.exists(), f"Directory {landmarks3d_dir} does not exist"
        xyz_json_path: Path = landmarks3d_dir / f"{self.config.sequence_name}.json"
        assert xyz_json_path.exists(), f"File {xyz_json_path} does not exist"
        with open(xyz_json_path, "rb") as f:
            raw: dict[str, dict[str, list[list[float]]]] = json.loads(f.read())

        keys_sorted: list[str] = sorted(raw.keys(), key=int)
        num_frames: int = len(keys_sorted)
        xyz_stack_mm: Float32[ndarray, "num_frames 2 21 3"] = np.empty(
            (num_frames, 2, 21, 3), dtype=np.float32
        )
        for i, key in enumerate(keys_sorted):
            frame_dict: dict[str, list[list[float]]] = raw[key]
            xyz_stack_mm[i, 0] = frame_dict["0"]
            xyz_stack_mm[i, 1] = frame_dict["1"]

        # Convert millimetres → metres.
        xyz_stack: Float32[ndarray, "num_frames 2 21 3"] = xyz_stack_mm * np.float32(1e-3)
        xyzc_stack: Float32[ndarray, "num_frames 133 4"] = assembly21_to_coco133_batched(xyz_stack)
        return ExoEgoLabels(xyzc_stack=xyzc_stack)

    @classmethod
    def iter_episode_sequences(cls, cfg: Assembly101Config) -> Generator["Assembly101Sequence", None, None]:
        """
        Iterates over all episode sequences in the dataset specified by the given configuration.

        This class method yields `Assembly101Sequence` instances for each sequence found in the dataset directory structure.
        It expects the dataset to be organized with subject directories named "subject_*", each containing sequence directories.

        Args:
            cfg (Assembly101Config): Configuration object specifying the root directory and other parameters.

        Yields:
            Assembly101Sequence: An instance for each sequence found, with configuration updated for the current subject and sequence.

        Notes:
            - Uses natural sorting for subject and sequence directories.
            - Prints subject ID and sequence name for each iteration using `icecream.ic`.
            - Pauses execution for user input after each sequence (likely for debugging).
        """
        for sequence_dir in cls._iter_sequence_dirs(cfg):
            # print(sequence_dir.name)  # Optionally use logging here
            new_cfg = replace(
                cfg,
                sequence_name=sequence_dir.name,
            )

            try:
                seq = cls(new_cfg)  # may raise
            except Exception as e:
                tqdm.write(f"[skip] {sequence_dir.name}: {e}")
                continue  # go on to the next directory
            else:
                yield seq

    @classmethod
    def num_sequences_for_config(cls, cfg: Assembly101Config) -> int:
        return len(cls._iter_sequence_dirs(cfg))

    @classmethod
    def iter_sequence_configs(cls, cfg: Assembly101Config):
        """Yield one Assembly101Config per source sequence without constructing.

        Constructing an ``Assembly101Sequence`` loads ego/exo cameras and
        labels (~4s/seq). When the batch ingestor only needs the per-sequence
        config to dispatch to a worker, we can skip that work and just walk
        the sequence directories.
        """
        for sequence_dir in cls._iter_sequence_dirs(cfg):
            yield replace(cfg, sequence_name=sequence_dir.name)

    @staticmethod
    def _iter_sequence_dirs(cfg: Assembly101Config) -> list[Path]:
        root: Path = cfg.root_directory
        videos_dir: Path = root / "videos" / "av1-720-new"
        assert videos_dir.exists(), f"Directory {videos_dir} does not exist"
        return natsorted([d for d in videos_dir.iterdir() if d.is_dir()])

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        return rr.ViewCoordinates.BUL

    @property
    def image_plane_distance(self) -> int | float:
        """Get the image plane distance for the camera in meters."""
        return 0.035
