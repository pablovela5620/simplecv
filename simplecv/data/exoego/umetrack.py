from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
from jaxtyping import Float32
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates
from serde.json import from_json

from simplecv.apis.view_umetrack_data import UmeTrackAnnotation, hand_model_numpy_to_tensor
from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.ego.umetrack_ego import UmeTrackEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig
from simplecv.data.skeleton.assembly_hands import assembly21_to_coco133
from simplecv.umetrack_temp.generic_hand_model import HandModelTensor, SingleHandPose, landmarks_from_hand_pose


@dataclass
class UmeTrackConfig(BaseExoEgoDatasetConfig):
    _target: type = field(default_factory=lambda: UmeTrackSequence)
    root_directory: Path = Path("/mnt/8tb/data/umetrack-split")
    data_type: Literal["synthetic", "real"] = "real"
    split: Literal["training", "testing"] = "training"
    hand_interaction: Literal["separate_hand", "hand_hand"] = "separate_hand"
    user: int = 15
    recording_id: int = 0


class UmeTrackSequence(BaseExoEgoSequence[UmeTrackConfig]):
    """Assembly101 dataset adapter with 3D annotations expressed in meters."""

    def __getitem__(self, idx) -> None:
        return None

    def _build_ego(self) -> BaseEgoSequence[UmeTrackConfig] | None:
        return UmeTrackEgoSequence(cfg=self.config)

    def _build_exo(self) -> BaseExoSequence[UmeTrackConfig] | None:
        return None

    def load_labels(self) -> ExoEgoLabels:
        """Load COCO-133 hand keypoints in meters for the current sequence."""

        recording_dir: Path = (
            self.config.root_directory
            / self.config.data_type
            / self.config.hand_interaction
            / self.config.split
            / f"user_{self.config.user:02d}"
            / f"recording_{self.config.recording_id:02d}"
        )
        annotation_path: Path = recording_dir / f"recording_{self.config.recording_id:02d}.json"
        assert annotation_path.exists(), f"Annotation file {annotation_path} does not exist."

        annotation: UmeTrackAnnotation = from_json(UmeTrackAnnotation, annotation_path.read_text())
        hand_model_tensor: HandModelTensor = hand_model_numpy_to_tensor(annotation.hand_model)

        num_frames: int = annotation.joint_angles.shape[0]
        # initialize with NaNs and zero confidence
        xyzc_stack: Float32[ndarray, "num_frames 133 4"] = np.full((num_frames, 133, 4), np.nan, dtype=np.float32)
        xyzc_stack[:, :, 3] = np.float32(0.0)

        scale_to_meters: float = 1e-3
        prev_landmarks_lr: Float32[ndarray, "2 21 3"] = np.full((2, 21, 3), np.nan, dtype=np.float32)
        for frame_idx in range(num_frames):
            landmarks_lr: Float32[ndarray, "2 21 3"] = np.full((2, 21, 3), np.nan, dtype=np.float32)
            hand_confidences: Float32[ndarray, "2"] = annotation.hand_confidences[frame_idx].astype(
                np.float32, copy=False
            )
            for hand_idx in range(2):
                confidence: float = float(hand_confidences[hand_idx])
                if confidence > 0.0:
                    joint_angles: Float32[ndarray, "22"] = annotation.joint_angles[frame_idx, hand_idx].astype(
                        np.float32, copy=False
                    )
                    wrist_transform: Float32[ndarray, "4 4"] = annotation.wrist_transforms[frame_idx, hand_idx].astype(
                        np.float32, copy=False
                    )
                    hand_pose: SingleHandPose = SingleHandPose(
                        joint_angles=joint_angles,
                        wrist_xform=wrist_transform,
                        hand_confidence=confidence,
                    )
                    landmarks_world: Float32[ndarray, "21 3"] = landmarks_from_hand_pose(
                        hand_model_tensor, hand_pose, hand_idx
                    ).astype(np.float32, copy=False)
                    scaled_landmarks: Float32[ndarray, "21 3"] = landmarks_world * scale_to_meters
                    landmarks_lr[hand_idx] = scaled_landmarks
                    prev_landmarks_lr[hand_idx] = scaled_landmarks
                else:
                    landmarks_lr[hand_idx] = prev_landmarks_lr[hand_idx]

            xyzc_stack[frame_idx] = assembly21_to_coco133(landmarks_lr)
            visible_hands: Float32[ndarray, "2"] = np.maximum(hand_confidences, np.float32(0.0))
            adjustments: tuple[tuple[int, int], ...] = ((0, 91), (1, 112))
            wrist_indices: tuple[int, int] = (9, 10)
            thumb_base_indices: tuple[int, int] = (92, 113)
            for hand_idx, coco_offset in adjustments:
                confidence: float = float(visible_hands[hand_idx])
                if confidence <= 0.0:
                    xyzc_stack[frame_idx, coco_offset : coco_offset + 21, 3] = np.float32(0.0)
                    xyzc_stack[frame_idx, wrist_indices[hand_idx], 3] = np.float32(0.0)
                    xyzc_stack[frame_idx, thumb_base_indices[hand_idx], 3] = np.float32(0.0)
                else:
                    xyzc_stack[frame_idx, coco_offset : coco_offset + 21, 3] = np.float32(confidence)
                    xyzc_stack[frame_idx, wrist_indices[hand_idx], 3] = np.float32(confidence)
                    if not np.isnan(xyzc_stack[frame_idx, thumb_base_indices[hand_idx], :3]).all():
                        xyzc_stack[frame_idx, thumb_base_indices[hand_idx], 3] = np.float32(confidence)

        return ExoEgoLabels(
            xyzc_stack=xyzc_stack,
        )

    @classmethod
    def iter_episode_sequences(cls, cfg: UmeTrackConfig) -> Generator["UmeTrackSequence", None, None]:
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
        raise NotImplementedError("UmeTrack dataset does not support iterating over multiple sequences.")

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        return rr.ViewCoordinates.BUL

    @property
    def image_plane_distance(self) -> int | float:
        """Get the image plane distance for the camera in meters."""
        return 0.035
