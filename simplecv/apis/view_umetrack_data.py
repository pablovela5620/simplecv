import warnings
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from einops import rearrange
from jaxtyping import Float, Float32, Int64, UInt8
from numpy import ndarray
from rerun import AnnotationInfo, ClassDescription
from serde import field as serde_field
from serde import serde
from serde.json import from_json

from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.umetrack_temp.camera_models import FisheyeCameraParameter, PinholeCameraParameter
from simplecv.umetrack_temp.cameras import Camera
from simplecv.umetrack_temp.generic_hand_model import (
    LANDMARK,
    UME_HAND_CONNECTIONS,
    HandModelTensor,
    HandPoseLabels,
    SingleHandPose,
    landmarks_from_hand_pose,
)
from simplecv.umetrack_temp.perspective_cropping import (
    gen_crop_parameters_from_points,
    get_crop_points_from_hand_pose,
    rank_hand_visibility_in_cameras,
    warp_image_between_cameras,
)
from simplecv.umetrack_temp.projection import project_points
from simplecv.video_io import VideoReader

HAND_TYPE = ("left", "right")
KEYPOINT_IDS = np.array([landmark.value for landmark in LANDMARK], dtype=np.uint16)
CAMERA_PANEL_ORDER: list[tuple[str, int]] = [
    ("BL", 0),
    ("BR", 1),
    ("TL", 2),
    ("TR", 3),
]


@serde
class UmeTrackCameras:
    """Intrinsic parameters for a single UmeTrack fisheye camera.

    Notes:
        Synthetic sequences encode the fifth and sixth radial coefficients under
        the legacy keys `p3` and `p4`; see https://github.com/facebookresearch/UmeTrack_data/issues/4.
    """

    image_size_x: int = serde_field(rename="ImageSizeX")
    """Horizontal image resolution in pixels."""

    image_size_y: int = serde_field(rename="ImageSizeY")
    """Vertical image resolution in pixels."""

    fx: float
    """Focal length along the X axis in pixels."""

    fy: float
    """Focal length along the Y axis in pixels."""

    cx: float
    """Principal point X coordinate in pixels."""

    cy: float
    """Principal point Y coordinate in pixels."""

    distortion_model: Literal["FishEye62"] = serde_field(rename="DistortionModel")
    """Distortion model identifier exported by UmeTrack."""

    k1: float
    """First radial distortion coefficient."""

    k2: float
    """Second radial distortion coefficient."""

    k3: float
    """Third radial distortion coefficient."""

    k4: float
    """Fourth radial distortion coefficient."""

    p1: float
    """First tangential distortion coefficient."""

    p2: float
    """Second tangential distortion coefficient."""

    k5: float = serde_field(default=0.0)
    """Fifth radial distortion coefficient (defaults to 0.0 if not specified)."""

    k6: float = serde_field(default=0.0)
    """Sixth radial distortion coefficient (defaults to 0.0 if not specified)."""


@serde
class HandModelNumpy:
    """Hand model parameters stored as NumPy arrays.

    Notes:
        Serde loads each field as a NumPy ndarray with the dtype/shape indicated
        by the jaxtyping annotations below.
    """

    joint_rotation_axes: Float32[ndarray, "n_joints=22 3"]
    """Unit rotation axes for each joint frame."""
    joint_rest_positions: Float32[ndarray, "n_joints=22 3"]
    """Joint rest positions expressed in the hand root frame."""
    joint_frame_index: Int64[ndarray, "n_joints=22"]
    """Mapping from joint to the frame index used during skinning."""
    joint_parent: Int64[ndarray, "n_joints=22"]
    """Parent joint indices (negative values indicate the root)."""
    joint_first_child: Int64[ndarray, "n_joints=22"]
    """Index to the first child joint for hierarchical traversal."""
    joint_next_sibling: Int64[ndarray, "n_joints=22"]
    """Index to the next sibling joint for hierarchical traversal."""
    landmark_rest_positions: Float32[ndarray, "num_landmarks 3"]
    """Rest pose landmark coordinates in the hand model frame."""
    landmark_rest_bone_weights: Float32[ndarray, "num_landmarks max_landmark_weights"]
    """Bone blend weights per landmark."""
    landmark_rest_bone_indices: Int64[ndarray, "num_landmarks max_landmark_weights"]
    """Bone indices paired with `landmark_rest_bone_weights`."""
    hand_scale: Float32[ndarray, ""]
    """Global uniform hand scale factor."""
    mesh_vertices: Float32[ndarray, "num_mesh_vertices 3"]
    """Skinned mesh vertices at rest pose."""
    mesh_triangles: Int64[ndarray, "num_mesh_faces 3"]
    """Triangle indices defining the mesh topology."""
    dense_bone_weights: Float32[ndarray, "num_mesh_vertices num_joint_frames"]
    """Blend weights used for dense mesh skinning."""
    joint_limits: Float32[ndarray, "n_joints=22 joint_limit_bounds"]
    """Lower/upper joint angle limits in radians."""


def hand_model_numpy_to_tensor(hand_model: HandModelNumpy) -> HandModelTensor:
    """Convert a NumPy-backed hand model into its torch tensor counterpart.

    Args:
        hand_model: Serde-deserialized hand model containing NumPy arrays.

    Returns:
        A `HandModelTensor` with each field materialized as a torch tensor.

    Notes:
        The conversion preserves the dtype coming from NumPy; this assumes the
        serde loader already supplied the correct float32 / int64 arrays.
    """
    tensor_fields: dict[str, torch.Tensor] = {
        name: torch.from_numpy(value) for name, value in asdict(hand_model).items()
    }
    return HandModelTensor(**tensor_fields)


@serde
class UmeTrackAnnotation:
    cameras: list[UmeTrackCameras]
    """Intrinsic definitions for each UmeTrack fisheye camera."""
    camera_angles: list[float]
    hand_model: HandModelNumpy
    joint_angles: Float32[ndarray, "n_frames n_hands=2 n_joints=22"]
    hand_confidences: Float32[np.ndarray, "n_frames n_hands=2"]
    wrist_transforms: Float32[np.ndarray, "n_frames n_hands=2 4 4"]
    camera_to_world_transforms: Float32[ndarray, "n_frames n_cams 4 4"]


class DataStream:
    def __init__(self, video_path: Path, annotation_path: Path) -> None:
        """
        Datastream for a single sequence. This provides the following:
        - camera intrinsics
        - camera extrinsics (these change per frame)
        - multiview monocular images
        - hand pose labels
        """
        self.video_stream = VideoReader(video_path)
        annotation: UmeTrackAnnotation = from_json(UmeTrackAnnotation, annotation_path.read_text())
        # camera intrinsics
        self.fisheye_cameras = create_cameras(annotation.cameras)
        # camera extrinsics (slam pose of each camera)
        self.world_T_cam_all: Float32[ndarray, "n_frames n_cams 4 4"] = annotation.camera_to_world_transforms
        self.hand_model: HandModelTensor = hand_model_numpy_to_tensor(annotation.hand_model)

        self.hand_pose_labels = HandPoseLabels(
            camera_angles=[float(angle) for angle in annotation.camera_angles],
            camera_to_world_transforms=self.world_T_cam_all,
            hand_model=self.hand_model,
            joint_angles=annotation.joint_angles,
            wrist_transforms=annotation.wrist_transforms,
            hand_confidences=annotation.hand_confidences,
        )
        self._warned_singular_pose = False

    def __len__(self):
        return len(self.hand_pose_labels)

    def __iter__(self) -> Iterator[tuple[ndarray, ndarray, dict[int, SingleHandPose]]]:
        """
        Returns:
            multi_view_images: (n_cams, frame_h, single_cam_w, 3)
            cam_to_world: (n_cams, 4, 4)
            gt_tracking: dict[int, SingleHandPose]
        """
        for frame_idx in range(self.__len__()):
            gt_tracking = {}
            # (h, n_cams * w, 3)
            raw_mono_frame = self.video_stream[frame_idx]
            raw_mono_images: UInt8[ndarray, "frame_h frame_w 3"] = np.asarray(raw_mono_frame, dtype=np.uint8)
            frame_height, frame_width, _ = raw_mono_images.shape
            n_cams = 4
            if frame_width % n_cams != 0:
                raise ValueError(f"Video width {frame_width} is not divisible by expected camera count {n_cams}.")
            single_cam_width = frame_width // n_cams
            multi_view_images: UInt8[ndarray, "n_cams frame_h single_cam_w 3"] = rearrange(
                raw_mono_images,
                "frame_h (n_cams single_cam_w) channels -> n_cams frame_h single_cam_w channels",
                n_cams=n_cams,
                single_cam_w=single_cam_width,
            )
            multi_world_T_cam: Float32[ndarray, "n_cams 4 4"] = self.world_T_cam_all[frame_idx]

            # Skip frames where any camera pose is singular or invalid.
            if not np.all(np.isfinite(multi_world_T_cam)):
                if not self._warned_singular_pose:
                    warnings.warn(
                        "Encountered non-finite camera pose; skipping affected frames.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._warned_singular_pose = True
                continue

            if any(abs(np.linalg.det(world_T_cam[:3, :3])) < 1e-8 for world_T_cam in multi_world_T_cam):
                if not self._warned_singular_pose:
                    warnings.warn(
                        "Encountered singular camera pose; skipping affected frames.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._warned_singular_pose = True
                continue
            for hand_idx in range(0, 2):
                if self.hand_pose_labels.hand_confidences[frame_idx, hand_idx] > 0:
                    gt_tracking[hand_idx] = SingleHandPose(
                        joint_angles=self.hand_pose_labels.joint_angles[frame_idx, hand_idx].astype(
                            np.float32, copy=False
                        ),
                        wrist_xform=self.hand_pose_labels.wrist_transforms[frame_idx, hand_idx].astype(
                            np.float32, copy=False
                        ),
                        hand_confidence=float(self.hand_pose_labels.hand_confidences[frame_idx, hand_idx]),
                    )

            # set camera extrinsics as they change per frame (slam tracking)
            for cam_idx, world_T_cam in enumerate(multi_world_T_cam):
                cam_T_world = np.linalg.inv(world_T_cam)
                self.fisheye_cameras[cam_idx].camera_parameters.set_KRT(
                    K=None, R=cam_T_world[:3, :3], T=cam_T_world[:3, 3]
                )
                self.fisheye_cameras[cam_idx].set_extrinsic(cam_T_world)

            yield multi_view_images, multi_world_T_cam, gt_tracking


def create_cameras(umtrack_camera_list: list[UmeTrackCameras]) -> list[Camera]:
    """
    Given annotations, convert to FisheryCameraParameter. This does not include extrinsic as they change per frame.
    """
    cameras: list[Camera] = []
    for camera_name, umetrack_camera in enumerate(umtrack_camera_list):
        cam_params = FisheyeCameraParameter(name=f"camera_{camera_name}")
        cam_params.set_intrinsic(
            width=umetrack_camera.image_size_x,
            height=umetrack_camera.image_size_y,
            fx=umetrack_camera.fx,
            fy=umetrack_camera.fy,
            cx=umetrack_camera.cx,
            cy=umetrack_camera.cy,
        )
        # dist coeff k can have between 4-6 params
        # dist coeef p can have between 2-4 params, but we can only set 2 as p3/p4 dont make sense for tangential distortion
        cam_params.set_dist_coeff(
            dist_coeff_k=[
                umetrack_camera.k1,
                umetrack_camera.k2,
                umetrack_camera.k3,
                umetrack_camera.k4,
                umetrack_camera.k5,
                umetrack_camera.k6,
            ],
            dist_coeff_p=[
                umetrack_camera.p1,
                umetrack_camera.p2,
            ],
        )
        cameras.append(Camera(cam_params))

    return cameras


def log_camera(
    camera_log_path: str,
    *,
    image: UInt8[ndarray, "image_h image_w 3"],
    cam_params: FisheyeCameraParameter | PinholeCameraParameter,
    image_plane_distance: float,
    image_path_name: str = "image",
) -> None:
    """
    Logs image, camera pinhole, and camera extrinsic
    """
    image_log_path = f"{camera_log_path}/{image_path_name}"
    rr.log(
        image_log_path,
        rr.Pinhole(
            image_from_camera=cam_params.intrinsic33(),
            resolution=(cam_params.width, cam_params.height),
            image_plane_distance=image_plane_distance,
        ),
    )
    rr.log(
        camera_log_path,
        rr.Transform3D(
            translation=cam_params.extrinsic_t,
            mat3x3=cam_params.extrinsic_r,
            relation=rr.TransformRelation.ChildFromParent,
        ),
    )
    # resized_image = resize_image_if_needed(image, cam_params.width, cam_params.height)
    rr.log(image_log_path, rr.Image(image).compress(jpeg_quality=90))


def resize_image_if_needed(
    image: UInt8[ndarray, "image_h image_w 3"], target_width: int, target_height: int
) -> UInt8[ndarray, "target_h target_w 3"]:
    """
    Resizes the provided image to the target resolution if required.
    """
    if image.shape[1] == target_width and image.shape[0] == target_height:
        return image
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def setup_logging(parent_log_path: str = "world") -> str:
    """
    setup logging for rerun along with annotations context for each hand
    """
    rr.log(parent_log_path, rr.ViewCoordinates.RUB, static=True)
    class_descriptions = []
    for hand_idx, hand_type in enumerate(HAND_TYPE):
        class_descriptions.append(
            ClassDescription(
                info=AnnotationInfo(label=f"{hand_type} hand", id=hand_idx),
                keypoint_annotations=[AnnotationInfo(id=lm.value) for lm in LANDMARK],
                keypoint_connections=list(UME_HAND_CONNECTIONS),
            ),
        )
    rr.log(f"{parent_log_path}", rr.AnnotationContext(class_descriptions), static=True)
    return parent_log_path


def create_umetrack_view(
    log_path: str = "world", camera_filter: list[Literal["TL", "TR", "BL", "BR"]] | None = None
) -> None:
    """Send a Rerun blueprint tailored for the UmeTrack visualization layout."""

    if camera_filter is None:
        camera_filter = ["TL", "BR"]
    spatial_view = rrb.Spatial3DView(origin=log_path, name="3D View")

    camera_rows: list[rrb.ContainerLike] = []
    for display_name, camera_idx in CAMERA_PANEL_ORDER:
        if display_name not in camera_filter:
            continue
        crop_views = rrb.Vertical(
            contents=[
                rrb.Spatial2DView(
                    origin=f"{log_path}/recropped_camera_left_{camera_idx}/crop_left_{camera_idx}",
                    contents=[
                        "+ $origin/**",
                    ],
                    name=f"{display_name} Left Crop",
                ),
                rrb.Spatial2DView(
                    origin=f"{log_path}/recropped_camera_right_{camera_idx}/crop_right_{camera_idx}",
                    contents=[
                        "+ $origin/**",
                    ],
                    name=f"{display_name} Right Crop",
                ),
            ],
            row_shares=[1, 1],
            name=f"{display_name} Crops",
        )

        camera_view = rrb.Spatial2DView(
            origin=f"{log_path}/camera_{camera_idx}/image_{camera_idx}",
            contents=[
                "+ $origin/**",
            ],
            name=f"{display_name} Camera",
        )

        camera_rows.append(
            rrb.Horizontal(
                contents=[crop_views, camera_view],
                column_shares=[1, 2],
                name=f"{display_name} Panel",
            )
        )

    right_column = rrb.Vertical(contents=camera_rows, row_shares=[1] * len(camera_rows), name="Camera Panels")

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            contents=[spatial_view, right_column],
            column_shares=[3, 2],
            name="UmeTrack Layout",
        )
    )

    rr.send_blueprint(blueprint)


@dataclass
class UmeTrackVisualizeConfig:
    """Structured configuration for running the exo/ego visualization CLI."""

    rr_config: RerunTyroConfig
    """Command-line options for spawning and configuring the Rerun viewer."""
    data_path: Path
    """Path to data, should be a directory that looks like 'UmeTrack_data/raw_data/x/x/x/user_xx/'."""
    sequence_id: int = 1
    """Sequence ID to visualize (0-indexed)."""


def main(config: UmeTrackVisualizeConfig) -> None:
    if not config.data_path.exists():
        raise FileNotFoundError(config.data_path)

    video_path: Path = sorted(config.data_path.glob("*.mp4"))[config.sequence_id]
    annotation_path: Path = sorted(config.data_path.glob("*.json"))[config.sequence_id]
    datastream = DataStream(video_path, annotation_path)
    camera_angles: list[float] = datastream.hand_pose_labels.camera_angles

    parent_log_path: str = setup_logging()
    create_umetrack_view(parent_log_path)

    for frame_idx, frame_data in enumerate(datastream):
        multi_view_images, _multi_world_T_cam, hand_pose_dict = frame_data
        rr.set_time("frame", sequence=frame_idx)
        hand_model = datastream.hand_model
        landmarks_dict: dict[str, Float32[ndarray, "n_kpts=21 3"] | None] = {
            "left": None,
            "right": None,
        }

        # log hand pose data
        for hand_idx, hand_type in enumerate(HAND_TYPE):
            if hand_idx in hand_pose_dict:
                hand_pose: SingleHandPose = hand_pose_dict[hand_idx]
                landmark: Float32[ndarray, "n_kpts=21 3"] = landmarks_from_hand_pose(hand_model, hand_pose, hand_idx)
                class_ids = np.full(len(landmark), hand_idx, dtype=np.uint16)
                rr.log(
                    f"{parent_log_path}/{hand_type}/landmark",
                    rr.Points3D(landmark, keypoint_ids=KEYPOINT_IDS, class_ids=class_ids, show_labels=False),
                )

                ## Generating Crop based on 3d keypoints
                # first are gt, second are neutral, third are open
                crop_points = get_crop_points_from_hand_pose(hand_model, hand_pose, hand_idx, num_crop_points=63)
                cam_indices: list[int] = rank_hand_visibility_in_cameras(
                    cameras=datastream.fisheye_cameras,
                    hand_model=hand_model,
                    hand_pose=hand_pose,
                    hand_idx=hand_idx,
                    min_required_vis_landmarks=19,
                )
                # creating new perspective cameras
                for cam_idx in cam_indices:
                    current_cam: Camera = datastream.fisheye_cameras[cam_idx]
                    perspective_cam_params: PinholeCameraParameter = gen_crop_parameters_from_points(
                        current_cam,
                        crop_points,
                        new_image_size=(96, 96),
                        mirror_img_x=False,  # True if hand_type == "right" else False,
                        camera_angle=camera_angles[cam_idx],
                        focal_multiplier=0.95,
                    )
                    perspective_cam = Camera(perspective_cam_params)

                    # perform image warping from src camera to dst camera
                    current_image = resize_image_if_needed(
                        multi_view_images[cam_idx],
                        current_cam.camera_parameters.width,
                        current_cam.camera_parameters.height,
                    )
                    crop: UInt8[ndarray, "crop_h crop_w 3"] = warp_image_between_cameras(
                        current_cam, perspective_cam, current_image
                    )

                    cropped_cam_log_path: str = f"{parent_log_path}/recropped_camera_{hand_type}_{cam_idx}"
                    crop_log_path_name: str = f"crop_{hand_type}_{cam_idx}"

                    log_camera(
                        cropped_cam_log_path,
                        image=crop,
                        cam_params=perspective_cam_params,
                        image_plane_distance=50.0,
                        image_path_name=crop_log_path_name,
                    )
                    uv_cropped = project_points(landmark, perspective_cam)
                    rr.log(
                        f"{cropped_cam_log_path}/{crop_log_path_name}/{hand_type}_landmark",
                        rr.Points2D(uv_cropped, keypoint_ids=KEYPOINT_IDS, class_ids=class_ids, show_labels=False),
                    )

                landmarks_dict[hand_type] = landmark

        # log original camera camera data and projected landmarks
        for camera_idx, camera in enumerate(datastream.fisheye_cameras):
            cam_log_path: str = f"{parent_log_path}/camera_{camera_idx}"
            img_log_path_name: str = f"image_{camera_idx}"
            current_image: UInt8[ndarray, "h w 3"] = multi_view_images[camera_idx]
            log_camera(
                cam_log_path,
                image=current_image,
                cam_params=camera.camera_parameters,
                image_plane_distance=25.0,
                image_path_name=img_log_path_name,
            )
            for hand_idx, hand_type in enumerate(HAND_TYPE):
                landmarks: Float32[ndarray, "n_kpts=21 3"] | None = landmarks_dict[hand_type]
                if landmarks is not None:
                    hand_landmarks: Float32[ndarray, "n_kpts=21 3"] = landmarks
                    uv: Float[np.ndarray, "num_points 2"] = project_points(hand_landmarks, camera)
                    rr.log(
                        f"{cam_log_path}/{img_log_path_name}/{hand_type}_landmark",
                        rr.Points2D(
                            uv,
                            keypoint_ids=KEYPOINT_IDS,
                            class_ids=hand_idx,
                            show_labels=False,
                        ),
                    )
