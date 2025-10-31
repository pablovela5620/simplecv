import json
import warnings
from argparse import ArgumentParser
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rerun as rr
from jaxtyping import Float, UInt8
from rerun import AnnotationInfo, ClassDescription
from tqdm import tqdm

from simplecv.umetrack_temp.camera_models import FisheyeCameraParameter
from simplecv.umetrack_temp.cameras import Camera
from simplecv.umetrack_temp.generic_hand_model import (
    HAND_CONNECTIONS,
    LANDMARK,
    HandPoseLabels,
    SingleHandPose,
    landmarks_from_hand_pose,
    load_hand_model_from_dict,
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
        with open(annotation_path) as f:
            annotation = json.load(f)
        # camera intrinsics
        self.fisheye_cameras = create_cameras(annotation["cameras"])
        # camera extrinsics (slam pose of each camera)
        self.world_T_cam_all: Float[np.ndarray, "num_frames num_cameras 4 4"] = np.array(
            annotation["camera_to_world_transforms"]
        )
        self.hand_model = load_hand_model_from_dict(annotation["hand_model"])
        self.hand_pose_labels = HandPoseLabels(
            camera_angles=annotation["camera_angles"],
            camera_to_world_transforms=self.world_T_cam_all,
            hand_model=self.hand_model,
            joint_angles=np.array(annotation["joint_angles"]),
            wrist_transforms=np.array(annotation["wrist_transforms"]),
            hand_confidences=np.array(annotation["hand_confidences"]),
        )
        self._warned_singular_pose = False

    def __len__(self):
        return len(self.hand_pose_labels)

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, dict[int, SingleHandPose]]]:
        """
        Returns:
            multi_view_images: (num_cameras, frame_h, single_cam_w, 3)
            cam_to_world: (num_cameras, 4, 4)
            gt_tracking: dict[int, SingleHandPose]
        """
        for frame_idx in range(self.__len__()):
            gt_tracking = {}
            # (h, num_cameras * w, 3)
            raw_mono_images: UInt8[np.ndarray, "frame_h frame_w 3"] = self.video_stream[frame_idx]
            frame_height, frame_width, _ = raw_mono_images.shape
            num_cameras = 4
            if frame_width % num_cameras != 0:
                raise ValueError(f"Video width {frame_width} is not divisible by expected camera count {num_cameras}.")
            single_cam_width = frame_width // num_cameras
            multi_view_images: UInt8[np.ndarray, "num_cameras frame_h single_cam_w 3"] = raw_mono_images.reshape(
                frame_height,
                num_cameras,
                single_cam_width,
                3,
            ).transpose(1, 0, 2, 3)
            multi_world_T_cam = self.world_T_cam_all[frame_idx]

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
                        joint_angles=self.hand_pose_labels.joint_angles[frame_idx, hand_idx],
                        wrist_xform=self.hand_pose_labels.wrist_transforms[frame_idx, hand_idx],
                        hand_confidence=self.hand_pose_labels.hand_confidences[frame_idx, hand_idx],
                    )

            # set camera extrinsics as they change per frame (slam tracking)
            for cam_idx, world_T_cam in enumerate(multi_world_T_cam):
                cam_T_world = np.linalg.inv(world_T_cam)
                self.fisheye_cameras[cam_idx].camera_parameters.set_KRT(
                    K=None, R=cam_T_world[:3, :3], T=cam_T_world[:3, 3]
                )
                self.fisheye_cameras[cam_idx].set_extrinsic(cam_T_world)

            yield multi_view_images, multi_world_T_cam, gt_tracking


def create_cameras(intri_annotations: list[dict[str, Any]]) -> list[Camera]:
    """
    Given annotations, convert to FisheryCameraParameter. This does not include extrinsic as they change per frame.
    """
    cameras = []
    for camera_name, intri_dict in enumerate(intri_annotations):
        width = intri_dict["ImageSizeX"]
        height = intri_dict["ImageSizeY"]
        fx = intri_dict["fx"]
        fy = intri_dict["fy"]
        cx = intri_dict["cx"]
        cy = intri_dict["cy"]
        dist_coeff_k = [intri_dict[key] for key in intri_dict if key.startswith("k")]
        dist_coeff_p = [intri_dict[key] for key in intri_dict if key.startswith("p")]

        cam_params = FisheyeCameraParameter(name=f"camera_{camera_name}")
        cam_params.set_intrinsic(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)
        # dist coeff k can have between 4-6 params
        # dist coeef p can have between 2-4 params, but we can only set 2
        cam_params.set_dist_coeff(dist_coeff_k=dist_coeff_k, dist_coeff_p=dist_coeff_p[:2])
        cameras.append(Camera(cam_params))

    return cameras


def log_camera(
    camera_log_path: str,
    image: UInt8[np.ndarray, "image_h image_w 3"],
    cam_params: FisheyeCameraParameter,
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
    resized_image = resize_image_if_needed(image, cam_params.width, cam_params.height)
    rr.log(image_log_path, rr.Image(resized_image))


def resize_image_if_needed(
    image: UInt8[np.ndarray, "image_h image_w 3"], target_width: int, target_height: int
) -> UInt8[np.ndarray, "target_h target_w 3"]:
    """
    Resizes the provided image to the target resolution if required.
    """
    if image.shape[1] == target_width and image.shape[0] == target_height:
        return image
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def setup_logging(log_path: str = "world") -> str:
    """
    setup logging for rerun along with annotations context for each hand
    """
    # rr.log_view_coordinates(log_path, up="+Y", right_handed=True, timeless=True)
    rr.log(log_path, rr.ViewCoordinates.RUB, static=True)
    class_descriptions = []
    for hand_idx, hand_type in enumerate(HAND_TYPE):
        class_descriptions.append(
            ClassDescription(
                info=AnnotationInfo(label=f"{hand_type} hand", id=hand_idx),
                keypoint_annotations=[AnnotationInfo(id=lm.value) for lm in LANDMARK],
                keypoint_connections=HAND_CONNECTIONS,
            ),
        )
    rr.log(f"{log_path}", rr.AnnotationContext(class_descriptions), static=True)
    return log_path


def main(data_path: Path, sequence_id: int = 1) -> None:
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    video_path: Path = sorted(data_path.glob("*.mp4"))[sequence_id]
    annotation_path: Path = sorted(data_path.glob("*.json"))[sequence_id]
    datastream = DataStream(video_path, annotation_path)
    camera_angles = datastream.hand_pose_labels.camera_angles

    log_path = setup_logging()

    for frame_idx, (multi_view_images, _multi_world_T_cam, hand_pose_dict) in tqdm(
        enumerate(datastream), total=len(datastream), desc="Processing frames"
    ):
        rr.set_time("frame", sequence=frame_idx)
        hand_model = datastream.hand_model
        landmarks_dict = {"left": None, "right": None}

        # log hand pose data
        for hand_idx, hand_type in enumerate(HAND_TYPE):
            if hand_idx in hand_pose_dict:
                hand_pose = hand_pose_dict[hand_idx]
                landmark = landmarks_from_hand_pose(hand_model, hand_pose, hand_idx)
                class_ids = np.full(len(landmark), hand_idx, dtype=np.uint16)
                rr.log(
                    f"{log_path}/{hand_type}/landmark",
                    rr.Points3D(landmark, keypoint_ids=KEYPOINT_IDS, class_ids=class_ids, show_labels=False),
                )

                ## Generating Crop based on 3d keypoints
                # first are gt, second are neutral, third are open
                crop_points = get_crop_points_from_hand_pose(hand_model, hand_pose, hand_idx, num_crop_points=63)
                cam_indices = rank_hand_visibility_in_cameras(
                    cameras=datastream.fisheye_cameras,
                    hand_model=hand_model,
                    hand_pose=hand_pose,
                    hand_idx=hand_idx,
                    min_required_vis_landmarks=19,
                )
                # creating new perspective cameras
                for cam_idx in cam_indices:
                    current_cam = datastream.fisheye_cameras[cam_idx]
                    perspective_cam_params = gen_crop_parameters_from_points(
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
                    crop = warp_image_between_cameras(current_cam, perspective_cam, current_image)

                    cropped_cam_log_path = f"{log_path}/recropped_camera_{hand_type}_{cam_idx}"
                    crop_log_path_name = f"crop_{hand_type}_{cam_idx}"

                    log_camera(cropped_cam_log_path, crop, perspective_cam_params, image_path_name=crop_log_path_name)
                    uv_cropped = project_points(landmark, perspective_cam)
                    rr.log(
                        f"{cropped_cam_log_path}/{crop_log_path_name}/{hand_type}_landmark",
                        rr.Points2D(uv_cropped, keypoint_ids=KEYPOINT_IDS, class_ids=class_ids, show_labels=False),
                    )

                landmarks_dict[hand_type] = landmark

        # log original camera camera data and projected landmarks
        for camera_idx, camera in enumerate(datastream.fisheye_cameras):
            cam_log_path = f"{log_path}/camera_{camera_idx}"
            img_log_path_name = f"image_{camera_idx}"
            current_image = multi_view_images[camera_idx]
            log_camera(cam_log_path, current_image, camera.camera_parameters, img_log_path_name)
            for hand_idx, hand_type in enumerate(HAND_TYPE):
                if landmarks_dict[hand_type] is not None:
                    uv = project_points(landmarks_dict[hand_type], camera)
                    class_ids = np.full(len(uv), hand_idx, dtype=np.uint16)
                    rr.log(
                        f"{cam_log_path}/{img_log_path_name}/{hand_type}_landmark",
                        rr.Points2D(uv, keypoint_ids=KEYPOINT_IDS, class_ids=class_ids, show_labels=False),
                    )


if __name__ == "__main__":
    parser = ArgumentParser("Visualize data")
    parser.add_argument(
        "--data-path",
        type=Path,
        # default="/hdd/data/UmeTrack_data/raw_data/real/hand_hand/testing/user_12/",
        default="/mnt/12tbdrive/data/UmeTrack_data/raw_data/real/separate_hand/testing/user_19/",
        help="Path to data, should be a directory that looks like\
              'UmeTrack_data/raw_data/x/x/x/user_xx/",
    )
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "quest2_hand_tracking")
    main(args.data_path)
    rr.script_teardown(args)
