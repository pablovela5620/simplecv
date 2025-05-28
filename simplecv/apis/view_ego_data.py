from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from einops import rearrange
from jaxtyping import Float32, Int, UInt8
from numpy import ndarray
from tqdm import tqdm

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.data.exoego.skeleton.avp_fullbody import AVP_ID2NAME, AVP_IDS, AVP_LINKS
from simplecv.ops.triangulate import proj_3d_vectorized
from simplecv.rerun_log_utils import Points3DWithConfidence, RerunTyroConfig, log_pinhole, log_video
from simplecv.video_utils import reencode_video_optimal


@dataclass
class ViewEgoConfig:
    rr_config: RerunTyroConfig
    dataset: Literal["ego-dex"] = "ego-dex"
    root_directory: Path = Path("/home/pablo/0Dev/data/ego-dex/test/")
    sequence_name: str = "wipe_screen"
    num_videos_to_log: Literal[4, 8] = 8
    log_depths: bool = False
    send_as_batch: bool = True
    load_labels: bool = True


@dataclass
class EgoDataSequence:
    video_path: Path
    pinhole_list: list[PinholeParameters]
    llm_description: str
    xyz_stack: Float32[ndarray, "n_frames 68 3"]
    conf_stack: Float32[ndarray, "n_frames 68 1"]


def confidence_scores_to_rgb(
    confidence_scores: Float32[ndarray, "n_frames n_kpts 1"],
) -> UInt8[ndarray, "n_frames n_kpts 3"]:
    """Converts confidence scores to RGB colors using a Red-Yellow-Green gradient.

    The color mapping is as follows:
    - A confidence score of 0.0 is mapped to Red (255, 0, 0).
    - A confidence score of 0.5 is mapped to Yellow (255, 255, 0).
    - A confidence score of 1.0 is mapped to Green (0, 255, 0).
    Scores are linearly interpolated between these points. Values outside the
    [0.0, 1.0] range will be clipped by the function.

        confidence_scores (Float32[ndarray, "n_frames n_kpts 1"]):
            A NumPy array of shape (n_frames, n_kpts, 1) containing
            confidence values. Values are typically between 0.0 and 1.0.

        UInt8[ndarray, "n_frames n_kpts 3"]:
            A NumPy array of shape (n_frames, n_kpts, 3) containing
            the corresponding RGB colors as uint8 values. Each color is
            represented as an array of three integers [R, G, B]."""
    n_frames, n_kpts, _ = confidence_scores.shape
    clipped_confidences: Float32[ndarray, "n_frames n_kpts 1"] = np.clip(confidence_scores, a_min=0.0, a_max=1.0)
    clipped_confidences: Float32[ndarray, "n_frames n_kpts"] = np.squeeze(clipped_confidences, axis=-1)

    colors: UInt8[ndarray, "n_frames n_kpts 3"] = np.zeros((n_frames, n_kpts, 3), dtype=np.uint8)
    # Segment A: red → yellow for conf ≤ 0.5
    mask_low = clipped_confidences <= 0.5
    if mask_low.any():
        t_low = clipped_confidences[mask_low] * 2.0  # 0‥1
        colors[..., 0][mask_low] = 255  # red fixed
        colors[..., 1][mask_low] = (t_low * 255).astype(np.uint8)

    # Segment B: yellow → green for conf > 0.5
    mask_high = ~mask_low
    if mask_high.any():
        t_high = (clipped_confidences[mask_high] - 0.5) * 2.0
        colors[..., 0][mask_high] = ((1.0 - t_high) * 255).astype(np.uint8)
        colors[..., 1][mask_high] = 255  # green fixed

    # blue channel remains 0
    return colors


def set_pose_annotation_context() -> None:
    rr.log(
        "/",
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label="Apple Vison Pro Keypoints", color=(0, 0, 255)),
                    keypoint_annotations=[rr.AnnotationInfo(id=id, label=name) for id, name in AVP_ID2NAME.items()],
                    keypoint_connections=AVP_LINKS,
                ),
            ]
        ),
        static=True,
    )


def parse_hdf5_file(hdf5_path: Path, video_path: Path) -> EgoDataSequence:
    # files contains
    h5py_file = h5py.File(f"{hdf5_path}", "r")
    # contains intrinsics that are right now manually set
    camera = h5py_file["camera"]
    confidences = h5py_file["confidences"]
    transforms = h5py_file["transforms"]

    world_T_camera: Float32[ndarray, "n_frames 4 4"] = transforms.get("camera")[:]
    joints_list: list[Float32[ndarray, "n_frames 3"]] = []
    for joint_name in AVP_ID2NAME.values():
        joint_transform: Float32[ndarray, "n_frames 4 4"] = transforms.get(joint_name)[:]
        joint_xyz: Float32[ndarray, "n_frames 3"] = joint_transform[:, :3, 3]
        joints_list.append(joint_xyz)

    joints_xyz: Float32[ndarray, "n_frames 68 3"] = np.stack(joints_list, axis=1)

    conf_list: list[Float32[ndarray, "n_frames 3"]] = []
    for joint_name in AVP_ID2NAME.values():
        conf: Float32[ndarray, "n_frames"] = confidences.get(joint_name)[:]
        conf_list.append(conf)

    conf_stack: Float32[ndarray, "n_frames 68"] = np.stack(conf_list, axis=1)
    conf_stack: Float32[ndarray, "n_frames 68 1"] = rearrange(conf_stack, "n_frames n_joints -> n_frames n_joints 1")

    # there are some problems with the intrinsics files in hdf5, they're always the same so set to a default
    # fmt: off
    intrinsics: Float32[ndarray, "3 3"] = np.array(
        [[736.6339, 0.0, 960.0],
         [0.0, 736.6339, 540.0],
         [0.0, 0.0, 1.0]]).astype(np.float32)
    # fmt: on

    fl_x: float = float(intrinsics[0, 0])
    fl_y: float = float(intrinsics[1, 1])
    cx: float = float(intrinsics[0, 2])
    cy: float = float(intrinsics[1, 2])

    pinhole_list = []
    for i in range(world_T_camera.shape[0]):
        pinhole = PinholeParameters(
            name="AVP Camera",
            intrinsics=Intrinsics(fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy, camera_conventions="RDF"),
            extrinsics=Extrinsics(world_R_cam=world_T_camera[i][:3, :3], world_t_cam=world_T_camera[i][:3, 3]),
        )
        pinhole_list.append(pinhole)

    ego_sequence = EgoDataSequence(
        video_path=video_path,
        pinhole_list=pinhole_list,
        llm_description=h5py_file.attrs["llm_description"],
        xyz_stack=joints_xyz,
        conf_stack=conf_stack,
    )
    return ego_sequence


def view_ego(config: ViewEgoConfig) -> None:
    print("Starting ego data viewer...")
    sequence_path: Path = config.root_directory / config.sequence_name
    assert sequence_path.exists(), f"Sequence path {sequence_path} does not exist."
    video_paths = sorted(sequence_path.glob("*.mp4"))
    hdf5_paths = sorted(sequence_path.glob("*.hdf5"))
    # check that theres at least one video file
    if len(video_paths) == 0:
        raise ValueError("No video files found in the specified directory.")

    for video_path, hdf5_path in zip(video_paths, hdf5_paths, strict=True):
        assert video_path.stem == hdf5_path.stem
        ego_sequence: EgoDataSequence = parse_hdf5_file(hdf5_path, video_path)

        break

    parent_log_path = Path("world")
    timeline = "video_time"

    cam_log_path: Path = parent_log_path / "camera"
    pinhole_log_path: Path = cam_log_path / "pinhole"
    video_log_path: Path = pinhole_log_path / "video"

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(),
            rrb.Vertical(
                rrb.TextDocumentView(origin="llm_description"),
                rrb.Spatial2DView(origin=video_log_path),
                row_shares=[1, 10],
            ),
            column_shares=[2, 1],
        ),
        collapse_panels=True,
    )

    set_pose_annotation_context()
    rr.log("/", rr.ViewCoordinates.RUB, static=True)

    rr.send_blueprint(blueprint=blueprint)
    new_video_path: Path = reencode_video_optimal(input_video_path=ego_sequence.video_path)
    frame_timestamps_ns: Int[ndarray, "num_frames"] = log_video(
        new_video_path, video_log_path=video_log_path, timeline=timeline
    )

    joints_log_path: Path = parent_log_path / "avp_keypoints"
    xyz_stack: Float32[ndarray, "n_frames 68 3"] = ego_sequence.xyz_stack
    conf_stack: Float32[ndarray, "n_frames 68 1"] = ego_sequence.conf_stack

    all_colors_stack: UInt8[ndarray, "n_frames 68 3"] = confidence_scores_to_rgb(confidence_scores=conf_stack)
    rearranged_colors: UInt8[ndarray, "_ 3"] = rearrange(all_colors_stack, "n_frames kpts rgb -> (n_frames kpts) rgb")

    rr.log("llm_description", rr.TextDocument(text=ego_sequence.llm_description), static=True)
    # rr.log(
    #     f"{joints_log_path}",
    #     rr.Points3D.from_fields(
    #         radii=0.005,
    #         class_ids=0,
    #         keypoint_ids=AVP_IDS,
    #         show_labels=False,
    #     ),
    #     static=True,
    # )
    # rr.send_columns(
    #     f"{joints_log_path}",
    #     indexes=[rr.TimeNanosColumn(timeline, frame_timestamps_ns)],
    #     columns=[
    #         *rr.Points3D.columns(
    #             positions=rearrange(
    #                 xyz_stack,
    #                 "num_frames kpts dim -> (num_frames kpts) dim",
    #             ),
    #             colors=rearranged_colors,  # Added dynamic confidence based colors
    #         ).partition(lengths=[n_kpts] * len(frame_timestamps_ns)),
    #     ],
    # )
    for ts_idx, (ts, pinhole) in enumerate(
        tqdm(
            zip(frame_timestamps_ns, ego_sequence.pinhole_list, strict=True),
            desc="Logging pinhole cameras",
            total=len(frame_timestamps_ns),
        )
    ):
        rr.set_time_nanos(timeline=timeline, nanos=ts)
        cam_log_path = parent_log_path / "camera"
        log_pinhole(
            pinhole,
            cam_log_path=cam_log_path,
            image_plane_distance=0.1,
            static=False,
        )
        current_xyz: Float32[ndarray, "68 3"] = xyz_stack[ts_idx, ...]
        current_conf: Float32[ndarray, "68 1"] = conf_stack[ts_idx, ...]
        current_colors: UInt8[ndarray, "68 3"] = all_colors_stack[ts_idx]
        rr.log(
            f"{joints_log_path}",
            Points3DWithConfidence(
                positions=current_xyz,
                confidences=current_conf.squeeze(),
                colors=current_colors,
                class_ids=0,
                keypoint_ids=AVP_IDS,
                show_labels=False,
                radii=0.005,
            ),
        )
