from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float, Int
from numpy import ndarray

# from simplecv.apis.view_exoego_data import log_exo_ego_sequence_batch, set_pose_annotation_context
from simplecv.camera_parameters import PinholeParameters
from simplecv.configs.ego_dataset_configs import AnnotatedEgoDatasetUnion
from simplecv.data.ego.base_ego import BaseEgoSequence, CamNameType, EgoLabels
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence
from simplecv.data.skeleton.coco_133 import COCO_133_ID2NAME, COCO_133_IDS, COCO_133_LINKS
from simplecv.rerun_log_utils import (
    Points2DWithConfidence,
    Points3DWithConfidence,
    RerunTyroConfig,
    confidence_scores_to_rgb,
    log_pinhole,
    log_video,
)
from simplecv.video_io import MultiVideoReader

np.set_printoptions(suppress=True)


@dataclass
class VisualizeConfig:
    rr_config: RerunTyroConfig
    dataset: AnnotatedEgoDatasetUnion
    num_videos_to_log: Literal[4, 8] = 8


def set_annotation_context() -> None:
    rr.log(
        "/",
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label="Coco Wholebody", color=(0, 0, 255)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name) for id, name in COCO_133_ID2NAME.items()
                    ],
                    keypoint_connections=COCO_133_LINKS,
                ),
            ]
        ),
        static=True,
    )


def create_blueprint(exo_video_log_paths: list[Path], num_videos_to_log: Literal[4, 8] = 8) -> rrb.Blueprint:
    active_tab: int = 0  # 0 for video, 1 for images
    main_view = rrb.Vertical(
        contents=[
            rrb.Spatial3DView(
                origin="/",
            ),
            # take the first 4 video files
            rrb.Horizontal(
                contents=[
                    rrb.Tabs(
                        rrb.Spatial2DView(origin=f"{video_log_path.parent}"),
                        active_tab=active_tab,
                    )
                    for video_log_path in exo_video_log_paths[:4]
                ]
            ),
        ],
        row_shares=[3, 1],
    )

    contents = [main_view]

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            contents=contents,
            column_shares=[4, 1],
        ),
        collapse_panels=True,
    )
    return blueprint


def visualize_exo_ego(config: VisualizeConfig):
    exoego_sequence: BaseExoEgoSequence = config.dataset.setup()  # one-liner
    ego_sequence: BaseEgoSequence | None = exoego_sequence.ego_sequence
    exo_sequence: BaseExoSequence | None = exoego_sequence.exo_sequence

    rr.log("/", exoego_sequence.world_coordinate_system, static=True)
    set_annotation_context()

    parent_log_path = Path("world")
    timeline: str = "video_time"

    ego_video_readers: MultiVideoReader = ego_sequence.ego_video_readers
    ego_video_files: list[Path] = ego_video_readers.video_paths

    ego_cam_dict: dict[CamNameType, list[PinholeParameters]] = ego_sequence.ego_cam_dict
    ego_cam_log_paths: list[Path] = [parent_log_path / ego_cam_name for ego_cam_name in ego_cam_dict]
    ego_video_log_paths: list[Path] = [cam_log_paths / "pinhole" / "video" for cam_log_paths in ego_cam_log_paths]

    exo_video_readers: MultiVideoReader = exo_sequence.exo_video_readers
    exo_video_files: list[Path] = exo_video_readers.video_paths
    exo_cam_log_paths: list[Path] = [parent_log_path / exo_cam.name for exo_cam in exo_sequence.exo_cam_list]
    exo_video_log_paths: list[Path] = [cam_log_paths / "pinhole" / "video" for cam_log_paths in exo_cam_log_paths]

    blueprint: rrb.Blueprint = create_blueprint(exo_cam_log_paths, num_videos_to_log=config.num_videos_to_log)
    rr.send_blueprint(blueprint)

    # log stationary exo cameras and video assets
    for exo_cam in exo_sequence.exo_cam_list:
        cam_log_path: Path = parent_log_path / exo_cam.name
        log_pinhole(
            camera=exo_cam,
            cam_log_path=cam_log_path,
            image_plane_distance=exo_sequence.image_plane_distance,
            static=True,
        )

    exo_timestamps: list[Int[ndarray, "num_frames"]] = []  # noqa: UP037
    for video_file, exo_video_log_path in zip(exo_video_files, exo_video_log_paths, strict=True):
        assert video_file.suffix == ".mp4", f"Video file {video_file} is not an mp4."
        # Log video asset which is referred to by frame references.
        frame_timestamps_ns: Int[ndarray, "num_frames"] = log_video(  # noqa: UP037
            video_file, exo_video_log_path, timeline=timeline
        )
        exo_timestamps.append(frame_timestamps_ns)

    ego_timestamps: list[Int[ndarray, "num_frames"]] = []  # noqa: UP037
    for video_file, ego_video_log_path in zip(ego_video_files, ego_video_log_paths, strict=True):
        assert video_file.suffix == ".mp4", f"Video file {video_file} is not an mp4."
        # Log video asset which is referred to by frame references.
        frame_timestamps_ns: Int[ndarray, "num_frames"] = log_video(  # noqa: UP037
            video_file, ego_video_log_path, timeline=timeline
        )
        ego_timestamps.append(frame_timestamps_ns)

    # Find the timestamp list with the maximum length.
    shortest_timestamp: Int[ndarray, "num_frames"] = min(ego_timestamps, key=len)  # noqa: UP037
    assert len(shortest_timestamp) == len(ego_sequence), (
        f"Length of timestamps {len(shortest_timestamp)} and sequence {len(ego_sequence)} do not match"
    )

    ego_labels: EgoLabels = ego_sequence.ego_labels
    xyzc_stack: Float[ndarray, "num_frames 133 4"] = ego_labels.xyzc_stack
    # uvc_stack: Float[ndarray, "n_frames n_views 68 3"] = ego_labels.uvc_stack
    # uv_stack_dict: dict[str, Float[ndarray, "..."]] = ego_labels.uv_stack_dict
    # # assume all confidence scores are the same for all cameras
    # conf_stack: Float[ndarray, "n_frames 68 1"] = uvc_stack[:, 0, :, -1:]  # Keep the confidence scores
    # all_colors_stack: UInt8[ndarray, "n_frames 68 3"] = confidence_scores_to_rgb(confidence_scores=conf_stack)

    for ts_idx, ts in enumerate(shortest_timestamp):
        rr.set_time_nanos(timeline=timeline, nanos=ts)
        ego_cam_param_list: list[PinholeParameters]
        for cam_idx, (cam_name, ego_cam_param_list) in enumerate(ego_cam_dict.items()):
            ego_video_log_path = ego_video_log_paths[cam_idx]
            try:
                ego_cam_param: PinholeParameters = ego_cam_param_list[ts_idx]
            except IndexError:
                print(f"Index {ts_idx} out of bounds for camera {cam_name}")
                continue
            # get the cam log path that corresponds to the camera name, check cam_log_paths if it exists
            cam_name: CamNameType = ego_cam_param.name
            cam_log_matches: list[Path] = [
                cam_log_path for cam_log_path in ego_cam_log_paths if cam_name in cam_log_path.name
            ]
            if not cam_log_matches:
                raise ValueError(f"Camera name {cam_name} not found in all_logs: {ego_cam_log_paths}")
            cam_log_path = cam_log_matches[0]
            log_pinhole(
                camera=ego_cam_param,
                cam_log_path=cam_log_path,
                image_plane_distance=ego_sequence.image_plane_distance,  # Assuming a default value for image plane distance
                static=False,
            )

            # Log the 2D keypoints
            xyz: Float[ndarray, "133 3"] = xyzc_stack[
                ts_idx, ..., :3
            ]  # Get the keypoints for the current timestamp and camera
            rr.log(
                f"{parent_log_path}/keypoints",
                rr.Points3D(
                    positions=xyz,  # Remove the view dimension
                    colors=(0, 255, 0),  # Assuming a default color for the keypoints
                    class_ids=0,
                    keypoint_ids=COCO_133_IDS,
                    show_labels=False,
                ),
            )

    # rr.log(
    #     f"{video_log_path}/keypoints",
    #     Points2DWithConfidence(
    #         positions=uv[0, :, 0:2],  # Remove the view dimension
    #         confidences=uv[0, :, -1],  # Keep the confidence scores
    #         colors=current_colors,
    #         class_ids=0,
    #         keypoint_ids=AVP_IDS,
    #         show_labels=False,
    #     ),
    # )

    # log_exo_ego_sequence_batch(
    #     exo_sequence,
    #     shortest_timestamp=shortest_timestamp,
    #     parent_log_path=parent_log_path,
    #     timeline=timeline,
    #     log_depth=False,
    # )

    # if config.load_labels:
    #     if config.send_as_batch:
    #         log_exo_ego_sequence_batch(
    #             sequence,
    #             shortest_timestamp=shortest_timestamp,
    #             parent_log_path=parent_log_path,
    #             timeline=timeline,
    #             log_depth=config.log_depths,
    #         )
    #     else:
    #         log_exo_ego_sequence_incremental(
    #             sequence,
    #             shortest_timestamp=shortest_timestamp,
    #             parent_log_path=parent_log_path,
    #             timeline=timeline,
    #         )

    # print(f"Time taken to load data: {timer() - start_time:.2f} seconds")
