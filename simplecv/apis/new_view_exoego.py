from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import tyro
from jaxtyping import Int
from numpy import ndarray

from simplecv.apis.view_exoego_data import log_exo_ego_sequence_batch, set_pose_annotation_context
from simplecv.camera_parameters import PinholeParameters
from simplecv.configs.ego_dataset_configs import AnnotatedEgoDatasetUnion
from simplecv.data.exoego.assembly_101 import Assembly101Sequence
from simplecv.data.new_exoego.assembly_101_ego import Assembly101EgoSequence, EgoAssembly101Config
from simplecv.data.new_exoego.base_ego import BaseEgoSequence, CamNameType
from simplecv.data.new_exoego.hocap_ego import EgoHocapConfig
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole, log_video
from simplecv.video_io import MultiVideoReader

np.set_printoptions(suppress=True)


@dataclass
class VisualizeConfig:
    rr_config: RerunTyroConfig
    dataset: AnnotatedEgoDatasetUnion
    num_videos_to_log: Literal[4, 8] = 8


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
    ego_sequence: BaseEgoSequence = config.dataset.setup()  # one-liner

    # exo_sequence: Assembly101Sequence = Assembly101Sequence(
    #     data_path=ego_sequence.config.root_directory,
    #     sequence_name=ego_sequence.config.sequence_name,
    #     subject_id=None,
    #     load_labels=True,
    # )

    rr.log("/", ego_sequence.world_coordinate_system, static=True)
    # set_pose_annotation_context(exo_sequence)

    parent_log_path = Path("world")
    timeline: str = "video_time"

    ego_video_readers: MultiVideoReader = ego_sequence.ego_video_readers
    ego_video_files: list[Path] = ego_video_readers.video_paths

    ego_cam_dict: dict[CamNameType, list[PinholeParameters]] = ego_sequence.ego_cam_dict
    ego_cam_log_paths: list[Path] = [parent_log_path / ego_cam_name for ego_cam_name in ego_cam_dict]
    ego_video_log_paths: list[Path] = [cam_log_paths / "pinhole" / "video" for cam_log_paths in ego_cam_log_paths]

    # exo_video_readers: MultiVideoReader = exo_sequence.exo_video_readers
    # exo_video_files: list[Path] = exo_video_readers.video_paths
    # exo_cam_log_paths: list[Path] = [parent_log_path / exo_cam.name for exo_cam in exo_sequence.exo_cam_list]
    # exo_video_log_paths: list[Path] = [cam_log_paths / "pinhole" / "video" for cam_log_paths in exo_cam_log_paths]

    # # log stationary exo cameras and video assets
    # for exo_cam in exo_sequence.exo_cam_list:
    #     cam_log_path: Path = parent_log_path / exo_cam.name
    #     log_pinhole(
    #         camera=exo_cam,
    #         cam_log_path=cam_log_path,
    #         image_plane_distance=100.0,
    #         static=True,
    #     )

    # exo_timestamps: list[Int[ndarray, "num_frames"]] = []  # noqa: UP037
    # for video_file, exo_video_log_path in zip(exo_video_files, exo_video_log_paths, strict=True):
    #     assert video_file.suffix == ".mp4", f"Video file {video_file} is not an mp4."
    #     # Log video asset which is referred to by frame references.
    #     frame_timestamps_ns: Int[ndarray, "num_frames"] = log_video(  # noqa: UP037
    #         video_file, exo_video_log_path, timeline=timeline
    #     )
    #     exo_timestamps.append(frame_timestamps_ns)

    blueprint: rrb.Blueprint = create_blueprint(ego_video_log_paths, num_videos_to_log=config.num_videos_to_log)
    rr.send_blueprint(blueprint)

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

    for ts_idx, ts in enumerate(shortest_timestamp):
        rr.set_time_nanos(timeline=timeline, nanos=ts)
        ego_cam_param_list: list[PinholeParameters]
        for cam_name, ego_cam_param_list in ego_cam_dict.items():
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
                image_plane_distance=20.0,  # Assuming a default value for image plane distance
                static=False,
            )

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
