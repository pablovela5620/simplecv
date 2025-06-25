# from dataclasses import dataclass
# from pathlib import Path
# from timeit import default_timer as timer
# from typing import Literal

# import cv2
# import numpy as np
# import open3d as o3d
# import rerun as rr
# import rerun.blueprint as rrb
# from einops import rearrange
# from jaxtyping import Float32, Int, UInt16
# from numpy import ndarray
# from tqdm import tqdm

# from simplecv.data.exoego.assembly_101 import Assembly101Sequence
# from simplecv.data.exoego.base_exo_ego import BaseExoEgoSequence, ExoBatchData, ExoData
# from simplecv.data.new_exoego.multicam import MulticamSequence
# from simplecv.ops.tsdf_depth_fuser import Open3DFuser
# from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole, log_video
# from simplecv.video_io import MultiVideoReader

# np.set_printoptions(suppress=True)


# @dataclass
# class VisualzeConfig:
#     rr_config: RerunTyroConfig
#     dataset: Literal["hocap", "assembly101", "multicam"] = "hocap"
#     root_directory: Path = Path("data/hocap/sample")
#     # subject_id: SubjectIDs | None = "8"
#     sequence_name: str = "20231024_180733"
#     num_videos_to_log: Literal[4, 8] = 8
#     log_depths: bool = False
#     send_as_batch: bool = True
#     load_labels: bool = True


# def set_pose_annotation_context(sequence: BaseExoEgoSequence) -> None:
#     rr.log(
#         "/",
#         rr.AnnotationContext(
#             [
#                 rr.ClassDescription(
#                     info=rr.AnnotationInfo(id=0, label="Left Hand", color=(255, 0, 0)),
#                     keypoint_annotations=[
#                         rr.AnnotationInfo(id=id, label=name) for id, name in sequence.hand_id2name.items()
#                     ],
#                     keypoint_connections=sequence.hand_links,
#                 ),
#                 rr.ClassDescription(
#                     info=rr.AnnotationInfo(id=1, label="Right Hand", color=(0, 0, 255)),
#                     keypoint_annotations=[
#                         rr.AnnotationInfo(id=id, label=name) for id, name in sequence.hand_id2name.items()
#                     ],
#                     keypoint_connections=sequence.hand_links,
#                 ),
#             ]
#         ),
#         static=True,
#     )


# def create_blueprint(exo_video_log_paths: list[Path], num_videos_to_log: Literal[4, 8] = 8) -> rrb.Blueprint:
#     active_tab: int = 0  # 0 for video, 1 for images
#     main_view = rrb.Vertical(
#         contents=[
#             rrb.Spatial3DView(
#                 origin="/",
#             ),
#             # take the first 4 video files
#             rrb.Horizontal(
#                 contents=[
#                     rrb.Tabs(
#                         rrb.Spatial2DView(origin=f"{video_log_path.parent}"),
#                         rrb.Spatial2DView(
#                             origin=f"{video_log_path}".replace("video", "depth"),
#                         ),
#                         active_tab=active_tab,
#                     )
#                     for video_log_path in exo_video_log_paths[:4]
#                 ]
#             ),
#         ],
#         row_shares=[3, 1],
#     )
#     additional_views = rrb.Vertical(
#         contents=[
#             rrb.Tabs(
#                 rrb.Spatial2DView(origin=f"{video_log_path.parent}"),
#                 rrb.Spatial2DView(origin=f"{video_log_path}".replace("video", "depth")),
#                 active_tab=active_tab,
#             )
#             for video_log_path in exo_video_log_paths[4:]
#         ]
#     )
#     # do the last 4 videos
#     contents = [main_view]
#     if num_videos_to_log == 8:
#         contents.append(additional_views)

#     blueprint = rrb.Blueprint(
#         rrb.Horizontal(
#             contents=contents,
#             column_shares=[4, 1],
#         ),
#         collapse_panels=True,
#     )
#     return blueprint


# def log_exo_ego_sequence_batch(
#     sequence: BaseExoEgoSequence,
#     *,
#     shortest_timestamp: Int[ndarray, "num_frames"],
#     parent_log_path: Path,
#     timeline: str,
#     log_depth: bool = True,
# ) -> None:
#     exo_batch_data: ExoBatchData = sequence.exo_batch_data
#     ####################
#     # log 3d keypoints #
#     pbar = tqdm(
#         enumerate(
#             (
#                 ("left", (0, 255, 0), 0),
#                 ("right", (0, 255, 0), 1),
#             )
#         ),
#         desc="Logging hand keypoints",
#         total=2,
#     )
#     for hand_idx, (hand_side, color, class_id) in pbar:
#         xyz_stack: Float32[ndarray, "num_frames 21 3"] = exo_batch_data.xyz_stack[:, hand_idx, ...]
#         rr.log(
#             hand_side,
#             rr.Points3D.from_fields(
#                 colors=color,
#                 class_ids=class_id,
#                 keypoint_ids=sequence.hand_ids,
#                 show_labels=False,
#             ),
#             static=True,
#         )

#         rr.send_columns(
#             hand_side,
#             indexes=[rr.TimeNanosColumn(timeline, shortest_timestamp[0 : len(sequence)])],
#             columns=[
#                 *rr.Points3D.columns(
#                     positions=rearrange(
#                         xyz_stack,
#                         "num_frames kpts dim -> (num_frames kpts) dim",
#                     ),
#                 ).partition(lengths=[21] * len(sequence)),
#             ],
#         )
#         for exo_cam in sequence.exo_cam_list:
#             image_log_path: Path = parent_log_path / exo_cam.name / "pinhole" / "video"
#             uv_stack: Float32[ndarray, "num_frames 21 2"] = exo_batch_data.uv_stack_dict[exo_cam.name][:, hand_idx, ...]
#             # filter batch with invalid values
#             uv_stack[uv_stack == -1] = np.nan
#             rr.log(
#                 f"{image_log_path}/{hand_side}",
#                 rr.Points2D.from_fields(
#                     colors=color,
#                     class_ids=class_id,
#                     keypoint_ids=sequence.hand_ids,
#                     show_labels=False,
#                 ),
#                 static=True,
#             )

#             rr.send_columns(
#                 f"{image_log_path}/{hand_side}",
#                 indexes=[rr.TimeNanosColumn(timeline, shortest_timestamp[0 : len(sequence)])],
#                 columns=[
#                     *rr.Points2D.columns(
#                         positions=rearrange(
#                             uv_stack,
#                             "num_frames kpts dim -> (num_frames kpts) dim",
#                         ),
#                     ).partition(lengths=[21] * len(sequence)),
#                 ],
#             )

#     if log_depth:
#         log_depths(
#             sequence,
#             parent_log_path,
#             shortest_timestamp,
#             timeline,
#         )


# def log_exo_ego_sequence_incremental(
#     sequence: BaseExoEgoSequence,
#     *,
#     shortest_timestamp: Int[ndarray, "num_frames"],
#     parent_log_path: Path,
#     timeline: str,
#     log_img: bool = False,
# ) -> None:
#     exo_data: ExoData
#     pbar = tqdm(
#         zip(shortest_timestamp, sequence, strict=True),
#         total=len(shortest_timestamp),
#     )
#     for ts_idx, (timestamp, exo_data) in enumerate(pbar):
#         rr.set_time_nanos(timeline=timeline, nanos=timestamp)
#         # # log ego cameras
#         # current_ego_cam: PinholeParameters = sequence._ego_data[ts_idx]
#         # log_pinhole(
#         #     camera=current_ego_cam,
#         #     cam_log_path=parent_log_path / current_ego_cam.name,
#         #     image_plane_distance=0.1,
#         #     static=False,
#         # )

#         # log 3d keypoints
#         for hand_idx, (hand_side, color, class_id) in enumerate(
#             (
#                 ("left", (0, 255, 0), 0),
#                 ("right", (0, 255, 0), 1),
#             )
#         ):
#             xyz: Float32[ndarray, "21 3"] = exo_data.xyz[hand_idx]
#             rr.log(
#                 hand_side,
#                 rr.Points3D(
#                     xyz,
#                     colors=color,
#                     class_ids=class_id,
#                     keypoint_ids=sequence.hand_ids,
#                     show_labels=False,
#                 ),
#             )
#             # # log 2d keypoints in ego cameras
#             # xyz_hom: Float32[ndarray, "21 4"] = np.hstack((xyz, np.ones((21, 1)))).astype(np.float32)
#             # P_ego: Float32[ndarray, "3 4"] = current_ego_cam.projection_matrix.astype(np.float32)
#             # P_ego: Float32[ndarray, "1 3 4"] = rearrange(P_ego, "n m -> 1 n m")
#             # uvc_ego: Float32[ndarray, "1 21 3"] = projectN3(xyz_hom, P_ego).astype(np.float32)
#             # uv_ego: Float32[ndarray, "21 2"] = uvc_ego[0, :, :2]
#             # uv_ego[uv_ego == -1] = np.nan
#             # image_log_path: Path = parent_log_path / current_ego_cam.name / "pinhole" / "video"
#             # rr.log(
#             #     f"{image_log_path}/{hand_side}",
#             #     rr.Points2D(
#             #         uv_ego,
#             #         colors=color,
#             #         class_ids=class_id,
#             #         keypoint_ids=sequence.hand_ids,
#             #         show_labels=False,
#             #     ),
#             # )
#             # log 2d keypoints in exo cameras
#             for cam_param, bgr in zip(exo_data.cam_params_list, exo_data.bgr_list, strict=True):
#                 uv: Float32[ndarray, "21 2"] = exo_data.uv_dict[cam_param.name][hand_idx]
#                 uv[uv == -1] = np.nan
#                 image_log_path: Path = parent_log_path / cam_param.name / "pinhole" / "video"
#                 rr.log(
#                     f"{image_log_path}/{hand_side}",
#                     rr.Points2D(
#                         uv,
#                         colors=color,
#                         class_ids=class_id,
#                         keypoint_ids=sequence.hand_ids,
#                         show_labels=False,
#                     ),
#                 )
#                 if log_img:
#                     rr.log(
#                         f"{parent_log_path}/{cam_param.name}/pinhole/image",
#                         rr.Image(
#                             bgr,
#                         ).compress(jpeg_quality=75),
#                     )


# def log_depths(
#     sequence: BaseExoEgoSequence,
#     parent_log_path: Path,
#     shortest_timestamp: Int[ndarray, "num_frames"],
#     timeline: str,
# ) -> None:
#     depth_paths: list[dict[str, Path]] | None = sequence.depth_paths
#     if depth_paths is not None:
#         for idx, depths_dict in enumerate(tqdm(depth_paths, desc="Logging depth images")):
#             rr.set_time_nanos(timeline=timeline, nanos=shortest_timestamp[idx])
#             fuser = Open3DFuser(fusion_resolution=0.01, max_fusion_depth=1.25)
#             bgr_list = sequence.exo_video_readers[idx]
#             for exo_cam, bgr in zip(sequence.exo_cam_list, bgr_list, strict=True):
#                 depth_path = depths_dict[exo_cam.name]
#                 assert depth_path.exists(), f"Path {depth_path} does not exist."
#                 depth_image: UInt16[np.ndarray, "480 640"] = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
#                 rgb_hw3 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#                 # rr.log(
#                 #     f"{parent_log_path / exo_cam.name / 'pinhole' / 'depth'}",
#                 #     rr.DepthImage(depth_image, meter=1000),
#                 # )
#                 fuser.fuse_frames(
#                     depth_image,
#                     exo_cam.intrinsics.k_matrix,
#                     exo_cam.extrinsics.cam_T_world,
#                     rgb_hw3,
#                 )
#             mesh: o3d.geometry.TriangleMesh = fuser.get_mesh()
#             mesh.compute_vertex_normals()

#             rr.log(
#                 f"{parent_log_path}/mesh",
#                 rr.Mesh3D(
#                     vertex_positions=mesh.vertices,
#                     triangle_indices=mesh.triangles,
#                     vertex_normals=mesh.vertex_normals,
#                     vertex_colors=mesh.vertex_colors,
#                 ),
#             )
#     else:
#         print("No depth images found.")


# def visualize_exo_ego(config: VisualzeConfig):
#     start_time: float = timer()

#     match config.dataset:
#         case "hocap":
#             sequence: HOCapSequence = HOCapSequence(
#                 data_path=config.root_directory,
#                 sequence_name=config.sequence_name,
#                 subject_id=config.subject_id,
#                 load_labels=config.load_labels,
#             )
#         case "assembly101":
#             sequence: Assembly101Sequence = Assembly101Sequence(
#                 data_path=config.root_directory,
#                 sequence_name=config.sequence_name,
#                 subject_id=None,
#                 load_labels=config.load_labels,
#             )

#         case "multicam":
#             sequence: MulticamSequence = MulticamSequence(
#                 data_path=config.root_directory,
#                 sequence_name=config.sequence_name,
#                 subject_id=None,
#                 load_labels=config.load_labels,
#             )

#     set_pose_annotation_context(sequence)
#     rr.log("/", sequence.world_coordinate_system, static=True)

#     parent_log_path = Path("world")
#     timeline: str = "video_time"

#     exo_video_readers: MultiVideoReader = sequence.exo_video_readers
#     exo_video_files: list[Path] = exo_video_readers.video_paths
#     exo_cam_log_paths: list[Path] = [parent_log_path / exo_cam.name for exo_cam in sequence.exo_cam_list]
#     exo_video_log_paths: list[Path] = [cam_log_paths / "pinhole" / "video" for cam_log_paths in exo_cam_log_paths]

#     blueprint: rrb.Blueprint = create_blueprint(exo_video_log_paths, num_videos_to_log=config.num_videos_to_log)
#     rr.send_blueprint(blueprint)

#     # log stationary exo cameras and video assets
#     for exo_cam in sequence.exo_cam_list:
#         cam_log_path: Path = parent_log_path / exo_cam.name
#         match config.dataset:
#             case "hocap":
#                 image_plane_distance = 0.1
#             case "assembly101":
#                 image_plane_distance = 100.0
#             case "multicam":
#                 image_plane_distance = 25.0
#             case _:
#                 # Default or error case, though Literal type hint should prevent this
#                 raise ValueError(f"Unexpected dataset value: {config.dataset}")
#         log_pinhole(
#             camera=exo_cam,
#             cam_log_path=cam_log_path,
#             image_plane_distance=image_plane_distance,
#             static=True,
#         )

#     # log ego video assets
#     # current_ego_cam: PinholeParameters = sequence._ego_data[0]
#     # log_video(
#     #     video_path=sequence._ego_video_path,
#     #     video_log_path=parent_log_path / current_ego_cam.name / "pinhole" / "video",
#     #     timeline=timeline,
#     # )

#     all_timestamps: list[Int[ndarray, "num_frames"]] = []  # noqa: UP037
#     for video_file, video_log_path in zip(exo_video_files, exo_video_log_paths, strict=True):
#         assert video_file.suffix == ".mp4", f"Video file {video_file} is not an mp4."
#         # Log video asset which is referred to by frame references.
#         frame_timestamps_ns: Int[ndarray, "num_frames"] = log_video(  # noqa: UP037
#             video_file, video_log_path, timeline=timeline
#         )
#         all_timestamps.append(frame_timestamps_ns)

#     # Find the timestamp list with the maximum length.
#     shortest_timestamp: Int[ndarray, "num_frames"] = min(all_timestamps, key=len)  # noqa: UP037
#     assert len(shortest_timestamp) == len(sequence), (
#         f"Length of timestamps {len(shortest_timestamp)} and sequence {len(sequence)} do not match"
#     )

#     if config.load_labels:
#         if config.send_as_batch:
#             log_exo_ego_sequence_batch(
#                 sequence,
#                 shortest_timestamp=shortest_timestamp,
#                 parent_log_path=parent_log_path,
#                 timeline=timeline,
#                 log_depth=config.log_depths,
#             )
#         else:
#             log_exo_ego_sequence_incremental(
#                 sequence,
#                 shortest_timestamp=shortest_timestamp,
#                 parent_log_path=parent_log_path,
#                 timeline=timeline,
#             )

#     print(f"Time taken to load data: {timer() - start_time:.2f} seconds")
