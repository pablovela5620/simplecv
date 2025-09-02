from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from einops import rearrange
from jaxtyping import Float, Float32, Int, UInt8
from numpy import ndarray

from simplecv.camera_parameters import PinholeParameters
from simplecv.configs.exoego_dataset_configs import AnnotatedEgoDatasetUnion
from simplecv.data.ego.base_ego import BaseEgoSequence, CamNameType
from simplecv.data.exo.base_exo import BaseExoSequence, ManoStack
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels
from simplecv.data.skeleton.coco_133 import (
    COCO_133_ID2NAME,
    COCO_133_IDS,
    COCO_133_LINKS,
    LEFT_HAND_IDX,
    RIGHT_HAND_IDX,
)
from simplecv.ops.triangulate import proj_3d_vectorized
from simplecv.rerun_log_utils import (
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
    max_exo_videos_to_log: Literal[4, 8] = 8
    log_exo: bool = True
    log_ego: bool = True


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


def create_blueprint(
    *,
    ego_video_log_paths: list[Path] | None = None,
    exo_video_log_paths: list[Path] | None = None,
    max_exo_videos_to_log: Literal[4, 8] = 8,
) -> rrb.Blueprint:
    """Creates a Rerun blueprint for visualizing ego and exo-centric video streams.

    This function constructs a Rerun blueprint layout. It starts with a main 3D
    spatial view. If ego-centric video paths are provided, it adds a vertical
    panel on the right with a tab for each ego video. If exo-centric video
    paths are provided, it adds a horizontal panel at the bottom with a tab for
    each exo video.

    Args:
        ego_video_log_paths: Optional list of paths to ego-centric video logs.
            If provided, a vertical panel with tabs for each video's 2D view
            is added to the right of the main 3D view.
        exo_video_log_paths: Optional list of paths to exo-centric video logs.
            If provided, a horizontal panel with tabs for each video's 2D view
            is added below the main view.
        max_exo_videos_to_log: The maximum number of exo-centric videos to display
            in the blueprint. Defaults to 8.

    Returns:
        A `rrb.Blueprint` object defining the layout for the Rerun viewer.
    """
    main_view = rrb.Spatial3DView(
        origin="/",
    )

    if ego_video_log_paths is not None:
        ego_view = rrb.Vertical(
            contents=[
                rrb.Tabs(
                    rrb.Spatial2DView(origin=f"{video_log_path.parent}"),
                )
                for video_log_path in ego_video_log_paths
            ]
        )
        main_view = rrb.Horizontal(
            contents=[main_view, ego_view],
            column_shares=[4, 1],
        )

    if exo_video_log_paths is not None:
        exo_view = rrb.Horizontal(
            contents=[
                rrb.Tabs(
                    rrb.Spatial2DView(origin=f"{video_log_path.parent}"),
                )
                for video_log_path in exo_video_log_paths[:max_exo_videos_to_log]
            ]
        )
        main_view = rrb.Vertical(
            contents=[main_view, exo_view],
            row_shares=[4, 1],
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


def filter_out_of_bounds_keypoints(
    uv_stack: Float[ndarray, "... 2"],
    camera_params: PinholeParameters,
    margin_percentage: float = 0.2,
) -> Float[ndarray, "... 2"]:
    """Filters out-of-bounds 2D keypoints by setting them to NaN."""
    width: int | float = camera_params.intrinsics.width
    height: int | float = camera_params.intrinsics.height
    margin_x: float = margin_percentage * width
    margin_y: float = margin_percentage * height

    uv_stack[..., 0] = np.where(
        (uv_stack[..., 0] < -margin_x) | (uv_stack[..., 0] > width + margin_x), np.nan, uv_stack[..., 0]
    )
    uv_stack[..., 1] = np.where(
        (uv_stack[..., 1] < -margin_y) | (uv_stack[..., 1] > height + margin_y),
        np.nan,
        uv_stack[..., 1],
    )
    return uv_stack


def log_exoego_batch(
    exoego_sequence: BaseExoEgoSequence,
    parent_log_path: Path,
    timeline: str,
    shortest_timestamp: Int[ndarray, "n_frames"],
    log_ego: bool = True,
    log_exo: bool = True,
) -> None:
    exoego_labels: ExoEgoLabels | None = exoego_sequence.exoego_labels
    if exoego_labels is not None:
        ### Send XYZ coordinates
        xyzc_stack: Float[ndarray, "n_frames 133 4"] = exoego_labels.xyzc_stack
        xyz_stack: Float[ndarray, "n_frames 133 3"] = xyzc_stack[:, :, :3]
        xyz_hom_stack: Float[ndarray, "n_frames 133 4"] = np.concatenate(
            [xyz_stack, np.ones_like(xyz_stack[..., :1])], axis=-1
        )
        conf_stack: Float[ndarray, "n_frames 133"] = xyzc_stack[:, :, 3]
        colors: UInt8[ndarray, "n_frames 133 3"] = confidence_scores_to_rgb(
            confidence_scores=conf_stack[..., np.newaxis]
        )
        rr.log(
            f"{parent_log_path}/keypoints",
            rr.Points3D.from_fields(
                class_ids=0,
                keypoint_ids=COCO_133_IDS,
                show_labels=False,
            ),
            static=True,
        )
        rr.send_columns(
            f"{parent_log_path}/keypoints",
            indexes=[rr.TimeColumn(timeline, duration=1e-9 * shortest_timestamp[0 : len(xyzc_stack)])],
            columns=[
                *rr.Points3D.columns(
                    positions=rearrange(
                        xyz_stack,
                        "n_frames kpts dim -> (n_frames kpts) dim",
                    ),
                    colors=rearrange(
                        colors,
                        "n_frames kpts dim -> (n_frames kpts) dim",
                    ),
                ).partition(lengths=[len(COCO_133_IDS)] * len(xyzc_stack)),
            ],
        )

        ### Send MANO Data, this includes
        mano_stack: ManoStack | None = exoego_sequence.exoego_labels.mano_stack
        if mano_stack is not None:
            from simplecv.ops.mano_np import MANOLayerNP

            mano_layers = [
                MANOLayerNP(side="right", betas=mano_stack.betas),
                MANOLayerNP(side="left", betas=mano_stack.betas),
            ]
            mano_poses: Float32[ndarray, "n_frames n_hands=2 51"] = mano_stack.poses
            mano_poses: Float32[ndarray, "n_hands=2 n_frames 51"] = rearrange(
                mano_poses, "n_frames n_hands pose -> n_hands n_frames pose"
            )
            # Prepare a single COCO-133 buffer (both hands combined)
            n_frames_mano_total: int = min(mano_poses.shape[1], len(shortest_timestamp))
            xyz_coco_mano: Float32[ndarray, "n_frames n_joints_coco=133 3"] = np.full(
                (n_frames_mano_total, 133, 3), np.nan, dtype=np.float32
            )
            conf_coco_mano: Float32[ndarray, "n_frames n_joints_coco=133"] = np.zeros(
                (n_frames_mano_total, 133), dtype=np.float32
            )
            for mano_pose, mano_layer in zip(mano_poses, mano_layers, strict=True):
                poses: Float32[ndarray, "n_frames 48"] = mano_pose[:, :48]
                translations: Float32[ndarray, "n_frames 3"] = mano_pose[:, 48:51]
                mano_outputs: tuple[
                    Float32[ndarray, "n_frames n_verts=778 3"],
                    Float32[ndarray, "n_frames n_joints=21 3"],
                ] = mano_layer(poses, translations)
                verts: Float32[ndarray, "n_frames n_verts=778 3"] = mano_outputs[0]
                xyz_mano: Float32[ndarray, "n_frames n_joints=21 3"] = mano_outputs[1]

                # Aggregate MANO joints (21) → into single COCO-133 buffer
                xyz_mano_np: Float32[ndarray, "n_frames n_joints=21 3"] = xyz_mano
                hand_idx: ndarray = RIGHT_HAND_IDX if mano_layer.side == "right" else LEFT_HAND_IDX
                xyz_coco_mano[:, hand_idx, :] = xyz_mano_np[0:n_frames_mano_total]
                conf_coco_mano[:, hand_idx] = 1.0

                # send verts
                rr.log(
                    f"{parent_log_path}/mano_{mano_layer.side}_verts",
                    rr.Points3D.from_fields(
                        show_labels=False,
                    ),
                    static=True,
                )
                rr.send_columns(
                    f"{parent_log_path}/mano_{mano_layer.side}_verts",
                    indexes=[rr.TimeColumn(timeline, duration=1e-9 * shortest_timestamp[0 : len(xyzc_stack)])],
                    columns=[
                        *rr.Points3D.columns(
                            positions=rearrange(
                                verts,
                                "n_frames kpts dim -> (n_frames kpts) dim",
                            ),
                        ).partition(lengths=[778] * len(verts)),
                    ],
                )

                # Log MANO mesh: static faces from the MANO layer, dynamic per-frame vertices
                faces_np: Int[ndarray, "n_faces=1538 3"] = mano_layer.f.astype(np.int32)
                mesh_entity_path: Path = parent_log_path / f"mano_{mano_layer.side}_mesh"
                rr.log(
                    f"{mesh_entity_path}",
                    rr.Mesh3D.from_fields(
                        triangle_indices=faces_np,
                    ),
                    static=True,
                )

                # Stream vertex positions over time under the same entity using send_columns
                verts_np: Float32[ndarray, "n_frames 778 3"] = verts
                n_frames_mesh: int = min(len(verts_np), len(shortest_timestamp))
                rr.send_columns(
                    f"{mesh_entity_path}",
                    indexes=[rr.TimeColumn(timeline, duration=1e-9 * shortest_timestamp[0:n_frames_mesh])],
                    columns=[
                        *rr.Mesh3D.columns(
                            vertex_positions=rearrange(
                                verts_np[0:n_frames_mesh],
                                "n v d -> (n v) d",
                            ),
                        ).partition(lengths=[verts_np.shape[1]] * n_frames_mesh),
                    ],
                )

            # Log a single combined MANO keypoints stream (both hands)
            colors_coco: UInt8[ndarray, "n_frames 133 3"] = confidence_scores_to_rgb(
                confidence_scores=conf_coco_mano[..., np.newaxis]
            )
            rr.log(
                f"{parent_log_path}/mano_keypoints",
                rr.Points3D.from_fields(
                    class_ids=0,
                    keypoint_ids=COCO_133_IDS,
                    show_labels=False,
                ),
                static=True,
            )
            rr.send_columns(
                f"{parent_log_path}/mano_keypoints",
                indexes=[rr.TimeColumn(timeline, duration=1e-9 * shortest_timestamp[0:n_frames_mano_total])],
                columns=[
                    *rr.Points3D.columns(
                        positions=rearrange(
                            xyz_coco_mano,
                            "n_frames kpts dim -> (n_frames kpts) dim",
                        ),
                        colors=rearrange(
                            colors_coco,
                            "n_frames kpts dim -> (n_frames kpts) dim",
                        ),
                    ).partition(lengths=[len(COCO_133_IDS)] * n_frames_mano_total),
                ],
            )

    ###########################
    # batch send all exo cams #
    ###########################
    if exoego_sequence.exo_sequence is not None and log_exo:
        exo_cam_param_list: list[PinholeParameters] = exoego_sequence.exo_sequence.exo_cam_list
        Pall_exo: Float[ndarray, "n_views 3 4"] = np.stack(
            [pinhole.projection_matrix for pinhole in exo_cam_param_list]
        )
        uv_exo_stack: Float[ndarray, "n_frames n_views 133 2"] = proj_3d_vectorized(xyz_hom=xyz_hom_stack, P=Pall_exo)
        uv_exo_stack: Float[ndarray, "n_frames n_views 133 2"] = filter_out_of_bounds_keypoints(
            uv_exo_stack, exo_cam_param_list[0]
        )
        for exo_cam_idx, exo_cam in enumerate(exo_cam_param_list):
            exo_cam_path: Path = parent_log_path / exo_cam.name
            exo_pinhole_path: Path = exo_cam_path / "pinhole"
            uv_exo: Float[ndarray, "n_frames 133 2"] = uv_exo_stack[:, exo_cam_idx, :, :]
            # filter batch with invalid values
            rr.log(
                f"{exo_pinhole_path}/keypoints",
                rr.Points2D.from_fields(
                    class_ids=0,
                    keypoint_ids=COCO_133_IDS,
                    show_labels=False,
                ),
                static=True,
            )
            rr.send_columns(
                f"{exo_pinhole_path}/keypoints",
                indexes=[rr.TimeColumn(timeline, duration=1e-9 * shortest_timestamp[0 : len(uv_exo)])],
                columns=[
                    *rr.Points2D.columns(
                        positions=rearrange(
                            uv_exo,
                            "n_frames kpts dim -> (n_frames kpts) dim",
                        ),
                        colors=rearrange(
                            colors,
                            "n_frames kpts dim -> (n_frames kpts) dim",
                        ),
                    ).partition(lengths=[len(COCO_133_IDS)] * len(uv_exo)),
                ],
            )

    ###########################
    # batch send all ego cams #
    ###########################
    if exoego_sequence.ego_sequence is not None and log_ego:
        for cam_name, ego_cam_param_list in exoego_sequence.ego_sequence.ego_cam_dict.items():
            # We assume that all cameras have the intrinsics
            first_cam: PinholeParameters = ego_cam_param_list[0]
            cam_log_path: Path = parent_log_path / cam_name
            pinhole_log_path: Path = cam_log_path / "pinhole"
            rr.log(
                f"{pinhole_log_path}",
                rr.Pinhole(
                    image_from_camera=first_cam.intrinsics.k_matrix,
                    height=first_cam.intrinsics.height,
                    width=first_cam.intrinsics.width,
                    camera_xyz=getattr(
                        rr.ViewCoordinates,
                        first_cam.intrinsics.camera_conventions,
                    ),
                    image_plane_distance=exoego_sequence.ego_sequence.image_plane_distance,
                ),
                static=True,
            )
            batch_world_t_cam: Float[ndarray, "n_frames 3"] = np.array(
                [ego_cam_param.extrinsics.world_t_cam for ego_cam_param in ego_cam_param_list]
            )
            batch_world_R_cam: Float[ndarray, "n_frames 3 3"] = np.array(
                [ego_cam_param.extrinsics.world_R_cam for ego_cam_param in ego_cam_param_list]
            )
            # camera extrinsics, there's no from_parent=True so need to send as world_x_cam
            rr.send_columns(
                f"{cam_log_path}",
                indexes=[rr.TimeColumn(timeline, duration=1e-9 * shortest_timestamp[0 : len(batch_world_t_cam)])],
                columns=[
                    *rr.Transform3D.columns(
                        translation=rearrange(batch_world_t_cam, "f d -> (f) d"),
                        mat3x3=rearrange(batch_world_R_cam, "f r c -> (f) r c"),
                    ),
                ],
            )

            # make Pall for specific camera
            Pall: Float[ndarray, "n_frames 3 4"] = np.stack(
                [pinhole.projection_matrix for pinhole in ego_cam_param_list]
            )
            uv_ego_stack: Float[ndarray, "n_frames 133 2"] = np.zeros((len(xyz_hom_stack), 133, 2))

            # Process in batches to balance memory usage and performance
            batch_size = min(100, len(xyz_hom_stack))  # Adjust based on available memory
            for start_idx in range(0, len(xyz_hom_stack), batch_size):
                end_idx: int = min(start_idx + batch_size, len(xyz_hom_stack))

                # Get batch data
                xyz_hom_batch = xyz_hom_stack[start_idx:end_idx]  # (batch_frames, 133, 4)
                P_batch = Pall[start_idx:end_idx]  # (batch_frames, 3, 4)

                # Use the vectorized projection function on the batch
                uv_batch: Float[ndarray, "batch_frames batch_frames 133 2"] = proj_3d_vectorized(
                    xyz_hom=xyz_hom_batch, P=P_batch
                )

                # Extract diagonal to get frame-to-frame correspondence
                batch_len = end_idx - start_idx
                uv_batch_diagonal = uv_batch[np.arange(batch_len), np.arange(batch_len)]  # (batch_frames, 133, 2)

                # Store results
                uv_ego_stack[start_idx:end_idx] = uv_batch_diagonal

            uv_ego_stack = filter_out_of_bounds_keypoints(uv_ego_stack, first_cam)
            rr.log(
                f"{pinhole_log_path}/keypoints",
                rr.Points2D.from_fields(
                    class_ids=0,
                    keypoint_ids=COCO_133_IDS,
                    show_labels=False,
                ),
                static=True,
            )
            rr.send_columns(
                f"{pinhole_log_path}/keypoints",
                indexes=[rr.TimeColumn(timeline, duration=1e-9 * shortest_timestamp[0 : len(uv_ego_stack)])],
                columns=[
                    *rr.Points2D.columns(
                        positions=rearrange(
                            uv_ego_stack,
                            "n_frames kpts dim -> (n_frames kpts) dim",
                        ),
                        colors=rearrange(
                            colors,
                            "n_frames kpts dim -> (n_frames kpts) dim",
                        ),
                    ).partition(lengths=[len(COCO_133_IDS)] * len(uv_ego_stack)),
                ],
            )


def visualize_exo_ego(config: VisualizeConfig):
    start_time: float = timer()
    exoego_sequence: BaseExoEgoSequence = config.dataset.setup()  # one-liner
    ego_sequence: BaseEgoSequence | None = exoego_sequence.ego_sequence
    exo_sequence: BaseExoSequence | None = exoego_sequence.exo_sequence

    rr.log("/", exoego_sequence.world_coordinate_system, static=True)
    set_annotation_context()

    parent_log_path = Path("world")
    timeline: str = "video_time"

    ego_timestamps: list[Int[ndarray, "n_frames"]] = []
    ego_video_log_paths: list[Path] | None = None
    if ego_sequence is not None and config.log_ego:
        ego_video_readers: MultiVideoReader = ego_sequence.ego_video_readers
        ego_video_files: list[Path] = ego_video_readers.video_paths
        ego_cam_dict: dict[CamNameType, list[PinholeParameters]] = ego_sequence.ego_cam_dict
        ego_cam_log_paths: list[Path] = [parent_log_path / ego_cam_name for ego_cam_name in ego_cam_dict]
        ego_video_log_paths: list[Path] = [cam_log_paths / "pinhole" / "video" for cam_log_paths in ego_cam_log_paths]

        for video_file, ego_video_log_path in zip(ego_video_files, ego_video_log_paths, strict=True):
            assert video_file.suffix == ".mp4", f"Video file {video_file} is not an mp4."
            # Log video asset which is referred to by frame references.
            ego_timestamps_ns: Int[ndarray, "n_frames"] = log_video(video_file, ego_video_log_path, timeline=timeline)
            ego_timestamps.append(ego_timestamps_ns)

    exo_video_log_paths: list[Path] | None = None
    if exo_sequence is not None and config.log_exo:
        exo_video_readers: MultiVideoReader = exo_sequence.exo_video_readers
        exo_video_files: list[Path] = exo_video_readers.video_paths
        exo_cam_log_paths: list[Path] = [parent_log_path / exo_cam.name for exo_cam in exo_sequence.exo_cam_list]
        exo_video_log_paths: list[Path] = [cam_log_paths / "pinhole" / "video" for cam_log_paths in exo_cam_log_paths]

        # log stationary exo cameras and video assets
        for exo_cam in exo_sequence.exo_cam_list:
            cam_log_path: Path = parent_log_path / exo_cam.name
            log_pinhole(
                camera=exo_cam,
                cam_log_path=cam_log_path,
                image_plane_distance=exo_sequence.image_plane_distance,
                static=True,
            )

        for idx, (video_file, exo_video_log_path) in enumerate(zip(exo_video_files, exo_video_log_paths, strict=True)):
            if idx >= config.max_exo_videos_to_log:
                break
            assert video_file.suffix == ".mp4", f"Video file {video_file} is not an mp4."
            # Log video asset which is referred to by frame references.
            log_video(video_file, exo_video_log_path, timeline=timeline)

    blueprint: rrb.Blueprint = create_blueprint(
        exo_video_log_paths=exo_video_log_paths,
        ego_video_log_paths=ego_video_log_paths,
        max_exo_videos_to_log=config.max_exo_videos_to_log,
    )
    rr.send_blueprint(blueprint)

    if ego_sequence is not None and ego_timestamps:
        # Find the timestamp list with the maximum length.
        shortest_timestamp: Int[ndarray, "n_frames"] = min(ego_timestamps, key=len)  # noqa: UP037
        assert len(shortest_timestamp) == len(ego_sequence), (
            f"Length of timestamps {len(shortest_timestamp)} and sequence {len(ego_sequence)} do not match"
        )

        log_exoego_batch(
            exoego_sequence,
            parent_log_path=parent_log_path,
            timeline=timeline,
            shortest_timestamp=shortest_timestamp,
            log_ego=config.log_ego,
            log_exo=config.log_exo,
        )

    print(f"Total time taken: {timer() - start_time:.2f} seconds")
