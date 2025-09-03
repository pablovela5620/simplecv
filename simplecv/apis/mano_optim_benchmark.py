from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jax import jit
from jax import numpy as jnp
from jaxtyping import Array, Float, Float32, Int, UInt8
from numpy import ndarray
from tqdm import tqdm

from simplecv.apis.view_exoego import create_blueprint, filter_out_of_bounds_keypoints, log_exoego_batch
from simplecv.camera_parameters import PinholeParameters
from simplecv.configs.exoego_dataset_configs import AnnotatedEgoDatasetUnion
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels
from simplecv.data.skeleton.coco_133 import COCO_133_ID2NAME, COCO_133_LINKS
from simplecv.ops.mano.mano_jax import ManoSimpleLayerJAX
from simplecv.ops.triangulate import proj_3d_vectorized
from simplecv.rerun_log_utils import (
    RerunTyroConfig,
    log_pinhole,
    log_video,
)
from simplecv.video_io import MultiVideoReader

np.set_printoptions(suppress=True)


@dataclass
class ManoOptimBenchConfig:
    rr_config: RerunTyroConfig
    dataset: AnnotatedEgoDatasetUnion


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
                rr.AnnotationInfo(id=1, label="Left Hand"),
                rr.AnnotationInfo(id=2, label="Right Hand"),
            ]
        ),
        static=True,
    )


def main(cfg: ManoOptimBenchConfig):
    start_time: float = timer()
    exoego_sequence: BaseExoEgoSequence = cfg.dataset.setup()
    exo_sequence: BaseExoSequence | None = exoego_sequence.exo_sequence

    rr.log("/", exoego_sequence.world_coordinate_system, static=True)
    set_annotation_context()

    parent_log_path = Path("world")
    timeline: str = "video_time"

    exo_timestamps: list[Int[ndarray, "n_frames"]] = []

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

    for video_file, exo_video_log_path in zip(exo_video_files, exo_video_log_paths, strict=True):
        assert video_file.suffix == ".mp4", f"Video file {video_file} is not an mp4."
        # Log video asset which is referred to by frame references.
        exo_timestamps_ns: Int[ndarray, "n_frames"] = log_video(video_file, exo_video_log_path, timeline=timeline)
        exo_timestamps.append(exo_timestamps_ns)

    blueprint: rrb.Blueprint = create_blueprint(
        exo_video_log_paths=exo_video_log_paths,
    )
    rr.send_blueprint(blueprint)

    shortest_timestamp: Int[ndarray, "n_frames"] = min(exo_timestamps, key=len)

    print(f"Total time taken: {timer() - start_time:.2f} seconds")

    log_exoego_batch(
        exoego_sequence,
        parent_log_path=parent_log_path,
        timeline=timeline,
        shortest_timestamp=shortest_timestamp,
        log_ego=False,
        log_exo=True,
    )

    exoego_labels: ExoEgoLabels | None = exoego_sequence.exoego_labels
    exo_cam_param_list: list[PinholeParameters] = exo_sequence.exo_cam_list
    if exoego_labels is not None:
        xyzc_stack: Float[ndarray, "n_frames n_kpts=133 4"] = exoego_labels.xyzc_stack
        xyz_stack: Float[ndarray, "n_frames n_kpts=133 3"] = xyzc_stack[:, :, :3]
        xyz_hom_stack: Float[ndarray, "n_frames n_kpts=133 4"] = np.concatenate(
            [xyz_stack, np.ones_like(xyz_stack[..., :1])], axis=-1
        )
        Pall_exo: Float[ndarray, "n_views 3 4"] = np.stack(
            [pinhole.projection_matrix for pinhole in exo_cam_param_list]
        )
        uv_exo_stack: Float[ndarray, "n_frames n_views n_kpts=133 2"] = proj_3d_vectorized(
            xyz_hom=xyz_hom_stack, P=Pall_exo
        )
        uv_exo_stack: Float[ndarray, "n_frames n_views n_kpts=133 2"] = filter_out_of_bounds_keypoints(
            uv_exo_stack, exo_cam_param_list[0]
        )
        # beta values for mano
        gt_beta: Float[ndarray, "10"] | None = (
            exoego_labels.mano_stack.betas if exoego_labels.mano_stack is not None else None
        )
        gt_poses: Float32[ndarray, "n_frames n_hands=2 51"] | None = (
            exoego_labels.mano_stack.poses if exoego_labels.mano_stack is not None else None
        )

    mano_fwd_right = jit(ManoSimpleLayerJAX(mano_root=Path("data/"), side="right"))
    for ts_idx, timestamp in enumerate(tqdm(shortest_timestamp, desc="Logging frames", unit="frame")):
        rr.set_time(timeline="video_time", duration=1e-9 * timestamp)
        _bgr_list: list[UInt8[ndarray, "H W 3"]] = exo_video_readers[ts_idx]

        # lets optimize the right hand xyz keypoints only

        pose: Float32[Array, "b n_poses=48"] = jnp.array(gt_poses[ts_idx : ts_idx + 1, 0, 0:48], dtype=jnp.float32)
        th_trans: Float32[Array, "b dim=3"] = jnp.array(gt_poses[ts_idx : ts_idx + 1, 0, 48:51], dtype=jnp.float32)

        th_betas: Float32[Array, "b n_betas=10"] = (
            jnp.array(gt_beta[None, :], dtype=jnp.float32) if gt_beta is not None else jnp.zeros((1, 10))
        )
        mano_out: tuple[Float32[Array, "b n_verts=778 3"], Float32[Array, "b joints_and_tips=21 3"]] = mano_fwd_right(
            th_pose_coeffs=pose, th_betas=th_betas, th_trans=th_trans
        )
        right_verts: Float32[Array, "b n_verts=778 3"] = mano_out[0] / 1000
        right_mano_xyz: Float32[Array, "b n_kpts=21 3"] = mano_out[1] / 1000

        rr.log(f"{parent_log_path}/optimized_right_kpts", rr.Points3D(right_mano_xyz))
