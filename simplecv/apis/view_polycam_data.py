from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import rerun as rr
from jaxtyping import UInt8, UInt16
from tqdm import tqdm

from simplecv.data.polycam import (
    DepthConfidenceLevel,
    PolycamData,
    PolycamDataset,
    load_polycam_data,
)
from simplecv.ops.tsdf_depth_fuser import Open3DFuser
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole


@dataclass
class PolyViewConfig:
    polycam_zip_path: Path
    rr_config: RerunTyroConfig
    log_incremental_mesh: bool = True
    # logs the mesh incrementally, will take up more memory


def log_polycam_data(
    parent_path: Path,
    polycam_data: PolycamData,
) -> None:
    cam_path: Path = parent_path / "cam"
    pinhole_path: Path = cam_path / "pinhole"

    rgb: UInt8[np.ndarray, "h w 3"] = polycam_data.rgb_hw3
    depth: UInt16[np.ndarray, "h w"] = polycam_data.depth_hw
    confidence: UInt8[np.ndarray, "h w"] = polycam_data.confidence_hw

    log_pinhole(camera=polycam_data.pinhole_params, cam_log_path=cam_path)
    rr.log(f"{pinhole_path}/image", rr.Image(rgb))
    rr.log(f"{pinhole_path}/confidence", rr.SegmentationImage(confidence))
    rr.log(f"{pinhole_path}/gt_depth", rr.DepthImage(depth, meter=1000))


def view_polycam_data(config: PolyViewConfig) -> None:
    polycam_dataset: PolycamDataset = load_polycam_data(
        polycam_zip_or_directory_path=config.polycam_zip_path
    )

    depth_fuser = Open3DFuser(fusion_resolution=0.04, max_fusion_depth=3.0)

    parent_path: Path = Path("world")
    rr.log(f"{parent_path}", rr.ViewCoordinates.RUB, timeless=True)

    pbar = tqdm(polycam_dataset, total=len(polycam_dataset))
    polycam_data: PolycamData
    for idx, (polycam_data) in enumerate(pbar):
        rr.set_time_sequence("timestep", idx)

        # filter depthmaps based on confidence, only keep with max confidence
        polycam_data.depth_hw[
            polycam_data.confidence_hw != DepthConfidenceLevel.HIGH
        ] = 0

        depth_fuser.fuse_frames(
            polycam_data.depth_hw,
            polycam_data.pinhole_params.intrinsics.k_matrix,
            polycam_data.pinhole_params.extrinsics.cam_T_world,
            polycam_data.rgb_hw3,
        )

        log_polycam_data(
            parent_path=parent_path,
            polycam_data=polycam_data,
        )

        if config.log_incremental_mesh:
            gt_mesh: o3d.geometry.TriangleMesh = depth_fuser.get_mesh()
            gt_mesh.compute_vertex_normals()

            rr.log(
                f"{parent_path}/gt_mesh",
                rr.Mesh3D(
                    vertex_positions=gt_mesh.vertices,
                    triangle_indices=gt_mesh.triangles,
                    vertex_normals=gt_mesh.vertex_normals,
                    vertex_colors=gt_mesh.vertex_colors,
                ),
            )

    # export mesh
    gt_mesh: o3d.geometry.TriangleMesh = depth_fuser.get_mesh()
    gt_mesh.compute_vertex_normals()

    rr.log(
        f"{parent_path}/gt_mesh",
        rr.Mesh3D(
            vertex_positions=gt_mesh.vertices,
            triangle_indices=gt_mesh.triangles,
            vertex_normals=gt_mesh.vertex_normals,
            vertex_colors=gt_mesh.vertex_colors,
        ),
    )
