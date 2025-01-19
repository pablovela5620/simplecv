import numpy as np
import open3d as o3d
from jaxtyping import Float, UInt8, UInt16


class DepthFuser:
    def __init__(
        self, fusion_resolution: float = 0.04, max_fusion_depth: float = 3.0
    ) -> None:
        self.fusion_resolution: float = fusion_resolution
        self.max_fusion_depth: float = max_fusion_depth


class Open3DFuser(DepthFuser):
    """Open3D-based implementation of TSDF fusion for depth maps.

    This class provides functionality to fuse depth maps and RGB images into a TSDF volume
    using Open3D's integration pipeline, and extract triangle meshes from the fused volume.

    Args:
        fusion_resolution (float, optional): Resolution of the TSDF volume in meters. Defaults to 0.04.
        max_fusion_depth (float, optional): Maximum depth value to consider for fusion in meters. Defaults to 3.0.

    Attributes:
        fusion_max_depth (float): Maximum depth threshold for fusion.
        volume (o3d.pipelines.integration.ScalableTSDFVolume): Open3D TSDF volume for integration.

    Example:
        ```python
        fuser = Open3DFuser(fusion_resolution=0.04, max_fusion_depth=3.0)
        for frame in frames:
            fuser.fuse_frames(depth, intrinsics, pose, rgb)
        mesh = fuser.get_mesh()
        ```
    """

    def __init__(
        self,
        fusion_resolution: float = 0.04,
        max_fusion_depth: float = 3.0,
    ):
        super().__init__(
            fusion_resolution,
            max_fusion_depth,
        )

        self.fusion_max_depth = max_fusion_depth

        voxel_size: float = fusion_resolution * 100
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=float(voxel_size) / 100,
            sdf_trunc=3 * float(voxel_size) / 100,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    def fuse_frames(
        self,
        depth_hw: UInt16[np.ndarray, "h w"],
        K_33: Float[np.ndarray, "3 3"],
        cam_T_world_44: Float[np.ndarray, "4 4"],
        rgb_hw3: UInt8[np.ndarray, "h w 3"],
    ) -> None:
        height: int = depth_hw.shape[0]
        width: int = depth_hw.shape[1]

        rgbd: o3d.geometry.RGBDImage = (
            o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb_hw3),
                o3d.geometry.Image(depth_hw),
                depth_scale=1000.0,
                depth_trunc=self.fusion_max_depth,
                convert_rgb_to_intensity=False,
            )
        )

        self.volume.integrate(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                width=width,
                height=height,
                fx=K_33[0, 0],
                fy=K_33[1, 1],
                cx=K_33[0, 2],
                cy=K_33[1, 2],
            ),
            cam_T_world_44,
        )

    def export_mesh(self, path) -> None:
        o3d.io.write_triangle_mesh(path, self.volume.extract_triangle_mesh())

    def get_mesh(
        self, export_single_mesh=None, convert_to_trimesh=False
    ) -> o3d.geometry.TriangleMesh:
        mesh = self.volume.extract_triangle_mesh()

        return mesh
