import numpy as np
from jaxtyping import Float32
from numpy import ndarray

from simplecv.camera_parameters import (
    Extrinsics,
    Fisheye62Parameters,
    Intrinsics,
    PinholeParameters,
    apply_radial_tangential_distortion,
    project_kannala_brandt,
)


class Camera:
    """Lightweight wrapper exposing common projections for pinhole and fisheye models."""

    def __init__(self, camera_parameters: PinholeParameters | Fisheye62Parameters) -> None:
        self.camera_parameters: PinholeParameters | Fisheye62Parameters = camera_parameters
        self._refresh_extrinsics()

    def _refresh_extrinsics(self) -> None:
        extrinsics = self.camera_parameters.extrinsics
        self.cam_T_world: Float32[ndarray, "4 4"] = np.asarray(extrinsics.cam_T_world, dtype=np.float32)
        self.world_T_cam: Float32[ndarray, "4 4"] = np.asarray(extrinsics.world_T_cam, dtype=np.float32)

    def set_extrinsic(self, cam_T_world: Float32[ndarray, "4 4"]) -> None:
        cam_T_world = np.asarray(cam_T_world, dtype=np.float32)
        cam_R_world: Float32[ndarray, "3 3"] = cam_T_world[:3, :3]
        cam_t_world: Float32[ndarray, "3"] = cam_T_world[:3, 3]
        self.camera_parameters.extrinsics = Extrinsics(cam_R_world=cam_R_world, cam_t_world=cam_t_world)
        self.camera_parameters.compute_projection_matrix()
        self._refresh_extrinsics()

    def camera_to_image(self, points_3d: Float32[ndarray, "n_points 3"]) -> Float32[ndarray, "n_points 2"]:
        points_cam: Float32[ndarray, "n_points 3"] = np.asarray(points_3d, dtype=np.float32)
        intrinsics: Intrinsics = self.camera_parameters.intrinsics

        if isinstance(self.camera_parameters, PinholeParameters):
            z: Float32[ndarray, "n_points 1"] = np.clip(points_cam[:, 2:3], 1e-8, None)
            norm: Float32[ndarray, "n_points 2"] = points_cam[:, :2] / z
            distorted_norm: Float32[ndarray, "n_points 2"] = apply_radial_tangential_distortion(
                self.camera_parameters.distortion, norm
            )
            uv: Float32[ndarray, "n_points 2"] = distorted_norm.copy()
        else:
            norm = project_kannala_brandt(points_cam, self.camera_parameters.distortion)
            uv = np.array(norm, dtype=np.float64, copy=True)
            if self.camera_parameters.distortion is not None:
                p1 = float(self.camera_parameters.distortion.p1)
                p2 = float(self.camera_parameters.distortion.p2)
            x = uv[:, 0]
            y = uv[:, 1]
            x2 = x * x
            y2 = y * y
            xy = x * y
            r2 = x2 + y2
            uv[:, 0] = x + 2 * p2 * xy + p1 * (r2 + 2 * x2)
            uv[:, 1] = y + 2 * p1 * xy + p2 * (r2 + 2 * y2)
            uv = uv.astype(np.float32, copy=False)

        uv[:, 0] = uv[:, 0] * float(intrinsics.fl_x) + float(intrinsics.cx)
        uv[:, 1] = uv[:, 1] * float(intrinsics.fl_y) + float(intrinsics.cy)
        return uv.astype(np.float32, copy=False)

    def image_to_camera(self, points_2d: Float32[ndarray, "num_points 2"]) -> Float32[ndarray, "num_points 3"]:
        assert isinstance(self.camera_parameters, PinholeParameters), "Only pinhole cameras support back-projection"
        assert self.camera_parameters.distortion is None, "Inverse distortion not implemented for crop cameras"

        K_inv: Float32[ndarray, "3 3"] = np.linalg.inv(self.camera_parameters.intrinsics.k_matrix).astype(
            np.float32, copy=False
        )
        points_2d_hom: Float32[ndarray, "num_points 3"] = np.concatenate(
            [points_2d, np.ones((points_2d.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        points_3d_hom: Float32[ndarray, "3 num_points"] = K_inv @ points_2d_hom.T
        points_3d: Float32[ndarray, "num_points 3"] = points_3d_hom[:3, :].T
        norm: Float32[ndarray, "num_points 1"] = np.linalg.norm(points_3d, axis=1, keepdims=True)
        return points_3d / norm

    def camera_to_world(self, points_3d_cam: Float32[ndarray, "num_points 3"]) -> Float32[ndarray, "num_points 3"]:
        points3d_hom: Float32[ndarray, "num_points 4"] = np.ones((points_3d_cam.shape[0], 4), dtype=np.float32)
        points3d_hom[:, :3] = points_3d_cam
        points3d_world: Float32[ndarray, "num_points 3"] = (self.world_T_cam @ points3d_hom.T).T[:, :3]
        return points3d_world

    def world_to_camera(self, points_3d_world: Float32[ndarray, "num_points 3"]) -> Float32[ndarray, "num_points 3"]:
        points3d_hom: Float32[ndarray, "num_points 4"] = np.ones((points_3d_world.shape[0], 4), dtype=np.float32)
        points3d_hom[:, :3] = points_3d_world
        points3d_cam: Float32[ndarray, "num_points 3"] = (self.cam_T_world @ points3d_hom.T).T[:, :3]
        return points3d_cam
