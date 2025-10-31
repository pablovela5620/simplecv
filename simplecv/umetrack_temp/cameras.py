from typing import Union

import numpy as np
from jaxtyping import Float

from simplecv.umetrack_temp.camera_models import FisheyeCameraParameter, PinholeCameraParameter
from simplecv.umetrack_temp.utils import get_transformation_matrix


class Camera:
    def __init__(self, camera_parameters: PinholeCameraParameter | FisheyeCameraParameter) -> None:
        """
        Initializes a Camera object. This takes care of placing the camera and projecting/unprojecting points.

        Args:
            camera_parameters (Union[PinholeCameraParameter, FisheyeCameraParameter]): The camera parameters.
        """
        self.camera_parameters: PinholeCameraParameter | FisheyeCameraParameter = camera_parameters
        self.cam_T_world: Float[np.ndarray, "4 4"] = get_transformation_matrix(camera_parameters)
        self.world_T_cam: Float[np.ndarray, "4 4"] = np.linalg.inv(self.cam_T_world)

    def set_extrinsic(self, cam_T_world: Float[np.ndarray, "4 4"]) -> None:
        """
        Sets the extrinsic parameters of the camera.

        Args:
            world_T_cam (Float[np.ndarray, "4 4"]): The transformation matrix from world to camera space.
        """
        self.cam_T_world = cam_T_world
        self.world_T_cam = np.linalg.inv(cam_T_world)

    def camera_to_image(self, points_3d: Float[np.ndarray, "num_points 3"]) -> Float[np.ndarray, "num_points 2"]:
        if isinstance(self.camera_parameters, PinholeCameraParameter):
            points_2d = perspective_projection(points_3d, self.camera_parameters.intrinsic33())
        elif isinstance(self.camera_parameters, FisheyeCameraParameter):
            points_2d = arctan_projection(points_3d, self.camera_parameters.intrinsic33())
            # Apply the camera distortion parameters to the 2D image coordinates
            # normalize points before applying distortion
            cx, cy = self.camera_parameters.intrinsic33()[0, 2], self.camera_parameters.intrinsic33()[1, 2]
            fx, fy = self.camera_parameters.intrinsic33()[0, 0], self.camera_parameters.intrinsic33()[1, 1]
            points_2d[:, 0] -= cx
            points_2d[:, 1] -= cy
            points_2d[:, 0] /= fx
            points_2d[:, 1] /= fy
            points_2d = apply_radial_tangential_distortion(self.camera_parameters.get_dist_coeff(), points_2d)

            # denormalize points after applying distortion
            points_2d[:, 0] *= fx
            points_2d[:, 1] *= fy
            points_2d[:, 0] += cx
            points_2d[:, 1] += cy

        return points_2d

    def image_to_camera(self, points_2d: Float[np.ndarray, "num_points 2"]) -> Float[np.ndarray, "num_points 3"]:
        """
        Unproject 2d image coordinates to unit-length 3D camera coordinates
        """
        assert isinstance(self.camera_parameters, PinholeCameraParameter), "Only support PinholeCameraParameter"
        # perform inverse projection
        k_inv = np.linalg.inv(self.camera_parameters.intrinsic33())
        points_2d_hom = np.concatenate([points_2d, np.ones((points_2d.shape[0], 1), dtype=points_2d.dtype)], axis=1)
        points_3d_hom = k_inv @ points_2d_hom.T
        points_3d = points_3d_hom[:3, :].T
        # normalize points to unit length
        norm = np.linalg.norm(points_3d, axis=1, keepdims=True)
        points_3d = points_3d / norm
        return points_3d

    def camera_to_world(self, points_3d_cam: Float[np.ndarray, "num_points 3"]) -> Float[np.ndarray, "num_points 3"]:
        """
        Transform camera coordinates to world coordinates
        """
        points3d_hom = np.ones((points_3d_cam.shape[0], 4), dtype=points_3d_cam.dtype)
        points3d_hom[:, :3] = points_3d_cam
        points3d_world = (self.world_T_cam @ points3d_hom.T).T[:, :3]

        return points3d_world

    def world_to_camera(self, points_3d_world: Float[np.ndarray, "num_points 3"]) -> Float[np.ndarray, "num_points 3"]:
        """
        Transform world coordinates to camera coordinates
        """
        points3d_hom = np.ones((points_3d_world.shape[0], 4), dtype=points_3d_world.dtype)
        points3d_hom[:, :3] = points_3d_world
        points3d_cam = (self.cam_T_world @ points3d_hom.T).T[:, :3]
        return points3d_cam


def perspective_projection(
    points_3d: Float[np.ndarray, "num_points 3"], K: Float[np.ndarray, "3 3"]
) -> Float[np.ndarray, "num_points 2"]:
    """
    Project 3D points in camera coordinates to 2D using perspective projection

    Args:
        points_3d: A numpy array of shape (num_points, 3) representing the 3D points in camera coordinates to project
        K: A numpy array of shape (3, 3) representing the camera intrinsic matrix

    Returns:
        A numpy array of shape (num_points, 2) representing the 2D image coordinates of the projected points
    """
    assert points_3d.shape[1] == 3, "points_3d must have shape (num_points, 3)"
    assert K.shape == (3, 3), "K must have shape (3, 3)"
    # Apply the camera intrinsic matrix to the 3D points to obtain the 2D image coordinates in homogeneous coordinates
    points_2d_hom = (K @ points_3d.T).T
    # Convert the homogeneous coordinates to Euclidean coordinates by dividing by the third coordinate
    points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2:]
    return points_2d


def arctan_projection(
    points_3d: Float[np.ndarray, "num_points 3"], K: Float[np.ndarray, "3 3"]
) -> Float[np.ndarray, "num_points 2"]:
    """
    Project 3D points in camera coordinates to 2D using arctan projection

    Args:
        points_3d: A numpy array of shape (num_points, 3) representing the 3D points in camera coordinates to project
        K: A numpy array of shape (3, 3) representing the camera intrinsic matrix

    Returns:
        A numpy array of shape (num_points, 2) representing the 2D image coordinates of the projected points
    """
    assert points_3d.shape[1] == 3, "points_3d must have shape (num_points, 3)"
    assert K.shape == (3, 3), "K must have shape (3, 3)"
    # Compute the radial distance of each 3D point from the camera center
    r = np.sqrt(np.sum(np.square(points_3d[:, :2]), axis=-1))
    eps = 2.0**-128
    # Compute the angles of the 2D image coordinates with respect to the camera center using arctan2
    s = np.arctan2(r, points_3d[:, 2]) / np.maximum(r, eps)
    # Scale the angles by the radial distance to obtain the final 2D image coordinates in camera coordinates
    points_2d_cam = np.zeros((points_3d.shape[0], 2))
    points_2d_cam[:, 0] = points_3d[:, 0] * s
    points_2d_cam[:, 1] = points_3d[:, 1] * s
    # Convert the camera coordinates to homogeneous coordinates
    points_2d_hom = np.ones((points_2d_cam.shape[0], 3), dtype=points_2d_cam.dtype)
    points_2d_hom[:, :2] = points_2d_cam
    # Apply the camera intrinsic matrix to the homogeneous coordinates to obtain the final 2D image coordinates in homogeneous coordinates
    points_2d = (K @ points_2d_hom.T).T
    # Convert the homogeneous coordinates to Euclidean coordinates by dividing by the third coordinate
    points_2d = points_2d[:, :2] / points_2d[:, 2:]
    return points_2d


def apply_radial_tangential_distortion(
    dist_coeffs: Float[np.ndarray, "8"], points2d: Float[np.ndarray, "num_points 2"]
) -> Float[np.ndarray, "num_points 2"]:
    """
    Applies radial and tangential distortion to normalized 2D points.

    Args:
        dist_coeffs (Float[np.ndarray, "8"]): The distortion coefficients.
        points2d (Float[np.ndarray, "num_points 2"]): A numpy array containing the normalized 2D coordinates of the points.

    Returns:
        Float[np.ndarray, "num_points 2"]: A numpy array containing the 2D coordinates of the distorted points.

    Note:
        The points2d input should be normalized before being passed to this function.
    """
    k1, k2, p1, p2, k3, k4, k5, k6 = dist_coeffs
    # radial component
    r2 = (points2d * points2d).sum(axis=-1, keepdims=True)
    r2 = np.clip(r2, -(np.pi**2), np.pi**2)
    r4 = r2 * r2
    r6 = r2 * r4
    r8 = r4 * r4
    r10 = r4 * r6
    r12 = r6 * r6
    radial = 1 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8 + k5 * r10 + k6 * r12
    uv = points2d * radial

    # tangential component
    x, y = uv[..., 0], uv[..., 1]
    x2 = x * x
    y2 = y * y
    xy = x * y
    r2 = x2 + y2
    x += 2 * p2 * xy + p1 * (r2 + 2 * x2)
    y += 2 * p1 * xy + p2 * (r2 + 2 * y2)
    return np.stack((x, y), axis=-1)
