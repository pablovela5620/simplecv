from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from einops import rearrange
from jaxtyping import Bool, Float
from numpy import ndarray


@dataclass
class Distortion:
    """
    Brown Conrady distortion model
    """

    k1: float
    k2: float
    p1: float
    p2: float
    k3: float
    k4: float | None = None
    k5: float | None = None
    k6: float | None = None


@dataclass
class Extrinsics:
    # Rotation and translation can be provided for both world-to-camera
    # and camera-to-world transformations
    world_R_cam: Float[ndarray, "3 3"] | None = None
    world_t_cam: Float[ndarray, "3"] | None = None
    cam_R_world: Float[ndarray, "3 3"] | None = None
    cam_t_world: Float[ndarray, "3"] | None = None
    # The projection matrix and transformation matrices will be computed in post-init
    world_T_cam: Float[ndarray, "4 4"] = field(init=False)
    cam_T_world: Float[ndarray, "4 4"] = field(init=False)

    def __post_init__(self) -> None:
        self.compute_transformation_matrices()

    def compute_transformation_matrices(self) -> None:
        # If world-to-camera is provided, compute the transformation matrix and its inverse
        if self.world_R_cam is not None and self.world_t_cam is not None:
            self.world_T_cam: Float[ndarray, "4 4"] = self.compose_transformation_matrix(
                self.world_R_cam, self.world_t_cam
            )
            self.cam_T_world: Float[ndarray, "4 4"] = np.linalg.inv(self.world_T_cam)
            # Extract camera-to-world rotation and translation from the inverse matrix
            self.cam_R_world, self.cam_t_world = self.decompose_transformation_matrix(self.cam_T_world)
        # If camera-to-world is provided, compute the transformation matrix and its inverse
        elif self.cam_R_world is not None and self.cam_t_world is not None:
            self.cam_T_world: Float[ndarray, "4 4"] = self.compose_transformation_matrix(
                self.cam_R_world, self.cam_t_world
            )
            self.world_T_cam: Float[ndarray, "4 4"] = np.linalg.inv(self.cam_T_world)
            # Extract world-to-camera rotation and translation from the inverse matrix
            self.world_R_cam, self.world_t_cam = self.decompose_transformation_matrix(self.world_T_cam)
        else:
            raise ValueError("Either world-to-camera or camera-to-world rotation and translation must be provided.")

    def compose_transformation_matrix(self, R: Float[ndarray, "3 3"], t: Float[ndarray, "3"]) -> Float[ndarray, "4 4"]:
        Rt: Float[ndarray, "3 4"] = np.hstack([R, rearrange(t, "c -> c 1")])
        T: Float[ndarray, "4 4"] = np.vstack([Rt, np.array([0, 0, 0, 1])])
        return T

    def decompose_transformation_matrix(
        self, T: Float[ndarray, "4 4"]
    ) -> tuple[Float[ndarray, "3 3"], Float[ndarray, "3"]]:
        R: Float[ndarray, "3 3"] = T[:3, :3]
        t: Float[ndarray, "3 "] = T[:3, 3]
        return R, t


@dataclass
class Intrinsics:
    camera_conventions: Literal["RDF", "RUB"]
    """RDF(OpenCV): X Right - Y Down - Z Front | RUB (OpenGL): X Right- Y Up - Z Back"""
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    height: int | None = None
    width: int | None = None
    k_matrix: Float[ndarray, "3 3"] = field(init=False)

    def __post_init__(self):
        self.compute_k_matrix()
        if self.height is None or self.width is None:
            self.height = 2 * self.cy
            self.width = 2 * self.cx

    def compute_k_matrix(self):
        # Compute the camera matrix using the focal length and principal point
        self.k_matrix = np.array(
            [
                [self.fl_x, 0, self.cx],  # noqa: E501
                [0, self.fl_y, self.cy],
                [0, 0, 1],
            ]
        )

    def __repr__(self):
        return (
            f"Intrinsics(camera_conventions={self.camera_conventions}, "
            f"fl_x={self.fl_x}, fl_y={self.fl_y}, cx={self.cx}, cy={self.cy}, "
            f"height={self.height}, width={self.width})"
        )


@dataclass
class PinholeParameters:
    name: str
    extrinsics: Extrinsics
    intrinsics: Intrinsics
    projection_matrix: Float[ndarray, "3 4"] = field(init=False)
    distortion: Distortion | None = None

    def __post_init__(self) -> None:
        self.compute_projection_matrix()

    def compute_projection_matrix(self) -> None:
        # Compute the projection matrix using k_matrix and world_T_cam
        self.projection_matrix: Float[ndarray, "3 4"] = self.intrinsics.k_matrix @ self.extrinsics.cam_T_world[:3, :]


@dataclass
class Fisheye62Parameters:
    """
    Has 6 radial (k) parameters and 2 tangential (p) distortion parameters
    """

    name: str
    extrinsics: Extrinsics
    intrinsics: Intrinsics
    projection_matrix: Float[ndarray, "3 4"] = field(init=False)
    distortion: Distortion | None = None

    def __post_init__(self) -> None:
        self.compute_projection_matrix()

    def compute_projection_matrix(self) -> None:
        # Compute the projection matrix using k_matrix and world_T_cam
        self.projection_matrix: Float[ndarray, "3 4"] = self.intrinsics.k_matrix @ self.extrinsics.cam_T_world[:3, :]


def to_homogeneous(
    points: Float[np.ndarray, "num_points _"],
) -> Float[np.ndarray, "num_points _"]:
    """
    Converts a set of 3D points to homogeneous coordinates.

    Args:
        points (Float[np.ndarray, "num_points 3"]): A numpy array containing the 3D coordinates of the points.

    Returns:
        Float[np.ndarray, "num_points 4"]: A numpy array containing the homogeneous coordinates of the points.
    """
    ones_column: Float[ndarray, "num_points 1"] = np.ones((points.shape[0], 1), dtype=points.dtype)
    return np.hstack([points, ones_column])


def from_homogeneous(
    points_hom: Float[np.ndarray, "num_points _"],
) -> Float[np.ndarray, "num_points _"]:
    """
    Converts a set of 3D points from homogeneous coordinates to Euclidean coordinates.

    Args:
        points (Float[np.ndarray, "num_points 4"]): A numpy array containing the homogeneous coordinates of the points.

    Returns:
        Float[np.ndarray, "num_points 3"]: A numpy array containing the 3D coordinates of the points.
    """
    points = points_hom / points_hom[:, 3:]
    return points[:, :3]


def rescale_intri(camera_intrinsics: Intrinsics, *, target_width: int, target_height: int) -> Intrinsics:
    """
    Rescales the input image and intrinsic matrix by a given scale factor.

    Args:
        cam (PinholeCameraParameter): The pinhole camera parameter.

    Returns:
        : The rescaled image frame and intrinsic matrix.
    """
    assert camera_intrinsics.height is not None, "Set Camera Height, currently None"
    assert camera_intrinsics.width is not None, "Set Camera Width, currently None"
    x_scale: float = target_width / camera_intrinsics.width
    y_scale: float = target_height / camera_intrinsics.height

    new_fl_x: float = camera_intrinsics.fl_x * x_scale
    new_fl_y: float = camera_intrinsics.fl_y * y_scale

    rescaled_intri = Intrinsics(
        camera_conventions=camera_intrinsics.camera_conventions,
        fl_x=new_fl_x,
        fl_y=new_fl_y,
        cx=camera_intrinsics.cx * x_scale,
        cy=camera_intrinsics.cy * y_scale,
        height=target_height,
        width=target_width,
    )

    return rescaled_intri


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
    # Compute the radial distance of each 3D point from the camera center
    r: Float[ndarray, "num_points"] = np.sqrt(
        np.sum(np.square(points_3d[:, :2]), axis=-1)
    )
    eps: float = 2.0**-128
    # Compute the angles of the 2D image coordinates with respect to the camera center using arctan2
    s: Float[ndarray, "num_points"] = np.arctan2(r, points_3d[:, 2]) / np.maximum(
        r, eps
    )
    # Scale the angles by the radial distance to obtain the final 2D image coordinates in camera coordinates
    points_2d_cam: Float[ndarray, "num_points 2"] = np.zeros((points_3d.shape[0], 2))
    points_2d_cam[:, 0] = points_3d[:, 0] * s
    points_2d_cam[:, 1] = points_3d[:, 1] * s
    # Convert the camera coordinates to homogeneous coordinates
    points_2d_hom: Float[ndarray, "num_points 3"] = to_homogeneous(points_2d_cam)
    # Apply the camera intrinsic matrix to the homogeneous coordinates to obtain the final 2D image coordinates in homogeneous coordinates
    points_2d: Float[ndarray, "num_points 3"] = (K @ points_2d_hom.T).T
    # Convert the homogeneous coordinates to Euclidean coordinates by dividing by the third coordinate
    points_2d: Float[ndarray, "num_points 2"] = points_2d[:, :2] / points_2d[:, 2:]
    return points_2d


def apply_radial_tangential_distortion(
    dist_coeffs: Distortion, points2d: Float[np.ndarray, "num_points 2"]
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
    # radial component
    r2 = (points2d * points2d).sum(axis=-1, keepdims=True)
    r2 = np.clip(r2, -(np.pi**2), np.pi**2)
    r4 = r2 * r2
    r6 = r2 * r4
    r8 = r4 * r4
    r10 = r4 * r6
    r12 = r6 * r6
    radial = (
        1
        + dist_coeffs.k1 * r2
        + dist_coeffs.k2 * r4
        + dist_coeffs.k3 * r6
        + dist_coeffs.k4 * r8
        + dist_coeffs.k5 * r10
        + dist_coeffs.k6 * r12
    )
    uv = points2d * radial

    # tangential component
    x, y = uv[..., 0], uv[..., 1]
    x2 = x * x
    y2 = y * y
    xy = x * y
    r2 = x2 + y2
    x += 2 * dist_coeffs.p2 * xy + dist_coeffs.p1 * (r2 + 2 * x2)
    y += 2 * dist_coeffs.p1 * xy + dist_coeffs.p2 * (r2 + 2 * y2)
    return np.stack((x, y), axis=-1)


def fisheye_projection(
    points_3d_world: Float[ndarray, "num_points 3"], camera: Fisheye62Parameters
) -> Float[ndarray, "num_points 2"]:
    # world to camera
    points_3d_hom_world: Float[ndarray, "num_points 4"] = to_homogeneous(points_3d_world)
    points_3d_hom_cam: Float[ndarray, "num_points 4"] = (camera.extrinsics.cam_T_world @ points_3d_hom_world.T).T
    points_3d_cam: Float[ndarray, "num_points 3"] = from_homogeneous(points_3d_hom_cam)
    # camera to image
    points_2d_undist: Float[ndarray, "num_points 2"] = arctan_projection(points_3d_cam, camera.intrinsics.k_matrix)
    # normalize points for distortion
    points_2d_undist[:, 0] -= camera.intrinsics.cx
    points_2d_undist[:, 1] -= camera.intrinsics.cy
    points_2d_undist[:, 0] /= camera.intrinsics.fl_x
    points_2d_undist[:, 1] /= camera.intrinsics.fl_y

    points_2d_distorted = apply_radial_tangential_distortion(camera.distortion, points_2d_undist)

    # denormalize points after applying distortion
    points_2d_distorted[:, 0] *= camera.intrinsics.fl_x
    points_2d_distorted[:, 1] *= camera.intrinsics.fl_y
    points_2d_distorted[:, 0] += camera.intrinsics.cx
    points_2d_distorted[:, 1] += camera.intrinsics.cy

    # make sure points are within image bounds
    out_of_bounds: Bool[ndarray, "num_points"] = np.logical_or(
        points_2d_distorted[:, 0] >= camera.intrinsics.width,
        points_2d_distorted[:, 1] >= camera.intrinsics.height,
    )
    out_of_bounds: Bool[ndarray, "num_points"] = np.logical_or(out_of_bounds, points_2d_distorted[:, 0] < 0)
    out_of_bounds: Bool[ndarray, "num_points"] = np.logical_or(out_of_bounds, points_2d_distorted[:, 1] < 0)
    # make sure points are in front of camera
    out_of_bounds: Bool[ndarray, "num_points"] = np.logical_or(out_of_bounds, points_3d_cam[:, 2] < 0)

    # if out of bounds, set to -1
    points_2d_distorted[out_of_bounds, :] = np.nan
    return points_2d_distorted
