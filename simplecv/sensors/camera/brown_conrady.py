"""Camera projection helpers for the Brown–Conrady distortion model."""

import numpy as np
from einops import rearrange
from jaxtyping import Float
from numpy import ndarray

from simplecv.camera_parameters import PinholeParameters
from simplecv.sensors.camera.base_camera import filter_out_of_bounds, world_to_cam_batched


def cam_to_image_batched(
    xyz_cam: Float[ndarray, "n_frames n_views n_points 3"],
    K: Float[ndarray, "n_views 3 3"],
) -> Float[ndarray, "n_frames n_views n_points 2"]:
    """Project batched camera-frame points into pixel space using intrinsics.

    Args:
        xyz_cam: Camera-frame coordinates ``[n_frames, n_views, n_points, 3]``.
        K: Intrinsic matrices per view ``[n_views, 3, 3]``.

    Returns:
        Pixel coordinates ``[n_frames, n_views, n_points, 2]``.
    """
    xyz_cam: Float[ndarray, "n_frames n_views 3 n_points"] = rearrange(
        xyz_cam,
        "n_frames n_views n_points dim -> n_frames n_views dim n_points",
        dim=3,
    )
    # [1, n_views, 3, 3] @ [1, n_views, 3, n_points] -> [n_frames, n_views, 3, n_points]
    uv_hom: Float[ndarray, "n_frames n_views 3 n_points"] = K @ xyz_cam
    uv_hom: Float[ndarray, "n_frames n_views n_points 3"] = rearrange(
        uv_hom, "n_frames n_views dim n_points -> n_frames n_views n_points dim", dim=3
    )
    uv: Float[ndarray, "n_frames n_views n_points 2"] = uv_hom[..., :2] / uv_hom[..., 2:]
    return uv


def project_brown_conrady_batched(
    xyz_stack_world: Float[ndarray, "n_frames n_points 3"],
    pinhole_param_list: list[PinholeParameters],
    *,
    filter_invalid: bool = True,
) -> Float[ndarray, "n_frames n_views n_points 2"]:
    """Project world-frame keypoints through a batch of Brown–Conrady pinhole cameras.

    Args:
        xyz_stack_world: World-frame coordinates ``[n_frames, n_points, 3]`` to reproject.
        pinhole_param_list: Ordered camera models defining extrinsics/intrinsics (one per view).
        filter_invalid: When ``True`` (default) mask pixels that fall outside the image bounds or
            behind the camera. Disable to obtain the raw projection for debugging/comparison.

    Returns:
        Distorted pixel coordinates ``[n_frames, n_views, n_points, 2]``. When ``filter_invalid`` is
        ``True`` (default) points falling behind the camera or outside the image bounds are masked via
        ``filter_out_of_bounds``.

    Notes:
        * Assumes all cameras share identical image dimensions and currently applies only the
          pinhole transform (no per-view Brown–Conrady coefficients yet).
        * Uses ``filter_out_of_bounds`` to drop pixels when ``filter_invalid`` is enabled.
    """
    # 0. Prepare intrinsics and extrinsics stacks
    cam_T_world: Float[ndarray, "n_views 4 4"] = np.stack(
        [pinhole.extrinsics.cam_T_world for pinhole in pinhole_param_list]
    )
    K_stack: Float[ndarray, "n_views 3 3"] = np.stack([pinhole.intrinsics.k_matrix for pinhole in pinhole_param_list])
    # TODO currently assumes same distortion coeffs for all cameras, should be extended to support per-camera coeffs
    # 1. Transform world coordinates to camera coordinates
    xyz_stack_cam: Float[ndarray, "n_frames n_views n_points 3"] = world_to_cam_batched(xyz_stack_world, cam_T_world)
    # 2. Project camera coordinates to image coordinates
    uv_stack: Float[ndarray, "n_frames n_views n_points 2"] = cam_to_image_batched(xyz_cam=xyz_stack_cam, K=K_stack)
    # TODO currently assumes same distortion coeffs for all cameras, should be extended to support per-camera coeffs
    # 3. Apply Brown–Conrady distortion if coefficients are provided
    if pinhole_param_list[0].distortion is not None:
        raise NotImplementedError("Brown–Conrady distortion is not implemented yet in batched projection.")

    if not filter_invalid:
        return uv_stack

    # 4. Filter out-of-bounds points (if needed)
    # check that all cameras have same image size for now, could be extended later
    h: int = pinhole_param_list[0].intrinsics.height
    w: int = pinhole_param_list[0].intrinsics.width
    assert all((pinhole.intrinsics.height == h and pinhole.intrinsics.width == w) for pinhole in pinhole_param_list), (
        "All pinhole cameras must have the same image size for batched Brown–Conrady projection."
    )
    uv_filtered: Float[ndarray, "n_frames n_views n_points 2"] = filter_out_of_bounds(
        uv_batch=uv_stack, xyz_cam_batch=xyz_stack_cam, h=h, w=w
    )
    return uv_filtered
