"""Batched helpers for projecting points through Kannala–Brandt fisheye models."""

from collections.abc import Sequence

import numpy as np
from einops import rearrange
from jaxtyping import Float
from numpy import ndarray

from simplecv.camera_parameters import Fisheye62Parameters, KannalaBrandtDistortion, apply_radial_tangential_distortion
from simplecv.sensors.camera.base_camera import filter_out_of_bounds, world_to_cam_batched


def arctan_cam_to_image_batched(
    xyz_cam: Float[ndarray, "n_frames n_views n_points 3"],
    K: Float[ndarray, "n_views 3 3"],
) -> Float[ndarray, "n_frames n_views n_points 2"]:
    """Map camera-frame points onto the image plane using the Kannala–Brandt arctan model.

    Args:
        xyz_cam: Camera-space coordinates ``[n_frames, n_views, n_points, 3]``.
        K: Intrinsic matrices per view ``[n_views, 3, 3]``.

    Returns:
        Undistorted pixel coordinates ``[n_frames, n_views, n_points, 2]`` produced by the arctan
        parameterisation (prior to applying the polynomial distortion terms).
    """

    x_cam: Float[ndarray, "n_frames n_views n_points"] = xyz_cam[..., 0]
    y_cam: Float[ndarray, "n_frames n_views n_points"] = xyz_cam[..., 1]
    z_cam: Float[ndarray, "n_frames n_views n_points"] = xyz_cam[..., 2]

    r_xy: Float[ndarray, "n_frames n_views n_points"] = np.sqrt(x_cam * x_cam + y_cam * y_cam)
    eps: float = float(2.0**-128)
    denom: Float[ndarray, "n_frames n_views n_points"] = np.maximum(r_xy, eps)
    theta: Float[ndarray, "n_frames n_views n_points"] = np.arctan2(r_xy, z_cam)
    scale: Float[ndarray, "n_frames n_views n_points"] = theta / denom

    xy_cam: Float[ndarray, "n_frames n_views n_points 2"] = np.zeros_like(xyz_cam[..., :2])
    xy_cam[..., 0] = x_cam * scale
    xy_cam[..., 1] = y_cam * scale

    ones: Float[ndarray, "n_frames n_views n_points 1"] = np.ones_like(scale)[..., None]
    xy_cam_hom: Float[ndarray, "n_frames n_views n_points 3"] = np.concatenate([xy_cam, ones], axis=-1)

    xy_cam_hom_batched: Float[ndarray, "n_frames n_views 3 n_points"] = rearrange(
        xy_cam_hom, "n_frames n_views n_points xyz -> n_frames n_views xyz n_points"
    )
    K_batched: Float[ndarray, "1 n_views 3 3"] = rearrange(K, "n_views n m -> 1 n_views n m")
    uv_hom: Float[ndarray, "n_frames n_views 3 n_points"] = K_batched @ xy_cam_hom_batched
    uv_hom = rearrange(uv_hom, "n_frames n_views xyz n_points -> n_frames n_views n_points xyz")

    denom_uv: Float[ndarray, "n_frames n_views n_points 1"] = uv_hom[..., 2:3]
    denom_safe: Float[ndarray, "n_frames n_views n_points 1"] = np.where(
        np.abs(denom_uv) < eps, np.sign(denom_uv) * eps, denom_uv
    )
    uv: Float[ndarray, "n_frames n_views n_points 2"] = uv_hom[..., :2] / denom_safe

    return uv


def apply_kannala_brandt_distortion_batch(
    uv_stack: Float[ndarray, "n_frames n_views n_kpts 2"],
    intrinsics_stack: Float[ndarray, "n_views 3 3"],
    distortions: Sequence[KannalaBrandtDistortion | None],
) -> Float[ndarray, "n_frames n_views n_kpts 2"]:
    """Apply per-view Kannala–Brandt distortion polynomials to image coordinates."""

    if all(distortion is None for distortion in distortions):
        return uv_stack

    uv_distorted: Float[ndarray, "n_frames n_views n_kpts 2"] = uv_stack.copy()
    K_views: Float[ndarray, "n_views 3 3"] = np.asarray(intrinsics_stack)

    fx: Float[ndarray, "n_views"] = K_views[:, 0, 0]
    fy: Float[ndarray, "n_views"] = K_views[:, 1, 1]
    cx: Float[ndarray, "n_views"] = K_views[:, 0, 2]
    cy: Float[ndarray, "n_views"] = K_views[:, 1, 2]

    uv_normalized: Float[ndarray, "n_frames n_views n_kpts 2"] = uv_distorted.copy()
    uv_normalized[..., 0] = (uv_normalized[..., 0] - cx[None, :, None]) / fx[None, :, None]
    uv_normalized[..., 1] = (uv_normalized[..., 1] - cy[None, :, None]) / fy[None, :, None]

    n_frames: int = uv_stack.shape[0]
    n_kpts: int = uv_stack.shape[2]

    for view_idx, distortion in enumerate(distortions):
        if distortion is None:
            continue
        view_norm: Float[ndarray, "n_frames n_kpts 2"] = uv_normalized[:, view_idx, :, :]
        view_norm_flat: Float[ndarray, "_ 2"] = view_norm.reshape(n_frames * n_kpts, 2)
        distorted_flat: Float[ndarray, "_ 2"] = apply_radial_tangential_distortion(distortion, view_norm_flat)
        uv_normalized[:, view_idx, :, :] = distorted_flat.reshape(n_frames, n_kpts, 2)

    uv_distorted[..., 0] = uv_normalized[..., 0] * fx[None, :, None] + cx[None, :, None]
    uv_distorted[..., 1] = uv_normalized[..., 1] * fy[None, :, None] + cy[None, :, None]

    return uv_distorted


def project_kannala_brandt_batched(
    xyz_stack_world: Float[ndarray, "n_frames n_points 3"],
    pinhole_param_list: list[Fisheye62Parameters],
    filter_invalid: bool = True,
) -> Float[ndarray, "n_frames n_views n_points 2"]:
    """Project world-frame keypoints through a batch of Kannala–Brandt fisheye cameras.

    Args:
        xyz_stack_world: World-frame coordinates ``[n_frames, n_points, 3]`` to reproject.
        pinhole_param_list: Ordered camera models defining extrinsics/intrinsics (one per view).
        filter_invalid: When ``True`` (default) mask pixels that fall outside the image bounds or
            behind the camera. Disable to obtain the raw projection for debugging/comparison.

    Returns:
        Distorted pixel coordinates ``[n_frames, n_views, n_points, 2]``. When ``filter_invalid`` is
        ``True`` (default) points falling outside the image bounds or behind the camera are filtered.

    Notes:
        * Assumes all cameras share identical image dimensions.
        * Uses ``filter_out_of_bounds`` to drop invalid pixels when ``filter_invalid`` is enabled.
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
    uv_stack: Float[ndarray, "n_frames n_views n_points 2"] = arctan_cam_to_image_batched(
        xyz_cam=xyz_stack_cam, K=K_stack
    )
    # TODO currently assumes same distortion coeffs for all cameras, should be extended to support per-camera coeffs
    # 3. Apply Kannala–Brandt distortion if coefficients are provided
    if pinhole_param_list[0].distortion is not None:
        distortions: list[KannalaBrandtDistortion | None] = [pinhole.distortion for pinhole in pinhole_param_list]
        uv_stack = apply_kannala_brandt_distortion_batch(
            uv_stack=uv_stack, intrinsics_stack=K_stack, distortions=distortions
        )
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
