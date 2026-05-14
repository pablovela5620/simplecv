"""Shared camera-space projection utilities."""

import numpy as np
from einops import rearrange
from jaxtyping import Float
from numpy import ndarray


def world_to_cam_batched(
    xyz_world: Float[ndarray, "n_frames n_points 3"],
    cam_T_world: Float[ndarray, "n_views 4 4"],
) -> Float[ndarray, "n_frames n_views n_points 3"]:
    """Transform world-frame points into each camera's coordinate system.

    Args:
        xyz_world: World-frame coordinates ``[n_frames, n_points, 3]``.
        cam_T_world: Camera-from-world transforms for each view ``[n_views, 4, 4]``.

    Returns:
        Camera-frame coordinates ``[n_frames, n_views, n_points, 3]`` with per-view poses applied.

    Notes:
        ``cam_T_world`` is treated as an affine transform; we slice out the
        ``[:3, :3]`` rotation and ``[:3, 3]`` translation and compute
        ``R @ xyz + t`` directly. That avoids materializing the homogeneous
        (n_frames, n_views, n_points, 4) tensor and the trailing divide by
        the homogeneous coordinate, which on Assembly101 (n_views=8,
        n_points=133, n_frames~16k) was ~1.3 s/sequence.
    """

    R: Float[ndarray, "n_views 3 3"] = cam_T_world[:, :3, :3]
    t: Float[ndarray, "n_views 3"] = cam_T_world[:, :3, 3]
    # Einsum semantics:
    #   v=n_views, i=cam axis (output 3), j=world axis (input 3),
    #   f=n_frames, p=n_points.
    xyz_cam: Float[ndarray, "n_frames n_views n_points 3"] = np.einsum(
        "vij,fpj->fvpi", R, xyz_world
    ) + t[None, :, None, :]
    return xyz_cam


def cam_to_world_batched(
    xyz_cam: Float[ndarray, "n_frames n_views n_points 3"],
) -> Float[ndarray, "n_frames n_views n_points 3"]:
    raise NotImplementedError("cam_to_world_batched is not implemented yet.")


def filter_out_of_bounds(
    uv_batch: Float[ndarray, "n_frames n_views n_points 2"],
    xyz_cam_batch: Float[ndarray, "n_frames n_views n_points 3"],
    h: int,
    w: int,
):
    """Mask pixels projected outside the image plane or behind the camera."""

    # make sure points are within image bounds
    out_of_bounds = np.logical_or(uv_batch[..., 0] >= w, uv_batch[..., 1] >= h)
    out_of_bounds = np.logical_or(out_of_bounds, uv_batch[..., 0] < 0)
    out_of_bounds = np.logical_or(out_of_bounds, uv_batch[..., 1] < 0)
    # make sure points are in front of camera
    out_of_bounds = np.logical_or(out_of_bounds, xyz_cam_batch[..., 2] < 0)

    # if out of bounds, set to nan
    uv_batch[out_of_bounds, :] = np.nan
    return uv_batch
