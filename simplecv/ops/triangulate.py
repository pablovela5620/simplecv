import numpy as np
from einops import rearrange
from jaxtyping import Float, Int
from numpy import ndarray


def projectN3(
    kpts3d: Float[ndarray, "n_views n_kpts 4"],
    Pall: Float[ndarray, "n_views 3 4"],
) -> Float[ndarray, "nViews nJoints 3"]:
    nViews: int = len(Pall)
    # convert to homogenous
    kp3d = np.hstack((kpts3d[:, :3], np.ones((kpts3d.shape[0], 1))))
    kp2ds = []
    for nv in range(nViews):
        kp2d = Pall[nv] @ kp3d.T
        kp2d[:2, :] /= kp2d[2:, :]
        kp2ds.append(kp2d.T[None, :, :])
    kp2ds = np.vstack(kp2ds)
    kp2ds[..., -1] = kp2ds[..., -1] * (kpts3d[None, :, -1] > 0.0)
    return kp2ds


def batch_triangulate(
    keypoints_2d: Float[ndarray, "nViews nJoints 3"],
    projection_matrices: Float[ndarray, "nViews 3 4"],
    min_views: int = 2,
) -> Float[ndarray, "nJoints 4"]:
    """
    Camera MUST be in OPENCV convention
    """
    num_joints: int = keypoints_2d.shape[1]

    # Count views where each joint is visible
    visibility_count: Int[ndarray, "nJoints"] = (keypoints_2d[:, :, -1] > 0).sum(axis=0)  # noqa: UP037
    valid_joints = np.where(visibility_count >= min_views)[0]

    # Filter keypoints by valid joints
    filtered_keypoints: Float[ndarray, "nViews nJoints 3"] = keypoints_2d[
        :, valid_joints
    ]
    conf3d = filtered_keypoints[:, :, -1].sum(axis=0) / visibility_count[valid_joints]

    P0: Float[ndarray, "1 nViews 4"] = projection_matrices[None, :, 0, :]
    P1: Float[ndarray, "1 nViews 4"] = projection_matrices[None, :, 1, :]
    P2: Float[ndarray, "1 nViews 4"] = projection_matrices[None, :, 2, :]

    # x-coords homogenous
    u: Float[ndarray, "nJoints nViews 1"] = rearrange(
        filtered_keypoints[..., 0], "c j -> j c 1"
    )
    uP2: Float[ndarray, "nJoints nViews 4"] = u * P2

    # y-coords homogenous
    v: Float[ndarray, "nJoints nViews 1"] = rearrange(
        filtered_keypoints[..., 1], "c j -> j c 1"
    )
    vP2: Float[ndarray, "nJoints nViews 4"] = v * P2

    confidences: Float[ndarray, "nJoints nViews 1"] = rearrange(
        filtered_keypoints[..., 2], "c j -> j c 1"
    )

    Au: Float[ndarray, "nJoints nViews 4"] = confidences * (uP2 - P0)
    Av: Float[ndarray, "nJoints nViews 4"] = confidences * (vP2 - P1)
    A: Float[ndarray, "nJoints _ 4"] = np.hstack([Au, Av])

    # Solve using SVD
    _, _, Vh = np.linalg.svd(A)
    triangulated_points = Vh[:, -1, :]
    triangulated_points /= triangulated_points[:, 3, None]

    # Construct result
    result: Float[ndarray, "nJoints 4"] = np.zeros((num_joints, 4))
    # convert from homogenous to euclidean and add confidence
    result[valid_joints, :3] = triangulated_points[:, :3]
    result[valid_joints, 3] = conf3d

    return result
