import numpy as np
from jaxtyping import Float


def projectN3(
    kpts3d: Float[np.ndarray, "n_views n_kpts 4"],
    Pall: Float[np.ndarray, "n_views 3 4"],
) -> Float[np.ndarray, "nViews nJoints 3"]:
    nViews: int = len(Pall)
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
    keypoints_2d: Float[np.ndarray, "nViews nJoints 3"],
    projection_matrices: Float[np.ndarray, "nViews 3 4"],
    min_views: int = 2,
) -> Float[np.ndarray, "nJoints 4"]:
    """
    Camera has to be in OPENCV convention
    """
    num_joints = keypoints_2d.shape[1]

    # Count views where each joint is visible
    visibility_count = (keypoints_2d[:, :, -1] > 0).sum(axis=0)
    valid_joints = np.where(visibility_count >= min_views)[0]

    # Filter keypoints by valid joints
    filtered_keypoints = keypoints_2d[:, valid_joints]
    conf3d = filtered_keypoints[:, :, -1].sum(axis=0) / visibility_count[valid_joints]

    # (1, nViews, 1, 4)
    P0 = projection_matrices[None, :, 0, :]
    P1 = projection_matrices[None, :, 1, :]
    P2 = projection_matrices[None, :, 2, :]

    # Triangulation calculations
    uP2 = filtered_keypoints[:, :, 0].T[:, :, None] * P2
    vP2 = filtered_keypoints[:, :, 1].T[:, :, None] * P2
    confidences = filtered_keypoints[:, :, 2].T[:, :, None]

    Au = confidences * (uP2 - P0)
    Av = confidences * (vP2 - P1)
    A = np.hstack([Au, Av])

    # Solve using SVD
    _, _, Vh = np.linalg.svd(A)
    triangulated_points = Vh[:, -1, :]
    triangulated_points /= triangulated_points[:, 3, None]

    # Construct result
    result = np.zeros((num_joints, 4))
    result[valid_joints, :3] = triangulated_points[:, :3]
    result[valid_joints, 3] = conf3d

    return result
