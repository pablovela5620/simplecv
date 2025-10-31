import numpy as np
from jaxtyping import Float

from simplecv.umetrack_temp.camera_models import FisheyeCameraParameter, PinholeCameraParameter


def print_tensor_info(tensor):
    print(f"shape: {tensor.shape}")
    print(f"dtype: {tensor.dtype}")


def get_transformation_matrix(
    camera: FisheyeCameraParameter | PinholeCameraParameter,
) -> Float[np.ndarray, "4 4"]:
    """
    Calculate and return the transformation matrix for the given camera.

    Args:
        camera (Union[FisheyeCameraParameter, PinholeCameraParameter]): An instance representing
                either a FisheyeCameraParameter or PinholeCameraParameter.
        world2cam (bool): A boolean indicating the direction of transformation. If True, the
                transformation is from world to camera. If False, the transformation is from
                camera to world. Default is True.

    Returns:
        Float[np.ndarray, "4 4"]: A 4x4 transformation matrix as a NumPy array of float64.
    """
    # Extract rotation matrix and translation vector from camera
    if hasattr(camera, "get_extrinsic_r"):
        r_mat = np.array(camera.get_extrinsic_r())
    else:
        r_mat = np.array(camera.extrinsic_r)

    if hasattr(camera, "get_extrinsic_t"):
        t_vec = np.array(camera.get_extrinsic_t())
    else:
        t_vec = np.array(camera.extrinsic_t)

    # Initialize the transformation matrix as an identity matrix
    xform_mat = np.eye(4)
    xform_mat[:3, :3] = r_mat  # Place the rotation matrix in the top left of the transformation matrix
    xform_mat[:3, 3] = t_vec  # Place the translation vector in the top right of the transformation matrix

    return xform_mat
