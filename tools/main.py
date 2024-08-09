import numpy as np

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters


def main():
    intri_params = Intrinsics(
        camera_conventions="RDF",
        fl_x=1.0,
        fl_y=1.0,
        cx=1.0,
        cy=1.0,
        height=1,
        width=1,
    )
    extri_params = Extrinsics(world_R_cam=np.eye(3), world_t_cam=np.zeros(3))
    cam_params = PinholeParameters(
        name="test", extrinsics=extri_params, intrinsics=intri_params
    )
    print(cam_params.projection_matrix)


if __name__ == "__main__":
    main()
