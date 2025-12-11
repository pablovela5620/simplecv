"""Parity checks for Brown–Conrady batched projection against OpenCV."""

from __future__ import annotations

import cv2
import hypothesis.extra.numpy as hnp
import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, composite

from simplecv.camera_parameters import BrownConradyDistortion, Extrinsics, Intrinsics, PinholeParameters
from simplecv.sensors.camera.brown_conrady import project_brown_conrady_grid


@composite
def brown_conrady_case(draw: DrawFn) -> tuple[np.ndarray, list[PinholeParameters]]:
    """Random batched scenario with per-view Brown–Conrady coefficients."""

    n_frames = draw(st.integers(min_value=1, max_value=2))
    n_points = draw(st.integers(min_value=1, max_value=48))
    n_views = draw(st.integers(min_value=1, max_value=3))

    xyz_world = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(n_frames, n_points, 3),
            elements=st.floats(min_value=-1.5, max_value=1.5, allow_nan=False, allow_infinity=False),
        )
    )
    # keep points in front of camera to avoid divide-by-zero in projection
    xyz_world[..., 2] = np.abs(xyz_world[..., 2]) + 0.5

    pinholes: list[PinholeParameters] = []
    for _ in range(n_views):
        fx = draw(st.floats(min_value=200.0, max_value=1500.0))
        fy = draw(st.floats(min_value=200.0, max_value=1500.0))
        cx = draw(st.floats(min_value=100.0, max_value=1200.0))
        cy = draw(st.floats(min_value=100.0, max_value=800.0))
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        intrinsics = Intrinsics.from_k_matrix(camera_conventions="RDF", k_matrix=K, height=1080, width=1920)

        dist_vals = draw(
            hnp.arrays(
                dtype=np.float64,
                shape=(14,),
                elements=st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False),
            )
        )
        distortion = BrownConradyDistortion(
            k1=float(dist_vals[0]),
            k2=float(dist_vals[1]),
            p1=float(dist_vals[2]),
            p2=float(dist_vals[3]),
            k3=float(dist_vals[4]),
            k4=float(dist_vals[5]),
            k5=float(dist_vals[6]),
            k6=float(dist_vals[7]),
            s1=float(dist_vals[8]),
            s2=float(dist_vals[9]),
            s3=float(dist_vals[10]),
            s4=float(dist_vals[11]),
            tau_x=float(dist_vals[12]),
            tau_y=float(dist_vals[13]),
        )

        extrinsics = Extrinsics(cam_R_world=np.eye(3), cam_t_world=np.zeros(3))
        pinholes.append(PinholeParameters(name="view", intrinsics=intrinsics, extrinsics=extrinsics, distortion=distortion))

    return xyz_world, pinholes


@settings(deadline=None, max_examples=25, suppress_health_check=[HealthCheck.too_slow])
@given(brown_conrady_case())
def test_brown_conrady_matches_opencv(case: tuple[np.ndarray, list[PinholeParameters]]) -> None:
    """NumPy Brown–Conrady projection should match OpenCV's projectPoints for each view."""

    xyz_world, pinholes = case
    n_frames, n_points, _ = xyz_world.shape
    n_views = len(pinholes)

    uv_bc = project_brown_conrady_grid(
        xyz_stack_world=xyz_world.astype(np.float64),
        pinholes_per_view=pinholes,
        filter_invalid=False,
    )
    assert uv_bc.shape == (n_frames, n_views, n_points, 2)

    uv_cv = np.empty_like(uv_bc)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)

    for view_idx, pinhole in enumerate(pinholes):
        assert pinhole.distortion is not None
        pts3d = xyz_world.reshape(-1, 3)
        dist_vec = np.array(
            [
                pinhole.distortion.k1,
                pinhole.distortion.k2,
                pinhole.distortion.p1,
                pinhole.distortion.p2,
                pinhole.distortion.k3,
                pinhole.distortion.k4,
                pinhole.distortion.k5,
                pinhole.distortion.k6,
                pinhole.distortion.s1,
                pinhole.distortion.s2,
                pinhole.distortion.s3,
                pinhole.distortion.s4,
                pinhole.distortion.tau_x,
                pinhole.distortion.tau_y,
            ],
            dtype=np.float64,
        )
        uv_flat, _ = cv2.projectPoints(
            pts3d,
            rvec,
            tvec,
            np.asarray(pinhole.intrinsics.k_matrix, dtype=np.float64),
            dist_vec,
        )
        uv_cv[:, view_idx, :, :] = uv_flat.reshape(n_frames, n_points, 2)

    np.testing.assert_allclose(uv_bc, uv_cv, rtol=1e-9, atol=1e-9)
