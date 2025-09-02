from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import torch

from simplecv.ops import mano_np, mano_torch

# ----------------------
# Small functional parity
# ----------------------


@given(
    quat=arrays(
        dtype=np.float32,
        shape=(1, 4),
        elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=50)
def test_quat2mat_parity(quat: np.ndarray):
    q_t = torch.from_numpy(quat)
    m_t = mano_torch.quat2mat(q_t).detach().cpu().numpy()
    m_n = mano_np.quat2mat(quat)
    np.testing.assert_allclose(m_n, m_t, rtol=1e-5, atol=1e-5)


@given(
    aa=arrays(
        dtype=np.float32,
        shape=(1, 3),
        elements=st.floats(min_value=-3.14, max_value=3.14, allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=50)
def test_batch_rodrigues_parity(aa: np.ndarray):
    a_t = torch.from_numpy(aa)
    r_t = mano_torch.batch_rodrigues(a_t).detach().cpu().numpy()
    r_n = mano_np.batch_rodrigues(aa)
    np.testing.assert_allclose(r_n, r_t, rtol=1e-5, atol=1e-5)


@given(
    pose=arrays(
        dtype=np.float32,
        shape=(1, 48),
        elements=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=25)
def test_posemap_and_subflatid_parity(pose: np.ndarray):
    p_t = torch.from_numpy(pose)
    rm_t = mano_torch.th_posemap_axisang(p_t).detach().cpu().numpy()
    pm_t = mano_torch.subtract_flat_id(torch.from_numpy(rm_t)).detach().cpu().numpy()

    rm_n = mano_np.th_posemap_axisang(pose)
    pm_n = mano_np.subtract_flat_id(rm_n)

    np.testing.assert_allclose(rm_n, rm_t, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(pm_n, pm_t, rtol=1e-5, atol=1e-5)


@given(
    mats=arrays(
        dtype=np.float32,
        shape=(1, 3, 4),
        elements=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=25)
def test_with_zeros_parity(mats: np.ndarray):
    m_t = mano_torch.th_with_zeros(torch.from_numpy(mats)).detach().cpu().numpy()
    m_n = mano_np.th_with_zeros(mats)
    np.testing.assert_allclose(m_n, m_t, rtol=1e-6, atol=1e-6)


# ----------------------
# Integration on HOCAP
# ----------------------


def _find_hocap_sample() -> tuple[np.ndarray, np.ndarray] | None:
    base = Path("data/hocap/sample")
    if not base.exists():
        return None

    # Betas YAMLs are under calibration/mano/subject_*.yaml
    # We'll read subject_8 (present in sample)
    try:
        from serde.yaml import from_yaml
        from simplecv.data.exoego.hocap import CalibratedMano
    except Exception:
        return None

    beta_yaml = base / "calibration" / "mano" / "subject_8.yaml"
    if not beta_yaml.exists():
        return None
    betas = from_yaml(CalibratedMano, beta_yaml.read_text()).betas.astype(np.float32)

    # Locate poses_m.npy (sample stores under poses/subject_8/<seq>/)
    poses_candidates = list((base / "poses").glob("subject_8/*/poses_m.npy"))
    if not poses_candidates:
        return None
    poses_m = np.load(poses_candidates[0]).astype(np.float32)  # [2, N, 51]
    poses_m = np.transpose(poses_m, (1, 0, 2))  # [N, 2, 51]
    return betas, poses_m


@pytest.mark.slow
def test_mano_np_matches_torch_on_hocap_sample():
    res = _find_hocap_sample()
    if res is None:
        pytest.skip("HoCap sample not available; skipping integration test")
    betas, poses_m = res

    mano_root = Path("data")
    assert (mano_root / "MANO_RIGHT.pkl").exists() and (mano_root / "MANO_LEFT.pkl").exists()

    # Compare for both hands on a few frames
    n = min(3, poses_m.shape[0])
    for side, idx in [("right", 0), ("left", 1)]:
        # Torch layer
        layer_t = mano_torch.MANOLayerTorch(side=side, betas=betas, mano_root_dir=mano_root)
        poses = poses_m[:n, idx, :48]
        trans = poses_m[:n, idx, 48:51]
        vt_t, jt_t = layer_t(torch.from_numpy(poses), torch.from_numpy(trans))
        vt_t = vt_t.detach().cpu().numpy()
        jt_t = jt_t.detach().cpu().numpy()

        # NumPy layer
        layer_n = mano_np.MANOLayerNP(side=side, betas=betas, mano_root_dir=mano_root)
        vt_n, jt_n = layer_n(poses, trans)

        np.testing.assert_allclose(vt_n, vt_t, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(jt_n, jt_t, rtol=1e-3, atol=1e-3)
