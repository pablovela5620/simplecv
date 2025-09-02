from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Float32
from numpy import ndarray

from simplecv.ops import mano_np, mano_torch

ROOT = Path("/mnt/8tb/data/hocap/datasets")
SUBJECT = "8"
SEQUENCES = [
    "20231024_180111",
    "20231024_180651",
    "20231024_180733",
    "20231024_181413",
]

"""
Beartype + pytest-beartype enforce function annotations (args/returns) at runtime.
For locals, prefer checking at function boundaries. Avoid intentionally wrong
local annotations; rely on return annotations for runtime validation.
"""


def _load_betas() -> Float32[ndarray, "10"] | None:
    try:
        from serde.yaml import from_yaml
        from simplecv.data.exoego.hocap import CalibratedMano
    except Exception:
        return None
    yaml_path = ROOT / "calibration" / "mano" / f"subject_{SUBJECT}.yaml"
    if not yaml_path.exists():
        return None
    return from_yaml(CalibratedMano, yaml_path.read_text()).betas.astype(np.float32)


def _load_poses(seq: str) -> Float32[ndarray, "n_frames 2 51"] | None:
    # primary location (full dataset)
    p = ROOT / f"subject_{SUBJECT}" / seq / "poses_m.npy"
    if p.exists():
        poses: Float32[ndarray, "n_hands=2 n_frames 51"] = np.load(p).astype(np.float32)
        poses: Float32[ndarray, "n_frames 2 51"] = np.transpose(poses, (1, 0, 2))
        return poses
    # fallback like the sample layout
    p2 = ROOT / "poses" / f"subject_{SUBJECT}" / seq / "poses_m.npy"
    if p2.exists():
        poses: Float32[ndarray, "n_hands=2 n_frames 51"] = np.load(p2).astype(np.float32)
        poses: Float32[ndarray, "n_frames 2 51"] = np.transpose(poses, (1, 0, 2))
        return poses
    return None


@pytest.mark.slow
@pytest.mark.parametrize("sequence", SEQUENCES)
def test_mano_np_matches_torch_on_full_hocap(sequence: str):
    torch = pytest.importorskip("torch")

    if not ROOT.exists():
        pytest.skip(f"Hocap root {ROOT} not present")

    betas = _load_betas()
    if betas is None:
        pytest.skip("Could not load subject betas; skipping")

    poses_m = _load_poses(sequence)
    if poses_m is None:
        pytest.skip(f"Could not load poses for sequence {sequence}")

    mano_root = Path("data")  # use local MANO pkl files
    assert (mano_root / "MANO_RIGHT.pkl").exists() and (mano_root / "MANO_LEFT.pkl").exists()

    n = min(3, poses_m.shape[0])
    for side, idx in [("right", 0), ("left", 1)]:
        # Torch
        layer_t = mano_torch.MANOLayerTorch(side=side, betas=betas, mano_root_dir=mano_root)
        poses = poses_m[:n, idx, :48]
        trans = poses_m[:n, idx, 48:51]
        vt_t, jt_t = layer_t(torch.from_numpy(poses), torch.from_numpy(trans))
        vt_t = vt_t.detach().cpu().numpy()
        jt_t = jt_t.detach().cpu().numpy()

        # NumPy
        layer_n = mano_np.MANOLayerNP(side=side, betas=betas, mano_root_dir=mano_root)
        vt_n, jt_n = layer_n(poses, trans)

        np.testing.assert_allclose(vt_n, vt_t, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(jt_n, jt_t, rtol=1e-3, atol=1e-3)
