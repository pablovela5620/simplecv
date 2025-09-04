import enum
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import jax
import jax.numpy as npj
import numpy as np
from einops import rearrange
from jax import jit
from jaxopt import LevenbergMarquardt
from jaxopt._src.levenberg_marquardt import LevenbergMarquardtState
from jaxtyping import Array, Bool, Float
from numpy import ndarray

from simplecv.ops.mano.mano_jax import ManoSimpleLayerJAX


@dataclass
class OptimizationResults:
    """Results for a single hand."""

    xyz_mano: Float[ndarray, "21 3"]  # meters
    so3: Float[ndarray, "48"]
    trans: Float[ndarray, "3"]
    betas: Float[ndarray, "10"] | None = None


class LossWeights(TypedDict):
    keypoint_2d: float
    depth: float
    temp: float


@jit
def proj_3d_vectorized(
    xyz_hom: Float[Array, "n_frames n_kpts 4"],
    P: Float[Array, "n_views 3 4"],
) -> Float[Array, "n_frames n_views n_kpts 2"]:
    """
    Projects 3D points to 2D using the projection matrix for a batch of frames and views.

    xyz_hom: [n_frames, 21, 4]  -> [x, y, z, 1]
    P:       [n_views, 3, 4]    -> K [R|t]

    returns: [n_frames, n_views, n_kpts, 2]
    """
    xyz_hom: Float[Array, "n_frames 1 4 n_kpts"] = rearrange(
        xyz_hom, "n_frames n_kpts xyz_hom -> n_frames 1 xyz_hom n_kpts"
    )
    P_b: Float[Array, "1 n_views 3 4"] = rearrange(P, "n_views n m -> 1 n_views n m")
    # [1, n_views, 3, 4] @ [n_frames, 1, 4, n_kpts] -> [n_frames, n_views, 3, n_kpts]
    uv_hom: Float[Array, "n_frames n_views 3 n_kpts"] = P_b @ xyz_hom
    uv_hom = rearrange(uv_hom, "n_frames n_views xyz_hom n_kpts -> n_frames n_views n_kpts xyz_hom")
    # Robust division to avoid Inf/NaN during early iterations (z ≈ 0)
    denom = uv_hom[..., 2:]
    eps = npj.array(1e-8, dtype=denom.dtype)
    denom_safe = npj.where(npj.abs(denom) < eps, npj.sign(denom) * eps, denom)
    uv = uv_hom[..., :2] / denom_safe
    return uv


type FwdKinematics = Callable[
    [Float[Array, "b 48"], Float[Array, "b 10"], Float[Array, "b 3"]],
    tuple[Float[Array, "b n_verts=778 3"], Float[Array, "b joints_and_tips=21 3"]],
]

# jaxopt residual signature
type ResidualFn = Callable[
    [
        Float[Array, "_"],  # flattened params (51)
        Float[Array, "b 10"],  # betas
        Float[Array, "b 3 4"],  # Pall
        Float[Array, "b n_views 21 2"],  # uv_pred
        "LossWeights",
        bool | Bool[Array, ""],  # unused (kept for interface parity)
    ],
    Float[Array, "_"],  # flat residual vector
]


def make_mv_scaled_residual(side: Literal["left", "right"]) -> tuple[ResidualFn, FwdKinematics]:
    """Factory: residual function + FK for the requested hand side."""
    mano_fwd = jit(ManoSimpleLayerJAX(side=side, mano_root=Path("data/")))

    @jit
    def mv_2d_scaled_residual(
        param_to_optimize: Float[Array, "_"],
        beta: Float[Array, "b 10"],
        Pall: Float[Array, "b 3 4"],
        uv_pred: Float[Array, "b n_views 21 2"],
        loss_weights: LossWeights,
        is_left: bool | Bool[Array, ""],  # not used in single-hand path; kept for signature
    ) -> Float[Array, "_"]:
        batch_size: int = uv_pred.shape[0]
        params_2d: Float[Array, "b 51"] = param_to_optimize.reshape(batch_size, 51)
        so3: Float[Array, "b 48"] = params_2d[:, 0:48]
        trans: Float[Array, "b 3"] = params_2d[:, 48:51]

        # MANO forward (mm), convert to meters for projection
        _, xyz_mano_mm = mano_fwd(so3, beta, trans)
        xyz_mano: Float[Array, "b 21 3"] = xyz_mano_mm / 1000.0

        xyz_mano_hom: Float[Array, "b 21 4"] = npj.concatenate([xyz_mano, npj.ones_like(xyz_mano)[..., 0:1]], axis=-1)
        uv_mano: Float[Array, "b n_views 21 2"] = proj_3d_vectorized(xyz_hom=xyz_mano_hom, P=Pall)

        res_2d: Float[Array, "b n_views 21 2"] = uv_mano - uv_pred
        res_2d = npj.nan_to_num(res_2d * loss_weights["keypoint_2d"], nan=0.0, posinf=0.0, neginf=0.0)
        return res_2d.flatten()

    return mv_2d_scaled_residual, mano_fwd


class SingleHandOptimization:
    def __init__(
        self,
        beta: Float[ndarray, "10"],
        Pall: Float[ndarray, "n_views 3 4"],
        hand_side: Literal["left", "right"],
        loss_weights: LossWeights,
        num_iters: int = 30,
    ) -> None:
        """
        Single-hand optimizer over MANO pose (axis-angle 48) + translation (3).
        Should avoid chaning the number of views, this causes retraces in jax jit which is bad
        instead use a validity mask
        """
        self.hand_side = hand_side
        self.num_iters = num_iters
        self.beta: Float[Array, "1 10"] = npj.array(beta)[npj.newaxis, ...]
        self.Pall: Float[Array, "n_views 3 4"] = npj.array(Pall)
        self.loss_weights: LossWeights = loss_weights

        # Init previous state (warm-start)
        self.so3_prev: Float[Array, "1 48"] = npj.zeros((1, 48))
        # Sensible depth prior (meters) to aid convergence
        self.trans_prev: Float[Array, "1 3"] = npj.array([[0.0, 0.0, 0.6]])
        # Track last finite pose to fall back to if needed
        self.last_finite_so3: Float[Array, "1 48"] = self.so3_prev.copy()
        self.last_finite_trans: Float[Array, "1 3"] = self.trans_prev.copy()

        residual_fn, self.mano_fwd = make_mv_scaled_residual(hand_side)

        self.optimizer = LevenbergMarquardt(
            residual_fun=residual_fn,
            maxiter=self.num_iters,
            solver="cholesky",
            jit=True,
            xtol=1e-6,
            gtol=1e-6,
        )

        # Trace JIT once
        print("Tracing JIT, can take a while...")
        n_views = Pall.shape[0]
        init_params: Float[Array, "1 51"] = npj.concatenate([self.so3_prev, self.trans_prev], axis=-1)
        uv_batch_init: Float[Array, "1 n_views 21 2"] = npj.zeros((1, n_views, 21, 2))
        _ = self.optimizer.run(
            init_params.flatten(),
            beta=self.beta,
            Pall=self.Pall,
            uv_pred=uv_batch_init,
            loss_weights=self.loss_weights,
            is_left=(self.hand_side == "left"),
        )
        self.optimizer = jit(self.optimizer.run)
        print("Trace Done")

    def __call__(
        self,
        uv_pred_batch: Float[ndarray, "n_views 21 2"],
    ) -> tuple[OptimizationResults, LevenbergMarquardtState]:
        uv_pred_batched: Float[Array, "1 n_views 21 2"] = npj.array(uv_pred_batch)[npj.newaxis, ...]

        # Try multiple inits to escape poor local minima in single-view scenarios
        init_candidates: list[Float[Array, "1 51"]] = []
        so3_prev = self.so3_prev.copy()
        trans_prev = self.trans_prev.copy()
        init_candidates.append(npj.concatenate([so3_prev, trans_prev], axis=-1))
        init_candidates.append(npj.concatenate([so3_prev, npj.array([[0.0, 0.0, 0.6]])], axis=-1))
        init_candidates.append(npj.concatenate([so3_prev, npj.array([[0.0, 0.0, 0.8]])], axis=-1))

        best_opt: Float[Array, "1 51"] | None = None
        best_residual: float = float("inf")
        best_state: LevenbergMarquardtState | None = None

        for init_params in init_candidates:
            opt_params, st = self.optimizer(
                init_params.flatten(),
                beta=self.beta,
                Pall=self.Pall,
                uv_pred=uv_pred_batched,
                loss_weights=self.loss_weights,
                is_left=(self.hand_side == "left"),
            )
            opt_params_2d: Float[Array, "1 51"] = opt_params.reshape(1, 51)
            so3_cand: Float[Array, "1 48"] = opt_params_2d[:, 0:48]
            trans_cand: Float[Array, "1 3"] = opt_params_2d[:, 48:51]
            so3_cand = npj.nan_to_num(so3_cand, nan=0.0, posinf=0.0, neginf=0.0)
            trans_cand = npj.nan_to_num(trans_cand, nan=0.0, posinf=0.0, neginf=0.0)
            # Evaluate 2D reprojection error of this candidate
            _, joints_mm_cand = self.mano_fwd(so3_cand, self.beta, trans_cand)
            xyz_mano_cand: Float[Array, "1 21 3"] = joints_mm_cand / 1000.0
            xyz_mano_hom: Float[Array, "1 21 4"] = npj.concatenate(
                [xyz_mano_cand, npj.ones_like(xyz_mano_cand)[..., 0:1]], axis=-1
            )
            uv_cand: Float[Array, "1 n_views 21 2"] = proj_3d_vectorized(xyz_hom=xyz_mano_hom, P=self.Pall)
            diff = uv_cand - uv_pred_batched
            diff = npj.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
            res = float(npj.mean(diff * diff))
            if np.isfinite(res) and res < best_residual:
                best_residual = res
                best_opt = opt_params_2d
                best_state = st

        # Use best candidate
        assert best_opt is not None and best_state is not None
        so3: Float[Array, "1 48"] = best_opt[:, 0:48]
        trans: Float[Array, "1 3"] = best_opt[:, 48:51]
        # Sanitize any potential NaNs/Infs from the optimizer
        so3 = npj.nan_to_num(so3, nan=0.0, posinf=0.0, neginf=0.0)
        trans = npj.nan_to_num(trans, nan=0.0, posinf=0.0, neginf=0.0)

        # Warm start for next call
        self.so3_prev = so3
        self.trans_prev = trans

        # Forward pass to get joints in meters
        _, joints_mm = self.mano_fwd(so3, self.beta, trans)
        xyz_mano_m: Float[ndarray, "21 3"] = np.array((joints_mm / 1000.0)[0])

        # Safety: if still non-finite, fall back to last finite state
        if not np.isfinite(xyz_mano_m).all():
            _, joints_mm_prev = self.mano_fwd(self.last_finite_so3, self.beta, self.last_finite_trans)
            xyz_mano_m = np.array((joints_mm_prev / 1000.0)[0])
            so3 = self.last_finite_so3
            trans = self.last_finite_trans

        # Update last finite state if valid
        if np.isfinite(xyz_mano_m).all():
            self.last_finite_so3 = so3
            self.last_finite_trans = trans

        results = OptimizationResults(
            xyz_mano=xyz_mano_m,
            so3=np.array(so3[0]),
            trans=np.array(trans[0]),
            betas=np.array(self.beta[0]),
        )
        return results, best_state
