from __future__ import annotations

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
from simplecv.ops.mano.optim_jax_single import proj_3d_vectorized


class LossWeights(TypedDict):
    keypoint_2d: float
    depth: float
    temp: float


@dataclass
class ShapeOptimizationResults:
    """Results for single-hand, multi-frame shape optimization.

    - xyz_mano in meters (per-frame joints)
    - `so3` and `trans` per frame
    - shared `betas` across frames
    """

    xyz_mano: Float[ndarray, "n_frames 21 3"]
    so3: Float[ndarray, "n_frames 48"]
    trans: Float[ndarray, "n_frames 3"]
    betas: Float[ndarray, "10"]


type ResidualFn = (
    # params_flat, P, uv_pred, loss_weights, is_left
    tuple[Float[Array, "_"], Float[Array, "n_views 3 4"], Float[Array, "n_frames n_views 21 2"], LossWeights, bool]
    | Float[Array, "_"]
)


class SingleHandOptimUnknownShape:
    """Optimizes MANO betas + per-frame pose/translation over a fixed window.

    Shapes are fixed for JIT:
      - `n_frames` (window length) is constant per instance
      - `n_views` is taken from `Pall` and constant per instance

    Residual masking is done inside the residual to avoid view slicing and JIT retraces:
      - Invalid 2D entries (NaN/Inf) contribute zero cost and zero gradient
      - Optional stereo gating: frames with <2 valid views are masked out entirely
    """

    def __init__(
        self,
        *,
        hand_side: Literal["left", "right"],
        Pall: Float[ndarray, "n_views 3 4"],
        n_frames: int,
        loss_weights: LossWeights,
        num_iters: int = 30,
        stereo_gate: bool = True,
    ) -> None:
        self.hand_side = hand_side
        self.Pall: Float[Array, "n_views 3 4"] = npj.array(Pall)
        self.n_views: int = int(Pall.shape[0])
        self.n_frames: int = int(n_frames)
        self.loss_weights: LossWeights = loss_weights
        self.stereo_gate: bool = stereo_gate
        self.num_iters: int = num_iters

        # Prepare MANO forward and residual
        self._mano_fwd = jit(ManoSimpleLayerJAX(side=self.hand_side, mano_root=Path("data/")))

        def residual_fun(
            params_flat: Float[Array, "_"],
            Pall_in: Float[Array, "n_views 3 4"],
            uv_pred: Float[Array, "n_frames n_views 21 2"],
            loss_weights: LossWeights,
            is_left: bool | Bool[Array, ""],  # kept for parity with other calls
        ) -> Float[Array, "_"]:
            # Unpack params: [betas(10), frames * (so3(48)+trans(3))]
            n_frames = self.n_frames
            betas: Float[Array, "10"] = params_flat[:10]
            frame_params: Float[Array, "n_frames 51"] = params_flat[10:].reshape((n_frames, 51))
            so3: Float[Array, "n_frames 48"] = frame_params[:, :48]
            trans: Float[Array, "n_frames 3"] = frame_params[:, 48:51]

            # Broadcast betas per frame
            betas_b: Float[Array, "n_frames 10"] = npj.repeat(betas[None, :], repeats=n_frames, axis=0)

            # MANO forward (mm) → meters
            _, joints_mm = self._mano_fwd(so3, betas_b, trans)
            xyz_m: Float[Array, "n_frames 21 3"] = joints_mm / 1000.0
            xyz_hom: Float[Array, "n_frames 21 4"] = npj.concatenate(
                [xyz_m, npj.ones_like(xyz_m)[..., 0:1]], axis=-1
            )
            uv_proj: Float[Array, "n_frames n_views 21 2"] = proj_3d_vectorized(xyz_hom=xyz_hom, P=Pall_in)

            # Build masks from uv_pred finiteness (keep shape fixed)
            finite_mask: Float[Array, "n_frames n_views 21 1"] = npj.isfinite(uv_pred).all(axis=-1, keepdims=True)
            uv_target: Float[Array, "n_frames n_views 21 2"] = npj.nan_to_num(uv_pred, nan=0.0, posinf=0.0, neginf=0.0)
            res: Float[Array, "n_frames n_views 21 2"] = (uv_proj - uv_target) * finite_mask

            # Stereo gating per frame (mask entire frame if fewer than 2 valid views)
            if self.stereo_gate:
                views_ok: Bool[Array, "n_frames n_views"] = npj.isfinite(uv_pred).all(axis=(2, 3))
                has_stereo: Float[Array, "n_frames 1 1 1"] = (
                    (npj.sum(views_ok.astype(npj.int32), axis=1) >= 2).astype(npj.float32).reshape((n_frames, 1, 1, 1))
                )
                res = res * has_stereo

            res = npj.nan_to_num(res * loss_weights["keypoint_2d"], nan=0.0, posinf=0.0, neginf=0.0)
            return res.flatten()

        self._residual_fun = jit(residual_fun)
        self._lm = LevenbergMarquardt(
            residual_fun=self._residual_fun,
            maxiter=self.num_iters,
            solver="cholesky",
            jit=True,
            xtol=1e-6,
            gtol=1e-6,
        )

        # Warmup JIT trace with zeros
        print("Tracing JIT (learn-shape), can take a while...")
        init_params = self._init_params()
        uv_zeros: Float[Array, "n_frames n_views 21 2"] = npj.zeros((self.n_frames, self.n_views, 21, 2))
        _ = self._lm.run(
            init_params,
            Pall_in=self.Pall,
            uv_pred=uv_zeros,
            loss_weights=self.loss_weights,
            is_left=(self.hand_side == "left"),
        )
        self._lm_run = jit(self._lm.run)
        print("Trace Done (learn-shape)")

    def _init_params(self, beta_init: Float[ndarray, "10"] | None = None) -> Float[Array, "_"]:
        # Default betas=0 if not provided
        if beta_init is None:
            betas0: Float[Array, "10"] = npj.zeros((10,), dtype=npj.float32)
        else:
            betas0 = npj.array(beta_init, dtype=npj.float32)
        # Per-frame pose/trans init
        so30: Float[Array, "n_frames 48"] = npj.zeros((self.n_frames, 48), dtype=npj.float32)
        trans0: Float[Array, "n_frames 3"] = npj.zeros((self.n_frames, 3), dtype=npj.float32)
        trans0 = trans0.at[:, 2].set(0.6)  # z-prior in meters
        frame_params: Float[Array, "n_frames 51"] = npj.concatenate([so30, trans0], axis=-1)
        return npj.concatenate([betas0, frame_params.flatten()])

    def __call__(
        self,
        uv_pred: Float[ndarray, "n_frames n_views 21 2"],
        beta_init: Float[ndarray, "10"] | None = None,
    ) -> tuple[ShapeOptimizationResults, LevenbergMarquardtState]:
        # Sanity: fixed shapes expected per instance
        assert uv_pred.shape[0] == self.n_frames, "uv_pred n_frames mismatch"
        assert uv_pred.shape[1] == self.n_views, "uv_pred n_views mismatch"

        init_params = self._init_params(beta_init)
        optimized_params, state = self._lm_run(
            init_params,
            Pall_in=self.Pall,
            uv_pred=npj.array(uv_pred),
            loss_weights=self.loss_weights,
            is_left=(self.hand_side == "left"),
        )

        # Unpack
        betas: Float[Array, "10"] = optimized_params[:10]
        frame_params: Float[Array, "n_frames 51"] = optimized_params[10:].reshape((self.n_frames, 51))
        so3: Float[Array, "n_frames 48"] = frame_params[:, :48]
        trans: Float[Array, "n_frames 3"] = frame_params[:, 48:51]

        # Final forward to joints in meters
        betas_b: Float[Array, "n_frames 10"] = npj.repeat(betas[None, :], repeats=self.n_frames, axis=0)
        _, joints_mm = self._mano_fwd(so3, betas_b, trans)
        xyz_m: Float[ndarray, "n_frames 21 3"] = np.array(joints_mm / 1000.0)

        # Safety: ensure finiteness
        if not np.isfinite(xyz_m).all():
            # Fall back to NaN-to-num rather than raising, to keep downstream logging resilient
            xyz_m = np.nan_to_num(xyz_m, nan=0.0, posinf=0.0, neginf=0.0)

        results = ShapeOptimizationResults(
            xyz_mano=xyz_m,
            so3=np.array(so3),
            trans=np.array(trans),
            betas=np.array(betas),
        )
        return results, state

