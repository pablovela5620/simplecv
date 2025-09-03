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
    """
    Stores results from bounding box detection and hand pose estimation
    """

    xyz_mano: Float[ndarray, "21 3"]
    so3: Float[ndarray, "48"]
    trans: Float[ndarray, "3"]
    betas: Float[ndarray, "10"] | None = None


class LossWeights(TypedDict):
    keypoint_2d: float
    depth: float
    temp: float


@jit
def proj_3d_vectorized(
    xyz_hom: Float[Array, "n_frames n_kpts 4"], P: Float[Array, "n_views 3 4"]
) -> Float[Array, "n_frames n_views n_kpts 2"]:
    """
    Projects 3D points to 2D using the projection matrix for a batch of frames and views.

    xyz_hom: [n_frames, 21, 4] [x, y, z, 1]
    P: [n_views, 3, 4] (projection matrix - includes extrensic (R, t) and intrinsic (K))

    return kp2d: [n_frames, n_views, n_kpts, 2] (squeeze out if 1)
    """
    # rearrange for batch matrix multiplication
    xyz_hom: Float[Array, "n_frames 1 4 21"] = rearrange(
        xyz_hom, "n_frames n_kpts xyz_hom -> n_frames 1 xyz_hom n_kpts"
    )
    P: Float[Array, "1 n_views 3 4"] = rearrange(P, "n_views n m -> 1 n_views n m")

    # [1 n_views, 3, 4] @ [n_frames, 1, 4, 21] -> [n_frames, n_views, 3, 21]
    uv_hom: Float[Array, "n_frames n_views 3 21"] = P @ xyz_hom
    uv_hom = rearrange(uv_hom, "n_frames n_views xyz_hom n_kpts -> n_frames n_views n_kpts xyz_hom")
    # convert back from homogeneous coordinates
    uv: Float[Array, "n_frames n_views 21 2"] = uv_hom[..., :2] / uv_hom[..., 2:]
    return uv


class HandSide(enum.IntEnum):
    """Represents the side of the hand."""

    LEFT = 0
    RIGHT = 1


type FwdKinematics = Callable[
    [Float[Array, "b 48"], Float[Array, "b 10"], Float[Array, "b 3"]],
    tuple[Float[Array, "b n_verts=778 3"], Float[Array, "b joints_and_tips=21 3"]],
]

# The residual you’ll hand to jaxopt
type ResidualFn = Callable[
    [
        Float[Array, "_"],  # flattened params + scale
        Float[Array, "b 3 4"],  # Pall
        Float[Array, "b n_views 21 2"],  # uv_pred
        "LossWeights",
        bool | Bool[Array, ""],
    ],
    Float[Array, "_"],  # flat residual vector
]


def make_mv_scaled_residual(side: Literal["left", "right"]) -> tuple[ResidualFn, FwdKinematics]:
    """
    Returns a JIT-compiled residual function that can be dropped straight into
    `jaxopt.LevenbergMarquardt`.  No globals leak out – the MANO forward
    functions are closed over the templates you pass in *once*.

    Example
    -------
    >>> mv_scaled_residual = make_mv_scaled_residual(
    ...     xyz_template_left , xyz_template_right
    ... )
    >>> solver = LevenbergMarquardt(residual_fun=mv_scaled_residual, ...)
    """

    # ------------------------------------------------------------------
    # build per-hand forward kinematics (static because templates are constant)
    # ------------------------------------------------------------------
    mano_fwd = jit(ManoSimpleLayerJAX(side=side, mano_root=Path("data/")))

    # ------------------------------------------------------------------
    # residual – declared once, re-used frame-to-frame
    # ------------------------------------------------------------------
    @jit
    def mv_2d_scaled_residual(
        param_to_optimize: Float[Array, "_"],
        beta: Float[Array, "b 10"],
        Pall: Float[Array, "b 3 4"],
        uv_pred: Float[Array, "b n_views 21 2"],
        loss_weights: LossWeights,
        is_left: bool | Bool[Array, ""],
    ) -> Float[Array, "_"]:
        """
        Calculates the residual error between projected MANO keypoints and target 2D keypoints.

        Args:
            param_to_optimize: Flattened MANO parameters (pose coefficients and translation).
                            Must be a 1D array (batch_size * 51) because jaxopt optimizers
                            like LevenbergMarquardt expect a flat vector of parameters.
            Pall: Projection matrices for each camera view, shape (b, 3, 4).
                'b' here refers to the batch size (number of frames/samples).
            uv_pred: Target 2D keypoints for each view and joint, shape (b, n_views, 21, 2).
                    'n_views' is the number of camera views.
            loss_weights: Dictionary containing weights for different loss components (e.g., 'keypoint_2d').
            is_left: Boolean indicating whether to use the left or right MANO model.

        Returns:
            A flattened 1D array containing the weighted residual errors for all keypoints, views, and batch items.
        """
        batch_size: int = uv_pred.shape[0]
        # extract parameters that are being optimized and add batch dimension
        param_to_optimize: Float[Array, "1 51"] = param_to_optimize.reshape(batch_size, 51)

        so3: Float[Array, "b 48"] = param_to_optimize[:, 0:48]
        trans: Float[Array, "b 3"] = param_to_optimize[:, 48:51]

        mano_output: tuple[Float[Array, "b n_verts=778 3"], Float[Array, "b joints_and_tips=21 3"]] = mano_fwd(
            so3, beta, trans
        )
        xyz_mano: Float[Array, "b n_kpts=21 3"] = mano_output[1]
        # ManoSimpleLayerJAX outputs joints in millimeters; convert to meters to
        # match the camera extrinsics/intrinsics units used in Pall.
        xyz_mano = xyz_mano / 1000.0

        xyz_mano_hom: Float[Array, "b n_kpts=21 4"] = npj.concatenate(
            [xyz_mano, npj.ones_like(xyz_mano)[..., 0:1]], axis=-1
        )

        uv_mano: Float[Array, "b n_views n_kpts=21 2"] = proj_3d_vectorized(xyz_hom=xyz_mano_hom, P=Pall)

        # calculate residuals
        res_2d: Float[Array, "b n_views n_kpts=21 2"] = uv_mano - uv_pred
        res_2d = npj.nan_to_num(res_2d * loss_weights["keypoint_2d"], nan=0.0)

        # Return the flattened vector of valid, weighted residuals
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
        beta - shape parameters for the entire sequence (for now we assume we have it)
        Pall - n, 3, 4 projection matrix
        loss_weights - dictionary containing how much value to give each portion
            of the cost function (2d, 3d, temporal)
        num_iters - how many iterations to optimize
        """

        batch_size = 1
        assert batch_size == 1, "Batch size must be 1 for this optimization"

        n_views: int = Pall.shape[0]

        self.num_iters: int = num_iters
        self.hand_side: Literal["left", "right"] = hand_side
        self.beta: Float[Array, "b=1 10"] = npj.array(beta)[npj.newaxis, ...]
        # Projection Matrix (n, 3, 4) where n is the number of cameras
        self.Pall: Float[Array, "n_views 3 4"] = npj.array(Pall)

        self.loss_weights: LossWeights = loss_weights
        # use previous values to initialize, there should only ever be 1
        # hand model per frame
        self.so3_prev: Float[Array, "b=1 48"] = npj.zeros((1, 48))
        self.trans_prev: Float[Array, "b=1 3"] = npj.zeros((1, 3))

        output_fns: tuple[ResidualFn, FwdKinematics, FwdKinematics] = make_mv_scaled_residual()

        residual_fn: ResidualFn = output_fns[0]
        self.mano_fwd_left: FwdKinematics = output_fns[1]
        self.mano_fwd_right: FwdKinematics = output_fns[2]

        # remove the need for two different optimizers, solvers ‘cholesky’, ‘inv’
        self.optimizer = LevenbergMarquardt(
            residual_fun=residual_fn, maxiter=self.num_iters, solver="cholesky", jit=True, xtol=1e-6, gtol=1e-6
        )
        # add jit
        print("Tracing JIT, can take a while...")
        init_params: Float[Array, "b=1 51"] = npj.concatenate([self.so3_prev, self.trans_prev], axis=-1)

        uv_batch_init: Float[Array, "n_frames n_views 21 2"] = npj.zeros((1, n_views, 21, 2))
        _, _ = self.optimizer.run(
            init_params.flatten(),
            beta=self.beta,
            Pall=self.Pall,
            uv_pred=uv_batch_init,
            loss_weights=loss_weights,
            is_left=True,
        )
        self.optimizer = jit(self.optimizer.run)

        print("Trace Done")

    def __call__(
        self,
        uv_pred_batch: Float[ndarray, "n_views 21 2"],
    ) -> tuple[OptimizationResults, LevenbergMarquardtState]:
        """
        pose_predictions_dict
            pose_predictions
            camera_dict
        """
        so3_optimized: Float[ndarray, "b=1 48"] = np.zeros((1, 48))
        trans_optimized: Float[ndarray, "b=1 3"] = np.zeros((1, 3))
        xyz_mano: Float[ndarray, "b=1 21 3"] = np.zeros((1, 21, 3))

        so3_prev: Float[Array, "b=1 48"] = self.so3_prev.copy()
        trans_prev: Float[Array, "b=1 3"] = self.trans_prev.copy()
        uv_pred_batch: Float[Array, "b=1 n_views 21 2"] = npj.array(uv_pred_batch)[npj.newaxis, ...]

        # TODO initialize only rotation from wrist form either 3d procustus or mano preds
        so3_init: Float[Array, "b=1 48"] = so3_prev
        trans_init: Float[Array, "b=1 3"] = trans_prev

        init_params: Float[Array, "b=1 51"] = npj.concatenate([so3_init, trans_init], axis=-1)

        optimized_params, state = self.optimizer(
            init_params.flatten(),
            beta=self.beta,
            Pall=self.Pall,
            uv_pred=uv_pred_batch,
            loss_weights=self.loss_weights,
            is_left=self.hand_side == "left",
        )

        optimized_params: Float[Array, "b=1 51"] = optimized_params.reshape(1, 51)

        so3: Float[Array, "b=1 48"] = optimized_params[:, 0:48]
        trans: Float[Array, "b=1 3"] = optimized_params[:, 48:51]

        so3_optimized = np.array(so3.copy())
        trans_optimized = np.array(trans.copy())

        # pass optimized values to mano to extract 3d joints
        self.so3_prev = so3
        self.trans_prev = trans

        optimization_results = OptimizationResults(
            xyz_mano=xyz_mano,
            so3=so3_optimized,
            trans=trans_optimized,
        )

        return optimization_results, state
