"""
MLEM (Shepp & Vardi 1982), plus the pieces it shares with map_em:
the sensitivity image / reconstruction support, and the forward model
ybar = mult * (A x) + contamination.
"""

from __future__ import annotations

import numpy as np


def sensitivity_and_support(A, mult=None, sens_floor_frac=0.025, eps=1e-8):
    """Sensitivity s = A^T(mult) (mult=None means 1) and the support mask.

    Voxels with s_j <= max(sens_floor_frac, eps) * max(s) are outside the
    support: MLEM divides by s_j, so tiny s_j (FOV edge/corners) would amplify
    noise into hot pixels. Returns (mult_arr, sens, support, sens_safe), where
    mult_arr is mult as an xp float32 array (ones if mult is None) and
    sens_safe = s on the support, 1 elsewhere (safe to divide by).
    """
    xp = getattr(A, "xp", np)
    mult_arr = xp.ones(A.out_shape, dtype=xp.float32) if mult is None \
        else xp.asarray(mult, dtype=xp.float32)
    sens = A.adjoint(mult_arr)
    thresh = max(sens_floor_frac, eps) * float(sens.max())
    support = sens > thresh
    sens_safe = xp.where(support, sens, 1.0)
    return mult_arr, sens, support, sens_safe


def expected_counts(A, x, mult=None, contamination=None):
    """Forward model ybar = mult * (A x) + contamination (None = 1 / 0)."""
    ybar = A(x)
    if mult is not None:
        ybar = ybar * mult
    if contamination is not None:
        ybar = ybar + contamination
    return ybar


def em_backprojection(A, x, y, ybar, mult=None, eps=1e-8):
    """c = x * A^T(mult * y / ybar). The MLEM update is c / s; MAP-EM uses c too."""
    xp = getattr(A, "xp", np)
    ratio = y / xp.maximum(ybar, eps)
    if mult is not None:
        ratio = ratio * mult
    return x * A.adjoint(ratio)


def mlem(A, y, n_iter=20, x0=None, mult=None, contamination=None,
         sens_floor_frac=0.025, eps=1e-8, verbose=False, callback=None):
    """Maximum-Likelihood Expectation Maximization (Shepp & Vardi 1982).

    Model:  ybar = mult * (A x) + contamination,   y ~ Poisson(ybar)

    Update: x <- x / sens * A^T( mult * y / ybar ),  sens = A^T(mult)

    Standard properties (theorems for the exact Poisson model):
      * each update does not decrease the Poisson log-likelihood;
      * count matching after every full update:
            sum(sens * x_k) = sum(y * Ax/(Ax + contam-part))  and with
            contamination == 0 exactly  sum(mult * A x_k) = sum(y)  for k >= 1;
      * convergence from below in contrast: bulk intensity appears in the
        first iterations, edges/peaks keep sharpening for tens of iterations
        (why peak values grow with n_iter even though totals are matched).

    Parameters
    ----------
    A : linear operator with __call__ (forward) and .adjoint.
    y : measured sinogram, shape A.out_shape, non-negative.
    mult : optional multiplicative factors, same shape as y (attenuation and/or
        normalization). None means 1.
    contamination : optional additive expectation, same shape as y (scatter
        and/or randoms estimate). None means 0. NOTE: with a contamination
        term, reconstruct the TOTAL (trues+scatter) sinogram against it; or
        reconstruct trues-only with contamination=None.
    sens_floor_frac : float, optional
        Voxels whose sensitivity s_j = A^T(mult) is below
        sens_floor_frac * max(s) are EXCLUDED from the support (held at 0).
        Rationale: the MLEM update divides by s_j, so FOV-edge/corner voxels
        with tiny s_j amplify backprojected noise into "hot pixels". Flooring
        the support removes the cause (default 2.5% of peak sensitivity). Set to
        0.0 to disable (recovers the old permissive behavior).
    callback : optional f(k, x) per iteration.
    """
    xp = getattr(A, "xp", np)
    y = xp.asarray(y, dtype=xp.float32)

    mult_arr, sens, support, sens_safe = sensitivity_and_support(
        A, mult, sens_floor_frac, eps)
    m = None if mult is None else mult_arr

    x = xp.ones(A.in_shape, dtype=xp.float32) if x0 is None \
        else xp.asarray(x0, dtype=xp.float32)
    x = xp.where(support, x, 0.0)

    for k in range(n_iter):
        ybar = expected_counts(A, x, m, contamination)
        x = xp.where(support, em_backprojection(A, x, y, ybar, m, eps) / sens_safe, 0.0)
        if verbose:
            print(f"  MLEM iter {k+1:3d}/{n_iter}  "
                  f"sum(model)={float(ybar.sum()):.6g}  sum(y)={float(y.sum()):.6g}")
        if callback is not None:
            callback(k, x)
    return x
