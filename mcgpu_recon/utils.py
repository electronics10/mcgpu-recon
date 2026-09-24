"""
Small helpers that are not part of the reconstruction algorithms:
scale_match (resolve MLEM's global-scale freedom before comparing images)
and binomial_thin (make low-dose data from full-count data).
"""

from __future__ import annotations

import numpy as np


def _namespace(*arrays):
    """Return the array-API namespace (numpy or cupy) of the given arrays."""
    try:
        import array_api_compat
        return array_api_compat.array_namespace(*arrays)
    except Exception:
        return np


def scale_match(x_ref, x, mask_frac=0.05):
    """Global least-squares scale c minimizing ||x_ref - c*x||, returning (c*x, c).

    MLEM reconstructions are correct only up to a global constant (no
    solid-angle/efficiency model), so two reconstructions of differently-scaled
    data (e.g. trues vs trues+scatter) sit at different levels even when their
    SHAPE agrees. Matching that one constant before differencing isolates shape
    error from a benign level offset.

    c is fit over voxels brighter than mask_frac*max(x_ref) ONLY, so the vast
    near-zero background (and any FOV-edge hot pixels) cannot drag the fit -- the
    scale is set where the signal is. Works with numpy or cupy arrays.

    Call it before handing images to metrics.evaluate_recon; the metric then 
    measures whatever it is given, with no hidden rescaling.
    """
    xp = _namespace(x_ref, x)
    m = x_ref > mask_frac * float(x_ref.max())
    c = float(xp.sum((x_ref * x)[m]) / xp.sum((x * x)[m]))
    return c * x, c



def binomial_thin(y, p, seed=None):
    """Low-dose data by binomial thinning: keep each count with probability p.
    If y ~ Poisson(lam), the result is exactly Poisson(p * lam).
    y must hold integer counts (e.g. from read_sinogram_ring_pairs). Works on
    the CPU; returns float32 numpy. Use a different seed per noise realization.
    """
    rng = np.random.default_rng(seed)
    y_int = np.asarray(y.get() if hasattr(y, "get") else y)
    if not np.allclose(y_int, np.round(y_int)):
        raise ValueError("binomial_thin needs integer counts")
    return rng.binomial(np.round(y_int).astype(np.int64), p).astype(np.float32)
