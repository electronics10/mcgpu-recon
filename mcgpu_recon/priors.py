"""
Neighbourhood penalties (priors) for MAP / penalized-likelihood reconstruction.

Notation (same as WORKING_PLAN_year2_v2.md, section 3)
-------------------------------------------------------
Image x has shape (Nz, Ny, Nx) (array-axis order, like the rest of mcgpu_recon).
Neighbours: the 26 voxels of the 3x3x3 cube around a voxel.

A *pair* {j, k} of neighbouring voxels carries one symmetric weight omega_jk >= 0.
We store the 13 "positive" offsets o only (the other 13 are their negatives), so
every unordered pair is stored exactly once:

    W[o][j] = omega between voxel j and voxel j+o     (0 if j+o is outside)

Penalties, summed over unordered pairs:

    quadratic : R(x) = 1/2 * sum_pairs omega_jk (x_j - x_k)^2
    RDP       : R(x) =       sum_pairs omega_jk (x_j - x_k)^2
                                   / (x_j + x_k + gamma |x_j - x_k|)

With this convention the quadratic gradient is the familiar
    dR/dx_j = sum_{k in N_j} omega_jk (x_j - x_k).

Weights
-------
uniform_weights : omega = 1 for every pair (optionally 1/distance).
bowsher_weights : MR-guided. Each voxel j picks the B neighbours whose MR value is
    closest to its own (a *directed* choice w_jk in {0,1}); the pair weight is the
    average omega_jk = (w_jk + w_kj)/2, in {0, 1/2, 1}. Averaging makes the
    weights symmetric, so R is a proper penalty, and uniform weights are the
    special case B = 26 (omega = 1 everywhere).

Arrays may be numpy or cupy (array_api_compat); the namespace `xp` is passed in.
"""

from __future__ import annotations

import itertools

import numpy as np


# ---------------------------------------------------------------------------
# Neighbourhood geometry
# ---------------------------------------------------------------------------

def _all_offsets():
    """The 26 offsets (dz, dy, dx) of the 3x3x3 neighbourhood."""
    return [o for o in itertools.product((-1, 0, 1), repeat=3) if o != (0, 0, 0)]


def _positive_offsets():
    """13 offsets, one from each +/- pair (lexicographically positive)."""
    return [o for o in _all_offsets() if o > (0, 0, 0)]


def _pair_slices(o, shape):
    """Slices (sl_j, sl_k) such that x[sl_j] are the voxels j whose neighbour
    j+o lies inside the grid, and x[sl_k] are those neighbours (same order)."""
    sl_j, sl_k = [], []
    for d, n in zip(o, shape):
        if d > 0:
            sl_j.append(slice(0, n - d)); sl_k.append(slice(d, n))
        elif d < 0:
            sl_j.append(slice(-d, n));    sl_k.append(slice(0, n + d))
        else:
            sl_j.append(slice(0, n));     sl_k.append(slice(0, n))
    return tuple(sl_j), tuple(sl_k)


def _distance(o, voxsize):
    return float(np.sqrt(sum((d * s) ** 2 for d, s in zip(o, voxsize))))


class NeighbourWeights:
    """Symmetric pair weights on a 3x3x3 neighbourhood.

    Attributes
    ----------
    shape   : image shape (Nz, Ny, Nx)
    offsets : list of the 13 positive offsets
    W       : list of 13 arrays of image shape; W[i][j] = omega(j, j+offsets[i])
    xp      : array namespace
    """

    def __init__(self, shape, offsets, W, xp=np):
        self.shape = tuple(shape)
        self.offsets = list(offsets)
        self.W = list(W)
        self.xp = xp

    def restrict(self, support):
        """Zero every pair that touches a voxel outside `support` (bool image).
        Used by map_em so the penalty does not pull edge voxels toward the
        (forced) zeros outside the reconstruction support."""
        xp = self.xp
        sup = xp.asarray(support)
        W_new = []
        for o, w in zip(self.offsets, self.W):
            sl_j, sl_k = _pair_slices(o, self.shape)
            w2 = xp.zeros_like(w)
            w2[sl_j] = w[sl_j] * (sup[sl_j] & sup[sl_k])
            W_new.append(w2)
        return NeighbourWeights(self.shape, self.offsets, W_new, xp)

    def total(self):
        """Omega_j = sum_k omega_jk (per voxel)."""
        xp = self.xp
        tot = xp.zeros(self.shape, dtype=xp.float32)
        for o, w in zip(self.offsets, self.W):
            sl_j, sl_k = _pair_slices(o, self.shape)
            tot[sl_j] += w[sl_j]
            tot[sl_k] += w[sl_j]
        return tot


def uniform_weights(shape, xp=np, voxsize=(1.0, 1.0, 1.0), distance_weighted=True):
    """omega_jk = 1 for all 26 neighbours (or d_min/d_jk if distance_weighted).

    voxsize is (dz, dy, dx) in mm; only matters when distance_weighted=True.
    """
    offs = _positive_offsets()
    dmin = min(_distance(o, voxsize) for o in offs)
    W = []
    for o in offs:
        val = dmin / _distance(o, voxsize) if distance_weighted else 1.0
        w = xp.zeros(tuple(shape), dtype=xp.float32)
        sl_j, _ = _pair_slices(o, shape)
        w[sl_j] = val
        W.append(w)
    return NeighbourWeights(shape, offs, W, xp)


def bowsher_weights(mr, B, xp=np, voxsize=(1.0, 1.0, 1.0),
                    distance_weighted=True, seed=0):
    """MR-guided (Bowsher) weights.

    For each voxel j, choose the B neighbours (out of up to 26 inside the grid)
    with the smallest |m_j - m_k|: w_jk = 1 for those, 0 otherwise. Pair weight
    omega_jk = (w_jk + w_kj)/2, times d_min/d_jk if distance_weighted.

    Ties. In a perfectly flat MR region all neighbours tie, and picking "the
    first B in array order" would always pick the same directions -> smoothing
    that is stronger along some axes (an artefact). Ties are therefore broken
    first by distance (closer first), then randomly (fixed `seed`). This changes
    nothing where MR values really differ.

    mr : (Nz, Ny, Nx) MR image on the PET voxel grid (numpy or cupy).
    B  : number of neighbours kept per voxel, 1..26.
    Computed on the CPU with numpy (once per reconstruction), then moved to xp.
    """
    m = np.asarray(mr.get() if hasattr(mr, "get") else mr, dtype=np.float64)
    shape = m.shape
    if not 1 <= B <= 26:
        raise ValueError("B must be in 1..26")
    all_offs = _all_offsets()
    dists = np.array([_distance(o, voxsize) for o in all_offs])
    rng = np.random.default_rng(seed)

    # key[n, j] = |m_j - m_{j+o_n}|, +inf where the neighbour is outside the grid
    key = np.full((26,) + shape, np.inf)
    for n, o in enumerate(all_offs):
        sl_j, sl_k = _pair_slices(o, shape)
        key[n][sl_j] = np.abs(m[sl_j] - m[sl_k])
    # tie-breaks: tiny compared with any real MR difference
    scale = max(float(np.nanmax(m) - np.nanmin(m)), 1e-30)
    tie = 1e-9 * scale
    dist_rank = np.argsort(np.argsort(dists, kind="stable"), kind="stable")
    key += tie * (dist_rank[:, None, None, None] / 26.0
                  + 0.01 * rng.random((26,) + shape))

    chosen = np.argpartition(key, B - 1, axis=0)[:B]          # (B, Nz, Ny, Nx)
    w_dir = np.zeros((26,) + shape, dtype=bool)                 # w_dir[n, j] = w_{j, j+o_n}
    np.put_along_axis(w_dir, chosen, True, axis=0)
    w_dir &= np.isfinite(key)                                   # never pick outside

    index = {o: n for n, o in enumerate(all_offs)}
    offs = _positive_offsets()
    dmin = min(_distance(o, voxsize) for o in offs)
    W = []
    for o in offs:
        n_pos = index[o]
        n_neg = index[tuple(-d for d in o)]
        sl_j, sl_k = _pair_slices(o, shape)
        w = np.zeros(shape, dtype=np.float32)
        # omega(j, j+o) = (w_{j -> j+o} + w_{j+o -> j}) / 2
        w[sl_j] = 0.5 * (w_dir[n_pos][sl_j].astype(np.float32)
                         + w_dir[n_neg][sl_k].astype(np.float32))
        if distance_weighted:
            w *= dmin / _distance(o, voxsize)
        W.append(xp.asarray(w))
    return NeighbourWeights(shape, offs, W, xp)


# ---------------------------------------------------------------------------
# Penalties
# ---------------------------------------------------------------------------

class QuadraticPrior:
    """R(x) = 1/2 sum_pairs omega_jk (x_j - x_k)^2.

    quadratic = True tells map_em it can use De Pierro's closed-form update.
    """
    quadratic = True

    def __init__(self, weights: NeighbourWeights):
        self.weights = weights

    def restrict(self, support):
        return QuadraticPrior(self.weights.restrict(support))

    def value(self, x):
        xp = self.weights.xp
        v = 0.0
        for o, w in zip(self.weights.offsets, self.weights.W):
            sl_j, sl_k = _pair_slices(o, self.weights.shape)
            d = x[sl_j] - x[sl_k]
            v += 0.5 * float(xp.sum(w[sl_j] * d * d))
        return v

    def gradient(self, x):
        xp = self.weights.xp
        g = xp.zeros(self.weights.shape, dtype=xp.float32)
        for o, w in zip(self.weights.offsets, self.weights.W):
            sl_j, sl_k = _pair_slices(o, self.weights.shape)
            t = w[sl_j] * (x[sl_j] - x[sl_k])
            g[sl_j] += t
            g[sl_k] -= t
        return g

    def depierro_terms(self, x):
        """Per voxel: Omega_j = sum_k omega_jk and S_j = sum_k omega_jk (x_j + x_k),
        the two sums needed by De Pierro's update (see map_em)."""
        xp = self.weights.xp
        Om = xp.zeros(self.weights.shape, dtype=xp.float32)
        S = xp.zeros(self.weights.shape, dtype=xp.float32)
        for o, w in zip(self.weights.offsets, self.weights.W):
            sl_j, sl_k = _pair_slices(o, self.weights.shape)
            wj = w[sl_j]
            t = wj * (x[sl_j] + x[sl_k])
            Om[sl_j] += wj; Om[sl_k] += wj
            S[sl_j] += t;   S[sl_k] += t
        return Om, S


class RDPrior:
    """Relative difference prior (Nuyts et al. 2002):

        R(x) = sum_pairs omega_jk (x_j - x_k)^2 / (x_j + x_k + gamma |x_j - x_k|)

    gamma >= 0 controls edge preservation (larger gamma = edges kept more).
    gamma = 2 is a common choice. Not quadratic, so map_em uses OSL for it.
    Defined for x >= 0; a pair with x_j = x_k = 0 contributes 0.
    """
    quadratic = False

    def __init__(self, weights: NeighbourWeights, gamma=2.0):
        self.weights = weights
        self.gamma = float(gamma)

    def restrict(self, support):
        return RDPrior(self.weights.restrict(support), self.gamma)

    def _pair_terms(self, x, o, w):
        xp = self.weights.xp
        sl_j, sl_k = _pair_slices(o, self.weights.shape)
        xj, xk = x[sl_j], x[sl_k]
        d = xj - xk
        ad = xp.abs(d)
        D = xj + xk + self.gamma * ad
        pos = D > 0
        Ds = xp.where(pos, D, 1.0)
        return sl_j, sl_k, w[sl_j], d, ad, Ds, pos

    def value(self, x):
        xp = self.weights.xp
        v = 0.0
        for o, w in zip(self.weights.offsets, self.weights.W):
            _, _, wj, d, _, Ds, pos = self._pair_terms(x, o, w)
            v += float(xp.sum(xp.where(pos, wj * d * (d / Ds), 0.0)))
        return v

    def gradient(self, x):
        """dR/dx_j. For one pair with d = x_j - x_k, D = x_j + x_k + gamma|d|:
            d/dx_j = d (x_j + 3 x_k + gamma|d|) / D^2
            d/dx_k = -d (3 x_j + x_k + gamma|d|) / D^2
        """
        xp = self.weights.xp
        g = xp.zeros(self.weights.shape, dtype=xp.float32)
        for o, w in zip(self.weights.offsets, self.weights.W):
            sl_j, sl_k, wj, d, ad, Ds, pos = self._pair_terms(x, o, w)
            xj, xk = x[sl_j], x[sl_k]
            # written as (d/D) * (.../D): both ratios are bounded for x >= 0
            # (|d| <= D), so nothing under/overflows even for x ~ 1e-30.
            # (Dividing by D*D would underflow to 0 in float32 -> 0/0 = NaN.)
            r = d / Ds
            gj = xp.where(pos, wj * r * ((xj + 3 * xk + self.gamma * ad) / Ds), 0.0)
            gk = xp.where(pos, -wj * r * ((3 * xj + xk + self.gamma * ad) / Ds), 0.0)
            g[sl_j] += gj
            g[sl_k] += gk
        return g


def beta_scale(prior, sens, x_ref, mask=None):
    """A reference value beta0 so that you can sweep a DIMENSIONLESS beta~:
    beta = beta~ * beta0 (e.g. beta~ in 0.01 ... 10, log-spaced).

    Heuristic [judgment]: compare the curvature of the two terms of the
    objective at a typical voxel of the object.
      * log-likelihood curvature ~ s / x        (s = sensitivity, x = activity)
      * quadratic penalty curvature ~ Omega     (sum of pair weights)
      * RDP curvature near d = 0     ~ Omega / x
    beta0 makes the penalty curvature equal to the likelihood curvature:
      quadratic: beta0 = mean(s) / (mean(x) * mean(Omega))
      RDP      : beta0 = mean(s) / mean(Omega)
    so beta~ = 1 means "prior and data pull about equally hard".
    Means are over `mask` (e.g. x_ref > 10% of max); x_ref is e.g. an MLEM image.
    Same beta~ does NOT mean same noise for different priors: compare methods
    on contrast-vs-noise curves.
    """
    xp = prior.weights.xp
    sens = xp.asarray(sens)
    x_ref = xp.asarray(x_ref)
    if mask is None:
        mask = x_ref > 0.1 * float(x_ref.max())
    Om = prior.weights.total()
    s_m = float(xp.mean(sens[mask]))
    x_m = float(xp.mean(x_ref[mask]))
    om_m = float(xp.mean(Om[mask]))
    if prior.quadratic:
        return s_m / (x_m * om_m)
    return s_m / om_m
