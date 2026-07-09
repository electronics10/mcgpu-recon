"""
3D MLEM reconstruction of MCGPU-PET span=1 sinograms with
parallelproj's LOW-LEVEL Joseph projectors (joseph3d_fwd / joseph3d_back).

Why low-level ("Path A")
------------------------
parallelproj's high-level RegularPolygonPETLORDescriptor enumerates sinogram
bins in ITS OWN (plane, view, radial) order with ITS OWN crystal indexing.
MCGPU-PET bins in (izm; ith, ir) with the kernel's crystal indexing. Feeding an
MCGPU sinogram to a descriptor-built projector silently mismatches almost every
bin (measured and predicted values refer to different physical LORs), which is
why MLEM on the raw MCGPU sinogram produced a near-black image.

The low-level projectors take explicit LOR endpoint coordinates instead. The
wrapper's lors.py already produces exact endpoints for every MCGPU bin (by
exhaustive replay of the kernel's binning arithmetic), so building the system
matrix directly from those endpoints makes the projector bin-order-identical to
the data BY CONSTRUCTION. There is no permutation to discover.

The forward model (mirror-symmetrized)
--------------------------------------
Fact (from the MCGPU-PET kernel, documented in the wrapper): the sinogram
plane labeled (a, b), a != b, holds a ~50/50 mix of the two LOR orientations,
because the kernel assigns the michelogram segment sign from photon tracking
order, which is isotropic. Concretely, for transverse crystal pair (c1, c2)
and ring set {a, b} there are two distinct physical lines:

    line alpha : (c1, ring a) -- (c2, ring b)
    line beta  : (c1, ring b) -- (c2, ring a)

and the expected count in BOTH plane (a, b) and plane (b, a) is
0.5 * (lambda_alpha + lambda_beta), where lambda_* is the line integral of the
activity along that line (times per-line sensitivity, not modeled here).

Model used here, exact under that 50/50 fact:

    (A x)[plane p, bin t] = 0.5 * ( J(x; line_alpha(p, t)) + J(x; line_beta(p, t)) )

where J is the Joseph line integral (mm-weighted). For direct planes (a == a)
alpha == beta and the mean reduces to the single line automatically, so the
formula is uniform over all planes -- no special-casing. The adjoint is the
mean of the two back-projections. Keeping both mirror planes (a,b) and (b,a)
as separate Poisson bins with the same mean is statistically equivalent to
merging them (Poisson additivity), so the data is used as read, unmerged.

Coordinate conventions (the seam where such integrations usually break)
-----------------------------------------------------------------------
parallelproj is convention-agnostic: image axis i, img_origin[i], voxsize[i]
and endpoint coordinate i just have to refer to the same physical axis. We
work in ARRAY-AXIS ORDER (z, y, x), matching the wrapper's (Nz, Ny, Nx)
images:

    voxsize    = (dz, dy, dx)
    img_origin = (-(Nz-1)/2*dz, -(Ny-1)/2*dy, -(Nx-1)/2*dx)   [scanner frame]
    endpoints  = (z, y, x) per LOR

img_origin is the center of voxel [0,0,0] in the scanner-centered frame; the
formula follows from the wrapper's conventions: voxel [k,j,i] center sits at
((i+.5)dx, (j+.5)dy, (k+.5)dz) in the origin-cornered voxel frame, and the
scanner (negative-radius convention) is centered on the bbox center
(Nx*dx/2, Ny*dy/2, Nz*dz/2); subtracting gives the formula above.
lors.transverse_endpoints_mm / ring_z_positions_mm are already scanner-
centered, so no further offset is needed.

Units and normalization: J returns sum_j x_j * (intersection length in mm), y
is in counts, and no solid-angle/efficiency/attenuation model is included by
default, so the reconstruction is correct up to a global scale (and shows mild
attenuation cupping unless attenuation factors are supplied; see mlem()'s
`mult` argument and attenuation_factors()).

Typical use
-----------
    import mcgpu_pet_wrapper as mpw
    from mcgpu_recon import MCGPUProjector, mlem

    cfg = mpw.load_config(run_dir / "config.json")
    y, r1, r2 = mpw.read_sinogram_ring_pairs(run_dir, cfg)   # requires span=1
    A = MCGPUProjector(cfg, r1, r2)                          # xp=numpy default
    x = mlem(A, y.astype("float32"), n_iter=20, verbose=True)

GPU notes: with the conda-forge parallelproj + CUDA, numpy inputs run in
"hybrid" mode (chunks are shipped to the GPU internally; tune num_chunks), or
pass xp=cupy to keep everything on the device. Endpoints are built per plane
chunk, so device memory stays bounded regardless of the total LOR count
(~174M LORs for the default 75-ring config).
"""

from __future__ import annotations

import numpy as np
import parallelproj

from mcgpu_pet_wrapper import lors
from mcgpu_pet_wrapper.config import voxel_space_shape_zyx, grid_size_mm


class MCGPUProjector:
    """Mirror-symmetrized Joseph projector for MCGPU-PET span=1 sinograms.

    Callable = forward (A), .adjoint = exact adjoint (A^T). Bin order of the
    output/input sinograms is exactly that of
    data_reader.read_sinogram_ring_pairs: axis 0 = filled planes in storage
    (izm) order, axes 1..2 = (angular ith, radial ir).

    Parameters
    ----------
    config : dict
        The run config (geometry source of truth).
    ring1, ring2 : int arrays (n_planes,)
        Ring labels per plane, as returned by read_sinogram_ring_pairs.
        Passing them (rather than recomputing) guarantees plane order matches
        the data they came with.
    xp : array namespace, optional
        numpy (default) or array_api_compat.cupy. Determines where images and
        sinograms live; parallelproj dispatches CPU/GPU accordingly.
    plane_chunk : int, optional
        Planes per endpoint-building chunk (memory bound ~
        plane_chunk * nbins * 48 bytes for the two endpoint arrays).
    num_chunks : int, optional
        Forwarded to parallelproj (sub-chunking in hybrid numpy+CUDA mode).
    """

    def __init__(self, config, ring1, ring2, xp=np, plane_chunk=128,
                 num_chunks=1):
        self.xp = xp
        self.num_chunks = int(num_chunks)
        self.plane_chunk = int(plane_chunk)

        # ---- image geometry, array-axis order (z, y, x) ------------------
        nz, ny, nx = voxel_space_shape_zyx(config)
        dx, dy, dz = grid_size_mm(config)
        self.in_shape = (nz, ny, nx)
        self.voxsize = xp.asarray([dz, dy, dx], dtype=xp.float32)
        self.img_origin = xp.asarray(
            [-(nz - 1) / 2.0 * dz, -(ny - 1) / 2.0 * dy, -(nx - 1) / 2.0 * dx],
            dtype=xp.float32,
        )

        # ---- LOR geometry from the kernel-exact inversion ----------------
        xy, hit = lors.transverse_endpoints_mm(config)   # (nang, nrad, 2, 2)
        self._nang, self._nrad = hit.shape
        if not hit.all():
            # General configs may leave unfillable bins; those carry no counts
            # and no LORs. We keep a mask and project only hit bins.
            import warnings
            warnings.warn(f"{(~hit).sum()} transverse bins receive no crystal "
                          "pair; they are excluded from the model.")
        self._hit = hit.ravel()                           # (nang*nrad,)
        # transverse endpoint templates restricted to hit bins, (nhit, {y, x})
        t0 = xy.reshape(-1, 2, 2)[self._hit]              # (nhit, side, {x,y})
        self._t0_yx = xp.asarray(t0[:, 0, ::-1], dtype=xp.float32)  # side ix1
        self._t1_yx = xp.asarray(t0[:, 1, ::-1], dtype=xp.float32)  # side ix2
        self._nhit = int(self._hit.sum())

        zpos = lors.ring_z_positions_mm(config)
        self._z1 = xp.asarray(zpos[np.asarray(ring1)], dtype=xp.float32)
        self._z2 = xp.asarray(zpos[np.asarray(ring2)], dtype=xp.float32)
        self.n_planes = int(len(ring1))
        self.out_shape = (self.n_planes, self._nang, self._nrad)

    # ---- endpoint builders ------------------------------------------------
    def _endpoints(self, sl, orientation):
        """Endpoints for planes[sl]; orientation 'a': side ix1 gets ring1,
        'b': side ix1 gets ring2 (the mirror line). Returns (xs, xe), each
        (P, nhit, 3) float32 in (z, y, x)."""
        xp = self.xp
        z1 = self._z1[sl]
        z2 = self._z2[sl]
        if orientation == "b":
            z1, z2 = z2, z1
        P = z1.shape[0]
        xs = xp.empty((P, self._nhit, 3), dtype=xp.float32)
        xe = xp.empty((P, self._nhit, 3), dtype=xp.float32)
        xs[..., 0] = z1[:, None]
        xs[..., 1:] = self._t0_yx[None, :, :]
        xe[..., 0] = z2[:, None]
        xe[..., 1:] = self._t1_yx[None, :, :]
        return xs, xe

    # ---- linear operator ---------------------------------------------------
    def __call__(self, x):
        """Forward: image (Nz, Ny, Nx) -> sinogram (n_planes, nang, nrad)."""
        xp = self.xp
        x = xp.asarray(x, dtype=xp.float32)
        out = xp.zeros((self.n_planes, self._nang * self._nrad),
                       dtype=xp.float32)
        hit_idx = xp.asarray(np.flatnonzero(self._hit))
        for lo in range(0, self.n_planes, self.plane_chunk):
            sl = slice(lo, min(lo + self.plane_chunk, self.n_planes))
            acc = None
            for o in ("a", "b"):
                xs, xe = self._endpoints(sl, o)
                v = parallelproj.joseph3d_fwd(
                    xs, xe, x, self.img_origin, self.voxsize,
                    num_chunks=self.num_chunks)
                acc = v if acc is None else acc + v
            out[sl, hit_idx] = 0.5 * acc
        return out.reshape(self.out_shape)

    def adjoint(self, y):
        """Adjoint: sinogram (n_planes, nang, nrad) -> image (Nz, Ny, Nx)."""
        xp = self.xp
        y = xp.asarray(y, dtype=xp.float32).reshape(self.n_planes, -1)
        img = xp.zeros(self.in_shape, dtype=xp.float32)
        hit_idx = xp.asarray(np.flatnonzero(self._hit))
        for lo in range(0, self.n_planes, self.plane_chunk):
            sl = slice(lo, min(lo + self.plane_chunk, self.n_planes))
            y_chunk = y[sl, hit_idx]
            for o in ("a", "b"):
                xs, xe = self._endpoints(sl, o)
                img = img + 0.5 * parallelproj.joseph3d_back(
                    xs, xe, self.in_shape, self.img_origin, self.voxsize,
                    y_chunk, num_chunks=self.num_chunks)
        return img


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

    ones = xp.ones(A.out_shape, dtype=xp.float32) if mult is None \
        else xp.asarray(mult, dtype=xp.float32)
    sens = A.adjoint(ones)
    thresh = max(sens_floor_frac, eps) * float(sens.max())
    support = sens > thresh
    sens_safe = xp.where(support, sens, 1.0)

    x = xp.ones(A.in_shape, dtype=xp.float32) if x0 is None \
        else xp.asarray(x0, dtype=xp.float32)
    x = xp.where(support, x, 0.0)

    for k in range(n_iter):
        ybar = A(x)
        if mult is not None:
            ybar = ybar * ones
        if contamination is not None:
            ybar = ybar + contamination
        ratio = y / xp.maximum(ybar, eps)
        if mult is not None:
            ratio = ratio * ones
        x = xp.where(support, x * A.adjoint(ratio) / sens_safe, 0.0)
        if verbose:
            print(f"  MLEM iter {k+1:3d}/{n_iter}  "
                  f"sum(model)={float(ybar.sum()):.6g}  sum(y)={float(y.sum()):.6g}")
        if callback is not None:
            callback(k, x)
    return x


def from_run(run_dir, config, scatter=False, **projector_kwargs):
    """Load a span=1 sinogram and build its matching projector in one step.

    Returns (y, A): y is float32 (n_planes, nang, nrad) in filled-izm order,
    A is an MCGPUProjector whose bin order matches y by construction (ring1/
    ring2 are taken from the same read_sinogram_ring_pairs call).
    """
    from mcgpu_pet_wrapper import read_sinogram_ring_pairs
    y, r1, r2 = read_sinogram_ring_pairs(run_dir, config, scatter=scatter)
    A = MCGPUProjector(config, r1, r2, **projector_kwargs)
    return y.astype(np.float32), A


def attenuation_factors(A, mu_map_per_mm):
    """Per-bin attenuation factors exp(-integral of mu along the LOR), using
    the SAME mirror-symmetrized geometry as A (so factors align with bins).

    mu_map_per_mm : (Nz, Ny, Nx) linear attenuation coefficients in 1/mm at
    511 keV (e.g. water ~ 0.0096/mm). Returns array of shape A.out_shape to
    pass as mlem(..., mult=...).

    Approximation note: the exact factor for a mixed-orientation bin is the
    count-weighted mix of exp(-int_alpha) and exp(-int_beta); we use
    exp(-0.5*(int_alpha+int_beta)), i.e. the geometric mean, consistent with
    the mean-line forward model and exact when the two mirror integrals are
    equal (always true for direct planes).
    """
    xp = getattr(A, "xp", np)
    line_int = A(xp.asarray(mu_map_per_mm, dtype=xp.float32))
    return xp.exp(-line_int)


# ---------------------------------------------------------------------------
# Reconstruction utilities (not metrics): scale-match and mu-map construction.
# These live here, beside the projector/mlem they serve, rather than in
# metrics.py, because neither MEASURES anything -- scale_match resolves MLEM's
# inherent global-scale freedom (a reconstruction concern), and
# attenuation_map_from_vox builds a forward-model input (paired with
# attenuation_factors below it). metrics.py is kept to pure measurements.
# ---------------------------------------------------------------------------

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


def attenuation_map_from_vox(vg, mu_rho):
    """Build a 511-keV linear attenuation map (1/mm) from a VoxelGrid.

    Pairs with attenuation_factors(): this makes the mu-map, that integrates it
    along the LORs. Reading mu straight from the simulation's own voxel grid
    gives an EXACT (oracle) attenuation map for simulation studies -- for real
    data you would instead derive mu from a CT.

    Parameters
    ----------
    vg : mcgpu_pet_wrapper VoxelGrid
        Has integer `material_id` and float `density` arrays, shape (Nz,Ny,Nx).
    mu_rho : dict {material_id: mass attenuation coefficient at 511 keV, cm^2/g}
        At 511 keV Compton scattering dominates, so all soft tissues are close
        to water (~0.096 cm^2/g) and the DENSITY term carries most of the
        variation; a uniform 0.096 is a reasonable first approximation. Use
        per-material values (e.g. NIST XCOM) for your material list to refine.
        Any material_id absent from the dict is left at mu = 0.

    Returns
    -------
    mu_per_mm : (Nz,Ny,Nx) float32, ready for attenuation_factors(A, mu_per_mm).
    """
    mat = np.asarray(vg.material_id)
    rho = np.asarray(vg.density, dtype=np.float32)
    mu_per_cm = np.zeros_like(rho, dtype=np.float32)
    for mid, mrho in mu_rho.items():
        sel = mat == int(mid)
        mu_per_cm[sel] = float(mrho) * rho[sel]      # (cm^2/g)*(g/cm^3) = 1/cm
    return (mu_per_cm / 10.0).astype(np.float32)     # 1/cm -> 1/mm (mm geometry)