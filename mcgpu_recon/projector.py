"""
System model for MCGPU-PET span=1 sinograms: the projector A (forward) and its
adjoint A^T (back-projection), built on parallelproj's LOW-LEVEL Joseph
projectors (joseph3d_fwd / joseph3d_back). Also: from_run (data + matching
projector in one call) and adjoint_test.

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


def adjoint_test(A, n_trials=3, seed=0):
    """Check that A.adjoint is the adjoint (transpose) of A:
        <A x, y> = <x, A^T y>   for random x >= 0, y >= 0.
    Returns the list of relative errors |<Ax,y> - <x,A^T y>| / |<Ax,y>|.
    Expect ~1e-6 .. 1e-4 in float32. Much larger means every EM step is wrong.
    """
    xp = getattr(A, "xp", np)
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n_trials):
        x = xp.asarray(rng.random(A.in_shape, dtype=np.float32))
        y = xp.asarray(rng.random(A.out_shape, dtype=np.float32))
        lhs = float(xp.sum(A(x) * y, dtype=xp.float64))
        rhs = float(xp.sum(x * A.adjoint(y), dtype=xp.float64))
        errs.append(abs(lhs - rhs) / abs(lhs))
    return errs
