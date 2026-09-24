"""
Attenuation correction: build a 511-keV mu-map and turn it into per-bin
attenuation factors exp(-integral of mu along the LOR), to pass to
mlem / map_em as `mult`.
"""

from __future__ import annotations

import numpy as np


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
