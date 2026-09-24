"""
Quick PI demo: MLEM vs MAP-EM (uniform quadratic, MR-guided Bowsher, RDP)
on a low-dose NEMA NU 4-2008 image-quality phantom simulated with MCGPU-PET.

What it shows
-------------
The NEMA IQ phantom has 5 hot rods (1-5 mm), a uniform region and 2 cold
inserts (water, air). We make an MR "surrogate" image from the phantom's own
materials, and deliberately erase ONE rod from it (default: the 4 mm rod). So:

  * rods 3 mm and 5 mm are MR-VISIBLE    -> Bowsher should help them
  * rod 4 mm is MR-INVISIBLE             -> Bowsher may smooth it away
  * the cold WATER insert looks exactly like the hot water around it in MR
    (same material), so it is also MR-invisible -> watch its spill-over ratio

Each method has one knob (Gaussian filter width for MLEM, beta for MAP). We
sweep it and plot contrast (rod recovery coefficient, RC) against noise
(%STD in the uniform region), so methods are compared at EQUAL NOISE.

Metrics follow NEMA NU 4-2008 in simplified form (details: rc_rod etc. below).
Noise is measured in ONE noise realization (NEMA practice); the full study will
use ensemble noise over several realizations.

Prerequisite: a span=1 NEMA IQ run, e.g. as in the README:
    cfg = mpw.default_config(); cfg["sinogram"]["span"] = 1
    vg  = mpw.nema_iq_preclinical(cfg, hot_activity_Bq_per_mL=20000)
    mpw.build_run(run_dir, cfg, vg); mpw.Runner()(run_dir, "overwrite")

Run (on the GPU machine, from the repo root):
    pixi run python examples/demo_map_em.py data/run_0 --dose 0.1

Runtime ~ (n_beta * 3 + 1) * n_iter_map * 2.6 s for the default geometry:
the defaults (5 betas, 60 iterations, warm start from MLEM) take ~40 min.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

import mcgpu_pet_wrapper as mpw
from mcgpu_pet_wrapper import phantoms as _ph
from mcgpu_recon import (from_run, mlem, map_em, attenuation_factors,
                         attenuation_map_from_vox, binomial_thin,
                         uniform_weights, bowsher_weights, QuadraticPrior,
                         RDPrior, beta_scale)

# Mass attenuation coefficients at 511 keV (cm^2/g) per material id; same as README.
MU_RHO = {1: 0.087, 2: 0.096, 3: 0.094, 4: 0.093}

METHODS = ["MLEM + Gaussian", "MAP quadratic (uniform)",
           "MAP quadratic (Bowsher, MR)", "MAP RDP"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed categorical order
MARKERS = ["o", "s", "^", "D"]


# ---------------------------------------------------------------------------
# NEMA IQ geometry (voxel indices), from the wrapper's own phantom constants
# ---------------------------------------------------------------------------

class IQGeometry:
    def __init__(self, cfg):
        nz, ny, nx = mpw.voxel_space_shape_zyx(cfg)
        dx, dy, dz = mpw.grid_size_mm(cfg)
        self.shape, self.d = (nz, ny, nx), (dz, dy, dx)
        cx, cy, cz = nx * dx / 2, ny * dy / 2, nz * dz / 2
        z0 = cz - _ph._IQ_BODY_LENGTH / 2
        # voxel-centre coordinates in mm (origin at the bbox corner, as the phantom)
        self.zc = (np.arange(nz) + 0.5) * dz
        self.yc = (np.arange(ny) + 0.5) * dy
        self.xc = (np.arange(nx) + 0.5) * dx
        self.cx, self.cy = cx, cy
        self.rod_z = (z0, z0 + _ph._IQ_ROD_REGION_LENGTH)
        self.uni_z = (z0 + _ph._IQ_ROD_REGION_LENGTH,
                      z0 + _ph._IQ_BODY_LENGTH - _ph._IQ_COLD_INSERT_LENGTH)
        self.cold_z = (z0 + _ph._IQ_BODY_LENGTH - _ph._IQ_COLD_INSERT_LENGTH,
                       z0 + _ph._IQ_BODY_LENGTH)
        n = len(_ph._IQ_ROD_DIAMETERS)
        self.rods = [(cx + _ph._IQ_ROD_PITCH * math.cos(2 * math.pi * k / n),
                      cy + _ph._IQ_ROD_PITCH * math.sin(2 * math.pi * k / n), dia)
                     for k, dia in enumerate(_ph._IQ_ROD_DIAMETERS)]
        h = _ph._IQ_COLD_INSERT_SEPARATION / 2
        self.cold = {"water": (cx + h, cy), "air": (cx - h, cy)}

    def zslab(self, zrange, length):
        """Boolean over z: a slab of `length` mm centred in zrange."""
        zm = 0.5 * (zrange[0] + zrange[1])
        return np.abs(self.zc - zm) <= length / 2

    def disk(self, x0, y0, r):
        Y, X = np.meshgrid(self.yc, self.xc, indexing="ij")
        return (X - x0) ** 2 + (Y - y0) ** 2 <= r ** 2

    def cylinder(self, x0, y0, r, zrange, length):
        return self.zslab(zrange, length)[:, None, None] & self.disk(x0, y0, r)[None]


def nema_metrics(img, g: IQGeometry):
    """Simplified NEMA NU 4-2008 IQ metrics.
    uniform: 22.5 mm diameter x 10 mm cylinder -> mean, %STD
    RC (rod d): average the central 10 mm of the rod region over z; in a disk of
        diameter 2d around the rod take the MAX; RC = max / uniform mean.
    SOR (cold insert): mean in a 4 mm diameter x 7.5 mm cylinder / uniform mean.
    """
    uni = g.cylinder(g.cx, g.cy, 22.5 / 2, g.uni_z, 10.0)
    mu = float(img[uni].mean())
    out = {"uniform_mean": mu, "pct_std": 100 * float(img[uni].std()) / mu}
    rod_avg = img[g.zslab(g.rod_z, 10.0)].mean(axis=0)
    for (x0, y0, dia) in g.rods:
        out[f"RC_{dia:g}mm"] = float(rod_avg[g.disk(x0, y0, dia)].max()) / mu
    for name, (x0, y0) in g.cold.items():
        cyl = g.cylinder(x0, y0, 2.0, g.cold_z, 7.5)
        out[f"SOR_{name}"] = float(img[cyl].mean()) / mu
    return out


# ---------------------------------------------------------------------------
# MR surrogate
# ---------------------------------------------------------------------------

def mr_surrogate(vg, g: IQGeometry, invisible_rods=(4.0,), snr=30.0, seed=0):
    """T1-weighted-like MR image from the phantom materials.
    water = 1.0, PMMA = 0.35, air = 0. The listed rods are painted as PMMA
    (invisible in MR, still hot in PET). Mild blur + Rician noise
    (|signal + complex Gaussian noise|, the noise of MR magnitude images)."""
    rng = np.random.default_rng(seed)
    rho = np.asarray(vg.density)
    m = np.zeros(g.shape, np.float32)
    m[rho > 0.5] = 1.0                               # water
    m[np.isclose(rho, 1.19, atol=0.02)] = 0.35       # PMMA (water id at 1.19 g/cc)
    zs = g.zslab(g.rod_z, g.rod_z[1] - g.rod_z[0])
    for (x0, y0, dia) in g.rods:
        if any(abs(dia - d) < 1e-6 for d in invisible_rods):
            cyl = zs[:, None, None] & g.disk(x0, y0, dia / 2 + 0.5)[None]
            m[cyl & (m > 0.5)] = 0.35
    m = gaussian_filter(m, 0.5)
    s = 1.0 / snr
    return np.sqrt((m + s * rng.standard_normal(m.shape)) ** 2
                   + (s * rng.standard_normal(m.shape)) ** 2).astype(np.float32)


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------

def run_demo(y_t, y_s, A, vg, cfg, out_dir, dose=0.1, n_iter_mlem=23,
             fwhms_mm=(0, 1, 2, 3, 4), betas_t=(0.01, 0.03, 0.1, 0.3, 1.0),
             n_iter_map=60, bowsher_B=6, rdp_gamma=2.0,
             invisible_rods=(4.0,), seed=0, xp=np):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    g = IQGeometry(cfg)
    asnp = (lambda a: a.get() if hasattr(a, "get") else np.asarray(a))

    # ---- low-dose data (binomial thinning, trues and scatter separately) --
    y_low = xp.asarray(binomial_thin(y_t, dose, seed=seed)
                       + binomial_thin(y_s, dose, seed=seed + 1))
    contam = xp.asarray(dose * np.asarray(asnp(y_s), np.float32))  # oracle scatter
    af = attenuation_factors(A, xp.asarray(attenuation_map_from_vox(vg, MU_RHO)))
    print(f"dose {dose}: {float(y_low.sum()):.4g} prompts "
          f"(scatter fraction {float(contam.sum() / y_low.sum()):.1%})")

    mr = mr_surrogate(vg, g, invisible_rods=invisible_rods, seed=seed)
    rec = {m: [] for m in METHODS}          # per method: list of (setting, metrics)
    images = {m: {} for m in METHODS}

    # ---- row A: MLEM + Gaussian post-filter --------------------------------
    x_ml = mlem(A, y_low, n_iter=n_iter_mlem, mult=af, contamination=contam)
    x_ml_np = asnp(x_ml)
    dz, dy, dx = g.d
    for f in fwhms_mm:
        img = x_ml_np if f == 0 else gaussian_filter(
            x_ml_np, [f / 2.3548 / s for s in (dz, dy, dx)])
        rec[METHODS[0]].append((f, nema_metrics(img, g))); images[METHODS[0]][f] = img

    # ---- MAP rows ---------------------------------------------------------
    sens = A.adjoint(af)
    shp = tuple(x_ml.shape)
    priors = {
        METHODS[1]: QuadraticPrior(uniform_weights(shp, xp=xp, voxsize=g.d)),
        METHODS[2]: QuadraticPrior(bowsher_weights(mr, bowsher_B, xp=xp, voxsize=g.d)),
        METHODS[3]: RDPrior(uniform_weights(shp, xp=xp, voxsize=g.d), gamma=rdp_gamma),
    }
    for name, prior in priors.items():
        b0 = beta_scale(prior, sens, x_ml)
        for bt in betas_t:
            print(f"{name}: beta~ = {bt} (beta = {bt * b0:.3g})")
            x = map_em(A, y_low, prior=prior, beta=bt * b0, n_iter=n_iter_map,
                       x0=x_ml, mult=af, contamination=contam)
            img = asnp(x)
            rec[name].append((bt, nema_metrics(img, g))); images[name][bt] = img

    # ---- save numbers -----------------------------------------------------
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump({k: [{"setting": s, **m} for s, m in v] for k, v in rec.items()},
                  fh, indent=1)
    plot_curves(rec, out_dir / "contrast_vs_noise.png", dose, invisible_rods)
    plot_images(rec, images, mr, g, out_dir / "images.png", dose)
    print("saved to", out_dir)
    return rec


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_curves(rec, path, dose, invisible_rods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("RC_3mm", "3 mm rod (MR-visible)"),
              (f"RC_{invisible_rods[0]:g}mm", f"{invisible_rods[0]:g} mm rod (MR-invisible)"),
              ("RC_5mm", "5 mm rod (MR-visible)"),
              ("SOR_water", "Cold water insert (MR-invisible), lower = better")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.3))
    for ax, (key, title) in zip(axes, panels):
        for name, c, mk in zip(METHODS, COLORS, MARKERS):
            # connect points in KNOB order (not sorted by noise)
            pts = [(m["pct_std"], m[key]) for _, m in rec[name]]
            ax.plot(*zip(*pts), color=c, marker=mk, markersize=6, linewidth=2,
                    label=name)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("noise: %STD in uniform region")
        ax.set_ylabel("spill-over ratio" if key.startswith("SOR") else "recovery coefficient")
        _style(ax)
    axes[0].legend(frameon=False, fontsize=9)
    fig.text(0.01, 0.005, "Points along a curve: MLEM = 23 it + Gaussian FWHM 0..4 mm; "
             "MAP = increasing beta. A curve that folds back = heavy smoothing "
             "spreads the phantom edge into the uniform ROI (or not converged).",
             fontsize=8, color="#555555")
    fig.suptitle(f"Contrast vs noise at {dose:.0%} dose "
                 "(each curve: one method, its knob swept)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path, "saved")


def plot_images(rec, images, mr, g, path, dose):
    """One image per method at (approximately) EQUAL noise: the setting whose
    %STD is closest to that of MLEM + 2 mm Gaussian (or the middle MLEM entry)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ref = dict(rec[METHODS[0]]).get(2, rec[METHODS[0]][len(rec[METHODS[0]]) // 2][1])
    target = ref["pct_std"]
    zr = g.zslab(g.rod_z, 10.0); zc = g.zslab(g.cold_z, 7.5)
    cols = [("MR surrogate", mr, None)]
    for name in METHODS:
        s, m = min(rec[name], key=lambda sm: abs(sm[1]["pct_std"] - target))
        cols.append((f"{name}\n(setting {s:g}, %STD {m['pct_std']:.1f})",
                     images[name][s], m["uniform_mean"]))
    # crop to the phantom (+ margin): 40 mm x 40 mm around the centre
    iy = np.abs(g.yc - g.cy) <= 20; ix = np.abs(g.xc - g.cx) <= 20
    fig, axes = plt.subplots(2, len(cols), figsize=(3.2 * len(cols), 6.6))
    for j, (title, img, mu) in enumerate(cols):
        for i, zsel in enumerate((zr, zc)):
            sl = img[zsel].mean(axis=0)[iy][:, ix]
            vmax = 1.3 * (mu if mu else float(sl.max()))
            axes[i, j].imshow(sl, cmap="gray", origin="lower", vmin=0, vmax=vmax)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
        axes[0, j].set_title(title, fontsize=9)
    axes[0, 0].set_ylabel("rod region"); axes[1, 0].set_ylabel("cold inserts")
    fig.suptitle(f"{dose:.0%} dose, methods shown at about equal noise "
                 f"(%STD ~ {target:.1f})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path, "saved")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--dose", type=float, default=0.1, help="kept fraction of counts")
    ap.add_argument("--betas", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3, 1.0],
                    help="dimensionless beta~ values (beta = beta~ * beta_scale)")
    ap.add_argument("--n-iter-map", type=int, default=60)
    ap.add_argument("--bowsher-B", type=int, default=6)
    ap.add_argument("--cpu", action="store_true", help="numpy instead of cupy")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.cpu:
        xp = np
    else:
        import array_api_compat.cupy as xp
    cfg = mpw.load_config(args.run_dir / "config.json")
    y_t, A = from_run(args.run_dir, cfg, xp=xp, plane_chunk=256)
    y_s, _ = from_run(args.run_dir, cfg, scatter=True, xp=xp, plane_chunk=256)
    vg = mpw.read_vox(args.run_dir, cfg)
    run_demo(y_t, y_s, A, vg, cfg, args.out or args.run_dir / "demo_map_em",
             dose=args.dose, betas_t=tuple(args.betas),
             n_iter_map=args.n_iter_map, bowsher_B=args.bowsher_B, xp=xp)


if __name__ == "__main__":
    main()
