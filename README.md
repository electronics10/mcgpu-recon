# A Reconstruction Tool for the MCGPU-PET Monte Carlo PET simulator

### Usage

This project depends on [Parallelproj](https://parallelproj.readthedocs.io/en/stable/), which has complex, non-Python dependencies. Personally, I preferred to use [pixi](https://pixi.prefix.dev/latest/) (instead of conda) to manage my environment. To use this tool, simply
```bash
git clone https://github.com/electronics10/mcgpu-recon.git
cd mcgpu-recon
pixi install
```

Then run the below example.

## Example

### 1 Perform a simple simulation

```Python
import mcgpu_pet_wrapper as mpw
from pathlib import Path

run_dir = Path("data/run_0")
cfg = mpw.default_config()
cfg["sinogram"]["span"] = 1
mpw.validate_config(cfg)
voxel_space = mpw.nema_iq_preclinical(cfg, hot_activity_Bq_per_mL=200000)
mpw.build_run(run_dir, cfg, voxel_space)
simulation = mpw.Runner()(run_dir, "overwrite")
```

### 2 Conduct reconstruction (MLEM)

```Python
from pathlib import Path
import numpy as np

import mcgpu_pet_wrapper as mpw
from mcgpu_recon import from_run, mlem
from mcgpu_recon.draw_tools import plot3Dimage

# ---- choose the array backend -------------------------------------------
# CPU (works everywhere; parallelproj uses OpenMP, or hybrid GPU if CUDA lib
# is present -- raise num_chunks if GPU memory is tight):
# import numpy as xp
# XP_KW = dict(xp=xp, plane_chunk=256, num_chunks=1)
# GPU (recommended -- everything stays on the device):
import array_api_compat.cupy as xp
XP_KW = dict(xp=xp, plane_chunk=256)

run_dir = Path("data/run_0")
cfg = mpw.load_config(run_dir / "config.json")
plot3Dimage(mpw.read_emission_image(run_dir, cfg), "recon_img/emission.png")

# ---- trues-only reconstruction ------------------------------------------
y, A = from_run(run_dir, cfg, **XP_KW)     # bin order matches by construction
x = mlem(A, xp.asarray(y), n_iter=20, verbose=True)

x = np.asarray(x) if xp is np else xp.asnumpy(x) if hasattr(xp, "asnumpy") \
    else np.asarray(x.get())
plot3Dimage(x, "recon_img/recon_mlem_trues_20.png")

# ---- total (trues+scatter) with the true scatter as known contamination ---
y, A = from_run(run_dir, cfg, **XP_KW)     # bin order matches by construction
y_s, _ = from_run(run_dir, cfg, scatter=True, **XP_KW)
x_tot = mlem(A, xp.asarray(y + y_s), n_iter=20, verbose=True)
x_tot = np.asarray(x_tot) if x_tot is np else xp.asnumpy(x_tot) if hasattr(xp, "asnumpy") \
    else np.asarray(x_tot.get())
plot3Dimage(x_tot, "recon_img/recon_mlem_total_20.png")
x_corr = mlem(A, xp.asarray(y + y_s), n_iter=20,
             contamination=xp.asarray(y_s), verbose=True)
x_corr = np.asarray(x_corr) if xp is np else xp.asnumpy(x_corr) if hasattr(xp, "asnumpy") \
    else np.asarray(x_corr.get())
plot3Dimage(x_corr, "recon_img/recon_mlem_sc_20.png")
```

## Package (Developer)

It is a little more complex to use it as a package directly (since the repo isn't released in conda-forge). One can try to paste the following toml text into the `pixi.toml` in there own project. First, create your own project if not yet created.

```bash
mkdir my-project
cd my-project
```

Then, initiallize pixi and intall Python>=3.12.
```bash
pixi init
pixi add python=3.12
```

You will see a file `pixi.toml` in your directory. Open it with a text editor and replace it with the following:
```toml
[workspace]
name = "my-project"
channels = ["conda-forge"]
platforms = ["linux-64"]

[dependencies]
python = "3.12.*"
parallelproj = ">=1.10.2,<2"
cupy = ">=14.1.1,<15"
cuda-version = "12.*"
matplotlib = ">=3.11.0,<4"

[system-requirements]
cuda = "12"

[pypi-dependencies]
mcgpu-recon = { git = "https://github.com/electronics10/mcgpu-recon.git" }
```

(Change the name and system requirements according to your own setup. Note that [mcgpu-pet-wrapper](https://github.com/electronics10/mcgpu-pet-wrapper.git), my another project/package, is NOT machine agnostic, check it out if you encounter any installation problem.)

Finally, enter
```bash
pixi install
```

and pixi will install everything in the environment.

## Canonical example: evaluate scatter correction (attenuation + metrics)

This is the end-to-end workflow the library is built for: reconstruct the same
measured data three ways, and score them against a common reference so the
numbers are interpretable.

**The bracket.** A scatter-correction result means nothing on its own; it is read
between two anchors:

- **floor** — reconstruct trues+scatter with *no* correction (worst case),
- **oracle** — reconstruct trues+scatter with the *true* scatter as the
  `contamination` term (best any method can do; you have it because this is a
  simulation),
- and your **model** (or SSS) sits between them. "Fraction of the floor→oracle
  gap closed" is the interpretable score.

**The reference for metrics is the trues-only reconstruction**, not the emission
map: every arm shares the same reconstruction artifacts, so differencing against
the trues recon cancels them (common-mode) and isolates *scatter handling*.

```python
from pathlib import Path
import numpy as np
import array_api_compat.cupy as xp          # GPU; use `import numpy as xp` for CPU

import mcgpu_pet_wrapper as mpw
from mcgpu_recon import (from_run, mlem, attenuation_factors,
                         attenuation_map_from_vox, scale_match)
from mcgpu_recon.metrics import rois_from_activity, object_bbox, evaluate_recon

XP_KW = dict(xp=xp, plane_chunk=256)
run_dir = Path("data/run_0")
cfg = mpw.load_config(run_dir / "config.json")

# --- attenuation map straight from the simulation's own voxel grid ---------
# mass attenuation coefficients at 511 keV (cm^2/g) per material id; look these
# up for YOUR material list (NIST XCOM). ~0.096 for soft tissue is a fine start.
vg = mpw.read_vox(run_dir, cfg)
MU_RHO = {1: 0.087, 2: 0.096, 3: 0.094, 4: 0.093}   # air, water, adipose, spongiosa
mu_per_mm = attenuation_map_from_vox(vg, MU_RHO)

# --- measured data + attenuation factors -----------------------------------
y,   A = from_run(run_dir, cfg, **XP_KW)             # trues
y_s, _ = from_run(run_dir, cfg, scatter=True, **XP_KW)  # true scatter
y, y_s = xp.asarray(y), xp.asarray(y_s)
af = attenuation_factors(A, xp.asarray(mu_per_mm))   # pass as mlem(mult=...)

NIT, FLOOR = 23, 0.07
# reference (target) + the two bracket arms, ALL with identical recon settings
x      = mlem(A, y,       n_iter=NIT, mult=af, sens_floor_frac=FLOOR)          # trues ref
floor  = mlem(A, y + y_s, n_iter=NIT, mult=af, sens_floor_frac=FLOOR)          # no correction
oracle = mlem(A, y + y_s, n_iter=NIT, mult=af, contamination=y_s,
              sens_floor_frac=FLOOR)                                           # exact scatter
# a model arm is identical with contamination = predicted_scatter (mlem-ready,
# same plane order as A.out_shape).

# --- ROIs from the known activity; reference = trues recon -----------------
hot, warm = rois_from_activity(np.asarray(vg.activity))   # bright / warm-tissue bg
bbox = object_bbox(np.asarray(vg.activity) > 0)
act  = np.asarray(vg.activity)                            # true ratio for CRC
ref  = xp.asnumpy(x)

print(f"{'arm':8s} {'PSNR':>7s} {'SSIM':>7s} {'CRC':>7s} {'CNR':>7s} {'SNR':>8s}")
for name, arm in [("floor", floor), ("oracle", oracle)]:
    arm_m, c = scale_match(x, arm)          # fix MLEM's global-scale freedom first
    m = evaluate_recon(xp.asnumpy(arm_m), ref, act, hot, warm, bbox=bbox)
    print(f"{name:8s} {m['psnr']:7.2f} {m['ssim']:7.3f} {m['crc']:7.3f} "
          f"{m['cnr']:7.3f} {m['snr']:8.3f}")
```

Reading the output — the metrics split into two groups that answer different
questions:

- **PSNR, SSIM, CRC** measure ACCURACY (bias). Oracle should beat floor: higher
  PSNR/SSIM, and CRC nearer 1.0 (full contrast recovery). A model arm's job is to
  land between floor and oracle; `(model − floor)/(oracle − floor)` on CRC is the
  fraction of achievable accuracy it recovered.
- **CNR, SNR** are NOISE-sensitive (they divide by background std). Scatter
  correction subtracts counts, which raises Poisson variance, so the corrected
  image is more accurate but NOISIER — oracle CNR/SNR may be FLAT or LOWER than
  floor. This is the bias–variance tradeoff, not a failure; report it as the
  noise cost of correction alongside the accuracy gain.

Notes:
- `scale_match` matters only for PSNR/SSIM (not scale-invariant); CRC/CNR/SNR are
  ratios and need no matching.
- In reconstructed PET the surrounding air is forced to ~0 by the sensitivity
  floor, so it is not a usable noise region; the warm in-object region serves as
  both the contrast reference and the noise region (NEMA-style).
- `evaluate_recon` never rescales internally — you scale-match, then measure — so
  a metric can't silently alter its input.