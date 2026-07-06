# A Reconstruction Tool for the MCGPU-PET Monte Carlo PET simulator

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
x_corr = mlem(A, xp.asarray(y + y_s), n_iter=20,
             contamination=xp.asarray(y_s), verbose=True)
x_tot = np.asarray(x_tot) if x_tot is np else xp.asnumpy(x_tot) if hasattr(xp, "asnumpy") \
    else np.asarray(x_tot.get())
plot3Dimage(x_tot, "recon_img/recon_mlem_total_20.png")
x_corr = np.asarray(x_corr) if xp is np else xp.asnumpy(x_corr) if hasattr(xp, "asnumpy") \
    else np.asarray(x_corr.get())
plot3Dimage(x_corr, "recon_img/recon_mlem_sc_20.png")
```