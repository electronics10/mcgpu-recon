import mcgpu_pet_wrapper as mpw
from pathlib import Path

run_dir = Path("data/run_0")

# Load configuration
cfg = mpw.default_config()
cfg["sinogram"]["span"] = 1
mpw.validate_config(cfg)

# Define voxel space object
voxel_space = mpw.nema_iq_preclinical(cfg, hot_activity_Bq_per_mL=200000)

# Build simulation directory and files
mpw.build_run(run_dir, cfg, voxel_space)

# Run simulation
simulation = mpw.Runner()(run_dir, "overwrite")
