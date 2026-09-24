"""mcgpu_recon: reconstruction for MCGPU-PET span=1 sinograms.

Modules
-------
projector    MCGPUProjector (A, A^T), from_run, adjoint_test
mlem         mlem (+ helpers shared with map_em)
map_em       map_em: penalized-likelihood (MAP) reconstruction
priors       neighbourhood penalties: QuadraticPrior, RDPrior; uniform / Bowsher weights
attenuation  attenuation_map_from_vox, attenuation_factors
utils        scale_match, binomial_thin
metrics      object_bbox, psnr_ssim, evaluate_recon
draw_tools   plot3Dimage
"""
from .projector import MCGPUProjector, from_run, adjoint_test
from .mlem import mlem
from .map_em import map_em
from .priors import (
    uniform_weights, bowsher_weights, QuadraticPrior, RDPrior, beta_scale,
)
from .attenuation import attenuation_map_from_vox, attenuation_factors
from .utils import scale_match, binomial_thin
from .metrics import object_bbox, psnr_ssim, evaluate_recon

__all__ = [
    # system model
    "MCGPUProjector", "from_run", "adjoint_test",
    # reconstruction
    "mlem", "map_em",
    "uniform_weights", "bowsher_weights", "QuadraticPrior", "RDPrior", "beta_scale",
    # corrections and helpers
    "attenuation_map_from_vox", "attenuation_factors", "scale_match", "binomial_thin",
    # region selection + metrics
    "object_bbox", "psnr_ssim", "evaluate_recon",
]
