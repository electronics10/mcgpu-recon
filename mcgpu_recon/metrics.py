"""
metrics.py -- image-domain reconstruction quality metrics.

ONLY measurements live here. Transforms (scale_match) and forward-model inputs
(attenuation_map_from_vox) are reconstruction concerns and live in
mcgpu_recon.py, so a metric never silently rescales its input.

Reference convention for scatter studies: use the TRUES reconstruction as `ref`
for PSNR/SSIM (all arms share the same recon artifacts, so differencing against
the trues recon cancels them and isolates scatter handling). CRC compares to the
known ACTIVITY map's true ratio instead.

ROIs: two regions from the activity map -- `hot` (bright structure) and `warm`
(the uniform-ish tissue background). In reconstructed PET the surrounding AIR is
forced to ~0 by the sensitivity floor, so it is NOT a usable noise region (unlike
MRI, where air carries real noise); the warm in-object region serves as both the
contrast reference (S2) and the noise region (sigma). This matches NEMA-style PET
background-variability practice.

A NOTE ON WHAT EACH METRIC SEES (important for reading the numbers):
  * PSNR/SSIM, CRC  -- accuracy/fidelity (BIAS). Scatter correction improves
    these: floor worst, oracle best.
  * CNR, SNR        -- noise-sensitive (they divide by std_warm). Scatter
    correction SUBTRACTS counts, which raises Poisson variance, so the corrected
    image is more accurate but NOISIER. CNR/SNR can therefore be FLAT or WORSE
    after correction even though the image is better -- a genuine bias-variance
    tradeoff, not a broken metric. Report CRC for "does correction help"; report
    CNR because it is conventional, and read floor>=oracle CNR as the noise cost.
"""

from __future__ import annotations

import numpy as np


def _np(a):
    """Coerce to a host numpy array (accepts numpy or cupy)."""
    if isinstance(a, np.ndarray):
        return a
    if hasattr(a, "get"):          # cupy ndarray
        return a.get()
    return np.asarray(a)


def object_bbox(mask):
    """Bounding-box slices of a boolean mask's True region (crop for PSNR/SSIM)."""
    mask = _np(mask)
    idx = np.where(mask)
    if len(idx[0]) == 0:
        raise ValueError("object_bbox: mask is empty")
    return tuple(slice(int(i.min()), int(i.max()) + 1) for i in idx)


def rois_from_activity(activity, hi=75, lo=40):
    """Hot / warm-background ROI masks from a known activity map, by percentile.

    hot  : voxels >= `hi`-th percentile of nonzero activity (bright structure).
    warm : nonzero voxels <= `lo`-th percentile (uniform-ish tissue background),
           used as BOTH the contrast reference and the noise region (see module
           docstring for why air is not usable in reconstructed PET).
    Assumes a hot-vs-warm separation exists; for a near-uniform phantom the two
    collapse and CNR/CRC are meaningless (that run is simply not a contrast test).
    """
    activity = _np(activity)
    nz = activity[activity > 0]
    if nz.size == 0:
        raise ValueError("rois_from_activity: activity is all zero")
    hot = activity >= np.percentile(nz, hi)
    warm = (activity > 0) & (activity <= np.percentile(nz, lo))
    return hot, warm


def cnr(img, hot, warm):
    """Contrast-to-noise ratio: (mean_hot - mean_warm) / std_warm.

    Warm region is both the contrast reference and the noise region (PET
    convention). NOISE-SENSITIVE: scatter correction adds variance, so this can
    be flat/worse after correction even when accuracy improves -- see module doc.
    """
    img, hot, warm = _np(img), _np(hot), _np(warm)
    sd = float(img[warm].std())
    return (float(img[hot].mean()) - float(img[warm].mean())) / sd \
        if sd > 0 else float("nan")


def snr(img, hot, warm):
    """Signal-to-noise ratio of the hot region: mean_hot / std_warm.
    Noise-sensitive, same caveat as cnr."""
    img, hot, warm = _np(img), _np(hot), _np(warm)
    sd = float(img[warm].std())
    return float(img[hot].mean()) / sd if sd > 0 else float("nan")


def crc(img, activity, hot, warm):
    """NEMA hot contrast recovery coefficient (BIAS only, scale-invariant):

        CRC = (rec_ratio - 1) / (true_ratio - 1),
        rec_ratio  = mean_hot(recon)    / mean_warm(recon),
        true_ratio = mean_hot(activity) / mean_warm(activity).

    1.0 = contrast perfectly recovered. This is the "does correction help"
    metric: no noise term, so floor should score worst and oracle best. `activity`
    is the KNOWN ground-truth activity map (defines the true ratio).
    """
    img, act = _np(img), _np(activity)
    mw_rec = float(img[warm].mean())
    mw_true = float(act[warm].mean())
    if mw_rec <= 0 or mw_true <= 0:
        return float("nan")
    true_ratio = float(act[hot].mean()) / mw_true
    rec_ratio = float(img[hot].mean()) / mw_rec
    return (rec_ratio - 1.0) / (true_ratio - 1.0) \
        if true_ratio != 1.0 else float("nan")


def psnr_ssim(img, ref, bbox=None):
    """(PSNR, SSIM) of img vs ref. Expects SCALE-MATCHED, cropped images.
    skimage is imported here so the core library needs no scikit-image."""
    from skimage.metrics import peak_signal_noise_ratio as _psnr
    from skimage.metrics import structural_similarity as _ssim
    img, ref = _np(img), _np(ref)
    if bbox is not None:
        img, ref = img[bbox], ref[bbox]
    dr = float(ref.max() - ref.min())
    if dr <= 0:
        return float("nan"), float("nan")
    p = float(_psnr(ref, img, data_range=dr))
    win = min(7, *(s for s in ref.shape))
    if win % 2 == 0:
        win -= 1
    s = float(_ssim(ref, img, data_range=dr, win_size=max(win, 3)))
    return p, s


def evaluate_recon(img, ref, activity, hot, warm, bbox=None):
    """{psnr, ssim, crc, cnr, snr} for one reconstruction.

    img      : reconstruction to score (scale-matched to ref for PSNR/SSIM;
               CRC/CNR/SNR are scale-invariant so matching is harmless).
    ref      : reference reconstruction (trues recon for scatter studies).
    activity : ground-truth activity map (defines CRC's true ratio).
    hot, warm: ROI masks from rois_from_activity.
    bbox     : object bounding box (crop for PSNR/SSIM).
    """
    p, s = psnr_ssim(img, ref, bbox=bbox)
    return {"psnr": p, "ssim": s,
            "crc": crc(img, activity, hot, warm),
            "cnr": cnr(img, hot, warm),
            "snr": snr(img, hot, warm)}