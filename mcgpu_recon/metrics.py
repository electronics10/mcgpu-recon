"""
metrics.py -- image-domain reconstruction quality metrics.

ONLY measurements live here. Things that transform an image (scale_match) or
build a forward-model input (attenuation_map_from_vox) are reconstruction
concerns and live in mcgpu_recon.py instead. Keeping this module measurement-only
means a metric never silently rescales or alters its input -- you hand it images
and it reports numbers.

Reference convention for scatter-correction studies: use the TRUES-only
reconstruction as the reference (not the emission activity map). All arms (floor,
oracle, model) and the trues reference share the same reconstruction artifacts
(finite iterations, sensitivity floor, attenuation handling, discretization), so
differencing against the trues recon makes those common-mode and cancels them,
isolating SCATTER handling -- which is what you are testing. Differencing against
the activity map instead folds in resolution/convergence blur that dwarfs the
scatter effect. (Pass the activity map as `ref` if you instead want absolute
fidelity, e.g. for a resolution study.)

Full-reference metrics (PSNR, SSIM) are NOT scale-invariant: scale-match the
image to the reference first (mcgpu_recon.scale_match), then pass it here. CNR is
a ratio, hence scale-invariant, so it needs no scale-match.

skimage is imported lazily inside psnr_ssim, so importing this module (and the
core recon library) never requires scikit-image -- only calling PSNR/SSIM does.
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
    """Bounding-box slices (kz, ky, kx) of a boolean mask's True region.

    Used to crop PSNR/SSIM to the object: without it, the huge near-zero
    background makes every method score near-identically (mostly matching
    zeros), destroying discriminative power.
    """
    mask = _np(mask)
    idx = np.where(mask)
    if len(idx[0]) == 0:
        raise ValueError("object_bbox: mask is empty")
    return tuple(slice(int(i.min()), int(i.max()) + 1) for i in idx)


def rois_from_activity(activity, hi=75, lo=40):
    """Hot / background ROI masks from a known activity map, by percentile.

    hot : voxels >= `hi`-th percentile of nonzero activity (the bright region).
    bg  : nonzero voxels <= `lo`-th percentile (the warm background).
    Assumes a hot-vs-background separation exists; for a near-uniform phantom
    the two collapse and CNR is meaningless (that run is simply not a CNR test).
    """
    activity = _np(activity)
    nz = activity[activity > 0]
    if nz.size == 0:
        raise ValueError("rois_from_activity: activity is all zero")
    hot = activity >= np.percentile(nz, hi)
    bg = (activity > 0) & (activity <= np.percentile(nz, lo))
    return hot, bg


def cnr(img, hot, bg):
    """Contrast-to-noise ratio: (mean_hot - mean_bg) / std_bg. Scale-invariant.

    The physically meaningful scatter metric: scatter raises the background mean
    and its spread, depressing CNR; correction should restore it toward the
    trues-reference CNR.
    """
    img, hot, bg = _np(img), _np(hot), _np(bg)
    mh, mb, sb = float(img[hot].mean()), float(img[bg].mean()), float(img[bg].std())
    return (mh - mb) / sb if sb > 0 else float("nan")


def psnr_ssim(img, ref, bbox=None):
    """(PSNR, SSIM) of img vs ref. Both expect SCALE-MATCHED, cropped images.

    Pass `bbox` (from object_bbox) to restrict to the object. skimage is
    imported here so the rest of the library needs no scikit-image.
    """
    from skimage.metrics import peak_signal_noise_ratio as _psnr
    from skimage.metrics import structural_similarity as _ssim
    img, ref = _np(img), _np(ref)
    if bbox is not None:
        img, ref = img[bbox], ref[bbox]
    dr = float(ref.max() - ref.min())
    if dr <= 0:
        return float("nan"), float("nan")
    p = float(_psnr(ref, img, data_range=dr))
    # win_size=7 (default) requires each cropped dim >= 7; shrink if a thin
    # object crops smaller.
    win = min(7, *(s for s in ref.shape))
    if win % 2 == 0:
        win -= 1
    s = float(_ssim(ref, img, data_range=dr, win_size=max(win, 3)))
    return p, s


def evaluate_recon(img, ref, hot, bg, bbox=None):
    """Convenience: {psnr, ssim, cnr} for one reconstruction.

    img : reconstruction to score (scale-matched to ref for PSNR/SSIM).
    ref : reference reconstruction (trues recon for scatter studies).
    hot, bg : ROI masks from rois_from_activity.
    bbox : object bounding box from object_bbox (crop for PSNR/SSIM).
    """
    p, s = psnr_ssim(img, ref, bbox=bbox)
    return {"psnr": p, "ssim": s, "cnr": cnr(img, hot, bg)}