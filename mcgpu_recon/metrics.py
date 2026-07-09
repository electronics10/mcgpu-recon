from __future__ import annotations

import numpy as np


def _np(a):
    """Coerce to a host numpy array (accepts numpy or cupy)."""
    if isinstance(a, np.ndarray):
        return a
    if hasattr(a, "get"):
        return a.get()
    return np.asarray(a)


def object_bbox(mask):
    """Bounding-box slices of a boolean mask's True region (crop for PSNR/SSIM)."""
    mask = _np(mask)
    idx = np.where(mask)
    if len(idx[0]) == 0:
        raise ValueError("object_bbox: mask is empty")
    return tuple(slice(int(i.min()), int(i.max()) + 1) for i in idx)


def psnr_ssim(img, ref, bbox=None):
    """(PSNR, SSIM) of img vs ref. Expects SCALE-MATCHED, cropped images.
    skimage imported here so the core library needs no scikit-image."""
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


def evaluate_recon(img, ref, bbox=None):
    """{psnr, ssim} for one reconstruction.

    img      : reconstruction (scale-matched to ref for PSNR/SSIM).
    ref      : reference reconstruction (trues recon).
    bbox     : object bbox (crop for PSNR/SSIM).
    """
    p, s = psnr_ssim(img, ref, bbox=bbox)
    return {"psnr": p, "ssim": s}