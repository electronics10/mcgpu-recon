from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot3Dimage(img: np.ndarray, save_path: str | Path, name: str | None = None,
                pmax: float = 100, cmap: str | None = "gray") -> None:
    zz, yy, xx = img.shape
    # vmax from a high percentile, NOT img.max(): a few FOV-edge hot pixels
    # would otherwise stretch the grayscale and make the real object look dim.
    # pmax=100 recovers the old img.max() behavior.
    vmin = float(img.min())
    vmax = float(np.percentile(img, pmax))
    if vmax <= vmin:                     # degenerate (flat image) guard
        vmax = float(img.max()) or 1.0
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))

    im00 = axes[0][0].imshow(img[zz//2, :, :], origin="lower", cmap=cmap,)
                            #  vmin=vmin, vmax=vmax)
    axes[0][0].set_title(f"axial plane {zz//2}")
    fig.colorbar(im00, ax=axes[0][0])

    im01 = axes[0][1].imshow(img.sum(axis=0), origin="lower", cmap=cmap,)
                            #  vmin=vmin, vmax=vmax)
    axes[0][1].set_title("flatten (sum) axial")
    fig.colorbar(im01, ax=axes[0][1])

    im10 = axes[1][0].imshow(img[:, yy//2, :], origin="lower", cmap=cmap,)
                            #  vmin=vmin, vmax=vmax)
    axes[1][0].set_title(f"coronal plane {yy//2}")
    fig.colorbar(im10, ax=axes[1][0])

    im11 = axes[1][1].imshow(img[:, :, xx//2], origin="lower", cmap=cmap,)
                            #  vmin=vmin, vmax=vmax)
    axes[1][1].set_title(f"sagittal plane {xx//2}")
    fig.colorbar(im11, ax=axes[1][1])

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not name: name = str(save_path)
    fig.suptitle(name)
    plt.tight_layout()
    plt.savefig(save_path)
    print(save_path, "saved")
    plt.close()
