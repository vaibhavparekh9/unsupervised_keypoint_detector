"""Human-facing diagnostic visualizations (non-blocking, reviewed later)."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def denorm_image(t):
    """(3,H,W) normalized tensor -> (H,W,3) uint8-ish float array in [0,1]."""
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    arr = t.detach().cpu().numpy() * std + mean
    return np.clip(arr.transpose(1, 2, 0), 0, 1)


def save_warp_overlay(im_a, im_b, grid, mask, out_path, n_points=60, title=""):
    """Scatter matched points: source pixels in A colored, their warped
    locations in B in the same colors. grid: (G,G,2) normalized; mask: (G,G)."""
    A = denorm_image(im_a)
    B = denorm_image(im_b)
    G = grid.shape[0]
    S = A.shape[0]

    vis = np.argwhere(mask.numpy() > 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(A)
    axes[1].imshow(B)
    if len(vis):
        idx = np.random.default_rng(0).choice(
            len(vis), min(n_points, len(vis)), replace=False)
        pts = vis[idx]  # (n, 2) as (v, u) in grid space
        colors = plt.cm.hsv(np.linspace(0, 1, len(pts)))
        scale = S / G
        for (v, u), c in zip(pts, colors):
            gu, gv = grid[v, u].numpy()
            tu = (gu + 1) / 2 * (G - 1) * scale
            tv = (gv + 1) / 2 * (G - 1) * scale
            axes[0].scatter([u * scale], [v * scale], color=c, s=12)
            axes[1].scatter([tu], [tv], color=c, s=12)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
