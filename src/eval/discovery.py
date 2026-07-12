"""Unsupervised landmark discovery: spherical k-means on canonical coords.

Landmarks = clusters on the predicted canonical sphere. A landmark's
detection in a view is the pixel with maximum cosine similarity to the
cluster centroid. Consistency metric: detections in view A, projected into
view B via the GT reprojection warp, must land near the detections in B.
"""

import numpy as np
import torch
import torch.nn.functional as F


def spherical_kmeans(vectors, k, iters=50, seed=0):
    """vectors: (N, 3) unit vectors. Returns (k, 3) unit centroids."""
    rng = np.random.default_rng(seed)
    v = F.normalize(vectors, dim=1)
    idx = rng.choice(len(v), k, replace=False)
    c = v[idx].clone()
    for _ in range(iters):
        sim = v @ c.t()                       # (N, k)
        assign = sim.argmax(dim=1)
        for j in range(k):
            sel = v[assign == j]
            if len(sel):
                c[j] = F.normalize(sel.mean(dim=0), dim=0)
    return c


@torch.no_grad()
def detect_landmarks(sphere_map, centroids, min_sim=0.9):
    """sphere_map: (3, H, W); centroids: (K, 3).
    Returns (K, 2) pixel coords (u, v) and (K,) peak similarities."""
    _, H, W = sphere_map.shape
    flat = F.normalize(sphere_map.reshape(3, -1), dim=0)     # (3, HW)
    sim = centroids @ flat                                    # (K, HW)
    peak, idx = sim.max(dim=1)
    uv = torch.stack([(idx % W).float(), (idx // W).float()], dim=1)
    return uv, peak


@torch.no_grad()
def landmark_consistency(model, pairs_ds, centroids, device, num_pairs=30,
                         min_sim=0.9):
    """Cross-view consistency of discovered landmarks on GT-warped pairs."""
    errs, n_repeat, n_total = [], 0, 0
    n = min(num_pairs, len(pairs_ds))
    for i in range(n):
        item = pairs_ds[i]
        out = model(torch.stack([item["feat_a"], item["feat_b"]]).to(device))
        sph_a, sph_b = out["sphere"][0], out["sphere"][1]
        G = sph_a.shape[-1]
        uv_a, sim_a = detect_landmarks(sph_a, centroids)
        uv_b, sim_b = detect_landmarks(sph_b, centroids)
        grid, mask = item["grid"], item["mask"]
        if grid.shape[0] != G:
            continue
        for k in range(len(centroids)):
            if sim_a[k] < min_sim or sim_b[k] < min_sim:
                continue
            u, v = int(uv_a[k, 0]), int(uv_a[k, 1])
            n_total += 1
            if mask[v, u] < 0.5:
                continue
            tu = (grid[v, u, 0] + 1) / 2 * (G - 1)
            tv = (grid[v, u, 1] + 1) / 2 * (G - 1)
            e = float(torch.hypot(tu - uv_b[k, 0], tv - uv_b[k, 1]))
            errs.append(e)
            n_repeat += 1
    return {
        "num_pairs": n,
        "mean_err_feat": float(np.mean(errs)) if errs else -1.0,
        "median_err_feat": float(np.median(errs)) if errs else -1.0,
        "repeat_rate": n_repeat / max(n_total, 1),
        "n_detections_checked": n_total,
    }
