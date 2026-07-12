"""Cross-instance canonical-frame alignment.

Nothing forces car #12 and car #500 to agree which end of the canonical
frame is "front". Mechanism (default, 'pseudo'): mutual-NN pseudo-matches
between frozen backbone tokens of *different* cars must receive consistent
canonical sphere coordinates. Match pairs are gated to similar predicted
orientations, because raw DINO matches carry exactly the L/R confusion we
are fighting — at similar orientations the confusion is minimal.

The 'exchange' alternative (DVE-style, through auxiliary instances) lives in
losses/correspondence.py::dense_correlation_loss_dve.
"""

import torch
import torch.nn.functional as F

from ..models.rotation import geodesic_distance


def _token_centers(ht, wt, res, device):
    """Descriptor-grid coords of token centers, normalized to [-1, 1]."""
    ys = (torch.arange(ht, device=device, dtype=torch.float32) + 0.5) / ht
    xs = (torch.arange(wt, device=device, dtype=torch.float32) + 0.5) / wt
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx * 2 - 1, yy * 2 - 1], dim=-1).reshape(-1, 2)


def pseudo_match_loss(tokens, spheres, R_pred, car_idx,
                      sim_thresh=0.7, max_orient_diff_deg=45.0,
                      max_matches=300):
    """tokens: (B, Ht, Wt, C) frozen backbone tokens (no grad needed);
    spheres: (B, 3, R, R) predicted canonical coords; R_pred: (B, 3, 3);
    car_idx: (B,) long. Pairs are consecutive batch items from different cars.
    """
    B, Ht, Wt, C = tokens.shape
    device = spheres.device
    centers = _token_centers(Ht, Wt, spheres.shape[-1], device)

    with torch.no_grad():
        flat = F.normalize(tokens.reshape(B, Ht * Wt, C).to(device), dim=-1)
        rel_deg = torch.rad2deg(geodesic_distance(
            R_pred.detach().unsqueeze(1), R_pred.detach().unsqueeze(0)))

    total = spheres.new_zeros(())
    n_pairs = 0
    for i in range(B):
        j = (i + 1) % B
        if car_idx[i] == car_idx[j] or i == j:
            continue
        if rel_deg[i, j] > max_orient_diff_deg:
            continue
        with torch.no_grad():
            sim = flat[i] @ flat[j].t()                     # (N, N)
            best_j = sim.argmax(dim=1)
            best_i = sim.argmax(dim=0)
            mutual = best_i[best_j] == torch.arange(len(best_j), device=device)
            strong = sim.gather(1, best_j[:, None]).squeeze(1) > sim_thresh
            keep = torch.nonzero(mutual & strong).squeeze(1)
            if len(keep) < 10:
                continue
            if len(keep) > max_matches:
                keep = keep[torch.randperm(len(keep), device=device)[:max_matches]]
        pts_i = centers[keep]                                # (M, 2)
        pts_j = centers[best_j[keep]]
        s_i = F.grid_sample(spheres[i:i + 1], pts_i.reshape(1, 1, -1, 2),
                            mode="bilinear", align_corners=True).squeeze()
        s_j = F.grid_sample(spheres[j:j + 1], pts_j.reshape(1, 1, -1, 2),
                            mode="bilinear", align_corners=True).squeeze()
        cos = (F.normalize(s_i, dim=0) * F.normalize(s_j, dim=0)).sum(0)
        total = total + (1.0 - cos).mean()
        n_pairs += 1
    if n_pairs == 0:
        return spheres.new_zeros(())
    return total / n_pairs
