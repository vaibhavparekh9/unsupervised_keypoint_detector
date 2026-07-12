"""Training-time validation on held-out view pairs (GT-warp based, no labels).

Metrics:
  desc_pck10 / desc_med_err : NN descriptor matching vs the GT reprojection
                              warp, visible pixels only, 64x64 grid.
  rot_err_deg_median        : geodesic error of predicted relative rotation.
  rot_spearman              : rank correlation of predicted vs ARKit relative
                              rotation magnitudes (gate S2b criterion).
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from ..models.rotation import geodesic_distance


@torch.no_grad()
def validate(model, loader, device, max_pts_per_pair=200):
    model.eval()
    errs, gt_angles, pred_angles, rot_errs = [], [], [], []
    for batch in loader:
        feat_a = batch["feat_a"].to(device)
        feat_b = batch["feat_b"].to(device)
        out_a = model(feat_a)
        out_b = model(feat_b)
        B = feat_a.shape[0]

        pred_rel = torch.matmul(out_a["R_pred"], out_b["R_pred"].transpose(-1, -2))
        gt_rel = batch["R_rel"].to(device)
        rot_errs.extend(torch.rad2deg(
            geodesic_distance(pred_rel, gt_rel)).cpu().tolist())
        # magnitudes for rank correlation
        eye = torch.eye(3, device=device).expand(B, 3, 3)
        gt_angles.extend(torch.rad2deg(
            geodesic_distance(gt_rel, eye)).cpu().tolist())
        pred_angles.extend(torch.rad2deg(
            geodesic_distance(pred_rel, eye)).cpu().tolist())

        desc_a, desc_b = out_a["desc"], out_b["desc"]
        G = desc_a.shape[-1]
        grid = batch["grid"].to(device)
        mask = batch["mask"].to(device)
        for b in range(B):
            vis = torch.nonzero(mask[b] > 0.5)
            if len(vis) < 10:
                continue
            sel = vis[torch.randperm(len(vis))[:max_pts_per_pair]]
            f1 = F.normalize(desc_a[b, :, sel[:, 0], sel[:, 1]], dim=0)  # (C,M)
            f2 = F.normalize(desc_b[b].reshape(desc_b.shape[1], -1), dim=0)
            nn = (f1.t() @ f2).argmax(dim=1)
            pu, pv = (nn % G).float(), (nn // G).float()
            gt = grid[b][sel[:, 0], sel[:, 1]]                # (M,2) normalized
            gu = (gt[:, 0] + 1) / 2 * (G - 1)
            gv = (gt[:, 1] + 1) / 2 * (G - 1)
            e = torch.sqrt((pu - gu) ** 2 + (pv - gv) ** 2)
            errs.append(e.cpu())

    model.train()
    if errs:
        errs = torch.cat(errs).numpy()
    else:
        errs = np.array([np.inf])
    rho = spearmanr(gt_angles, pred_angles).statistic if len(gt_angles) > 5 else 0.0
    return {
        "desc_pck10": float((errs < 10).mean() * 100),
        "desc_med_err": float(np.median(errs)),
        "rot_err_deg_median": float(np.median(rot_errs)) if rot_errs else -1.0,
        "rot_spearman": float(rho if rho == rho else 0.0),
        "n_pairs_rot": len(gt_angles),
    }
