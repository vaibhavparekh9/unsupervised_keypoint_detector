"""DVE-protocol light regressor (ported from thesis keypoint_prediction.py).

50 virtual keypoints (1x1xC filters, soft-argmax) -> per-annotated-point
linear regressor, trained on FROZEN descriptor maps (no backprop into them).
Comparable with the 2018-2024 landmark literature.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class IntermediateKeypointPredictor(nn.Module):
    def __init__(self, descriptor_dim, num_annotated=24, num_virtual=50,
                 softargmax_mul=50.0):
        super().__init__()
        self.nA = num_annotated
        self.nI = num_virtual
        self.softargmax_mul = softargmax_mul
        latent = self.nA * self.nI
        self.inner_conv = nn.Conv2d(descriptor_dim, latent, 1, bias=False)
        self.reg_conv = nn.Conv2d(latent * 2, 2 * self.nA, 1,
                                  groups=self.nA, bias=False)

    def forward(self, desc):
        """desc: (B, C, H, W) frozen descriptor maps (detached).
        Returns (B, nA, 2) predictions in [-1, 1] image coords."""
        desc = desc.detach()
        B, C, H, W = desc.shape
        xi = torch.linspace(-1, 1, W, device=desc.device)
        yi = torch.linspace(-1, 1, H, device=desc.device)
        yy, xx = torch.meshgrid(yi, xi, indexing="ij")

        corr = self.inner_conv(desc).view(B, self.nA * self.nI, H * W)
        smcorr = F.softmax(self.softargmax_mul * corr, dim=2)
        smcorr = smcorr.reshape(B, self.nA, self.nI, H, W)
        mass = smcorr.sum(dim=(3, 4))
        xpred = (smcorr * xx.view(1, 1, 1, H, W)).sum(dim=(3, 4)) / mass
        ypred = (smcorr * yy.view(1, 1, 1, H, W)).sum(dim=(3, 4)) / mass
        inter = torch.stack((xpred, ypred), dim=3)         # (B, nA, nI, 2)
        pred = self.reg_conv(inter.view(B, -1, 1, 1)).view(B, self.nA, 2)
        return pred


def train_regressor(desc_maps, kps_norm, vis, steps=400, lr=1e-3,
                    num_virtual=50, device="cpu", log_every=100):
    """desc_maps: (N, C, H, W); kps_norm: (N, 24, 2) in [-1,1]; vis: (N, 24).
    Returns trained predictor."""
    model = IntermediateKeypointPredictor(
        desc_maps.shape[1], num_annotated=kps_norm.shape[1],
        num_virtual=num_virtual).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = desc_maps.shape[0]
    for s in range(steps):
        idx = torch.randint(0, N, (min(16, N),))
        pred = model(desc_maps[idx].to(device))
        gt = kps_norm[idx].to(device)
        m = vis[idx].to(device).unsqueeze(-1).float()
        loss = (F.smooth_l1_loss(pred * m, gt * m, reduction="sum")
                / m.sum().clamp(min=1.0))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and (s + 1) % log_every == 0:
            print(f"  regressor step {s+1}: loss {float(loss):.4f}")
    return model


@torch.no_grad()
def eval_regressor(model, desc_maps, kps_norm, vis, feature_res=64,
                   device="cpu"):
    """PCK@10 on the 64-grid (thesis convention): normalized error scaled to
    feature-grid pixels."""
    pred = model(desc_maps.to(device)).cpu()            # (N, nA, 2) in [-1,1]
    err_grid = ((pred - kps_norm).pow(2).sum(-1).sqrt()
                * (feature_res - 1) / 2.0)              # normalized -> grid px
    m = vis.bool()
    errs = err_grid[m].numpy()
    return {
        "pck@10_feat64": float((errs < 10).mean() * 100),
        "pck@5_feat64": float((errs < 5).mean() * 100),
        "median_err_feat64": float(np.median(errs)),
        "n_kps": int(m.sum()),
    }
