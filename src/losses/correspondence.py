"""Masked dense correspondence losses (ported thesis/DVE machinery).

The grid is computed directly at descriptor resolution (no striding, unlike
the original which strided an input-res grid down). Convention unchanged:
grid[v, u] = normalized target coords for source pixel (u, v); mask gates
which source pixels may produce gradients (never let an invisible pixel
produce a gradient).
"""

import torch
import torch.nn.functional as F


def _unnormalize(grid, H, W):
    scale = torch.tensor([W - 1.0, H - 1.0], dtype=grid.dtype,
                         device=grid.device).reshape(1, 1, 1, 2)
    return (grid + 1.0) / 2.0 * scale


def _pixel_grid(H, W, device):
    yy, xx = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                            torch.arange(W, device=device, dtype=torch.float32),
                            indexing="ij")
    return torch.stack([xx, yy], dim=-1)          # (H, W, 2) as (u, v)


def dense_correlation_loss(feats_a, feats_b, grid, mask,
                           pow_=0.5, temp=20.0, normalize=True):
    """Softmax-matching expected-distance loss, visible pixels only.

    feats_a/b : (B, C, H, W)
    grid      : (B, H, W, 2) normalized A->B correspondence
    mask      : (B, H, W) float
    """
    B, C, H, W = feats_a.shape
    device = feats_a.device
    grid_u = _unnormalize(grid.to(device), H, W)
    xxyy = _pixel_grid(H, W, device)
    mask = mask.to(device)

    loss = feats_a.new_zeros(())
    denom = 0.0
    for b in range(B):
        f1 = feats_a[b].reshape(C, H * W)
        f2 = feats_b[b].reshape(C, H * W)
        if normalize:
            f1 = F.normalize(f1, p=2, dim=0) * temp
            f2 = F.normalize(f2, p=2, dim=0) * temp
        corr = f1.t() @ f2                                     # (HW, HW)
        smcorr = F.softmax(corr, dim=1).reshape(H, W, H, W)
        with torch.no_grad():
            diff = grid_u[b].reshape(H, W, 1, 1, 2) - \
                xxyy.reshape(1, 1, H, W, 2)
            diff = diff.pow(2).sum(-1).sqrt().pow(pow_)        # (H, W, H, W)
        L = (diff * smcorr).sum(dim=(2, 3))                    # (H, W)
        vm = mask[b]
        loss = loss + (L * vm).sum()
        denom += vm.sum().item()
    return loss / max(denom, 1.0)


def dense_correlation_loss_dve(feats_a, feats_b, aux, grid, mask,
                               pow_=0.5, temp=20.0, normalize=True):
    """DVE exchange variant: A's descriptors are reconstructed through an
    auxiliary instance's feature map before matching into B — forces
    category-level (cross-instance exchangeable) descriptors.

    aux: (B, C, H, W) auxiliary feature maps (different car, e.g. batch
    rolled by one).
    """
    B, C, H, W = feats_a.shape
    device = feats_a.device
    grid_u = _unnormalize(grid.to(device), H, W)
    xxyy = _pixel_grid(H, W, device)
    mask = mask.to(device)

    loss = feats_a.new_zeros(())
    denom = 0.0
    for b in range(B):
        f1 = feats_a[b].reshape(C, H * W)
        f2 = feats_b[b].reshape(C, H * W)
        fa = aux[b].reshape(C, H * W)
        if normalize:
            f1 = F.normalize(f1, p=2, dim=0) * temp
            f2 = F.normalize(f2, p=2, dim=0) * temp
            fa = F.normalize(fa, p=2, dim=0) * temp
        smcorr = F.softmax(f1.t() @ fa, dim=1)                 # (HW_1, HW_a)
        f1_via_fa = smcorr @ fa.t()                            # (HW_1, C)
        smcorr2 = F.softmax(f1_via_fa @ f2, dim=1).reshape(H, W, H, W)
        with torch.no_grad():
            diff = grid_u[b].reshape(H, W, 1, 1, 2) - \
                xxyy.reshape(1, 1, H, W, 2)
            diff = diff.pow(2).sum(-1).sqrt().pow(pow_)
        L = (diff * smcorr2).sum(dim=(2, 3))
        vm = mask[b]
        loss = loss + (L * vm).sum()
        denom += vm.sum().item()
    return loss / max(denom, 1.0)


def warped_consistency_loss(map_a, map_b, grid, mask):
    """Direct consistency: map_b sampled at grid must equal map_a, on visible
    pixels (cosine distance). Used for canonical sphere coordinates."""
    B = map_a.shape[0]
    samp = F.grid_sample(map_b, grid.to(map_b.device), mode="bilinear",
                         align_corners=True, padding_mode="border")
    cos = (F.normalize(map_a, dim=1) * F.normalize(samp, dim=1)).sum(1)  # (B,H,W)
    vm = mask.to(map_a.device)
    return ((1.0 - cos) * vm).sum() / vm.sum().clamp(min=1.0)
