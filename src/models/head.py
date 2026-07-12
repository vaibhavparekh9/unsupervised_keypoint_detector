"""Orientation-conditioned head on frozen backbone tokens.

The paper's core mechanism: DINO patch tokens encode "wheel-ness" but not
"front-left-wheel-ness". A learned query pools a global orientation estimate
from all tokens via cross-attention; local descriptors are FiLM-conditioned
on it, turning symmetric part descriptors into side-disambiguated ones.

Outputs per image:
    desc   : (B, C_desc, R, R)  dense descriptors (R = descriptor_res)
    sphere : (B, 3, R, R)       canonical unit-sphere coordinates per pixel
    R_pred : (B, 3, 3)          global orientation (canonical -> camera)
    azim   : (B,)               azimuth angle (only for orientation=azimuth)

Trainable params < 10M. Resolution-agnostic (2D sincos position encoding),
so dev (37x37 tokens) and lab (64x64 tokens) checkpoints share code.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rotation import rotation_6d_to_matrix


def sincos_pos_embed_2d(dim, ht, wt, device):
    """(ht*wt, dim) fixed 2D sine-cosine position embedding."""
    assert dim % 4 == 0
    d = dim // 4
    omega = torch.arange(d, device=device, dtype=torch.float32) / d
    omega = 1.0 / (10000 ** omega)
    y = torch.arange(ht, device=device, dtype=torch.float32)
    x = torch.arange(wt, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    out_y = yy.reshape(-1, 1) * omega.reshape(1, -1)
    out_x = xx.reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat([torch.sin(out_x), torch.cos(out_x),
                      torch.sin(out_y), torch.cos(out_y)], dim=1)


class OrientationHead(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256, num_blocks=2, num_heads=8,
                 descriptor_dim=64, descriptor_res=64, orientation="6d",
                 film=True):
        super().__init__()
        self.orientation = orientation
        self.film = film
        self.descriptor_res = descriptor_res

        self.proj = nn.Sequential(nn.LayerNorm(in_dim),
                                  nn.Linear(in_dim, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=4 * hidden_dim,
            dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_blocks)

        # global orientation: learned query cross-attending over all tokens
        self.orient_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.orient_attn = nn.MultiheadAttention(hidden_dim, num_heads,
                                                 batch_first=True)
        out_dim = 6 if orientation == "6d" else 2
        self.orient_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, out_dim))

        if film:
            self.film_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 2 * hidden_dim))

        self.desc_branch = nn.Linear(hidden_dim, descriptor_dim)
        self.desc_refine = nn.Conv2d(descriptor_dim, descriptor_dim, 3, padding=1)
        self.sphere_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 3))
        self.sphere_refine = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, tokens):
        """tokens: (B, Ht, Wt, C_in) frozen backbone patch tokens."""
        B, Ht, Wt, _ = tokens.shape
        x = self.proj(tokens.reshape(B, Ht * Wt, -1))
        x = x + sincos_pos_embed_2d(x.shape[-1], Ht, Wt, x.device).unsqueeze(0)
        x = self.blocks(x)

        q = self.orient_query.expand(B, -1, -1)
        pooled, _ = self.orient_attn(q, x, x)
        pooled = pooled.squeeze(1)                       # (B, D)
        o = self.orient_mlp(pooled)

        azim = None
        if self.orientation == "6d":
            R_pred = rotation_6d_to_matrix(o)
        else:  # azimuth ablation: S1 embedding, R about the vertical axis
            v = F.normalize(o, dim=-1)
            azim = torch.atan2(v[:, 0], v[:, 1])
            c, s = torch.cos(azim), torch.sin(azim)
            zero, one = torch.zeros_like(c), torch.ones_like(c)
            R_pred = torch.stack([
                torch.stack([c, zero, s], -1),
                torch.stack([zero, one, zero], -1),
                torch.stack([-s, zero, c], -1)], -2)

        if self.film:
            gb = self.film_mlp(pooled)                   # (B, 2D)
            gamma, beta = gb.chunk(2, dim=-1)
            xf = x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        else:
            xf = x

        R = self.descriptor_res

        desc = self.desc_branch(xf).reshape(B, Ht, Wt, -1).permute(0, 3, 1, 2)
        desc = F.interpolate(desc, size=(R, R), mode="bilinear",
                             align_corners=True)
        desc = desc + self.desc_refine(desc)

        sph = self.sphere_branch(xf).reshape(B, Ht, Wt, 3).permute(0, 3, 1, 2)
        sph = F.interpolate(sph, size=(R, R), mode="bilinear",
                            align_corners=True)
        sph = sph + self.sphere_refine(sph)
        sph = F.normalize(sph, p=2, dim=1)

        return {"desc": desc, "sphere": sph, "R_pred": R_pred, "azim": azim,
                "tokens": x}
