"""Frozen dense-feature backbones (never trained, never fine-tuned).

Default: DINOv2 ViT-B/14 with registers (auto-downloads via torch.hub).
Optional: DINOv3 ViT-B/16 — weights are license-gated; pass the downloaded
checkpoint path via config `backbone.dinov3_weights`.
"""

import torch


class FrozenBackbone(torch.nn.Module):
    def __init__(self, name="dinov2_vitb14_reg", dinov3_weights=None):
        super().__init__()
        self.name = name
        if name.startswith("dinov2"):
            self.model = torch.hub.load("facebookresearch/dinov2", name)
            self.patch = 14
            self.dim = self.model.embed_dim
        elif name.startswith("dinov3"):
            if not dinov3_weights:
                raise RuntimeError(
                    "DINOv3 weights are gated. Download them (see README) and "
                    "set backbone.dinov3_weights in the config.")
            self.model = torch.hub.load(
                "facebookresearch/dinov3", name, weights=dinov3_weights)
            self.patch = 16
            self.dim = self.model.embed_dim
        else:
            raise ValueError(f"unknown backbone {name}")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images):
        """images: (B, 3, S, S) normalized. Returns (B, Ht, Wt, C) tokens."""
        B, _, H, W = images.shape
        ht, wt = H // self.patch, W // self.patch
        out = self.model.forward_features(images)
        tokens = out["x_norm_patchtokens"]          # (B, ht*wt, C)
        return tokens.reshape(B, ht, wt, self.dim)
