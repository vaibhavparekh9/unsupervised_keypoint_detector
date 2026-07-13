"""Visualize canonical sphere coordinates as color overlays.

Each pixel's predicted canonical coordinate (unit vector on S^2) is mapped to
RGB. If the method works: (1) the same car part keeps the same color as the
camera orbits (view consistency), (2) different cars share the color scheme
(canonical-frame agreement), (3) left and right sides have DIFFERENT colors
(symmetry broken) — all three visible at a glance.

Usage:
  # filmstrip of one car across its walkaround + a second car for comparison
  python scripts/vis_sphere.py --ckpt outputs/runs/full/ckpt_last.pth \
      --cars 0400 0450 --num-frames 6
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import geometry as geo
from src.data.realcar import scan_car, image_transform

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sphere_to_rgb(sph):
    """(3, H, W) unit vectors -> (H, W, 3) colors in [0,1]."""
    return ((sph.permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0).clip(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/runs/full/ckpt_last.pth")
    ap.add_argument("--cars", nargs="+", required=True)
    ap.add_argument("--num-frames", type=int, default=6)
    ap.add_argument("--input-res", type=int, default=518)
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--device", default=None)
    ap.add_argument("--image-root", default="/home/vaibhav/3DRealCars-English")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    from scripts.gate_s3 import load_model
    model, cfg = load_model(os.path.join(REPO, args.ckpt), device)
    cache_root = os.path.join(REPO, cfg.backbone.cache_dir,
                              f"{cfg.backbone.name}_{args.input_res}")

    backbone = None
    tx = image_transform()
    S = args.input_res

    fig, axes = plt.subplots(len(args.cars), args.num_frames,
                             figsize=(3 * args.num_frames, 3 * len(args.cars)))
    axes = np.atleast_2d(axes)
    for r, car in enumerate(args.cars):
        recs = scan_car(args.image_root, car)
        if not recs:
            sys.exit(f"no frames for car {car}")
        step = max(len(recs) // args.num_frames, 1)
        picks = recs[::step][:args.num_frames]
        for c, rec in enumerate(picks):
            img = geo.prepare_image(
                Image.open(rec.img_path).convert("RGB"), S)
            feat_path = os.path.join(cache_root, car, rec.name + ".npy")
            if os.path.exists(feat_path):
                tok = torch.from_numpy(np.load(feat_path)).float().unsqueeze(0)
            else:
                if backbone is None:
                    from src.models.backbone import FrozenBackbone
                    backbone = FrozenBackbone(
                        cfg.backbone.name,
                        cfg.backbone.get("dinov3_weights")).to(device)
                tok = backbone(tx(img).unsqueeze(0).to(device)).cpu()
            with torch.no_grad():
                sph = model(tok.to(device))["sphere"].squeeze(0)
            rgb = sphere_to_rgb(F.interpolate(
                sph.unsqueeze(0), size=(S, S), mode="bilinear",
                align_corners=True).squeeze(0))
            base = np.asarray(img, dtype=np.float32) / 255.0
            overlay = (1 - args.alpha) * base + args.alpha * rgb
            ax = axes[r, c]
            ax.imshow(overlay)
            ax.set_title(f"{car}/{rec.name}", fontsize=7)
            ax.axis("off")
    fig.suptitle("canonical sphere coordinates as RGB — same part should "
                 "keep its color across views AND across cars; left/right "
                 "should differ", fontsize=10)
    fig.tight_layout()
    out = args.out or os.path.join(
        REPO, "outputs/diagnostics",
        f"sphere_{'_'.join(args.cars)}.jpg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    main()
