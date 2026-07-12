"""Visualize descriptor NN matches for an arbitrary frame pair — same car or
two different cars (cross-instance).

Points are sampled on a grid in image A (no ground truth needed); each is
matched into image B by nearest-neighbour descriptor similarity. Same color =
matched pair. Low-confidence matches (cosine sim below --min-sim) are skipped.

Usage:
  # cross-instance: two different test cars, roughly similar viewpoints
  python scripts/vis_matches.py --ckpt outputs/runs/smoke/ckpt_last.pth \
      --frame-a 0500:frame_00000 --frame-b 0501:frame_00013
  # same car, two views
  python scripts/vis_matches.py --ckpt ... --frame-a 0500:frame_00000 --frame-b 0500:frame_00246
"""

import argparse
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
from src.data.realcar import image_transform

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_frame(image_root, spec, input_res, cache_root):
    car, frame = spec.split(":")
    img = Image.open(os.path.join(image_root, car, frame + ".jpg")).convert("RGB")
    img = geo.prepare_image(img, input_res)
    feat_path = os.path.join(cache_root, car, frame + ".npy")
    tok = None
    if os.path.exists(feat_path):
        tok = torch.from_numpy(np.load(feat_path)).float()
    return img, tok, f"{car}/{frame}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/runs/smoke/ckpt_last.pth")
    ap.add_argument("--frame-a", required=True, help="car_id:frame_name")
    ap.add_argument("--frame-b", required=True, help="car_id:frame_name")
    ap.add_argument("--input-res", type=int, default=518)
    ap.add_argument("--grid-step", type=int, default=5,
                    help="sample every Nth descriptor-grid cell in A")
    ap.add_argument("--min-sim", type=float, default=0.6)
    ap.add_argument("--device", default=None)
    ap.add_argument("--image-root", default="/home/vaibhav/3DRealCars-English")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    from scripts.gate_s3 import load_model
    model, cfg = load_model(os.path.join(REPO, args.ckpt), device)
    cache_root = os.path.join(REPO, cfg.backbone.cache_dir,
                              f"{cfg.backbone.name}_{args.input_res}")

    ims, toks, names = [], [], []
    tx = image_transform()
    backbone = None
    for spec in (args.frame_a, args.frame_b):
        img, tok, name = load_frame(args.image_root, spec, args.input_res,
                                    cache_root)
        if tok is None:  # frame not in cache -> run backbone live
            if backbone is None:
                from src.models.backbone import FrozenBackbone
                backbone = FrozenBackbone(
                    cfg.backbone.name, cfg.backbone.get("dinov3_weights")
                ).to(device)
            tok = backbone(tx(img).unsqueeze(0).to(device)).squeeze(0).cpu()
        ims.append(img)
        toks.append(tok)
        names.append(name)

    with torch.no_grad():
        out = model(torch.stack(toks).to(device))
    desc = F.normalize(out["desc"], p=2, dim=1)
    G = desc.shape[-1]
    S = args.input_res

    # grid-sample source points, NN-match into target
    idx = torch.arange(0, G, args.grid_step)
    vv, uu = torch.meshgrid(idx, idx, indexing="ij")
    uu, vv = uu.reshape(-1), vv.reshape(-1)
    f1 = desc[0][:, vv, uu].t().to(device)                     # (M, C)
    f2 = desc[1].reshape(desc.shape[1], -1)                    # (C, G*G)
    sim = f1 @ f2
    conf, nn = sim.max(dim=1)
    keep = conf > args.min_sim
    uu, vv, nn = uu[keep.cpu()], vv[keep.cpu()], nn[keep].cpu()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(ims[0])
    axes[1].imshow(ims[1])
    colors = plt.cm.hsv(np.linspace(0, 1, max(len(uu), 1)))
    s = S / G
    for k in range(len(uu)):
        axes[0].scatter([float(uu[k]) * s], [float(vv[k]) * s],
                        color=colors[k], s=14)
        axes[1].scatter([float(nn[k] % G) * s], [float(nn[k] // G) * s],
                        color=colors[k], s=14)
    for ax, name in zip(axes, names):
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(f"descriptor NN matches (sim > {args.min_sim}), "
                 f"{int(keep.sum())} shown")
    fig.tight_layout()
    out_path = args.out or os.path.join(
        REPO, "outputs/diagnostics",
        f"matches_{names[0].replace('/', '_')}__{names[1].replace('/', '_')}.jpg")
    fig.savefig(out_path, dpi=120)
    print("saved", out_path)


if __name__ == "__main__":
    main()
