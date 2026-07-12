"""External-benchmark harness (S4): zero-shot eval of a trained checkpoint.

Modes:
  --bench spair     SPair-71k cars keypoint transfer, PCK@0.1 (bbox), read
                    directly from the extracted SPair-71k directory. Subsets:
                    --subset all | viewpoint (viewpoint_variation >= 1;
                    geometry-aware-style subset — the exact TLfR keypoint
                    list can be plugged in later via --kps-file).
  --bench freiburg  Freiburg Static Cars 52: zero-shot relative-azimuth
                    consistency of the orientation head vs the dataset's
                    viewpoint annotations (the annotation SphericalMaps
                    binned; no keypoints ship with this dataset).
  --bench fixture   tiny 3DRealCar-based benchmark in the SPair pair format
                    (built by gate_s4) — proves the code path mechanically.

Usage:
  python scripts/bench_external.py --bench spair --data data/SPair-71k \
      --ckpt outputs/runs/smoke/ckpt_last.pth [--max-pairs 30]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.realcar import image_transform
from src.models.backbone import FrozenBackbone
from src.models.rotation import geodesic_distance

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_RES = 518


def letterbox(img, size=INPUT_RES):
    """Resize longest side to `size`, pad to square. Returns (PIL, scale,
    pad_x, pad_y): orig -> letterboxed via p' = p * scale + pad."""
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (124, 116, 104))
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img, (px, py))
    return canvas, scale, px, py


class Extractor:
    def __init__(self, ckpt_path, device, backbone_name=None, dinov3_weights=None):
        ck = torch.load(ckpt_path, map_location=device)
        from src.utils.config import Cfg
        from src.models.head import OrientationHead
        cfg = Cfg(ck["cfg"])
        self.model = OrientationHead(
            in_dim=cfg.backbone.dim, hidden_dim=cfg.model.hidden_dim,
            num_blocks=cfg.model.num_blocks, num_heads=cfg.model.num_heads,
            descriptor_dim=cfg.model.descriptor_dim,
            descriptor_res=cfg.model.descriptor_res,
            orientation=cfg.model.orientation, film=cfg.model.film).to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.backbone = FrozenBackbone(
            backbone_name or cfg.backbone.name,
            dinov3_weights or cfg.backbone.get("dinov3_weights")).to(device)
        self.device = device
        self.tx = image_transform()

    @torch.no_grad()
    def __call__(self, img_pil):
        """-> dict(desc (C,G,G) L2-normalized, R_pred (3,3))."""
        x = self.tx(img_pil).unsqueeze(0).to(self.device)
        tok = self.backbone(x)
        out = self.model(tok)
        return {"desc": F.normalize(out["desc"].squeeze(0), p=2, dim=0),
                "R_pred": out["R_pred"].squeeze(0)}


def eval_spair(args, ex):
    pair_files = sorted(glob.glob(
        os.path.join(args.data, "PairAnnotation/test/*:car.json")))
    if args.max_pairs:
        pair_files = pair_files[:args.max_pairs]
    img_dir = os.path.join(args.data, "JPEGImages/car")

    errs_alpha = []   # error / (0.1 * max(bbox_wh)), <=1 means correct
    n_pairs = 0
    for pf in pair_files:
        with open(pf) as f:
            p = json.load(f)
        if args.subset == "viewpoint" and p.get("viewpoint_variation", 0) < 1:
            continue
        src = Image.open(os.path.join(img_dir, p["src_imname"])).convert("RGB")
        trg = Image.open(os.path.join(img_dir, p["trg_imname"])).convert("RGB")
        src_lb, ss, sx, sy = letterbox(src)
        trg_lb, ts, tx_, ty = letterbox(trg)
        out_s, out_t = ex(src_lb), ex(trg_lb)
        desc_s, desc_t = out_s["desc"], out_t["desc"]
        G = desc_s.shape[-1]

        src_kps = np.array(p["src_kps"], dtype=np.float64) * ss + [sx, sy]
        trg_kps = np.array(p["trg_kps"], dtype=np.float64)
        # sample source descriptors, NN in target map
        g = torch.tensor(2.0 * src_kps / (INPUT_RES - 1) - 1.0,
                         dtype=torch.float32).reshape(1, 1, -1, 2).to(ex.device)
        sd = F.grid_sample(desc_s.unsqueeze(0), g, mode="bilinear",
                           align_corners=True).squeeze(0).squeeze(1).t()
        sd = F.normalize(sd, p=2, dim=1)
        sim = sd @ desc_t.reshape(desc_t.shape[0], -1)
        nn = sim.argmax(dim=1).cpu().numpy()
        pu = (nn % G) * (INPUT_RES - 1) / (G - 1)
        pv = (nn // G) * (INPUT_RES - 1) / (G - 1)
        # letterbox -> original target image coords
        pu = (pu - tx_) / ts
        pv = (pv - ty) / ts
        bx1, by1, bx2, by2 = p["trg_bndbox"]
        thr = 0.1 * max(bx2 - bx1, by2 - by1)
        e = np.sqrt((pu - trg_kps[:, 0]) ** 2 + (pv - trg_kps[:, 1]) ** 2) / thr
        errs_alpha.extend(e.tolist())
        n_pairs += 1

    errs_alpha = np.array(errs_alpha)
    return {
        "bench": "spair_car", "subset": args.subset,
        "num_pairs": n_pairs, "num_kps": len(errs_alpha),
        "pck@0.1": float((errs_alpha <= 1.0).mean() * 100) if len(errs_alpha) else 0.0,
        "pck@0.05": float((errs_alpha <= 0.5).mean() * 100) if len(errs_alpha) else 0.0,
    }


def eval_freiburg(args, ex):
    from scipy.stats import spearmanr
    annot_files = sorted(glob.glob(os.path.join(args.data, "annotations/*_annot.txt")))
    rng = np.random.default_rng(0)
    preds, gts = [], []
    n_pairs_per_seq = max(1, (args.max_pairs or 40) // max(len(annot_files), 1))
    for af in annot_files:
        rows = []
        with open(af) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 6:
                    continue
                relpath, bbox, az = parts[0], list(map(float, parts[1:5])), float(parts[5])
                img_path = os.path.join(args.data, relpath)
                for ext in (".png", ".jpg"):
                    cand = os.path.splitext(img_path)[0] + ext
                    if os.path.exists(cand):
                        rows.append((cand, bbox, az))
                        break
        if len(rows) < 2:
            continue
        for _ in range(n_pairs_per_seq):
            i, j = rng.choice(len(rows), 2, replace=False)
            outs = []
            for k in (i, j):
                path, bbox, az = rows[k]
                img = Image.open(path).convert("RGB")
                x1, y1, x2, y2 = bbox
                m = 0.15 * max(x2 - x1, y2 - y1)  # margin around the car
                img = img.crop((max(x1 - m, 0), max(y1 - m, 0),
                                min(x2 + m, img.width), min(y2 + m, img.height)))
                lb, _, _, _ = letterbox(img)
                outs.append(ex(lb)["R_pred"])
            gt = abs((rows[i][2] - rows[j][2] + 180.0) % 360.0 - 180.0)
            pred = float(torch.rad2deg(geodesic_distance(
                outs[0].unsqueeze(0), outs[1].unsqueeze(0))))
            preds.append(pred)
            gts.append(gt)
    rho = spearmanr(gts, preds).statistic if len(gts) > 5 else float("nan")
    return {
        "bench": "freiburg_cars52_relative_azimuth",
        "num_pairs": len(gts),
        "spearman_pred_vs_gt_angle": float(rho if rho == rho else 0.0),
        "median_abs_err_deg": float(np.median(np.abs(np.array(preds) - np.array(gts)))) if gts else -1.0,
        "note": "no keypoints ship with this dataset; keypoint PCK requires "
                "external annotations (see README)",
    }


def eval_fixture(args, ex):
    """Fixture dir uses the SPair pair-json format under pairs/ + images/."""
    pair_files = sorted(glob.glob(os.path.join(args.data, "pairs/*.json")))
    class A:  # reuse eval_spair by mimicking its layout expectations
        pass
    errs = []
    n = 0
    for pf in pair_files:
        with open(pf) as f:
            p = json.load(f)
        src = Image.open(os.path.join(args.data, "images", p["src_imname"])).convert("RGB")
        trg = Image.open(os.path.join(args.data, "images", p["trg_imname"])).convert("RGB")
        src_lb, ss, sx, sy = letterbox(src)
        trg_lb, ts, tx_, ty = letterbox(trg)
        out_s, out_t = ex(src_lb), ex(trg_lb)
        desc_s, desc_t = out_s["desc"], out_t["desc"]
        G = desc_s.shape[-1]
        src_kps = np.array(p["src_kps"], dtype=np.float64) * ss + [sx, sy]
        trg_kps = np.array(p["trg_kps"], dtype=np.float64)
        g = torch.tensor(2.0 * src_kps / (INPUT_RES - 1) - 1.0,
                         dtype=torch.float32).reshape(1, 1, -1, 2).to(ex.device)
        sd = F.grid_sample(desc_s.unsqueeze(0), g, mode="bilinear",
                           align_corners=True).squeeze(0).squeeze(1).t()
        sd = F.normalize(sd, p=2, dim=1)
        nn = (sd @ desc_t.reshape(desc_t.shape[0], -1)).argmax(dim=1).cpu().numpy()
        pu = ((nn % G) * (INPUT_RES - 1) / (G - 1) - tx_) / ts
        pv = ((nn // G) * (INPUT_RES - 1) / (G - 1) - ty) / ts
        bx1, by1, bx2, by2 = p["trg_bndbox"]
        thr = 0.1 * max(bx2 - bx1, by2 - by1)
        e = np.sqrt((pu - trg_kps[:, 0]) ** 2 + (pv - trg_kps[:, 1]) ** 2) / thr
        errs.extend(e.tolist())
        n += 1
    errs = np.array(errs)
    return {"bench": "fixture", "num_pairs": n, "num_kps": len(errs),
            "pck@0.1": float((errs <= 1.0).mean() * 100) if len(errs) else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True,
                    choices=["spair", "freiburg", "fixture"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subset", default="all", choices=["all", "viewpoint"])
    ap.add_argument("--max-pairs", type=int, default=30)  # TOBECHANGED 0=all (3090)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.max_pairs == 0:
        args.max_pairs = None

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ex = Extractor(os.path.join(REPO, args.ckpt), device)

    fn = {"spair": eval_spair, "freiburg": eval_freiburg,
          "fixture": eval_fixture}[args.bench]
    res = fn(args, ex)
    print(json.dumps(res, indent=1))

    out = args.out or os.path.join(
        REPO, "outputs/paper", f"bench_{args.bench}_{args.subset}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
