"""Cache frozen-backbone patch tokens to disk as fp16 .npy files.

The backbone never trains, so this is the single biggest speed/VRAM lever.
Cache layout: <cache_dir>/<backbone>_<res>/<car_id>/<frame>.npy  (Ht, Wt, C) fp16

Usage:
    python scripts/cache_features.py --cars dev_smoke dev_test_smoke \
        [--max-frames 80] [--device cpu]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.realcar import RealCarFrames
from src.models.backbone import FrozenBackbone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cache_dir_for(cache_root, backbone_name, input_res):
    return os.path.join(cache_root, f"{backbone_name}_{input_res}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cars", nargs="+", default=["dev_smoke"],
                    help="split keys or explicit car ids")
    ap.add_argument("--backbone", default="dinov2_vitb14_reg")
    ap.add_argument("--dinov3-weights", default=None)
    ap.add_argument("--input-res", type=int, default=518)  # TOBECHANGED 896 (3090)
    ap.add_argument("--max-frames", type=int, default=80)  # TOBECHANGED none: pass 0 (3090)
    ap.add_argument("--batch", type=int, default=2)        # TOBECHANGED 16 (3090)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--image-root", default="/home/vaibhav/3DRealCars-English")
    ap.add_argument("--cache-root", default=os.path.join(REPO, "outputs/cache"))
    args = ap.parse_args()

    with open(os.path.join(REPO, "configs/split.json")) as f:
        split = json.load(f)
    car_ids = []
    for c in args.cars:
        car_ids.extend(split[c] if c in split else [c])
    car_ids = list(dict.fromkeys(car_ids))

    out_root = cache_dir_for(args.cache_root, args.backbone, args.input_res)
    os.makedirs(out_root, exist_ok=True)

    max_frames = args.max_frames if args.max_frames > 0 else None
    ds = RealCarFrames(args.image_root, car_ids, input_res=args.input_res,
                       max_frames_per_car=max_frames)
    todo = [i for i, r in enumerate(ds.records)
            if not os.path.exists(os.path.join(out_root, r.car_id, r.name + ".npy"))]
    print(f"{len(ds)} frames, {len(todo)} to cache -> {out_root}")
    if not todo:
        return

    bb = FrozenBackbone(args.backbone, args.dinov3_weights).to(args.device)
    torch.set_num_threads(os.cpu_count() or 8)

    for k in tqdm(range(0, len(todo), args.batch), ncols=80):
        idxs = todo[k:k + args.batch]
        ims = torch.stack([ds[i]["image"] for i in idxs]).to(args.device)
        tok = bb(ims).cpu().numpy().astype(np.float16)
        for b, i in enumerate(idxs):
            rec = ds.records[i]
            os.makedirs(os.path.join(out_root, rec.car_id), exist_ok=True)
            np.save(os.path.join(out_root, rec.car_id, rec.name + ".npy"), tok[b])


if __name__ == "__main__":
    main()
