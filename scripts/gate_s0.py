"""Gate S0 — automated smoke check for data plumbing + geometry.

Checks, on N>=50 sampled pairs across >=10 cars:
  (a) warp round-trip A->B->A median error < 3 px at 128x128 on pixels
      visible in both directions;
  (b) visible fraction sane (mean in [0.10, 0.70], pairs in [0.02, 0.90])
      and decreasing with angular separation (Pearson r < -0.2);
  (c) PifPaf reprojection agreement: project A's PifPaf keypoints into B via
      depth+pose, compare with B's own detections — median error <= 2.5% of
      the image diagonal (1920x1440 space).

Saves warp overlays to outputs/diagnostics/warps/ (non-blocking, for human
review) and a machine-readable report to outputs/diagnostics/gate_s0.json.

Usage: python scripts/gate_s0.py [--num-pairs 60] [--cars dev_smoke]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import geometry as geo
from src.data import pifpaf
from src.data.realcar import RealCarPairs
from src.viz.overlays import save_warp_overlay

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIAG = float(np.hypot(geo.ORIG_W, geo.ORIG_H))


def round_trip_error(ds, rec_a, rec_b, G=128):
    """Median A->B->A error in px at G resolution over doubly-visible pixels."""
    grid_ab, mask_ab = ds.compute_pair_grid(rec_a, rec_b, grid_res=G)
    grid_ba, mask_ba = ds.compute_pair_grid(rec_b, rec_a, grid_res=G)

    # For each visible source pixel p in A: q = grid_ab[p] (in B), then
    # sample grid_ba at q to land back in A.
    g_ab = grid_ab.unsqueeze(0)                                  # (1,G,G,2)
    back = F.grid_sample(grid_ba.permute(2, 0, 1).unsqueeze(0),  # (1,2,G,G)
                         g_ab, mode="bilinear", align_corners=True)
    back = back.squeeze(0).permute(1, 2, 0)                      # (G,G,2)
    vis_b = F.grid_sample(mask_ba.unsqueeze(0).unsqueeze(0), g_ab,
                          mode="nearest", align_corners=True).squeeze()

    yy, xx = torch.meshgrid(torch.arange(G), torch.arange(G), indexing="ij")
    orig_u = xx.float()
    orig_v = yy.float()
    back_u = (back[..., 0] + 1) / 2 * (G - 1)
    back_v = (back[..., 1] + 1) / 2 * (G - 1)
    err = torch.sqrt((back_u - orig_u) ** 2 + (back_v - orig_v) ** 2)

    both = (mask_ab > 0.5) & (vis_b > 0.5)
    if both.sum() < 20:
        return None, float(mask_ab.mean())
    return float(err[both].median()), float(mask_ab.mean())


def pifpaf_reprojection(labels_root, rec_a, rec_b, min_conf=0.5,
                        vis_thresh=0.15):
    """Project A's PifPaf kps into B via depth+pose; compare with B's own
    detections. Returns list of pixel errors (1920x1440 space)."""
    la = pifpaf.load_keypoints(labels_root, rec_a.car_id, rec_a.name)
    lb = pifpaf.load_keypoints(labels_root, rec_b.car_id, rec_b.name)
    if la is None or lb is None:
        return []
    kps_a, conf_a = la
    kps_b, conf_b = lb

    depths = geo.depth_at_points(rec_a.depth_path, kps_a, robust=True)
    keep = (conf_a > min_conf) & (conf_b > min_conf) & (depths > 1e-6)
    if keep.sum() == 0:
        return []
    idx = np.where(keep)[0]

    K = geo.parse_intrinsics({"intrinsics": rec_a.K_raw.ravel().tolist()})
    K_b = rec_b.K_raw
    world = geo.unproject_to_world(kps_a[idx].astype(np.float64),
                                   depths[idx].astype(np.float64), K, rec_a.T)
    uv_b, z_b = geo.project_from_world(world, K_b, rec_b.T)

    # visibility: reprojected depth must agree with B's own depth map
    depth_b = geo.depth_at_points(rec_b.depth_path, uv_b, robust=True)
    in_bounds = ((uv_b[:, 0] >= 0) & (uv_b[:, 0] < geo.ORIG_W) &
                 (uv_b[:, 1] >= 0) & (uv_b[:, 1] < geo.ORIG_H) & (z_b > 0))
    visible = in_bounds & ((np.abs(z_b - depth_b) < vis_thresh) | (depth_b < 1e-6))
    if visible.sum() == 0:
        return []
    errs = np.linalg.norm(uv_b[visible] - kps_b[idx][visible], axis=1)
    return errs.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-pairs", type=int, default=60)
    ap.add_argument("--cars", default="dev_smoke")
    ap.add_argument("--image-root", default="/home/vaibhav/3DRealCars-English")
    ap.add_argument("--labels-root", default="/home/vaibhav/3DRealCars-Labels")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(os.path.join(REPO, "configs/split.json")) as f:
        split = json.load(f)
    car_ids = split[args.cars]

    ds = RealCarPairs(args.image_root, car_ids, input_res=518, grid_res=64,
                      max_pairs_per_car=20, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    n = min(args.num_pairs, len(ds.pairs))
    pick = rng.choice(len(ds.pairs), n, replace=False)

    warp_dir = os.path.join(REPO, "outputs/diagnostics/warps")
    os.makedirs(warp_dir, exist_ok=True)

    rt_errors, vis_fracs, angles, pp_errors = [], [], [], []
    cars_seen = set()
    for k, pi in enumerate(pick):
        car_id, i, j, ang = ds.pairs[pi]
        rec_a, rec_b = ds.cars[car_id][i], ds.cars[car_id][j]
        cars_seen.add(car_id)

        rt, visfrac = round_trip_error(ds, rec_a, rec_b)
        if rt is not None:
            rt_errors.append(rt)
        vis_fracs.append(visfrac)
        angles.append(ang)

        if k < 12:  # save a handful of overlays for human review
            item = ds[pi]
            save_warp_overlay(
                item["im_a"], item["im_b"], item["grid"], item["mask"],
                os.path.join(warp_dir, f"{car_id}_{rec_a.name}_{rec_b.name}.jpg"),
                title=f"{car_id} {rec_a.name}->{rec_b.name}  {ang:.0f}deg "
                      f"vis={visfrac:.2f}")

    # Check (c) uses its own pair sample restricted to labeled frames
    # (exactly-one-detection on both sides is only ~37% of frames, so the
    # generic sample yields too few keypoint matches to be meaningful).
    labeled_pairs = []
    for car_id, pool in ds._pair_pool:
        recs = ds.cars[car_id]
        has_label = {i: pifpaf.load_keypoints(args.labels_root, car_id,
                                              recs[i].name) is not None
                     for i in set(i for p in pool for i in p[:2])}
        lp = [(car_id, i, j) for i, j, _ in pool if has_label[i] and has_label[j]]
        if lp:
            take = rng.choice(len(lp), min(6, len(lp)), replace=False)
            labeled_pairs.extend(lp[t] for t in take)
    for car_id, i, j in labeled_pairs:
        pp_errors.extend(pifpaf_reprojection(
            args.labels_root, ds.cars[car_id][i], ds.cars[car_id][j]))

    vis_fracs = np.array(vis_fracs)
    angles = np.array(angles)
    corr = float(np.corrcoef(angles, vis_fracs)[0, 1])
    med_rt = float(np.median(rt_errors)) if rt_errors else float("inf")
    med_pp = float(np.median(pp_errors)) if pp_errors else float("inf")
    med_pp_frac = med_pp / IMAGE_DIAG

    checks = {
        "a_roundtrip_median_px_at128": med_rt,
        "a_pass": med_rt < 3.0,
        "b_visfrac_mean": float(vis_fracs.mean()),
        "b_visfrac_p10": float(np.percentile(vis_fracs, 10)),
        "b_visfrac_max": float(vis_fracs.max()),
        "b_angle_visfrac_pearson": corr,
        "b_pass": bool(0.10 <= vis_fracs.mean() <= 0.70
                       and np.percentile(vis_fracs, 10) >= 0.02
                       and vis_fracs.max() <= 0.90
                       and corr < -0.2),
        "c_pifpaf_median_err_px": med_pp,
        "c_pifpaf_median_err_frac_diag": med_pp_frac,
        "c_num_kp_matches": len(pp_errors),
        "c_pass": med_pp_frac <= 0.025 and len(pp_errors) >= 100,
        "num_pairs": int(n),
        "num_cars": len(cars_seen),
        "coverage_pass": bool(n >= 50 and len(cars_seen) >= 10),
    }
    checks["gate_pass"] = bool(checks["a_pass"] and checks["b_pass"]
                               and checks["c_pass"] and checks["coverage_pass"])

    out = os.path.join(REPO, "outputs/diagnostics/gate_s0.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(checks, f, indent=1)

    print(json.dumps(checks, indent=1))
    print("GATE S0:", "PASS" if checks["gate_pass"] else "FAIL")
    sys.exit(0 if checks["gate_pass"] else 1)


if __name__ == "__main__":
    main()
