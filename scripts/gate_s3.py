"""Gate S3 — evaluation suite runs end-to-end on a smoke checkpoint.

(a) thesis-protocol PCK@10 intra/cross at 64x64 on held-out cars (model
    descriptors);
(b) DVE-protocol light regressor (50 virtual keypoints -> linear regressor
    to PifPaf keypoints, frozen embeddings);
(c) unsupervised landmark discovery: spherical k-means on canonical coords ->
    cross-view detection consistency;
(d) symmetry-confusion re-run with the trained model (before/after figure).

Gate = mechanical correctness: all four emit tables/figures to outputs/paper/.
Absolute numbers are meaningless at smoke scale.

Usage: python scripts/gate_s3.py --ckpt outputs/runs/smoke/ckpt_last.pth
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.config import Cfg
from src.models.head import OrientationHead
from src.eval.probe import (BackboneDescriptors, ModelDescriptors,
                            load_labeled_frames, run_probe)
from src.eval.regressor import train_regressor, eval_regressor
from src.eval.discovery import spherical_kmeans, landmark_consistency, \
    detect_landmarks

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    mc = Cfg(ck["cfg"]).model
    bc = Cfg(ck["cfg"]).backbone
    model = OrientationHead(
        in_dim=bc.dim, hidden_dim=mc.hidden_dim, num_blocks=mc.num_blocks,
        num_heads=mc.num_heads, descriptor_dim=mc.descriptor_dim,
        descriptor_res=mc.descriptor_res, orientation=mc.orientation,
        film=mc.film).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, Cfg(ck["cfg"])


def collect_desc_maps(provider, car_frames, input_res, feature_res=64,
                      max_frames_per_car=25):
    """Stack descriptor maps + normalized kps for regressor training.
    max_frames_per_car bounds memory (1 MB per frame at 64x64x64)."""
    descs, kps, vis = [], [], []
    for frames in car_frames.values():
        for fr in frames[:max_frames_per_car]:
            descs.append(provider(fr))
            kps.append(fr["kps"] / (input_res - 1) * 2.0 - 1.0)
            vis.append(fr["visible"])
    return (torch.stack(descs), torch.tensor(np.stack(kps)),
            torch.tensor(np.stack(vis)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/runs/smoke/ckpt_last.pth")
    ap.add_argument("--cars-train", default="dev_smoke")   # TOBECHANGED train_pool (3090)
    ap.add_argument("--cars-test", default="dev_test_smoke")  # TOBECHANGED test (3090)
    ap.add_argument("--input-res", type=int, default=518)
    ap.add_argument("--device", default=None)
    # large enough that the high-azimuth confusion bins have usable n
    ap.add_argument("--num-intra", type=int, default=800)
    ap.add_argument("--num-cross", type=int, default=400)
    ap.add_argument("--k-landmarks", type=int, default=16)
    ap.add_argument("--image-root", default="/home/vaibhav/3DRealCars-English")
    ap.add_argument("--labels-root", default="/home/vaibhav/3DRealCars-Labels")
    ap.add_argument("--tag", default=None,
                    help="suffix for output files (default: run-dir name of "
                         "the checkpoint, e.g. 'full', 'no_film')")
    args = ap.parse_args()

    tag = args.tag or os.path.basename(
        os.path.dirname(os.path.join(REPO, args.ckpt))) or "run"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(os.path.join(REPO, args.ckpt), device)
    cache_root = os.path.join(REPO, cfg.backbone.cache_dir,
                              f"{cfg.backbone.name}_{args.input_res}")
    paper_dir = os.path.join(REPO, "outputs/paper")
    os.makedirs(paper_dir, exist_ok=True)

    with open(os.path.join(REPO, "configs/split.json")) as f:
        split = json.load(f)

    test_frames = load_labeled_frames(
        args.image_root, args.labels_root, split[args.cars_test],
        args.input_res, cache_root)
    train_frames = load_labeled_frames(
        args.image_root, args.labels_root, split[args.cars_train],
        args.input_res, cache_root)
    print(f"test: {len(test_frames)} cars; train: {len(train_frames)} cars")

    status = {}
    provider = ModelDescriptors(model, device)

    # ---------- (a) thesis-protocol PCK ----------
    try:
        results, conf_curve, group_rates, _ = run_probe(
            provider, test_frames, args.input_res,
            args.num_intra, args.num_cross)
        with open(os.path.join(paper_dir, f"model_pck_{tag}.json"), "w") as f:
            json.dump(results, f, indent=1)
        with open(os.path.join(paper_dir, f"model_pck_{tag}.md"), "w") as f:
            f.write("| method | intra PCK@10 | cross PCK@10 |\n|---|---|---|\n")
            f.write(f"| ours ({tag}) "
                    f"| {results['intra']['pck@10_feat64']:.1f} "
                    f"| {results['cross']['pck@10_feat64']:.1f} |\n")
        status["a_pck"] = {"ok": True, **{k: v["pck@10_feat64"]
                                          for k, v in results.items()}}
    except Exception:
        traceback.print_exc()
        status["a_pck"] = {"ok": False}

    # ---------- (d) symmetry confusion with trained model ----------
    try:
        from scripts.gate_s1 import save_confusion_plot
        with open(os.path.join(paper_dir, f"symmetry_confusion_model_{tag}.json"),
                  "w") as f:
            json.dump({"confusion_vs_azimuth": conf_curve,
                       "confusion_by_group": group_rates}, f, indent=1)
        save_confusion_plot(conf_curve, f"ours ({tag})",
                            os.path.join(paper_dir,
                                         f"symmetry_confusion_model_{tag}.png"))
        status["d_confusion"] = {"ok": True}
    except Exception:
        traceback.print_exc()
        status["d_confusion"] = {"ok": False}

    # ---------- (b) DVE-protocol regressor ----------
    try:
        d_tr, k_tr, v_tr = collect_desc_maps(provider, train_frames,
                                             args.input_res)
        d_te, k_te, v_te = collect_desc_maps(provider, test_frames,
                                             args.input_res)
        reg = train_regressor(d_tr, k_tr, v_tr, steps=400, device=device)
        res_tr = eval_regressor(reg, d_tr, k_tr, v_tr, device=device)
        res_te = eval_regressor(reg, d_te, k_te, v_te, device=device)
        with open(os.path.join(paper_dir, f"regressor_pck_{tag}.json"), "w") as f:
            json.dump({"train": res_tr, "test": res_te}, f, indent=1)
        status["b_regressor"] = {"ok": True,
                                 "test_pck10": res_te["pck@10_feat64"]}
    except Exception:
        traceback.print_exc()
        status["b_regressor"] = {"ok": False}

    # ---------- (c) landmark discovery ----------
    try:
        from scripts.train import build_dataset
        vecs = []
        with torch.no_grad():
            for frames in list(test_frames.values()):
                for fr in frames[:10]:
                    tok = torch.from_numpy(
                        np.load(fr["feat_path"])).float().unsqueeze(0).to(device)
                    sph = model(tok)["sphere"].squeeze(0)
                    v = sph.reshape(3, -1).t()
                    sel = torch.randperm(v.shape[0])[:200]
                    vecs.append(v[sel].cpu())
        vecs = torch.cat(vecs)
        centroids = spherical_kmeans(vecs, args.k_landmarks)

        pairs_ds = build_dataset(cfg, args.cars_test, seed=0, max_pairs=5)
        cons = landmark_consistency(model, pairs_ds, centroids.to(device),
                                    device)
        with open(os.path.join(paper_dir, f"landmark_discovery_{tag}.json"), "w") as f:
            json.dump({"k": args.k_landmarks, **cons}, f, indent=1)

        # qualitative: landmarks on 4 frames of one test car
        from PIL import Image
        from src.data import geometry as geo
        car0 = list(test_frames.keys())[0]
        frames = test_frames[car0][:4]
        fig, axes = plt.subplots(1, len(frames), figsize=(4 * len(frames), 4))
        for ax, fr in zip(np.atleast_1d(axes), frames):
            tok = torch.from_numpy(
                np.load(fr["feat_path"])).float().unsqueeze(0).to(device)
            sph = model(tok)["sphere"].squeeze(0)
            uv, sim = detect_landmarks(sph, centroids.to(device))
            G = sph.shape[-1]
            img = geo.prepare_image(
                Image.open(fr["rec"].img_path).convert("RGB"), args.input_res)
            ax.imshow(img)
            s = args.input_res / G
            colors = plt.cm.tab20(np.linspace(0, 1, len(uv)))
            for k in range(len(uv)):
                if sim[k] > 0.9:
                    ax.scatter([float(uv[k, 0]) * s], [float(uv[k, 1]) * s],
                               color=colors[k], s=40)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(paper_dir, f"landmarks_qualitative_{tag}.jpg"),
                    dpi=110)
        plt.close(fig)
        status["c_discovery"] = {"ok": True, **cons}
    except Exception:
        traceback.print_exc()
        status["c_discovery"] = {"ok": False}

    gate_pass = all(v.get("ok") for v in status.values())
    report = {"status": status, "gate_pass": bool(gate_pass)}
    os.makedirs(os.path.join(REPO, "outputs/diagnostics"), exist_ok=True)
    with open(os.path.join(REPO, f"outputs/diagnostics/gate_s3_{tag}.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    print("GATE S3:", "PASS" if gate_pass else "FAIL")
    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
