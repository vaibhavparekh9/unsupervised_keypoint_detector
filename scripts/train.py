"""Config-driven training loop (head only; backbone features come from cache).

Usage:
    python scripts/train.py --config configs/base.yaml
    python scripts/train.py --config configs/base.yaml -o train.total_steps=100
    python scripts/train.py --config configs/base.yaml --resume outputs/runs/base/ckpt_last.pth
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.data.realcar import RealCarPairs
from src.models.head import OrientationHead
from src.losses.correspondence import (dense_correlation_loss,
                                       dense_correlation_loss_dve,
                                       warped_consistency_loss)
from src.losses.orientation import (relative_orientation_loss,
                                    relative_azimuth_loss)
from src.losses.cross_instance import pseudo_match_loss
from src.eval.validation import validate

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_dataset(cfg, cars_key, seed=0, max_pairs=None):
    with open(os.path.join(REPO, cfg.data.split_file)) as f:
        split = json.load(f)
    if cars_key in split:
        car_ids = split[cars_key]
    else:  # explicit id or comma-separated ids (used by the overfit gate)
        car_ids = cars_key.split(",")
    cache_dir = os.path.join(
        REPO, cfg.backbone.cache_dir,
        f"{cfg.backbone.name}_{cfg.data.input_res}")
    return RealCarPairs(
        cfg.data.image_root, car_ids,
        input_res=cfg.data.input_res, grid_res=cfg.data.grid_res,
        theta_min=cfg.data.theta_min, theta_max=cfg.data.theta_max,
        occlusion_thresh=cfg.data.occlusion_thresh,
        depth_scale=cfg.data.depth_scale,
        motion_quality_thresh=cfg.data.motion_quality_thresh,
        max_frames_per_car=cfg.data.max_frames_per_car,
        max_pairs_per_car=max_pairs or cfg.data.max_pairs_per_car,
        return_images=False, feature_cache_dir=cache_dir, seed=seed)


def compute_losses(cfg, model, batch, device):
    feat_a = batch["feat_a"].to(device)
    feat_b = batch["feat_b"].to(device)
    grid = batch["grid"].to(device)
    if cfg.loss.visibility_masking:
        mask = batch["mask"].to(device)
    else:  # ablation: in-bounds pixels only (grid is zeroed where invalid,
        # so fall back to all-ones — the thesis showed this is much worse)
        mask = torch.ones_like(batch["mask"]).to(device)

    B = feat_a.shape[0]
    out = model(torch.cat([feat_a, feat_b], dim=0))
    out_a = {k: (v[:B] if torch.is_tensor(v) else v) for k, v in out.items()}
    out_b = {k: (v[B:] if torch.is_tensor(v) else v) for k, v in out.items()}

    losses = {}
    # 1. descriptor correspondence (optionally DVE exchange through aux car)
    if cfg.loss.cross_instance == "exchange" and B > 1:
        aux = out_a["desc"].roll(1, dims=0)
        losses["corr"] = dense_correlation_loss_dve(
            out_a["desc"], out_b["desc"], aux, grid, mask,
            pow_=cfg.loss.corr_pow, temp=cfg.loss.corr_temp,
            normalize=cfg.loss.corr_normalize)
    else:
        losses["corr"] = dense_correlation_loss(
            out_a["desc"], out_b["desc"], grid, mask,
            pow_=cfg.loss.corr_pow, temp=cfg.loss.corr_temp,
            normalize=cfg.loss.corr_normalize)

    # 2. relative orientation consistency
    if cfg.model.orientation == "azimuth":
        losses["orient"] = relative_azimuth_loss(
            out_a["azim"], out_b["azim"], batch["rel_azimuth"].to(device))
    else:
        losses["orient"] = relative_orientation_loss(
            out_a["R_pred"], out_b["R_pred"], batch["R_rel"].to(device))

    # 3. canonical sphere: cross-view consistency + matching (anti-collapse:
    #    sphere coords must *discriminate* correspondence, not just agree)
    losses["sphere"] = (
        warped_consistency_loss(out_a["sphere"], out_b["sphere"], grid, mask)
        + dense_correlation_loss(
            out_a["sphere"], out_b["sphere"], grid, mask,
            pow_=cfg.loss.corr_pow, temp=cfg.loss.sphere_temp,
            normalize=True))

    # 4. cross-instance canonical alignment (pseudo-matches)
    if cfg.loss.cross_instance == "pseudo" and B > 1:
        losses["cross"] = pseudo_match_loss(
            feat_a, out_a["sphere"], out_a["R_pred"], batch["car_idx"],
            sim_thresh=cfg.loss.pseudo_sim_thresh)
    else:
        losses["cross"] = torch.zeros((), device=device)

    total = (cfg.loss.w_corr * losses["corr"]
             + cfg.loss.w_orient * losses["orient"]
             + cfg.loss.w_sphere * losses["sphere"]
             + cfg.loss.w_cross * losses["cross"])
    return total, {k: float(v) for k, v in losses.items()}


def save_match_vis(cfg, model, ds, device, out_path):
    """Descriptor NN-match overlay on a fixed pair (human review)."""
    import torch.nn.functional as F
    from src.viz.overlays import denorm_image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from src.data import geometry as geo
    from src.data.realcar import image_transform

    item = ds[0]
    car_id, i, j, _ = ds.pairs[0]
    rec_a, rec_b = ds.cars[car_id][i], ds.cars[car_id][j]
    tx = image_transform()
    im_a = tx(geo.prepare_image(Image.open(rec_a.img_path).convert("RGB"), 518))
    im_b = tx(geo.prepare_image(Image.open(rec_b.img_path).convert("RGB"), 518))

    with torch.no_grad():
        out = model(torch.stack([item["feat_a"], item["feat_b"]]).to(device))
    desc_a, desc_b = out["desc"][0], out["desc"][1]
    G = desc_a.shape[-1]
    vis = torch.nonzero(item["mask"] > 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(denorm_image(im_a))
    axes[1].imshow(denorm_image(im_b))
    if len(vis) >= 10:
        sel = vis[torch.randperm(len(vis))[:40]]
        f1 = F.normalize(desc_a[:, sel[:, 0], sel[:, 1]], dim=0)
        f2 = F.normalize(desc_b.reshape(desc_b.shape[0], -1), dim=0)
        nn = (f1.t() @ f2).argmax(dim=1).cpu()
        colors = plt.cm.hsv(np.linspace(0, 1, len(sel)))
        s = 518 / G
        for n, c in enumerate(colors):
            axes[0].scatter([sel[n, 1] * s], [sel[n, 0] * s], color=c, s=12)
            axes[1].scatter([(nn[n] % G) * s], [(nn[n] // G) * s], color=c, s=12)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("-o", "--override", action="append", default=[])
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    device = torch.device(args.device)
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    out_dir = os.path.join(REPO, cfg.train.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "vis"), exist_ok=True)
    with open(os.path.join(out_dir, "config_used.yaml"), "w") as f:
        import yaml
        yaml.safe_dump(json.loads(json.dumps(cfg)), f)

    train_ds = build_dataset(cfg, cfg.data.train_cars, seed=cfg.train.seed)
    val_ds = build_dataset(cfg, cfg.data.test_cars, seed=0, max_pairs=10)
    print(f"train pairs: {len(train_ds)}  val pairs: {len(val_ds)}")

    loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                        num_workers=cfg.data.num_workers, drop_last=True,
                        persistent_workers=cfg.data.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size,
                            num_workers=0)

    model = OrientationHead(
        in_dim=cfg.backbone.dim, hidden_dim=cfg.model.hidden_dim,
        num_blocks=cfg.model.num_blocks, num_heads=cfg.model.num_heads,
        descriptor_dim=cfg.model.descriptor_dim,
        descriptor_res=cfg.model.descriptor_res,
        orientation=cfg.model.orientation, film=cfg.model.film).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"head params: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)

    def lr_lambda(step):
        if step < cfg.train.warmup_steps:
            return step / max(cfg.train.warmup_steps, 1)
        p = (step - cfg.train.warmup_steps) / max(
            cfg.train.total_steps - cfg.train.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step = ck["step"]
        print(f"resumed from {args.resume} at step {step}")

    log_path = os.path.join(out_dir, "log.jsonl")
    log_f = open(log_path, "a")

    def save_ckpt(name):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step,
                    "cfg": json.loads(json.dumps(cfg))},
                   os.path.join(out_dir, name))

    model.train()
    epoch = 0
    t0 = time.time()
    running = {}
    while step < cfg.train.total_steps:
        train_ds.resample_pairs(seed=cfg.train.seed + epoch)
        for batch in loader:
            if step >= cfg.train.total_steps:
                break
            with torch.autocast(device_type=device.type, enabled=use_amp):
                total, parts = compute_losses(cfg, model, batch, device)
            if not torch.isfinite(total):
                raise RuntimeError(f"non-finite loss at step {step}: {parts}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1

            for k, v in {**parts, "total": float(total)}.items():
                running[k] = running.get(k, 0.0) + v
            if step % cfg.train.log_every == 0:
                avg = {k: v / cfg.train.log_every for k, v in running.items()}
                running = {}
                rec = {"step": step, "lr": sched.get_last_lr()[0],
                       "sec_per_step": (time.time() - t0) / cfg.train.log_every,
                       **{f"loss_{k}": round(v, 5) for k, v in avg.items()}}
                t0 = time.time()
                print(rec)
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
            if step % cfg.train.eval_every == 0 or step == cfg.train.total_steps:
                metrics = validate(model, val_loader, device)
                rec = {"step": step, "val": metrics}
                print(rec)
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
            if step % cfg.train.ckpt_every == 0 or step == cfg.train.total_steps:
                save_ckpt("ckpt_last.pth")
            if step % cfg.train.vis_every == 0:
                try:
                    save_match_vis(cfg, model, val_ds, device, os.path.join(
                        out_dir, "vis", f"match_{step:06d}.jpg"))
                except Exception as e:  # vis is non-blocking
                    print("vis failed:", e)
        epoch += 1

    save_ckpt("ckpt_last.pth")
    print("done at step", step)


if __name__ == "__main__":
    main()
