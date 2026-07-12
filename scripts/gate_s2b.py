"""Gate S2b — smoke train: dev-scale run completes without OOM/NaN,
correspondence loss decreases, and predicted relative rotations correlate
with ARKit relative rotations on held-out pairs (rank correlation).

Usage: python scripts/gate_s2b.py [--steps 600] [--grid-res 32] [--device cpu]
       (GPU dev: --steps 3000 --grid-res 64;  lab 3090: full config defaults)
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)   # TOBECHANGED 3000+ (GPU)
    ap.add_argument("--grid-res", type=int, default=32)  # TOBECHANGED 64 (GPU)
    ap.add_argument("--batch", type=int, default=2)      # TOBECHANGED 4+ (GPU)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-train", action="store_true",
                    help="only evaluate an existing outputs/runs/smoke log")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = "outputs/runs/smoke"
    log_path = os.path.join(REPO, out_dir, "log.jsonl")

    if not args.skip_train:
        if os.path.exists(log_path):
            os.remove(log_path)
        overrides = [
            f"data.grid_res={args.grid_res}",
            f"model.descriptor_res={args.grid_res}",
            "data.num_workers=4",
            f"train.batch_size={args.batch}",
            f"train.total_steps={args.steps}",
            f"train.eval_every={max(args.steps // 3, 1)}",
            f"train.ckpt_every={max(args.steps // 3, 1)}",
            f"train.vis_every={max(args.steps // 3, 1)}",
            f"train.out_dir={out_dir}",
        ]
        cmd = [PY, "scripts/train.py", "--config", "configs/base.yaml",
               "--device", device]
        for ov in overrides:
            cmd += ["-o", ov]
        print(" ".join(cmd))
        r = subprocess.run(cmd, cwd=REPO)
        if r.returncode != 0:
            print("GATE S2B: FAIL (training crashed / NaN)")
            sys.exit(1)

    corr, vals = [], []
    with open(log_path) as f:
        for line in f:
            rec = json.loads(line)
            if "loss_corr" in rec:
                corr.append(rec["loss_corr"])
            if "val" in rec:
                vals.append(rec["val"])
    k = max(len(corr) // 5, 1)
    corr_first = float(np.mean(corr[:k]))
    corr_last = float(np.mean(corr[-k:]))
    decreasing = corr_last < corr_first

    final_val = vals[-1] if vals else {}
    spearman = final_val.get("rot_spearman", -1.0)
    rot_ok = spearman > 0.3

    # diagnostics plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 3.2))
    plt.plot(corr)
    plt.xlabel("log window")
    plt.ylabel("correspondence loss")
    plt.tight_layout()
    os.makedirs(os.path.join(REPO, "outputs/diagnostics"), exist_ok=True)
    plt.savefig(os.path.join(REPO, "outputs/diagnostics/smoke_corr_loss.png"),
                dpi=130)

    report = {
        "corr_first": corr_first, "corr_last": corr_last,
        "corr_decreasing": bool(decreasing),
        "final_val": final_val,
        "rot_spearman": spearman, "rot_spearman_ok": bool(rot_ok),
        "gate_pass": bool(decreasing and rot_ok),
    }
    with open(os.path.join(REPO, "outputs/diagnostics/gate_s2b.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    print("GATE S2B:", "PASS" if report["gate_pass"] else "FAIL")
    sys.exit(0 if report["gate_pass"] else 1)


if __name__ == "__main__":
    main()
