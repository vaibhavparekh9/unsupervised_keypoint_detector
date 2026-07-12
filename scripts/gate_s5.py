"""Gate S5 — every ablation config launches and trains >=50 steps on dev.

The dinov3 config additionally requires its own feature cache; if the gated
weights are missing the config is recorded as a documented SKIP (flagged).

Usage: python scripts/gate_s5.py [--steps 50] [--grid-res 32|64]
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--grid-res", type=int, default=None,
                    help="override for CPU (32); GPU uses config default")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    configs = sorted(glob.glob(os.path.join(REPO, "configs/ablations/*.yaml")))
    status = {}
    for cfg in configs:
        name = os.path.basename(cfg)[:-5]
        from src.utils.config import load_config
        c = load_config(cfg)
        if c.backbone.name.startswith("dinov3"):
            w = c.backbone.get("dinov3_weights")
            if not w or not os.path.exists(os.path.join(REPO, w)):
                status[name] = {"ok": True,
                                "skipped": "dinov3 weights missing (gated); "
                                           "see README"}
                continue
            cache = os.path.join(REPO, c.backbone.cache_dir,
                                 f"{c.backbone.name}_{c.data.input_res}")
            if not os.path.isdir(cache) or not glob.glob(cache + "/*/*.npy"):
                r = subprocess.run(
                    [PY, "scripts/cache_features.py", "--cars", "dev_smoke",
                     "dev_test_smoke", "--max-frames", "80",
                     "--backbone", c.backbone.name,
                     "--input-res", str(c.data.input_res),
                     "--dinov3-weights", w, "--device", device,
                     "--batch", "4"], cwd=REPO)
                if r.returncode != 0:
                    status[name] = {"ok": False, "err": "dinov3 cache failed"}
                    continue

        overrides = [
            f"train.total_steps={args.steps}",
            "train.warmup_steps=10",
            f"train.log_every=10",
            f"train.eval_every={args.steps}",
            f"train.ckpt_every={args.steps}",
            f"train.vis_every={args.steps + 1}",
            f"train.out_dir=outputs/runs/gate_s5_{name}",
            "data.num_workers=2",
            "train.batch_size=2",
        ]
        if args.grid_res:
            overrides += [f"data.grid_res={args.grid_res}",
                          f"model.descriptor_res={args.grid_res}"]
        cmd = [PY, "scripts/train.py", "--config", cfg, "--device", device]
        for ov in overrides:
            cmd += ["-o", ov]
        print("===", name)
        r = subprocess.run(cmd, cwd=REPO)
        status[name] = {"ok": r.returncode == 0}

    gate_pass = all(v.get("ok") for v in status.values())
    report = {"status": status, "gate_pass": bool(gate_pass)}
    os.makedirs(os.path.join(REPO, "outputs/diagnostics"), exist_ok=True)
    with open(os.path.join(REPO, "outputs/diagnostics/gate_s5.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    print("GATE S5:", "PASS" if gate_pass else "FAIL")
    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
