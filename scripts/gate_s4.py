"""Gate S4 — external benchmark harnesses run end-to-end on a smoke ckpt.

1. fixture: tiny 3DRealCar-based benchmark in the SPair pair format — always
   must run (proves harness mechanics with no external dependency).
2. SPair-71k cars PCK@0.1 (all + viewpoint subset) if data/SPair-71k exists.
3. Freiburg Static Cars 52 relative-azimuth eval if data/freiburg_cars exists.

Datasets that are absent are recorded as documented SKIPs (per CLAUDE.md),
but the fixture path must PASS. Numbers at smoke scale are mechanical only.

Usage: python scripts/gate_s4.py --ckpt outputs/runs/smoke/ckpt_last.pth
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def build_fixture(image_root, labels_root, out_dir, num_pairs=8,
                  cars_key="dev_test_smoke"):
    """SPair-format pairs from 3DRealCar test cars + PifPaf keypoints."""
    from PIL import Image
    from src.data import pifpaf
    from src.data.realcar import scan_car

    with open(os.path.join(REPO, "configs/split.json")) as f:
        split = json.load(f)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "pairs"), exist_ok=True)

    made = 0
    for car_id in split[cars_key]:
        if made >= num_pairs:
            break
        frames = []
        for rec in scan_car(image_root, car_id)[:30]:
            lab = pifpaf.load_keypoints(labels_root, car_id, rec.name)
            if lab is None:
                continue
            kps, conf = lab
            if (conf > 0.5).sum() >= 6:
                frames.append((rec, kps, conf))
            if len(frames) == 2:
                break
        if len(frames) < 2:
            continue
        (ra, ka, ca), (rb, kb, cb) = frames
        common = np.where((ca > 0.5) & (cb > 0.5))[0]
        if len(common) < 4:
            continue
        names = []
        for rec in (ra, rb):
            name = f"{car_id}_{rec.name}.jpg"
            img = Image.open(rec.img_path).convert("RGB")
            img.resize((960, 720)).save(os.path.join(out_dir, "images", name))
            names.append(name)
        s = 0.5  # 1920x1440 -> 960x720
        ksa = (ka[common] * s).tolist()
        ksb = (kb[common] * s).tolist()
        xs = [p[0] for p in ksb]
        ys = [p[1] for p in ksb]
        pair = {
            "src_imname": names[0], "trg_imname": names[1],
            "src_kps": ksa, "trg_kps": ksb,
            "trg_bndbox": [min(xs), min(ys), max(xs), max(ys)],
        }
        with open(os.path.join(out_dir, "pairs", f"{car_id}.json"), "w") as f:
            json.dump(pair, f)
        made += 1
    return made


def run_bench(bench, data, ckpt, subset="all", max_pairs=30):
    out = os.path.join(REPO, "outputs/paper", f"bench_{bench}_{subset}.json")
    if os.path.exists(out):
        os.remove(out)
    cmd = [PY, "scripts/bench_external.py", "--bench", bench, "--data", data,
           "--ckpt", ckpt, "--subset", subset, "--max-pairs", str(max_pairs)]
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode != 0 or not os.path.exists(out):
        return None
    with open(out) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/runs/smoke/ckpt_last.pth")
    ap.add_argument("--image-root", default="/home/vaibhav/3DRealCars-English")
    ap.add_argument("--labels-root", default="/home/vaibhav/3DRealCars-Labels")
    ap.add_argument("--fixture-cars", default="dev_test_smoke",
                    help="split key for fixture frames (lab PC: lab_test)")
    args = ap.parse_args()

    status = {}

    fixture_dir = os.path.join(REPO, "data/fixture_bench")
    if not os.path.isdir(os.path.join(fixture_dir, "pairs")):
        n = build_fixture(args.image_root, args.labels_root, fixture_dir,
                          cars_key=args.fixture_cars)
        print(f"fixture built: {n} pairs")
    res = run_bench("fixture", fixture_dir, args.ckpt)
    status["fixture"] = {"ok": res is not None, **(res or {})}

    spair_dir = os.path.join(REPO, "data/SPair-71k")
    if os.path.isdir(spair_dir):
        for subset in ("all", "viewpoint"):
            res = run_bench("spair", spair_dir, args.ckpt, subset=subset)
            status[f"spair_{subset}"] = {"ok": res is not None, **(res or {})}
    else:
        status["spair_all"] = {
            "ok": True, "skipped": "data/SPair-71k missing — run: "
            "tar xzf data/downloads/SPair-71k.tar.gz -C data/"}

    frei_dir = os.path.join(REPO, "data/freiburg_cars")
    if os.path.isdir(os.path.join(frei_dir, "annotations")):
        res = run_bench("freiburg", frei_dir, args.ckpt, max_pairs=40)
        status["freiburg"] = {"ok": res is not None, **(res or {})}
    else:
        status["freiburg"] = {
            "ok": True, "skipped": "data/freiburg_cars missing — run: "
            "python scripts/prepare_freiburg.py"}

    gate_pass = all(v.get("ok") for v in status.values())
    report = {"status": status, "gate_pass": bool(gate_pass)}
    os.makedirs(os.path.join(REPO, "outputs/diagnostics"), exist_ok=True)
    with open(os.path.join(REPO, "outputs/diagnostics/gate_s4.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    print("GATE S4:", "PASS" if gate_pass else "FAIL")
    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
