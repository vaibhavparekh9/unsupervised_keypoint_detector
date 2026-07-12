"""Gate S1 — frozen-feature probe: zero-shot PCK baseline + symmetry confusion.

Uses cached frozen-backbone tokens (run scripts/cache_features.py first) on
held-out test cars. Produces the paper's baseline row and motivation figure.

Gate: probe runs end-to-end (>=300 intra kp matches) AND left/right symmetry
confusion grows with azimuth separation. If frozen features show NO
confusion anywhere, the paper's premise is in question: exit code 2 and a
prominent STOP flag.

Usage: python scripts/gate_s1.py [--cars dev_test_smoke]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eval.probe import (AZ_BINS, BackboneDescriptors, load_labeled_frames,
                            run_probe)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evaluate_gate(results, conf_curve, min_intra_kps=300):
    rates = [v["confusion_rate"] for v in conf_curve.values()
             if v["confusion_rate"] is not None and v["n"] >= 20]
    first = next((v["confusion_rate"] for v in conf_curve.values()
                  if v["confusion_rate"] is not None and v["n"] >= 20), None)
    later = [v["confusion_rate"] for v in list(conf_curve.values())[2:]
             if v["confusion_rate"] is not None and v["n"] >= 20]
    ran_ok = results["intra"]["num_kps"] >= min_intra_kps and len(rates) >= 3
    grows = first is not None and later and max(later) > first + 0.05
    no_confusion = bool(rates) and max(rates) < 0.10
    return bool(ran_ok), bool(grows), bool(no_confusion)


def save_confusion_plot(conf_curve, title, out_path):
    xs = [f"{lo}-{hi}" for lo, hi in AZ_BINS]
    ys = [conf_curve[x]["confusion_rate"] for x in xs]
    plt.figure(figsize=(5, 3.4))
    plt.plot(xs, [y if y is not None else np.nan for y in ys], "o-")
    plt.xlabel("azimuth separation (deg)")
    plt.ylabel("L/R symmetry confusion rate")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cars", default="dev_test_smoke")  # TOBECHANGED test (3090)
    ap.add_argument("--backbone", default="dinov2_vitb14_reg")
    ap.add_argument("--input-res", type=int, default=518)
    # large enough that the high-azimuth confusion bins have usable n
    ap.add_argument("--num-intra", type=int, default=800)
    ap.add_argument("--num-cross", type=int, default=400)
    ap.add_argument("--image-root", default="/home/vaibhav/3DRealCars-English")
    ap.add_argument("--labels-root", default="/home/vaibhav/3DRealCars-Labels")
    args = ap.parse_args()

    cache_root = os.path.join(REPO, "outputs/cache",
                              f"{args.backbone}_{args.input_res}")
    with open(os.path.join(REPO, "configs/split.json")) as f:
        split = json.load(f)
    car_ids = split[args.cars] if args.cars in split else [args.cars]

    car_frames = load_labeled_frames(args.image_root, args.labels_root,
                                     car_ids, args.input_res, cache_root)
    n_frames = sum(len(v) for v in car_frames.values())
    print(f"{len(car_frames)} cars, {n_frames} labeled+cached frames")
    if n_frames == 0:
        print("No cached features — run scripts/cache_features.py first.")
        sys.exit(1)

    results, conf_curve, group_rates, _ = run_probe(
        BackboneDescriptors(), car_frames, args.input_res,
        args.num_intra, args.num_cross)
    print("PCK:", json.dumps(results, indent=1))
    print("confusion vs azimuth:", json.dumps(conf_curve, indent=1))
    print("confusion by group:", group_rates)

    ran_ok, grows, no_confusion = evaluate_gate(results, conf_curve)

    paper_dir = os.path.join(REPO, "outputs/paper")
    os.makedirs(paper_dir, exist_ok=True)
    tag = args.backbone
    report = {
        "backbone": args.backbone,
        "pck": results,
        "confusion_vs_azimuth": conf_curve,
        "confusion_by_group": group_rates,
        "ran_ok": ran_ok,
        "confusion_grows_with_separation": grows,
        "no_confusion_flag": no_confusion,
        "gate_pass": bool(ran_ok and grows and not no_confusion),
    }
    with open(os.path.join(paper_dir, f"symmetry_confusion_{tag}.json"), "w") as f:
        json.dump(report, f, indent=1)
    save_confusion_plot(conf_curve, f"frozen {args.backbone}",
                        os.path.join(paper_dir, f"symmetry_confusion_{tag}.png"))
    with open(os.path.join(paper_dir, f"baseline_pck_{tag}.md"), "w") as f:
        f.write("| backbone | intra PCK@10 | cross PCK@10 | intra n | cross n |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| frozen {args.backbone} (zero-shot NN) "
                f"| {results['intra']['pck@10_feat64']:.1f} "
                f"| {results['cross']['pck@10_feat64']:.1f} "
                f"| {results['intra']['num_kps']} "
                f"| {results['cross']['num_kps']} |\n")

    print(json.dumps({k: report[k] for k in
                      ["ran_ok", "confusion_grows_with_separation",
                       "no_confusion_flag", "gate_pass"]}, indent=1))
    if no_confusion:
        print("=" * 70)
        print("STOP AUTOPILOT: frozen features show NO symmetry confusion —")
        print("the paper's premise needs rechecking by the user.")
        print("=" * 70)
        sys.exit(2)
    print("GATE S1:", "PASS" if report["gate_pass"] else "FAIL")
    sys.exit(0 if report["gate_pass"] else 1)


if __name__ == "__main__":
    main()
