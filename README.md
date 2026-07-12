# Unsupervised 360° Keypoint Discovery

Fully unsupervised landmark discovery for 360° rigid objects (3DRealCar).
Frozen DINO dense features + an orientation-conditioned head mapping pixels to
canonical spherical coordinates, trained from multi-view reprojection
correspondence + visibility masking + relative-pose orientation consistency.
See `CLAUDE.md` for the full plan and `research_direction.md` for rationale.

## Setup (identical on dev PC and lab PC — plain venv, no conda, no docker)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(dev PC: the venv in this checkout is named `kp/` instead of `.venv/`.)

Requirements are fully pinned; every pin ships cp310 and cp312 manylinux
wheels (dev = Python 3.10.12, lab = 3.12.3). torch is the default PyPI wheel
bundling the CUDA 13 runtime — works with driver 580.x on both machines.

## Before a lab (3090) run

```bash
grep -rn TOBECHANGE configs/ src/ scripts/
```

and apply the lab values noted in the inline comments. Do not commit lab
values as defaults.

## Pipeline (stages S0–S5, each with an automated smoke gate)

```bash
python scripts/make_split.py          # committed as configs/split.json
python scripts/gate_s0.py             # data plumbing + geometry gate
python scripts/cache_features.py --cars dev_smoke --cars2 dev_test_smoke
python scripts/gate_s1.py             # frozen-feature probe + symmetry confusion
python scripts/train.py --config configs/base.yaml
python scripts/gate_s2a.py            # overfit test (2 cars)
python scripts/gate_s2b.py            # smoke train check
python scripts/gate_s3.py --ckpt outputs/runs/base/ckpt_last.pth   # eval suite
python scripts/gate_s4.py --ckpt ...  # external benchmark harnesses
python scripts/gate_s5.py             # ablation configs launch check
bash scripts/run_all_lab.sh           # full ablation matrix (3090)
```

Diagnostics (human review, non-blocking) land in `outputs/diagnostics/`;
paper-facing tables/figures in `outputs/paper/`.

## DINOv3

DINOv3 weights are gated (Meta license). Download `dinov3_vitb16` weights
after requesting access, place the checkpoint path in
`configs/base.yaml: backbone.dinov3_weights`, and set `backbone.name: dinov3_vitb16`.
Without weights the code falls back to / defaults to DINOv2-B/14 (registers),
which auto-downloads via torch.hub.

## External benchmarks (S4)

- Freiburg Cars: `python scripts/download_freiburg_cars.py` (or follow the
  manual step it prints if the host blocks scripted downloads).
- Telling-Left-from-Right geometry-aware subset: built on SPair-71k —
  `python scripts/download_spair.py`.
