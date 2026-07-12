#!/usr/bin/env bash
# Full ablation matrix for the lab 3090 — sequential, checkpoint-resume safe.
#
# BEFORE RUNNING:
#   1. grep -rn TOBECHANGE configs/ scripts/   and apply lab values in
#      configs/base.yaml. The lab PC only has cars 0000-0499 on disk, so use
#      the lab split: train_cars: lab_train (400 cars), test_cars: lab_test
#      (100 held-out cars) — NOT train_pool/test.
#   2. Check data paths (configs/base.yaml image_root/labels_root) match this
#      machine.
#   3. Check disk: feature cache is ~2.1 MB/frame at 518 px (fp16), 6.3 MB at
#      896 px. Defaults below (120 frames/car, 400+100 cars, 518 px) need
#      ~125 GB. Tune CACHE_FRAMES / input_res to your free space.
#
# USAGE:
#   bash scripts/run_all_lab.sh                 # everything (days)
#   bash scripts/run_all_lab.sh full            # one ablation (overnight)
#   bash scripts/run_all_lab.sh no_film azimuth # a chosen subset
#   bash scripts/run_all_lab.sh bench           # external benchmarks only
# Caching + baseline probe run (cheap/skipped-if-done) unless 'bench' only.
# Every training is checkpoint-resume safe: rerun the same command to continue.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
DEVICE=${DEVICE:-cuda}
CACHE_FRAMES=${CACHE_FRAMES:-120}   # frames per car to cache; 0 = all (disk!)

ALL=(full no_masking no_film cross_none cross_exchange azimuth dinov3)
if [ $# -eq 0 ]; then
  SELECTED=("${ALL[@]}"); RUN_BENCH=1
else
  SELECTED=(); RUN_BENCH=0
  for a in "$@"; do
    if [ "$a" = "bench" ]; then RUN_BENCH=1; else SELECTED+=("$a"); fi
  done
fi

# 1) feature caches (skips already-cached frames). lab_train = 400-car
#    training set, lab_test = 100 held-out cars (all within available 0000-0499).
if [ ${#SELECTED[@]} -gt 0 ]; then
  $PY scripts/cache_features.py --cars lab_train lab_test \
      --max-frames "$CACHE_FRAMES" --device "$DEVICE" --batch 16
  if printf '%s\n' "${SELECTED[@]}" | grep -q '^dinov3$'; then
    $PY scripts/cache_features.py --cars lab_train lab_test \
        --max-frames "$CACHE_FRAMES" --batch 12 --device "$DEVICE" \
        --backbone dinov3_vitb16 --input-res 512 \
        --dinov3-weights data/downloads/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
  fi

  # 2) baseline probe at full eval scale (baseline row + motivation figure)
  $PY scripts/gate_s1.py --cars lab_test || echo "gate_s1 FAILED (non-fatal)"
fi

# 3) training (resume-safe)
for name in "${SELECTED[@]}"; do
  cfg="configs/ablations/$name.yaml"
  [ -f "$cfg" ] || { echo "unknown ablation: $name"; exit 1; }
  ckpt="outputs/runs/$name/ckpt_last.pth"
  resume=()
  [ -f "$ckpt" ] && resume=(--resume "$ckpt")
  echo "=== training $name ==="
  $PY scripts/train.py --config "$cfg" --device "$DEVICE" "${resume[@]}" \
      || { echo "TRAIN FAILED: $name"; exit 1; }
  echo "=== eval $name ==="
  $PY scripts/gate_s3.py --ckpt "$ckpt" --device "$DEVICE" \
      --cars-train lab_train --cars-test lab_test \
      || echo "EVAL FAILED (non-fatal): $name"
done

# 4) external benchmarks on the main model (full pair sets)
if [ "$RUN_BENCH" -eq 1 ]; then
  if [ ! -f outputs/runs/full/ckpt_last.pth ]; then
    echo "bench skipped: outputs/runs/full/ckpt_last.pth missing (train 'full' first)"
  else
    $PY scripts/gate_s4.py --ckpt outputs/runs/full/ckpt_last.pth \
        --fixture-cars lab_test
    $PY scripts/bench_external.py --bench spair --data data/SPair-71k \
        --ckpt outputs/runs/full/ckpt_last.pth --max-pairs 0 --subset all
    $PY scripts/bench_external.py --bench spair --data data/SPair-71k \
        --ckpt outputs/runs/full/ckpt_last.pth --max-pairs 0 --subset viewpoint
    $PY scripts/bench_external.py --bench freiburg --data data/freiburg_cars \
        --ckpt outputs/runs/full/ckpt_last.pth --max-pairs 300
  fi
fi
echo "ALL DONE"
