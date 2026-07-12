#!/usr/bin/env bash
# Full ablation matrix for the lab 3090 — sequential, checkpoint-resume safe.
# BEFORE RUNNING: grep -rn TOBECHANGE configs/ src/ scripts/  and apply lab values.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
DEVICE=${DEVICE:-cuda}

# 1) feature caches (skips already-cached frames)
$PY scripts/cache_features.py --cars train_pool test --max-frames 0 \
    --device "$DEVICE" --batch 16
$PY scripts/cache_features.py --cars train_pool test --max-frames 0 \
    --backbone dinov3_vitb16 --input-res 512 --batch 16 --device "$DEVICE" \
    --dinov3-weights data/downloads/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth

# 2) training matrix (resume-safe: pass --resume if a checkpoint exists)
CONFIGS=(
  configs/ablations/full.yaml
  configs/ablations/no_masking.yaml
  configs/ablations/no_film.yaml
  configs/ablations/cross_none.yaml
  configs/ablations/cross_exchange.yaml
  configs/ablations/azimuth.yaml
  configs/ablations/dinov3.yaml
)
for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  ckpt="outputs/runs/$name/ckpt_last.pth"
  resume=()
  [ -f "$ckpt" ] && resume=(--resume "$ckpt")
  echo "=== training $name ==="
  $PY scripts/train.py --config "$cfg" --device "$DEVICE" "${resume[@]}" \
      || { echo "TRAIN FAILED: $name"; exit 1; }
  echo "=== eval $name ==="
  $PY scripts/gate_s3.py --ckpt "$ckpt" --device "$DEVICE" \
      || echo "EVAL FAILED (non-fatal): $name"
done

# 3) external benchmarks on the main model
$PY scripts/gate_s4.py --ckpt outputs/runs/full/ckpt_last.pth
echo "ALL DONE"
