#!/usr/bin/env bash
set -euo pipefail

REF_IMAGE="/home/linyuliu/jxmount/diffusion_custom/test/ref3.png"
COMP_IMAGE="/home/linyuliu/jxmount/diffusion_custom/test/ref.png"
MODEL_PATH="/home/linyuliu/jxmount/diffusion_custom/models/stabilityai/stable-diffusion-3.5-large"

R_LIST=(10 15 20 30 40 60 80)
LOG_DIR="logs_r_sweep"
mkdir -p "$LOG_DIR"

for R in "${R_LIST[@]}"; do
  echo "======================================"
  echo "Running explore_r=${R}"
  echo "Start time: $(date)"
  echo "======================================"

  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0,1,2,3 \
  python -m scripts.24_multigpu_testR \
    --ref_image "$REF_IMAGE" \
    --comp_image "$COMP_IMAGE" \
    --model_path "$MODEL_PATH" \
    --explore_r "$R" \
    2>&1 | tee "$LOG_DIR/explore_r_${R}.log"

  echo "Finished explore_r=${R} at $(date)"
  echo "Waiting a bit before next run..."
  sleep 5

  echo "GPU status after R=${R}:"
  nvidia-smi
  echo
done