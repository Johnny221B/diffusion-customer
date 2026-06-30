#!/usr/bin/env bash
# Launch alpha=10, v=1, lambda=100 on only GPU0/GPU1.
# Intended to run concurrently with alpha=6 occupying GPU2/GPU3.
set -u
cd /home/linyuliu/jxmount/diffusion_custom

for a in "$@"; do case "$a" in STAMP=*) export "${a}";; esac; done
STAMP=${STAMP:-$(date +%m%d_%H%M)}

MODEL=models/stabilityai/stable-diffusion-3.5-large
OUT="outputs/cmts_a10_v1_lam100_bbright_d16_B8_T1000_${STAMP}"
mkdir -p "$OUT"

launch () {  # gpu seed_start seed_end partial_id
  local G=$1 SS=$2 SE=$3 PID=$4
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=$G \
    setsid conda run -n diverse --no-capture-output \
    python scripts/73_cmts_dreamsim.py \
      --model_path "$MODEL" --device cuda:0 \
      --B_word bright --B_seed 18 \
      --seed_start "$SS" --seed_end "$SE" \
      --dim 16 --T 1000 --B 8 --n0 24 \
      --v 1.0 --S 8.0 --lam 100 --alpha 10 \
      --save_img_every 50 \
      --partial_id "$PID" --tag cmts --out_root "$OUT" \
      >> "$OUT/launch_T1000_g${G}.log" 2>&1 </dev/null &
  echo "GPU$G alpha=10 v=1.0 lam=100 sims=[$SS,$SE) pid=$! -> $OUT"
}

echo "=== launch alpha=10 on GPU0/GPU1 only; STAMP=$STAMP ==="
launch 0 0 3 80
launch 1 3 5 81
echo "=== two detached workers launched ==="
