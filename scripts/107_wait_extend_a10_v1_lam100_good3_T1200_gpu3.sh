#!/usr/bin/env bash
# Wait for GPU3 to become free, then extend the good alpha=10, v=1, lambda=100
# trajectories sim001/sim002/sim003 from T=1000 to T=1200.
set -u
cd /home/linyuliu/jxmount/diffusion_custom

OUT=outputs/cmts_a10_v1_lam100_bbright_d16_B8_T1000_0625_2312
MODEL=models/stabilityai/stable-diffusion-3.5-large
LOG="$OUT/extend_T1200_good3_g3.log"
GPU=3
MEM_LIMIT_MIB=5000

echo "=== wait then extend alpha=10 v=1 lambda=100 good3 to T=1200 on GPU${GPU} ===" >> "$LOG"
date >> "$LOG"

while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
  echo "$(date '+%F %T') GPU${GPU} used=${used}MiB util=${util}%" >> "$LOG"
  if [ "${used:-999999}" -lt "$MEM_LIMIT_MIB" ]; then
    break
  fi
  sleep 300
done

echo "GPU${GPU} is free enough; launching extension" >> "$LOG"
date >> "$LOG"

env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$GPU" \
  conda run -n diverse --no-capture-output \
  python scripts/73_cmts_dreamsim.py \
    --model_path "$MODEL" --device cuda:0 \
    --B_word bright --B_seed 18 \
    --seed_start 1 --seed_end 4 \
    --dim 16 --T 1200 --B 8 --n0 24 \
    --v 1.0 --S 8.0 --lam 100 --alpha 10 \
    --save_img_every 50 \
    --partial_id 82 --tag cmts --out_root "$OUT" \
    >> "$LOG" 2>&1

echo "=== extension command finished ===" >> "$LOG"
date >> "$LOG"
