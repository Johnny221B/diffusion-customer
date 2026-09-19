#!/usr/bin/env bash
# Extend alpha=8, v=0.5, lambda=100 sim000/sim001 from T=1000 to T=1500.
set -u
cd /home/linyuliu/jxmount/diffusion_custom

OUT=outputs/cmts_a8_v0.5_lam100_bbright_d16_B8_T1000_0629_0842
MODEL=models/stabilityai/stable-diffusion-3.5-large
LOG="$OUT/extend_T1500_sim0_sim1.log"

echo "=== extend alpha=8 v=0.5 lambda=100 sim000/sim001 to T=1500 ===" >> "$LOG"
date >> "$LOG"

run_one () {  # gpu seed partial
  local G=$1
  local S=$2
  local PID=$3
  local RUNLOG="$OUT/extend_T1500_sim${S}_g${G}.log"
  echo "Launching sim${S} on GPU${G}" >> "$LOG"
  date >> "$LOG"
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$G" \
    conda run -n diverse --no-capture-output \
    python scripts/73_cmts_dreamsim.py \
      --model_path "$MODEL" --device cuda:0 \
      --B_word bright --B_seed 18 \
      --seed_start "$S" --seed_end "$((S+1))" \
      --dim 16 --T 1500 --B 8 --n0 24 \
      --v 0.5 --S 8.0 --lam 100 --alpha 8 \
      --save_img_every 50 \
      --partial_id "$PID" --tag cmts --out_root "$OUT" \
      >> "$RUNLOG" 2>&1
  echo "Finished sim${S} on GPU${G}" >> "$LOG"
  date >> "$LOG"
}
export -f run_one
export OUT MODEL LOG

setsid bash -lc 'cd /home/linyuliu/jxmount/diffusion_custom; run_one 0 0 120' >/tmp/a8_v05_T1500_sim0_launcher.log 2>&1 </dev/null &
echo "sim000 launcher pid=$!" >> "$LOG"

setsid bash -lc 'cd /home/linyuliu/jxmount/diffusion_custom; run_one 1 1 121' >/tmp/a8_v05_T1500_sim1_launcher.log 2>&1 </dev/null &
echo "sim001 launcher pid=$!" >> "$LOG"

echo "=== launched sim000/sim001 ===" >> "$LOG"
date >> "$LOG"
