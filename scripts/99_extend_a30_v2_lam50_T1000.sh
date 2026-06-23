#!/usr/bin/env bash
# Resume the five alpha=30, v=2, lambda=50 trajectories at t=500 and extend
# them through t=999. Existing checkpoints make this a paired continuation.
set -u
cd /home/linyuliu/jxmount/diffusion_custom

OUT=outputs/cmts_a30_v2_lam50_bbright_d16_B8_T200_0621_0437
MODEL=models/stabilityai/stable-diffusion-3.5-large

if pgrep -f "[7]3_cmts_dreamsim.py" >/dev/null; then
  echo "REFUSING: 73_cmts_dreamsim.py already running."
  pgrep -af "[7]3_cmts_dreamsim.py"
  exit 1
fi

launch () {  # gpu seed_start seed_end partial_id
  local G=$1 SS=$2 SE=$3 PID=$4
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=$G \
    setsid conda run -n diverse --no-capture-output \
    python scripts/73_cmts_dreamsim.py \
      --model_path "$MODEL" --device cuda:0 \
      --B_word bright --B_seed 18 \
      --seed_start "$SS" --seed_end "$SE" \
      --dim 16 --T 1000 --B 8 --n0 24 \
      --v 2.0 --S 8.0 --lam 50 --alpha 30 \
      --save_img_every 10 --partial_id "$PID" --tag cmts \
      --out_root "$OUT" \
      >> "$OUT/extend_T1000_g${G}.log" 2>&1 </dev/null &
  echo "GPU$G sims=[$SS,$SE) pid=$!"
}

echo "=== extend a30_v2_lam50: T500 -> T1000, five paired trajectories ==="
launch 0 0 2 30
launch 1 2 3 31
launch 2 3 4 32
launch 3 4 5 33
echo "=== four detached extension workers launched ==="
