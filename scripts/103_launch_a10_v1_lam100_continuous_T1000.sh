#!/usr/bin/env bash
# Continuous follow-up for low-alpha/high-lambda sweep:
#   alpha=10, v=1, lambda=100
#
# Same canonical environment as the alpha=5/6 tests:
#   bright/18, d=16, B=8, n0=24, S=8, T=1000, save image every 50 rounds.
#
# Four detached workers over five sim seeds:
#   GPU0: sims [0,2)
#   GPU1: sims [2,3)
#   GPU2: sims [3,4)
#   GPU3: sims [4,5)
set -u
cd /home/linyuliu/jxmount/diffusion_custom

for a in "$@"; do case "$a" in STAMP=*) export "${a}";; esac; done
STAMP=${STAMP:-$(date +%m%d_%H%M)}

if pgrep -f "[7]3_cmts_dreamsim.py" >/dev/null; then
  echo "REFUSING: 73_cmts_dreamsim.py already running."
  pgrep -af "[7]3_cmts_dreamsim.py"
  exit 1
fi

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

echo "=== launch continuous alpha=10, v=1, lambda=100; STAMP=$STAMP ==="
launch 0 0 2 70
launch 1 2 3 71
launch 2 3 4 72
launch 3 4 5 73
echo "=== four detached workers launched ==="
