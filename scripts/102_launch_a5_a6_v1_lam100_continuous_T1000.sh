#!/usr/bin/env bash
# Continuous test for the discrete-suggested low-alpha, high-lambda settings:
#   alpha=5, v=1, lambda=100
#   alpha=6, v=1, lambda=100
#
# Both use the current canonical environment:
#   competitor bright/18, d=16, B=8, n0=24, S=8, T=1000.
# Images are saved every 50 rounds.
#
# Four detached workers:
#   GPU0: alpha=5, sims [0,3)
#   GPU1: alpha=5, sims [3,5)
#   GPU2: alpha=6, sims [0,3)
#   GPU3: alpha=6, sims [3,5)
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
BWORD=bright
BSEED=18
T=1000
B=8
N0=24
DIM=16
S=8.0
V=1.0
LAM=100
SAVE_EVERY=50

launch () {  # gpu alpha seed_start seed_end partial_id
  local G=$1 ALPHA=$2 SS=$3 SE=$4 PID=$5
  local LABEL="a${ALPHA}_v1_lam100"
  local OUT="outputs/cmts_${LABEL}_bbright_d16_B8_T${T}_${STAMP}"
  mkdir -p "$OUT"
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=$G \
    setsid conda run -n diverse --no-capture-output \
    python scripts/73_cmts_dreamsim.py \
      --model_path "$MODEL" --device cuda:0 \
      --B_word "$BWORD" --B_seed "$BSEED" \
      --seed_start "$SS" --seed_end "$SE" \
      --dim "$DIM" --T "$T" --B "$B" --n0 "$N0" \
      --v "$V" --S "$S" --lam "$LAM" --alpha "$ALPHA" \
      --save_img_every "$SAVE_EVERY" \
      --partial_id "$PID" --tag cmts --out_root "$OUT" \
      >> "$OUT/launch_T1000_g${G}.log" 2>&1 </dev/null &
  echo "GPU$G alpha=$ALPHA v=$V lam=$LAM sims=[$SS,$SE) pid=$! -> $OUT"
}

echo "=== launch continuous low-alpha/high-lambda test; STAMP=$STAMP ==="
launch 0 5 0 3 60
launch 1 5 3 5 61
launch 2 6 0 3 62
launch 3 6 3 5 63
echo "=== four detached workers launched ==="
