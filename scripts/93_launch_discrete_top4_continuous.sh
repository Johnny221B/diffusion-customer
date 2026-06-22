#!/usr/bin/env bash
# Continuous validation of the top discrete CM-TS region.
#
# Four controlled configurations, one per GPU, five trajectories each:
#   GPU0: alpha=20, v=0.5, lam=50  (best discrete endpoint + calibration)
#   GPU1: alpha=20, v=1.0, lam=50  (v ablation; best cross-alpha average)
#   GPU2: alpha=20, v=2.0, lam=50  (v ablation; slower rising curve)
#   GPU3: alpha=30, v=2.0, lam=50  (alpha ablation at v=2)
#
# T=200 is the canonical first-stage budget. All runs checkpoint every round, so
# selected configurations can be extended later without repeating completed work.
#
# Start:  bash scripts/93_launch_discrete_top4_continuous.sh
# Resume: bash scripts/93_launch_discrete_top4_continuous.sh STAMP=mmdd_HHMM
set -u
cd /home/linyuliu/jxmount/diffusion_custom

for a in "$@"; do case "$a" in STAMP=*) export "${a}";; esac; done
STAMP=${STAMP:-$(date +%m%d_%H%M)}

if pgrep -f "[7]3_cmts_dreamsim.py" >/dev/null; then
  echo "REFUSING: 73_cmts_dreamsim.py already running."
  pgrep -af "[7]3_cmts_dreamsim.py"
  exit 1
fi

T=200
B=8
N0=24
DIM=16
S=8.0
LAM=50
SAVE_EVERY=10
BWORD=bright
BSEED=18
MODEL=models/stabilityai/stable-diffusion-3.5-large

launch () {  # gpu alpha v label
  local G=$1 ALPHA=$2 V=$3 LABEL=$4
  local OUT="outputs/cmts_${LABEL}_bbright_d16_B8_T${T}_${STAMP}"
  mkdir -p "$OUT"
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=$G \
    setsid conda run -n diverse --no-capture-output \
    python scripts/73_cmts_dreamsim.py \
      --model_path "$MODEL" --device cuda:0 \
      --B_word "$BWORD" --B_seed "$BSEED" \
      --seed_start 0 --seed_end 5 \
      --dim "$DIM" --T "$T" --B "$B" --n0 "$N0" \
      --v "$V" --S "$S" --lam "$LAM" --alpha "$ALPHA" \
      --save_img_every "$SAVE_EVERY" \
      --partial_id "$G" --tag cmts --out_root "$OUT" \
      >> "$OUT/launch_g${G}.log" 2>&1 </dev/null &
  echo "GPU$G alpha=$ALPHA v=$V lam=$LAM seeds=[0,5) pid=$! -> $OUT"
}

echo "=== top-4 continuous validation; STAMP=$STAMP; T=$T; 5 sims/config ==="
launch 0 20 0.5 a20_v0.5_lam50
launch 1 20 1.0 a20_v1_lam50
launch 2 20 2.0 a20_v2_lam50
launch 3 30 2.0 a30_v2_lam50
echo "=== four detached workers launched; resume with STAMP=$STAMP ==="
