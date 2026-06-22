#!/usr/bin/env bash
# lambda in {1/3, 1/2, 1, 5} sweep, v=1 FIXED, alpha=12, 2 seeds each. competitor=wine/34.
#
# GOAL: probe the SMALL-lambda regime (0.33..5, near discrete's prior_var=3 -> lam≈0.33) and
#   record per-round ||beta_hat|| to see where each lambda's norm settles vs the S=8 clip.
#   NOTE: v only scales the Thompson draw cov, NOT the MAP point estimate beta_hat -> v=1 does
#   not change ||beta_hat||; this sweep isolates lambda's effect on the norm.
#   Prior runs: lam5/lam16 railed at S=8 (saturated); lam50/lam100 sat at ~0.1 (de-saturated).
#   Expectation: these 4 small lambdas likely all rail at S=8 unless alpha=12 (softer labels,
#   weaker data term) pulls them down. The beta-norm line chart (scripts/83) is the deliverable.
#
#   v=1, alpha=12, S=8, T=200, B=8, n0=24, dim=16, competitor wine/34 (D_B=0.5198, 10th pct).
#   4 GPUs: one lambda each, seeds 0,1.
#
#   bash scripts/73p_lamsweep4_wine_a12_v1.sh                  # start (new stamp)
#   bash scripts/73p_lamsweep4_wine_a12_v1.sh STAMP=mmdd_HHMM  # resume
set -u
cd /home/linyuliu/jxmount/diffusion_custom

for a in "$@"; do case "$a" in STAMP=*) export "${a}";; esac; done
STAMP=${STAMP:-$(date +%m%d_%H%M)}

if pgrep -f "73_cmts_dreamsim.py" >/dev/null; then
  echo "REFUSING: 73_cmts_dreamsim.py already running. pids: $(pgrep -f 73_cmts_dreamsim.py | tr '\n' ' ')"
  exit 1
fi

VV=1.0; T=200; B=8; N0=24; DIM=16; S=8.0; ALPHA=12; SAVE_EVERY=10
BWORD=wine; BSEED=34        # D_B=0.5198, 10th percentile
MODEL=models/stabilityai/stable-diffusion-3.5-large
SS=0; SE=2                  # 2 seeds per lambda

LAMS=(0.3333 0.5 1.0 5.0)
echo "=== [$(date +%H:%M:%S)] lam={1/3,1/2,1,5}x2seed, v=$VV, B=wine/34, alpha=$ALPHA  STAMP=$STAMP"
for G in 0 1 2 3; do
  LAM=${LAMS[$G]}
  OUT="outputs/cmts_lam${LAM}_v1.0_bwine_a12_d16_B8_T200_${STAMP}"
  mkdir -p "$OUT"
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=$G \
    setsid conda run -n diverse --no-capture-output python scripts/73_cmts_dreamsim.py \
      --model_path $MODEL --device cuda:0 \
      --B_word $BWORD --B_seed $BSEED \
      --seed_start $SS --seed_end $SE \
      --dim $DIM --T $T --B $B --n0 $N0 --v $VV --S $S --lam $LAM --alpha $ALPHA \
      --save_img_every $SAVE_EVERY \
      --partial_id $G --tag cmts \
      --out_root "$OUT" \
      >> "$OUT/launch_g${G}.log" 2>&1 </dev/null &
  echo "    GPU$G  lam=$LAM  v=$VV  B=$BWORD/$BSEED  seeds[$SS,$SE)  detached pid=$!  ->  $OUT"
done
echo "=== all 4 workers detached (survive shell exit). STAMP=$STAMP ==="
echo "=== resume: bash scripts/73p_lamsweep4_wine_a12_v1.sh STAMP=$STAMP ==="
