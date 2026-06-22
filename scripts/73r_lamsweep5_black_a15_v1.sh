#!/usr/bin/env bash
# lambda in {0.5,1,5,10,50} sweep, v=1.0 FIXED, alpha=15, competitor=black/18 (strong, D_B=0.4704).
#
# RATIONALE (this session): v=0.25 (run 73q) killed exploration -- best_ds stuck ~0.437 for ALL lambda,
#   nobody reached black_best=0.2803. So bump v back to 1.0 (Thompson draw cov ∝ v²/λ -> x16 vs v=0.25).
#   Reading rule from cov ≈ (v²/λ)·I in the saturated regime: at v=1 the GOOD 0613 breadth (v=4,lam50,
#   v²/λ≈0.32, the trend-winner) corresponds to lam≈3 here; lam=50 -> v²/λ=0.02 (de-saturation probe,
#   under-explores). So this sweep spans both ends: lam{0.5,1}=big breadth, lam{10,50}=small breadth/probe.
#   alpha KEPT at 15 (user choice: isolate the v change vs run 73q, even though 15 is softer than the
#   28-30 sweet spot). competitor black/18 kept (strong -> win-rate headroom; weaker words pin high).
#
#   SCOUT pass: 1 seed per lambda (SE=1). 5 lambdas on 4 GPUs -> GPU0 runs 2 (0.5,50) serially,
#   GPU1/2/3 run 1 each. ~6-8h wall on GPU0, ~3-4h others. Deepen the winning lambda(s) afterwards.
#
#   v=1.0, alpha=15, S=8, T=200, B=8, n0=24, dim=16, competitor black/18 (D_B=0.4704, 3rd pct).
#
#   bash scripts/73r_lamsweep5_black_a15_v1.sh                  # start (new stamp)
#   bash scripts/73r_lamsweep5_black_a15_v1.sh STAMP=mmdd_HHMM  # resume
set -u
cd /home/linyuliu/jxmount/diffusion_custom

for a in "$@"; do case "$a" in STAMP=*) export "${a}";; esac; done
STAMP=${STAMP:-$(date +%m%d_%H%M)}

if pgrep -f "[7]3_cmts_dreamsim.py" >/dev/null; then
  echo "REFUSING: 73_cmts_dreamsim.py already running. pids: $(pgrep -f '[7]3_cmts_dreamsim.py' | tr '\n' ' ')"
  exit 1
fi

VV=1.0; T=200; B=8; N0=24; DIM=16; S=8.0; ALPHA=15; SAVE_EVERY=10
BWORD=black; BSEED=18        # D_B=0.4704, 3rd percentile (strong -> win-rate headroom)
MODEL=models/stabilityai/stable-diffusion-3.5-large
SS=0; SE=1                   # 1 seed per lambda (scout pass)

# per-GPU lambda queue (space-separated, run serially within the GPU)
declare -A QUEUE=( [0]="0.5 50" [1]="1" [2]="5" [3]="10" )

echo "=== [$(date +%H:%M:%S)] lam={0.5,1,5,10,50}x1seed, v=$VV, B=black/18, alpha=$ALPHA  STAMP=$STAMP"
for G in 0 1 2 3; do
  LAMS="${QUEUE[$G]}"
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=$G setsid bash -c '
    for LAM in '"$LAMS"'; do
      OUT="outputs/cmts_lam${LAM}_v1.0_bblack_a15_d16_B8_T200_'"$STAMP"'"
      mkdir -p "$OUT"
      conda run -n diverse --no-capture-output python scripts/73_cmts_dreamsim.py \
        --model_path '"$MODEL"' --device cuda:0 \
        --B_word '"$BWORD"' --B_seed '"$BSEED"' \
        --seed_start '"$SS"' --seed_end '"$SE"' \
        --dim '"$DIM"' --T '"$T"' --B '"$B"' --n0 '"$N0"' --v '"$VV"' --S '"$S"' --lam "$LAM" --alpha '"$ALPHA"' \
        --save_img_every '"$SAVE_EVERY"' \
        --partial_id '"$G"' --tag cmts \
        --out_root "$OUT" \
        >> "$OUT/launch_g'"$G"'.log" 2>&1
    done
  ' </dev/null &
  echo "    GPU$G  lam_queue=[$LAMS]  v=$VV  B=$BWORD/$BSEED  seeds[$SS,$SE)  detached pid=$!"
done
echo "=== all GPU queues detached (survive shell exit). STAMP=$STAMP ==="
echo "=== resume: bash scripts/73r_lamsweep5_black_a15_v1.sh STAMP=$STAMP ==="
