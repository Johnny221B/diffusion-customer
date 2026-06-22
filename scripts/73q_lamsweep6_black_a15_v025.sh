#!/usr/bin/env bash
# lambda in {0.5,1,5,10,20,50} sweep, v=0.25 FIXED, alpha=15, competitor=black/18 (strong, D_B=0.4704).
#
# RATIONALE (this session): with v small the Thompson draw cov (∝ v²/λ under saturation) shrinks, so
#   CM-TS approaches GREEDY argmax on beta_hat's direction -- which (like discrete ranking) is immune
#   to sigma-saturation. Discrete = (exploration_a=1.0 -> v=1, prior_var=3 -> lam≈1/3); at v=1 continuous
#   still scattered (manifold argmax amplifies cov that discrete ranking absorbs). So we go BELOW v=1.
#   v=0.25 kills the random-search scatter; then sweep lambda to find where beta de-saturates AND a
#   rising win-rate trend appears against the strong black/18 competitor (which gives win-rate headroom,
#   unlike wine/34 which pinned ~0.96). alpha=15 (softer than the 28-30 sweet spot but stronger than 12).
#
#   SCOUT pass: 1 seed per lambda (SE=1). 6 lambdas on 4 GPUs -> GPU0/1 run 2 lambdas serially,
#   GPU2/3 run 1 each. ~6-8h wall. Deepen the winning lambda(s) to more seeds afterwards.
#
#   v=0.25, alpha=15, S=8, T=200, B=8, n0=24, dim=16, competitor black/18 (D_B=0.4704, 3rd pct).
#
#   bash scripts/73q_lamsweep6_black_a15_v025.sh                  # start (new stamp)
#   bash scripts/73q_lamsweep6_black_a15_v025.sh STAMP=mmdd_HHMM  # resume
set -u
cd /home/linyuliu/jxmount/diffusion_custom

for a in "$@"; do case "$a" in STAMP=*) export "${a}";; esac; done
STAMP=${STAMP:-$(date +%m%d_%H%M)}

if pgrep -f "[7]3_cmts_dreamsim.py" >/dev/null; then
  echo "REFUSING: 73_cmts_dreamsim.py already running. pids: $(pgrep -f '[7]3_cmts_dreamsim.py' | tr '\n' ' ')"
  exit 1
fi

VV=0.25; T=200; B=8; N0=24; DIM=16; S=8.0; ALPHA=15; SAVE_EVERY=10
BWORD=black; BSEED=18        # D_B=0.4704, 3rd percentile (strong -> win-rate headroom)
MODEL=models/stabilityai/stable-diffusion-3.5-large
SS=0; SE=1                   # 1 seed per lambda (scout pass)

# per-GPU lambda queue (space-separated, run serially within the GPU)
declare -A QUEUE=( [0]="0.5 20" [1]="1 50" [2]="5" [3]="10" )

echo "=== [$(date +%H:%M:%S)] lam={0.5,1,5,10,20,50}x1seed, v=$VV, B=black/18, alpha=$ALPHA  STAMP=$STAMP"
for G in 0 1 2 3; do
  LAMS="${QUEUE[$G]}"
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=$G setsid bash -c '
    for LAM in '"$LAMS"'; do
      OUT="outputs/cmts_lam${LAM}_v0.25_bblack_a15_d16_B8_T200_'"$STAMP"'"
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
echo "=== resume: bash scripts/73q_lamsweep6_black_a15_v025.sh STAMP=$STAMP ==="
