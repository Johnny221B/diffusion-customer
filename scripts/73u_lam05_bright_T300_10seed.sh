#!/usr/bin/env bash
# SINGLE lambda=0.5 (rising-hard-trend branch), v=1.0, alpha=15, competitor=bright/18 (D_B=0.4705, 3rd pct).
# T=300, 10 seeds -> averaged trend with error band.
#
# RATIONALE (this session): the lambda-sweep scout (73r/73t) showed lam=0.5 is the cleanest RISING hard
#   win-rate branch (slope +0.191, 0.456->0.806 over 200 rounds) -- but on 1 seed only, so the slope may be
#   luck. User: run lam=0.5 alone on 10 seeds, T=300, average to confirm the trend is real (not 1-seed noise),
#   and see whether it keeps climbing past 200 or plateaus (bright hard ceiling ~0.78-0.81; last-20 already
#   0.806 at T=200, so likely plateau soon). CAVEAT: lam=0.5 is the SATURATED branch (beta=8, pred=1.0
#   overconfident) -- this run buys the hard trend, NOT calibration/true_p (that is the lam=50 branch).
#
#   10 seeds across 4 GPUs (disjoint ranges) into ONE shared out_root so diag reads all 10:
#     GPU0 seeds[0,3)  GPU1 seeds[3,6)  GPU2 seeds[6,8)  GPU3 seeds[8,10)
#   Each seed at T=300 ~ 5-6h; GPU0/1 run 3 seeds serial (~16-18h), GPU2/3 run 2 (~11-12h). Resumable
#   (per-sim _ckpt.pkl), so a kill/restart with the same STAMP picks up where it left off.
#
#   bash scripts/73u_lam05_bright_T300_10seed.sh                  # start (new stamp)
#   bash scripts/73u_lam05_bright_T300_10seed.sh STAMP=mmdd_HHMM  # resume
set -u
cd /home/linyuliu/jxmount/diffusion_custom

for a in "$@"; do case "$a" in STAMP=*) export "${a}";; esac; done
STAMP=${STAMP:-$(date +%m%d_%H%M)}

if pgrep -f "[7]3_cmts_dreamsim.py" >/dev/null; then
  echo "REFUSING: 73_cmts_dreamsim.py already running. pids: $(pgrep -f '[7]3_cmts_dreamsim.py' | tr '\n' ' ')"
  exit 1
fi

LAM=0.5; VV=1.0; T=300; B=8; N0=24; DIM=16; S=8.0; ALPHA=15; SAVE_EVERY=20
BWORD=bright; BSEED=18        # D_B=0.4705, 3rd percentile (strong -> un-pinned, warm hit ~0.58)
MODEL=models/stabilityai/stable-diffusion-3.5-large
OUT="outputs/cmts_lam${LAM}_v1.0_bbright_a15_d16_B8_T${T}_${STAMP}"
mkdir -p "$OUT"

# per-GPU disjoint seed ranges (10 seeds total): "start end"
declare -A QUEUE=( [0]="0 3" [1]="3 6" [2]="6 8" [3]="8 10" )

echo "=== [$(date +%H:%M:%S)] lam=$LAM x 10seed, v=$VV, B=bright/18, alpha=$ALPHA, T=$T  STAMP=$STAMP"
echo "    OUT=$OUT"
for G in 0 1 2 3; do
  read SS SE <<< "${QUEUE[$G]}"
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=$G setsid bash -c '
    conda run -n diverse --no-capture-output python scripts/73_cmts_dreamsim.py \
      --model_path '"$MODEL"' --device cuda:0 \
      --B_word '"$BWORD"' --B_seed '"$BSEED"' \
      --seed_start '"$SS"' --seed_end '"$SE"' \
      --dim '"$DIM"' --T '"$T"' --B '"$B"' --n0 '"$N0"' --v '"$VV"' --S '"$S"' --lam '"$LAM"' --alpha '"$ALPHA"' \
      --save_img_every '"$SAVE_EVERY"' \
      --partial_id '"$G"' --tag cmts \
      --out_root "'"$OUT"'" \
      >> "'"$OUT"'/launch_g'"$G"'.log" 2>&1
  ' </dev/null &
  echo "    GPU$G  seeds[$SS,$SE)  lam=$LAM  v=$VV  B=$BWORD/$BSEED  detached pid=$!"
done
echo "=== all 4 GPU queues detached (survive shell exit). STAMP=$STAMP ==="
echo "=== resume: bash scripts/73u_lam05_bright_T300_10seed.sh STAMP=$STAMP ==="
echo "=== diag:   conda run -n diverse --no-capture-output python scripts/91_lam05_bright_T300_diag.py $STAMP ==="
