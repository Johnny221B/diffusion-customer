#!/usr/bin/env bash
# lambda in {0.5,1,10,50} sweep, v=1.0 FIXED, alpha=15, competitor=low/36 (D_B=0.5394, 15th pct).
#
# RATIONALE (this session): competitor re-picked via joint (competitor,alpha) calibration (scripts/87).
#   GOAL = keep the HARD win-rate trend alive AND widen the soft winning basin. The pinning boundary is
#   17% (best word red wins all 40 seeds -> hard win-rate pins at 1.0). low/36 sits at 15% = the LARGEST
#   soft basin (5 words p_soft>0.5, soft ceiling 0.84) still compatible with an UN-pinned hard trend
#   (best-word hard ceiling 0.95, not 1.0). alpha=15 chosen because per-image soft-prob distribution is
#   most evenly spread there (alpha=30 polarizes into a 0/1 bimodal; see calib_dist_*.png). lambda swept
#   because v=1 + lam=50 was the only de-saturated+calibrated cell in run 73r (black); re-test that here
#   plus the small-lambda (saturated) ends for contrast against the new competitor.
#
#   SCOUT pass: 1 seed per lambda (SE=1). 4 lambdas on 4 GPUs (one each, no serial queue). ~3-4h.
#   Deepen the winning lambda(s) to more seeds afterwards.
#
#   v=1.0, alpha=15, S=8, T=200, B=8, n0=24, dim=16, competitor low/36 (D_B=0.5394, 15th pct).
#
#   bash scripts/73s_lamsweep4_low_a15_v1.sh                  # start (new stamp)
#   bash scripts/73s_lamsweep4_low_a15_v1.sh STAMP=mmdd_HHMM  # resume
set -u
cd /home/linyuliu/jxmount/diffusion_custom

for a in "$@"; do case "$a" in STAMP=*) export "${a}";; esac; done
STAMP=${STAMP:-$(date +%m%d_%H%M)}

if pgrep -f "[7]3_cmts_dreamsim.py" >/dev/null; then
  echo "REFUSING: 73_cmts_dreamsim.py already running. pids: $(pgrep -f '[7]3_cmts_dreamsim.py' | tr '\n' ' ')"
  exit 1
fi

VV=1.0; T=200; B=8; N0=24; DIM=16; S=8.0; ALPHA=15; SAVE_EVERY=10
BWORD=low; BSEED=36          # D_B=0.5394, 15th percentile (un-pinned hard trend + 5-word soft basin)
MODEL=models/stabilityai/stable-diffusion-3.5-large
SS=0; SE=1                   # 1 seed per lambda (scout pass)

# per-GPU lambda queue (one lambda each)
declare -A QUEUE=( [0]="0.5" [1]="1" [2]="10" [3]="50" )

echo "=== [$(date +%H:%M:%S)] lam={0.5,1,10,50}x1seed, v=$VV, B=low/36, alpha=$ALPHA  STAMP=$STAMP"
for G in 0 1 2 3; do
  LAMS="${QUEUE[$G]}"
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=$G setsid bash -c '
    for LAM in '"$LAMS"'; do
      OUT="outputs/cmts_lam${LAM}_v1.0_blow_a15_d16_B8_T200_'"$STAMP"'"
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
echo "=== resume: bash scripts/73s_lamsweep4_low_a15_v1.sh STAMP=$STAMP ==="
