#!/usr/bin/env bash
# lambda in {0.5,1,10,50} sweep, v=1.0 FIXED, alpha=15, competitor=bright/18 (D_B=0.4705, 3rd pct).
#
# RATIONALE (this session): the low/36 (15%) attempt PINNED -- warm hit-rate=1.000, hard win-rate dead-flat
#   from step 0, because the optimizer's typical achievable ds (~0.474) already beats any D_B>~0.49. So a
#   rising HARD trend FORCES a strong competitor with D_B ~= 0.47 (3-5%); bright/18 (D_B=0.4705) is that
#   (warm hit ~0.625 -> floor leaves room to climb, proven 0.77->0.82 with black/18 same D_B). Soft basin
#   is necessarily 1 word (red) at this strength -- accepted: headline = hard trend + beta de-saturation
#   + calibration (lam=50,v=1 was the de-saturated+calibrated cell in run 73r). lambda re-swept to confirm.
#   bright/18 == black/18 in D_B (0.4705 vs 0.4704); user picked the "bright" word.
#
#   SCOUT pass: 1 seed per lambda (SE=1). 4 lambdas on 4 GPUs (one each). ~3-4h. Deepen winner afterwards.
#
#   v=1.0, alpha=15, S=8, T=200, B=8, n0=24, dim=16, competitor bright/18 (D_B=0.4705, 3rd pct).
#
#   bash scripts/73t_lamsweep4_bright_a15_v1.sh                  # start (new stamp)
#   bash scripts/73t_lamsweep4_bright_a15_v1.sh STAMP=mmdd_HHMM  # resume
set -u
cd /home/linyuliu/jxmount/diffusion_custom

for a in "$@"; do case "$a" in STAMP=*) export "${a}";; esac; done
STAMP=${STAMP:-$(date +%m%d_%H%M)}

if pgrep -f "[7]3_cmts_dreamsim.py" >/dev/null; then
  echo "REFUSING: 73_cmts_dreamsim.py already running. pids: $(pgrep -f '[7]3_cmts_dreamsim.py' | tr '\n' ' ')"
  exit 1
fi

VV=1.0; T=200; B=8; N0=24; DIM=16; S=8.0; ALPHA=15; SAVE_EVERY=10
BWORD=bright; BSEED=18        # D_B=0.4705, 3rd percentile (strong -> un-pinned hard trend, warm hit ~0.625)
MODEL=models/stabilityai/stable-diffusion-3.5-large
SS=0; SE=1                    # 1 seed per lambda (scout pass)

# per-GPU lambda queue (one lambda each)
declare -A QUEUE=( [0]="0.5" [1]="1" [2]="10" [3]="50" )

echo "=== [$(date +%H:%M:%S)] lam={0.5,1,10,50}x1seed, v=$VV, B=bright/18, alpha=$ALPHA  STAMP=$STAMP"
for G in 0 1 2 3; do
  LAMS="${QUEUE[$G]}"
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=$G setsid bash -c '
    for LAM in '"$LAMS"'; do
      OUT="outputs/cmts_lam${LAM}_v1.0_bbright_a15_d16_B8_T200_'"$STAMP"'"
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
echo "=== resume: bash scripts/73t_lamsweep4_bright_a15_v1.sh STAMP=$STAMP ==="
