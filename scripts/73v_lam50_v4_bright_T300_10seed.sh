#!/usr/bin/env bash
# SINGLE lambda=50 (DE-SATURATED branch), v=4.0, alpha=15, competitor=bright/18 (D_B=0.4705, 3rd pct).
# T=300, 10 seeds -> averaged trend with error band. SAME config as 73u (0616_2245) except LAM 0.5->50, v 1->4.
#
# RATIONALE (this session): 73u (lam0.5, saturated) bought the cleanest RISING hard win-rate (0.42->0.79,
#   slope +0.135/100r over 10 seeds) but LOCKED true_p at 0.52 -- beta railed at S=8, direction never learned,
#   best-ds plateaued 0.396, predicted_p fake-1.0. User wants true_p LARGER. true_p = sigma(alpha*(D_B - ds))
#   depends ONLY on ds, so true_p only rises if the optimizer actually steers ds down -- which needs the
#   DE-SATURATED branch (beta leaves 8, Hessian informative, cov anisotropic, direction learnable).
#   Evidence 0613_0004 (lam50, v4, alpha=30, black/18) crossed true_p 0.5 with a still-rising win-rate.
#   This run tests lam50+v4 at the SAME alpha=15/bright/18 as 73u so the two are directly comparable.
#   CAVEAT: lam UP shrinks cov ~ v^2/lam (exploration), so v is raised 1->4 to keep breadth; the hard-trend
#   slope will likely be SHALLOWER than 73u (trade hard-slope for true_p/calibration). true_p ceiling at
#   alpha=15 is ds_reach~0.39 -> sigma(15*(0.4705-0.39))~0.77; we are at 0.52, so there IS headroom.
#
#   10 seeds across 4 GPUs (disjoint ranges) into ONE shared out_root so diag reads all 10:
#     GPU0 seeds[0,3)  GPU1 seeds[3,6)  GPU2 seeds[6,8)  GPU3 seeds[8,10)
#   Each seed at T=300 ~ 5-6h; GPU0/1 run 3 seeds serial (~16-18h), GPU2/3 run 2 (~11-12h). Resumable
#   (per-sim _ckpt.pkl), so a kill/restart with the same STAMP picks up where it left off.
#
#   bash scripts/73v_lam50_v4_bright_T300_10seed.sh                  # start (new stamp)
#   bash scripts/73v_lam50_v4_bright_T300_10seed.sh STAMP=mmdd_HHMM  # resume
set -u
cd /home/linyuliu/jxmount/diffusion_custom

for a in "$@"; do case "$a" in STAMP=*) export "${a}";; esac; done
STAMP=${STAMP:-$(date +%m%d_%H%M)}

if pgrep -f "[7]3_cmts_dreamsim.py" >/dev/null; then
  echo "REFUSING: 73_cmts_dreamsim.py already running. pids: $(pgrep -f '[7]3_cmts_dreamsim.py' | tr '\n' ' ')"
  exit 1
fi

LAM=50; VV=4.0; T=300; B=8; N0=24; DIM=16; S=8.0; ALPHA=15; SAVE_EVERY=20
BWORD=bright; BSEED=18        # D_B=0.4705, 3rd percentile (strong -> un-pinned, warm hit ~0.58)
MODEL=models/stabilityai/stable-diffusion-3.5-large
OUT="outputs/cmts_lam${LAM}_v${VV}_bbright_a15_d16_B8_T${T}_${STAMP}"
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
echo "=== resume: bash scripts/73v_lam50_v4_bright_T300_10seed.sh STAMP=$STAMP ==="
echo "=== diag:   conda run -n diverse --no-capture-output python scripts/91_lam05_bright_T300_diag.py $STAMP 300 LAM=50 VV=4.0 ==="
