#!/usr/bin/env bash
# lambda in {50,5} sweep, v=4 FIXED, alpha=15, 5 seeds each. competitor=wine/34.
#
# WHY wine/34 (from the rising-trend-band diagnostic, 0613):
#   black/18  (3rd pct, D_B=0.470): too STRONG -> best word wins only 0.78 of seeds
#             -> win-rate climbs but caps ~0.75 (floor-limited, true_p stuck ~0.5).
#   pigment/12(30th pct, D_B=0.581): too WEAK -> best word wins 1.00 -> win-rate PINNED at
#             1.0 from round 1 (ceiling-limited, no trend; run 0613_0931).
#   wine/34   (10th pct, D_B=0.520): the BAND in between -> floor 0.10, best word wins 0.93
#             (high but not pinned), only 1 strong word so argmax must FIND it.
#             Prediction: win-rate climbs ~0.3 -> ~0.85, and the stronger pressure (vs pigment)
#             should push best-so-far ds back toward black's 0.28.
#   NOTE: the best winnable word is "red" = the reference word itself -> learnable direction
#         ≈ "head toward R". Clean demo, but report it explicitly.
#   dreams_matrix UNCHANGED; ds_to_R comparable across all runs (panel d green = black best 0.2803).
#
# everything else = 73n: v=4, alpha=15, S=8, T=200, B=8, lam in {50,5}, 5 seeds.
#
#   bash scripts/73o_lamsweep_wine_a15.sh                  # start (new stamp)
#   bash scripts/73o_lamsweep_wine_a15.sh STAMP=mmdd_HHMM  # resume an existing run
set -u
cd /home/linyuliu/jxmount/diffusion_custom

for a in "$@"; do case "$a" in STAMP=*) export "${a}";; esac; done
STAMP=${STAMP:-$(date +%m%d_%H%M)}

if pgrep -f "73_cmts_dreamsim.py" >/dev/null; then
  echo "REFUSING: 73_cmts_dreamsim.py already running. pids: $(pgrep -f 73_cmts_dreamsim.py | tr '\n' ' ')"
  exit 1
fi

VV=4.0; T=200; B=8; N0=24; DIM=16; S=8.0; ALPHA=15; SAVE_EVERY=10
BWORD=wine; BSEED=34        # D_B=0.5198, 10th percentile
MODEL=models/stabilityai/stable-diffusion-3.5-large

launch () {  # $1=lam  $2=gpu  $3=seed_start  $4=seed_end  $5=partial_id  $6=out_root
  local LAM=$1 G=$2 SS=$3 SE=$4 PID=$5 OUT=$6
  mkdir -p "$OUT"
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=$G \
    setsid conda run -n diverse --no-capture-output python scripts/73_cmts_dreamsim.py \
      --model_path $MODEL --device cuda:0 \
      --B_word $BWORD --B_seed $BSEED \
      --seed_start $SS --seed_end $SE \
      --dim $DIM --T $T --B $B --n0 $N0 --v $VV --S $S --lam $LAM --alpha $ALPHA \
      --save_img_every $SAVE_EVERY \
      --partial_id $PID --tag cmts \
      --out_root "$OUT" \
      >> "$OUT/launch_g${G}.log" 2>&1 </dev/null &
  echo "    GPU$G  lam=$LAM  v=$VV  B=$BWORD/$BSEED  seeds[$SS,$SE)  detached pid=$!  ->  $OUT"
}

OUT50="outputs/cmts_lam50_v4.0_bwine_a15_d16_B8_T200_${STAMP}"
OUT5="outputs/cmts_lam5_v4.0_bwine_a15_d16_B8_T200_${STAMP}"
echo "=== [$(date +%H:%M:%S)] lam={50,5}x5seed, v=4, B=wine/34 (D_B=0.5198,10pct), alpha=$ALPHA  STAMP=$STAMP"
launch 50 0 0 3 0 "$OUT50"     # GPU0: lam50  seeds 0,1,2
launch 50 1 3 5 1 "$OUT50"     # GPU1: lam50  seeds 3,4
launch 5  2 0 3 2 "$OUT5"      # GPU2: lam5   seeds 0,1,2
launch 5  3 3 5 3 "$OUT5"      # GPU3: lam5   seeds 3,4
echo "=== all 4 workers detached (survive shell exit). STAMP=$STAMP ==="
echo "=== resume: bash scripts/73o_lamsweep_wine_a15.sh STAMP=$STAMP ==="
