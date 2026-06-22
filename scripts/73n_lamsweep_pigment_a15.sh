#!/usr/bin/env bash
# lambda in {50, 5} sweep, v=4 FIXED, alpha=15, 5 seeds each. competitor=pigment/12.
#
# WHY: with the fairer competitor pigment/12 (D_B=0.5813, 30th pct, ceiling 0.763) now in place,
#   probe TWO things at once on a SMALLER alpha:
#     alpha=15  -> softer labels (alpha*std_ds≈0.54, below the ≈1.0 sweet spot). Motive: ease the
#                 "model just camps a hair past the competitor" concern -- soft labels reward genuinely
#                 getting close to R, not merely crossing D_B. Risk: too soft -> signal washes out.
#     lambda    -> saturation knob, v=4 fixed (narrow radius, the trend-winner from the v-sweep):
#                   lam=50 = de-saturated (beta off the S=8 clip, direction alive)
#                   lam=5  = small prior -> beta rails the clip -> saturated (W=p(1-p)->0)
#                 Direct "saturated vs de-saturated" A/B at the new alpha+competitor.
#   Question: does lam=50 still show the rising true_p_soft, or does alpha=15 flatten it? does lam=5
#             collapse (saturation + soft label = no direction)?
#   dreams_matrix UNCHANGED; ds_to_R comparable across all runs (panel d vs black best=0.2803).
#
#   bash scripts/73n_lamsweep_pigment_a15.sh                  # start (new stamp)
#   bash scripts/73n_lamsweep_pigment_a15.sh STAMP=mmdd_HHMM  # resume an existing run
set -u
cd /home/linyuliu/jxmount/diffusion_custom

for a in "$@"; do case "$a" in STAMP=*) export "${a}";; esac; done
STAMP=${STAMP:-$(date +%m%d_%H%M)}

if pgrep -f "73_cmts_dreamsim.py" >/dev/null; then
  echo "REFUSING: 73_cmts_dreamsim.py already running. pids: $(pgrep -f 73_cmts_dreamsim.py | tr '\n' ' ')"
  exit 1
fi

VV=4.0; T=200; B=8; N0=24; DIM=16; S=8.0; ALPHA=15; SAVE_EVERY=10
BWORD=pigment; BSEED=12        # D_B=0.5813, 30th percentile
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

OUT50="outputs/cmts_lam50_v4.0_bpigment_a15_d16_B8_T200_${STAMP}"
OUT5="outputs/cmts_lam5_v4.0_bpigment_a15_d16_B8_T200_${STAMP}"
echo "=== [$(date +%H:%M:%S)] lam={50,5}x5seed, v=4, B=pigment/12, alpha=$ALPHA  STAMP=$STAMP"
launch 50 0 0 3 0 "$OUT50"     # GPU0: lam50  seeds 0,1,2
launch 50 1 3 5 1 "$OUT50"     # GPU1: lam50  seeds 3,4
launch 5  2 0 3 2 "$OUT5"      # GPU2: lam5   seeds 0,1,2
launch 5  3 3 5 3 "$OUT5"      # GPU3: lam5   seeds 3,4
echo "=== all 4 workers detached (survive shell exit). STAMP=$STAMP ==="
echo "=== resume: bash scripts/73n_lamsweep_pigment_a15.sh STAMP=$STAMP ==="
