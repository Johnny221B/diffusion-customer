#!/usr/bin/env bash
# lambda=50 FIXED, v in {4,12} sweep, 5 seeds each -- SAME as 73l EXCEPT competitor.
#
# WHY new competitor (from the oracle-ceiling diagnostic, 0613):
#   black/18 (D_B=0.4704) sits at the 3rd PERCENTILE of all 9120 images -> too strong.
#   Only 0.4% of words have word-level soft win > 0.5; top5-word ceiling = 0.438.
#   So true_p_soft capped ~0.53 regardless of surrogate -- it's a CEILING, not a fit problem.
#
#   New competitor = pigment/12, D_B=0.5813 = 30th PERCENTILE (30% of imgs beat it, 70% worse).
#     -> top5-word ceiling 0.763, random-word start mean_w=0.342 (room to rise),
#        16 words genuinely beat it (argmax has real positives to find).
#   dreams_matrix UNCHANGED (competitor only re-indexes the label threshold) -> 10-min rerun.
#   NOTE: ds_to_R is competitor-independent, so best-so-far ds is directly comparable to the
#         black run (0613_0004): does a fairer opponent push generation closer to R?
#
# everything else identical to 73l: lam=50, v={4,12}, alpha=30, S=8, T=200, B=8, 5 seeds.
#
#   bash scripts/73m_lam50_pigment.sh                  # start (new stamp)
#   bash scripts/73m_lam50_pigment.sh STAMP=mmdd_HHMM  # resume an existing run
set -u
cd /home/linyuliu/jxmount/diffusion_custom

for a in "$@"; do case "$a" in STAMP=*) export "${a}";; esac; done
STAMP=${STAMP:-$(date +%m%d_%H%M)}

if pgrep -f "73_cmts_dreamsim.py" >/dev/null; then
  echo "REFUSING: 73_cmts_dreamsim.py already running. pids: $(pgrep -f 73_cmts_dreamsim.py | tr '\n' ' ')"
  exit 1
fi

LAM=50; T=200; B=8; N0=24; DIM=16; S=8.0; ALPHA=30; SAVE_EVERY=10
BWORD=pigment; BSEED=12        # D_B=0.5813, 30th percentile
MODEL=models/stabilityai/stable-diffusion-3.5-large

launch () {  # $1=v  $2=gpu  $3=seed_start  $4=seed_end  $5=partial_id  $6=out_root
  local VV=$1 G=$2 SS=$3 SE=$4 PID=$5 OUT=$6
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
  echo "    GPU$G  v=$VV  lam=$LAM  B=$BWORD/$BSEED  seeds[$SS,$SE)  detached pid=$!  ->  $OUT"
}

OUT4="outputs/cmts_lam50_v4.0_bpigment_d16_B8_T200_${STAMP}"
OUT12="outputs/cmts_lam50_v12.0_bpigment_d16_B8_T200_${STAMP}"
echo "=== [$(date +%H:%M:%S)] lam=50, v={4,12}x5seed, B=pigment/12 (D_B=0.5813, 30pct)  alpha=$ALPHA  STAMP=$STAMP"
launch 4.0  0 0 3 0 "$OUT4"     # GPU0: v4  seeds 0,1,2
launch 4.0  1 3 5 1 "$OUT4"     # GPU1: v4  seeds 3,4
launch 12.0 2 0 3 2 "$OUT12"    # GPU2: v12 seeds 0,1,2
launch 12.0 3 3 5 3 "$OUT12"    # GPU3: v12 seeds 3,4
echo "=== all 4 workers detached (survive shell exit). STAMP=$STAMP ==="
echo "=== resume: bash scripts/73m_lam50_pigment.sh STAMP=$STAMP ==="
