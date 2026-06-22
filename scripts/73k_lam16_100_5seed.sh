#!/usr/bin/env bash
# lambda = {16, 100} comparison, 5 seeds each.
# LABEL IS NOW SOFT-Bernoulli (alpha=10): the simulate process draws
#   y ~ Bernoulli(sigma(alpha*(D_B - ds)))   -- matches cmts_sim.py's Bernoulli-of-sigmoid
# feedback, and gives graded win-prob so "barely beating B" no longer wins w.p. 1
# (kills the camp-just-past-B pathology of the old HARD 1[ds<D_B] label).
# Plots/metrics still use the HARD indicator y_hard = 1[ds<D_B].
#   true_p_soft   = sigma(alpha*(D_B - ds))   (alpha=10) -- now == the true expected label
#                   (the real expectation, NOT the surrogate's predicted_p=sigma(beta^T phi)).
#   cov_eig_max / cov_eig_min  = spectrum of THIS round's Thompson draw covariance
#                   cov = v^2 * 0.5 (H^-1 + H^-T). lam large -> H~=lam I -> cov ~=
#                   (v^2/lam) I: eigenvalues shrink AND max/min -> 1 (isotropic),
#                   i.e. exploration direction becomes near-random. This is the
#                   suspected mechanism for "lam big -> win-rate rises".
#
# WHY 16 vs 100 only: 16 (=d) is the SATURATED baseline (||beta|| pinned at clip S=8);
# 100 is the first clearly de-saturated setting. Two-way contrast keeps GPU cheap.
#
#   bash scripts/73k_lam16_100_5seed.sh                  # start (new stamp)
#   bash scripts/73k_lam16_100_5seed.sh STAMP=mmdd_HHMM  # resume an existing run
#
# 4 GPUs: each lambda split across 2 GPUs (seeds 0-2 / 3-4). DETACHED (setsid),
# RESUMABLE per-round (_ckpt.pkl), guard refuses to double-launch.
set -u
cd /home/linyuliu/jxmount/diffusion_custom

for a in "$@"; do case "$a" in STAMP=*) export "${a}";; esac; done
STAMP=${STAMP:-$(date +%m%d_%H%M)}

if pgrep -f "73_cmts_dreamsim.py" >/dev/null; then
  echo "REFUSING: 73_cmts_dreamsim.py already running. pids: $(pgrep -f 73_cmts_dreamsim.py | tr '\n' ' ')"
  exit 1
fi

V=4.0; T=200; B=8; N0=24; DIM=16; S=8.0; ALPHA=10; SAVE_EVERY=10
MODEL=models/stabilityai/stable-diffusion-3.5-large

launch () {  # $1=lambda  $2=gpu  $3=seed_start  $4=seed_end  $5=partial_id  $6=out_root
  local LAM=$1 G=$2 SS=$3 SE=$4 PID=$5 OUT=$6
  mkdir -p "$OUT"
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=$G \
    setsid conda run -n diverse --no-capture-output python scripts/73_cmts_dreamsim.py \
      --model_path $MODEL --device cuda:0 \
      --B_word black --B_seed 18 \
      --seed_start $SS --seed_end $SE \
      --dim $DIM --T $T --B $B --n0 $N0 --v $V --S $S --lam $LAM --alpha $ALPHA \
      --save_img_every $SAVE_EVERY \
      --partial_id $PID --tag cmts \
      --out_root "$OUT" \
      >> "$OUT/launch_g${G}.log" 2>&1 </dev/null &
  echo "    GPU$G  lam=$LAM  seeds[$SS,$SE)  detached pid=$!  ->  $OUT"
}

OUT16="outputs/cmts_lam16_v${V}_d${DIM}_B${B}_T${T}_${STAMP}"
OUT100="outputs/cmts_lam100_v${V}_d${DIM}_B${B}_T${T}_${STAMP}"
echo "=== [$(date +%H:%M:%S)] lam={16,100} x 5 seeds  v=$V S=$S alpha=$ALPHA T=$T  STAMP=$STAMP"
launch 16  0 0 3 0 "$OUT16"     # GPU0: lam16 seeds 0,1,2
launch 16  1 3 5 1 "$OUT16"     # GPU1: lam16 seeds 3,4
launch 100 2 0 3 2 "$OUT100"    # GPU2: lam100 seeds 0,1,2
launch 100 3 3 5 3 "$OUT100"    # GPU3: lam100 seeds 3,4
echo "=== all 4 workers detached (survive shell exit). STAMP=$STAMP ==="
echo "=== plot:   conda run -n diverse --no-capture-output python scripts/78_lam16_100_diag.py $STAMP"
echo "=== resume: bash scripts/73k_lam16_100_5seed.sh STAMP=$STAMP ==="
