#!/usr/bin/env bash
# PRIOR-PRECISION sweep (lambda), NOT a norm-cap (S) sweep.
#
# Diagnosis (M=85, v=4.0): per-round win-rate is flat because the logistic fit
# SATURATES -- ||beta|| pins at S=8 on separable data, so W=p(1-p)->0, which (a)
# freezes the fit DIRECTION (correctly-classified points drop out of the gradient)
# and (b) collapses the Laplace posterior to isotropic (H->lam*I). The argmax pick
# is scale-invariant in beta, so shrinking S alone does NOT change a round's ranking
# -- the real lever is keeping sigma in the SOFT band so the direction keeps adapting.
# That is what lambda (ridge prior precision in laplace_map) controls: larger lambda
# -> smaller ||beta|| -> W stays away from 0 -> direction keeps rotating with data.
#
# Each round now logs cos_beta_prev (cos(beta_t,beta_{t-1}); ->1 = frozen) and
# ts_cos_mean (Thompson angular breadth). We read those to SEE whether the direction
# unfreezes, instead of guessing.
#
#   bash scripts/73j_lamsweep.sh                  # start (new stamp)
#   bash scripts/73j_lamsweep.sh STAMP=mmdd_HHMM  # resume an existing sweep
#
# 4 GPUs, one lambda each: {16(=d, saturated baseline), 100, 1000, 10000}. v=4.0,
# S=8.0 (kept high so the PRIOR, not the clip, sets the regime). M=5 sims each,
# save_img_every=10. DETACHED (setsid </dev/null &): survive shell / harness reaping.
# RESUMABLE per-round via _ckpt.pkl; guard refuses to double-launch.
set -u
cd /home/linyuliu/jxmount/diffusion_custom

for a in "$@"; do case "$a" in STAMP=*) export "${a}";; esac; done
STAMP=${STAMP:-$(date +%m%d_%H%M)}

if pgrep -f "73_cmts_dreamsim.py" >/dev/null; then
  echo "REFUSING: 73_cmts_dreamsim.py already running. pids: $(pgrep -f 73_cmts_dreamsim.py | tr '\n' ' ')"
  exit 1
fi

V=4.0; T=200; B=8; N0=24; DIM=16; S=8.0; MSEED=5; SAVE_EVERY=10
LAMS=(16 100 1000 10000)
echo "=== [$(date +%H:%M:%S)] LAMBDA-sweep  lam={${LAMS[*]}}  v=$V S=$S M=$MSEED T=$T  STAMP=$STAMP"
for g in 0 1 2 3; do
    LAM=${LAMS[$g]}
    OUT="outputs/cmts_lam${LAM}_v${V}_d${DIM}_B${B}_T${T}_${STAMP}"
    mkdir -p "$OUT"
    env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=$g \
      setsid conda run -n diverse --no-capture-output python scripts/73_cmts_dreamsim.py \
        --model_path models/stabilityai/stable-diffusion-3.5-large \
        --device cuda:0 \
        --B_word black --B_seed 18 \
        --seed_start 0 --seed_end $MSEED \
        --dim $DIM --T $T --B $B --n0 $N0 --v $V --S $S --lam $LAM \
        --save_img_every $SAVE_EVERY \
        --partial_id $g --tag cmts \
        --out_root "$OUT" \
        >> "$OUT/launch_g${g}.log" 2>&1 </dev/null &
    echo "    GPU$g  lam=$LAM  detached pid=$!  ->  $OUT"
done
echo "=== all 4 workers detached (survive shell exit). ==="
echo "=== read regime: grep median_predicted_p / mean_cos_beta_prev in */sim*/summary.json ==="
echo "=== resume: re-run 'bash scripts/73j_lamsweep.sh STAMP=$STAMP' ==="
