#!/usr/bin/env bash
# Scale-up the WINNER v=4.0 to large M using ALL 4 GPUs in parallel, pooled into
# the existing v-sweep dir (sims 0..4 already done -> new sims 5..84).
#
# Each GPU runs a DISJOINT block of 20 sim seeds sequentially (~2h37m/trajectory),
# so in a 48h budget each GPU clears ~18 -> ~72 new trajectories pooled to M~77.
# Images are saved only every 10 rounds (--save_img_every 10): ~25M/sim instead of
# 223M, because disk is tight (was 99% full). Warm + final round always saved.
#
#   bash scripts/73i_scaleup_v4.sh                 # start / resume the scale-up
#   bash scripts/73i_scaleup_v4.sh STAMP=0603_0137 # explicit (default is this stamp)
#
# DETACHED (setsid </dev/null &): workers reparent to init and survive this shell /
# any harness reaping. RESUMABLE: re-run the same command; finished trajectories
# finalize-and-skip, interrupted ones resume from their per-round _ckpt.pkl.
# Guard refuses to double-launch if a 73_cmts worker is already alive.
set -u
cd /home/linyuliu/jxmount/diffusion_custom

for a in "$@"; do case "$a" in STAMP=*) export "${a}";; esac; done
STAMP=${STAMP:-0603_0137}
V=4.0; T=200; B=8; N0=24; DIM=16; S=8.0; SAVE_EVERY=10

OUT="outputs/cmts_v${V}_d${DIM}_B${B}_T${T}_${STAMP}"
[ -d "$OUT" ] || { echo "ERROR: $OUT not found (check STAMP)"; exit 1; }

if pgrep -f "73_cmts_dreamsim.py" >/dev/null; then
  echo "REFUSING: 73_cmts_dreamsim.py already running. pids: $(pgrep -f 73_cmts_dreamsim.py | tr '\n' ' ')"
  exit 1
fi

# 4 GPUs, disjoint seed blocks of 20: 5..24 / 25..44 / 45..64 / 65..84.
RANGES=( "5 25" "25 45" "45 65" "65 85" )
echo "=== [$(date +%H:%M:%S)] SCALE-UP v=$V on 4 GPUs -> $OUT  (save_img_every=$SAVE_EVERY)"
for g in 0 1 2 3; do
    read SS SE <<<"${RANGES[$g]}"
    env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=$g \
      setsid conda run -n diverse --no-capture-output python scripts/73_cmts_dreamsim.py \
        --model_path models/stabilityai/stable-diffusion-3.5-large \
        --device cuda:0 \
        --B_word black --B_seed 18 \
        --seed_start $SS --seed_end $SE \
        --dim $DIM --T $T --B $B --n0 $N0 --v $V --S $S \
        --save_img_every $SAVE_EVERY \
        --partial_id $((g + 10)) --tag cmts \
        --out_root "$OUT" \
        >> "$OUT/launch_scaleup_g${g}.log" 2>&1 </dev/null &
    echo "    GPU$g  v=$V  seeds $SS..$((SE-1))  detached pid=$!  ->  $OUT"
done
echo "=== all 4 workers detached (survive shell exit). ==="
echo "=== monitor: tail -f $OUT/launch_scaleup_g*.log   |  resume: re-run this script ==="
