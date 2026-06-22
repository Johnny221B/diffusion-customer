#!/usr/bin/env bash
# Detached v-sweep launcher. Same experiment as 73g_vsweep.sh, but each GPU worker
# is started with `setsid ... </dev/null &` so it lands in its OWN session/process
# group, reparented to init. It therefore SURVIVES the death of whatever shell (or
# harness background task) launched it. No parent `wait` holds the tree together,
# so nothing reaps the workers as a group.
#
#   bash scripts/73g_vsweep_detached.sh STAMP=0603_0137   # resume the in-flight sweep
#
# Resume semantics are identical: each worker reads its per-round _ckpt.pkl and
# continues from the last committed round. Re-running while workers are alive would
# double-launch -- guard below refuses if any 73_cmts proc is already running.
set -u
cd /home/linyuliu/jxmount/diffusion_custom

for a in "$@"; do case "$a" in STAMP=*) export "${a}";; esac; done
STAMP=${STAMP:-$(date +%m%d_%H%M)}

if pgrep -f "73_cmts_dreamsim.py" >/dev/null; then
  echo "REFUSING: 73_cmts_dreamsim.py already running. pids: $(pgrep -f 73_cmts_dreamsim.py | tr '\n' ' ')"
  exit 1
fi

T=200; B=8; N0=24; DIM=16; S=8.0
MSEED=5
VS=(0.5 1.0 2.0 4.0)
echo "=== [$(date +%H:%M:%S)] DETACHED v-sweep  v={${VS[*]}}  M=$MSEED T=$T B=$B S=$S  STAMP=$STAMP"

for g in 0 1 2 3; do
    V=${VS[$g]}
    OUT="outputs/cmts_v${V}_d${DIM}_B${B}_T${T}_${STAMP}"
    mkdir -p "$OUT"
    env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=$g \
      setsid conda run -n diverse --no-capture-output python scripts/73_cmts_dreamsim.py \
        --model_path models/stabilityai/stable-diffusion-3.5-large \
        --device cuda:0 \
        --B_word black --B_seed 18 \
        --seed_start 0 --seed_end $MSEED \
        --dim $DIM --T $T --B $B --n0 $N0 --v $V --S $S \
        --partial_id $g --tag cmts \
        --out_root "$OUT" \
        >> "$OUT/launch_g${g}.log" 2>&1 </dev/null &
    echo "    GPU$g  v=$V  detached pid=$!  ->  $OUT"
done
echo "=== all 4 workers detached. They survive this shell exiting. ==="
echo "=== monitor: tail outputs/cmts_v*_${STAMP}/launch_g*.log   |  resume: re-run this script same STAMP ==="
