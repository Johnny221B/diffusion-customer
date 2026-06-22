#!/usr/bin/env bash
# Extend a CHOSEN v from M=5 -> M=20 by running 15 NEW trajectories (seeds 5..19)
# into the SAME directory as the v-sweep run. Pooled automatically (glob sim*).
#
# Usage:
#   bash scripts/73h_extend.sh V=2.0 STAMP=0603_0137
# Config is hard-locked to the v-sweep's (T=200 B=8 n0=24 S=8 black/18) so the
# new trajectories are poolable with the original 5. RESUMABLE: re-run the same
# command; finished trajectories finalize-and-skip, interrupted ones resume.
set -u
cd /home/linyuliu/jxmount/diffusion_custom
for a in "$@"; do case "$a" in V=*|STAMP=*) export "${a}";; esac; done
: "${V:?must pass V=<winner v, e.g. 2.0>}"
: "${STAMP:?must pass STAMP=<original sweep stamp, e.g. 0603_0137>}"

T=200; B=8; N0=24; DIM=16; S=8.0
OUT="outputs/cmts_v${V}_d${DIM}_B${B}_T${T}_${STAMP}"
[ -d "$OUT" ] || { echo "ERROR: $OUT not found (check V and STAMP)"; exit 1; }
echo "=== [$(date +%H:%M:%S)] extend v=$V  seeds 5..19 -> $OUT  (pool to M=20)"

# 15 new seeds (5..19) split across 4 GPUs: 4,4,4,3.  Fresh partial_ids 4..7.
RANGES=( "5 9" "9 13" "13 17" "17 20" )
for g in 0 1 2 3; do
    read SS SE <<<"${RANGES[$g]}"
    env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=$g \
      conda run -n diverse --no-capture-output python scripts/73_cmts_dreamsim.py \
        --model_path models/stabilityai/stable-diffusion-3.5-large \
        --device cuda:0 \
        --B_word black --B_seed 18 \
        --seed_start $SS --seed_end $SE \
        --dim $DIM --T $T --B $B --n0 $N0 --v $V --S $S \
        --partial_id $((g + 4)) --tag cmts \
        --out_root "$OUT" \
        > "$OUT/launch_extend_g${g}.log" 2>&1 &
done
wait
echo "=== [$(date +%H:%M:%S)] extend done ==="

python - "$OUT" "$V" <<'PY'
import json, glob, sys, numpy as np
out, v = sys.argv[1], sys.argv[2]
S = [json.load(open(f)) for f in sorted(glob.glob(f"{out}/sim*/summary.json"))]
wf = np.array([s["winrate_first10"] for s in S])
wl = np.array([s["winrate_last10"]  for s in S])
bd = np.array([s["best_ds"] for s in S])
print(f"  v={v} POOLED N={len(S)} trajectories (target 20)")
print(f"    winrate {wf.mean():.3f} -> {wl.mean():.3f}  (Δ={wl.mean()-wf.mean():+.3f})")
print(f"    best_ds {bd.mean():.4f} ± {bd.std():.4f}")
PY
echo "=== re-run 'bash scripts/73h_extend.sh V=$V STAMP=$STAMP' to resume. ==="
