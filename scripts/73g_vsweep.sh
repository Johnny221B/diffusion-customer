#!/usr/bin/env bash
# v-sweep: tune the posterior-covariance EXPLORATION INDEX v.
#   draw  beta_tilde ~ N(beta_hat, v^2 * H^{-1})   -- v is the index.
# One v per GPU, SAME seeds 0..4 across all v (paired). M=5, T=200, B=8, S=8.
# Produces 4 learning curves; pick the v whose per-round win-rate actually rises.
#
# RESUMABLE: re-run with the SAME STAMP to continue any interrupted trajectory:
#   bash scripts/73g_vsweep.sh STAMP=<original_stamp>
# Finished trajectories finalize-and-skip; interrupted ones resume from their
# last committed round (round-level _ckpt.pkl).
set -u
cd /home/linyuliu/jxmount/diffusion_custom

# allow `STAMP=0603_1530 bash 73g_vsweep.sh`  AND  `bash 73g_vsweep.sh STAMP=...`
for a in "$@"; do case "$a" in STAMP=*) export "${a}";; esac; done
STAMP=${STAMP:-$(date +%m%d_%H%M)}

T=200; B=8; N0=24; DIM=16; S=8.0
MSEED=5                       # seeds 0..MSEED-1 per v
VS=(0.5 1.0 2.0 4.0)          # GPU 0,1,2,3
echo "=== [$(date +%H:%M:%S)] v-sweep  v={${VS[*]}}  M=$MSEED T=$T B=$B S=$S  STAMP=$STAMP"

declare -a OUTS
for g in 0 1 2 3; do
    V=${VS[$g]}
    OUT="outputs/cmts_v${V}_d${DIM}_B${B}_T${T}_${STAMP}"
    OUTS[$g]="$OUT"
    mkdir -p "$OUT"
    echo "    GPU$g  v=$V  ->  $OUT"
    env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=$g \
      conda run -n diverse --no-capture-output python scripts/73_cmts_dreamsim.py \
        --model_path models/stabilityai/stable-diffusion-3.5-large \
        --device cuda:0 \
        --B_word black --B_seed 18 \
        --seed_start 0 --seed_end $MSEED \
        --dim $DIM --T $T --B $B --n0 $N0 --v $V --S $S \
        --partial_id $g --tag cmts \
        --out_root "$OUT" \
        > "$OUT/launch_g${g}.log" 2>&1 &
done
wait
echo "=== [$(date +%H:%M:%S)] all 4 v partials returned ==="

echo ""
echo "=== v-sweep summary (per-round winrate first10 -> last10) ==="
for g in 0 1 2 3; do
  python - "${VS[$g]}" "${OUTS[$g]}" <<'PY'
import json, glob, sys, numpy as np
v, out = sys.argv[1], sys.argv[2]
S = [json.load(open(f)) for f in sorted(glob.glob(f"{out}/sim*/summary.json"))]
if not S:
    print(f"  v={v}: (no completed trajectories yet)"); raise SystemExit
wf = np.array([s["winrate_first10"] for s in S])
wl = np.array([s["winrate_last10"]  for s in S])
bd = np.array([s["best_ds"] for s in S])
bn = np.array([s["final_beta_norm"] for s in S])
print(f"  v={v:<4} N={len(S)}  winrate {wf.mean():.3f}->{wl.mean():.3f} "
      f"(Δ={wl.mean()-wf.mean():+.3f})  best_ds={bd.mean():.4f}  ||β||={bn.mean():.2f}")
PY
done
echo "=== done.  Re-run 'bash scripts/73g_vsweep.sh STAMP=$STAMP' to resume. ==="
