#!/usr/bin/env bash
# M=20 overnight run, RESUMABLE. Re-run this exact command after any kill:
# finished trajectories finalize-and-skip; interrupted ones resume from their
# last committed round (round-level _ckpt.pkl). Safe to run many times.
#
# Default: paired v-comparison over the SAME 13h wall-clock --
#   GPU0,1 -> v=0.4 (canonical)   seeds 0..9    -> outputs/cmts_v0.4_...
#   GPU2,3 -> v=1.5 (more explore) seeds 0..9   -> outputs/cmts_v1.5_...
# Same seeds on both sides => paired comparison. M=10 per v (B*M=80, smooth).
#
# To run ONE config at full M=20 instead: set V_LO=V_HI and it still works
# (both dirs identical v); or edit the 4 launch lines to one OUT + seeds 0..19.
set -u
cd /home/linyuliu/jxmount/diffusion_custom

STAMP=${STAMP:-$(date +%m%d_%H%M)}
T=200; B=8; N0=24; DIM=16; S=8.0
V_LO=0.4; V_HI=1.5
OUT_LO="outputs/cmts_v${V_LO}_d${DIM}_B${B}_T${T}_${STAMP}"
OUT_HI="outputs/cmts_v${V_HI}_d${DIM}_B${B}_T${T}_${STAMP}"
mkdir -p "$OUT_LO" "$OUT_HI"

launch () {   # gpu  out  v  seed_start  seed_end  partial_id
    local G=$1 OUT=$2 V=$3 SS=$4 SE=$5 PID=$6
    env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=$G \
      conda run -n diverse --no-capture-output python scripts/73_cmts_dreamsim.py \
        --model_path models/stabilityai/stable-diffusion-3.5-large \
        --device cuda:0 \
        --B_word black --B_seed 18 \
        --seed_start $SS --seed_end $SE \
        --dim $DIM --T $T --B $B --n0 $N0 --v $V --S $S \
        --partial_id $PID --tag cmts \
        --out_root "$OUT" \
        > "$OUT/launch_g${G}.log" 2>&1 &
}

echo "=== [$(date +%H:%M:%S)] M=20 launch  T=$T B=$B  v=${V_LO}|${V_HI}  STAMP=$STAMP"
echo "    v=${V_LO} -> $OUT_LO   (seeds 0-9 on GPU 0,1)"
echo "    v=${V_HI} -> $OUT_HI   (seeds 0-9 on GPU 2,3)"

launch 0 "$OUT_LO" $V_LO 0 5 0
launch 1 "$OUT_LO" $V_LO 5 10 1
launch 2 "$OUT_HI" $V_HI 0 5 2
launch 3 "$OUT_HI" $V_HI 5 10 3
wait
echo "=== [$(date +%H:%M:%S)] all 4 GPU partials returned ==="

for OUT in "$OUT_LO" "$OUT_HI"; do
  python - "$OUT" <<'PY'
import json, glob, sys, numpy as np
out = sys.argv[1]
S = []
for f in sorted(glob.glob(f"{out}/sim*/summary.json")):
    S.append(json.load(open(f)))
if S:
    wf = np.array([s["winrate_first10"] for s in S])
    wl = np.array([s["winrate_last10"]  for s in S])
    bd = np.array([s["best_ds"] for s in S])
    print(f"  {out}: N={len(S)}  winrate {wf.mean():.3f}->{wl.mean():.3f} "
          f"(Δ={wl.mean()-wf.mean():+.3f})  best_ds={bd.mean():.4f}±{bd.std():.4f}")
PY
done
echo "=== done.  Re-run this script to resume any interrupted trajectory. ==="
