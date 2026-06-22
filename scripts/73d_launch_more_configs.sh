#!/usr/bin/env bash
# 续 73c，再跑 3 个 config（D/E/F），跟前 3 个组成完整 2×3 grid：
#   dim ∈ {16, 32, 64}  ×  v ∈ {0.4, 2.0}
# 每个 config 内 4 GPU 并行，NSEED=12（每 GPU 3 trajectory）。
# 总耗时预计 ~5.4h。需在 73c 跑完之后启动（否则会争抢 4 张 GPU）。
set -u
cd /home/linyuliu/jxmount/diffusion_custom

STAMP=$(date +%m%d_%H%M)
NSEED=12
TRAJ_PER_GPU=$((NSEED / 4))   # = 3

run_config () {
    local TAG=$1 DIM=$2 V=$3 SC=$4
    local OUT="outputs/cmts_${TAG}_d${DIM}_v${V}_S${SC}_${STAMP}"
    mkdir -p "$OUT"
    echo ""
    echo "=== [$(date +%H:%M:%S)] config $TAG: d=$DIM v=$V S=$SC → $OUT"

    for g in 0 1 2 3; do
        env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=$g \
          conda run -n diverse --no-capture-output python scripts/73_cmts_dreamsim.py \
            --model_path models/stabilityai/stable-diffusion-3.5-large \
            --device cuda:0 \
            --B_word black --B_seed 18 \
            --seed_start $((g * TRAJ_PER_GPU)) \
            --seed_end   $((g * TRAJ_PER_GPU + TRAJ_PER_GPU)) \
            --dim "$DIM" --T 200 --n0 24 --v "$V" --S "$SC" \
            --partial_id $g --tag "$TAG" \
            --out_root "$OUT" \
            > "$OUT/launch_g${g}.log" 2>&1 &
    done
    wait
    echo "=== [$(date +%H:%M:%S)] config $TAG done"

    python - <<PY
import json, glob, numpy as np
summaries = []
for f in sorted(glob.glob("$OUT/summaries_partial*.json")):
    summaries.extend(json.load(open(f)))
if summaries:
    best = [s["best_ds"]    for s in summaries]
    hit  = [s["main_hit_rate"] for s in summaries]
    print(f"  N={len(summaries)}  best_ds mean={np.mean(best):.4f}±{np.std(best):.4f}")
    print(f"  main_hit_rate mean={np.mean(hit):.3f}±{np.std(hit):.3f}")
PY
}

# 2×3 grid 的第二行
run_config "D" 32 2.0 8.0     # dim×explore cross
run_config "E" 64 0.4 8.0     # max dim alone
run_config "F" 64 2.0 8.0     # dim+explore combined

echo ""
echo "=== [$(date +%H:%M:%S)] D/E/F DONE — 2×3 grid complete ==="
