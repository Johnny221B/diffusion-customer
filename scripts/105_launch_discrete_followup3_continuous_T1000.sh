#!/usr/bin/env bash
# Continuous follow-up from discrete fixed-seed sweep:
#   1) alpha=8,  v=1.0, lambda=100
#   2) alpha=10, v=0.5, lambda=100
#   3) alpha=8,  v=0.5, lambda=100
#
# Canonical environment:
#   bright/18, d=16, B=8, n0=24, S=8, T=1000, save image every 50 rounds.
#
# The 15 sim trajectories are balanced across 4 GPUs using per-GPU queues.
set -u
cd /home/linyuliu/jxmount/diffusion_custom

for a in "$@"; do case "$a" in STAMP=*) export "${a}";; esac; done
STAMP=${STAMP:-$(date +%m%d_%H%M)}

if pgrep -f "[7]3_cmts_dreamsim.py" >/dev/null; then
  echo "REFUSING: 73_cmts_dreamsim.py already running."
  pgrep -af "[7]3_cmts_dreamsim.py"
  exit 1
fi

MODEL=models/stabilityai/stable-diffusion-3.5-large
T=1000
B=8
N0=24
DIM=16
S=8.0
LAM=100
SAVE_EVERY=50
BWORD=bright
BSEED=18
export MODEL T B N0 DIM S LAM SAVE_EVERY BWORD BSEED STAMP

run_block () {  # gpu alpha v seed_start seed_end partial_id
  local G=$1 ALPHA=$2 V=$3 SS=$4 SE=$5 PID=$6
  local VLAB
  VLAB=$(python - <<PY
v=float("$V")
print(str(v).rstrip('0').rstrip('.'))
PY
)
  local LABEL="a${ALPHA}_v${VLAB}_lam100"
  local OUT="outputs/cmts_${LABEL}_bbright_d16_B8_T${T}_${STAMP}"
  mkdir -p "$OUT"
  echo "GPU$G alpha=$ALPHA v=$V lam=$LAM sims=[$SS,$SE) partial=$PID -> $OUT"
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=$G \
    conda run -n diverse --no-capture-output \
    python scripts/73_cmts_dreamsim.py \
      --model_path "$MODEL" --device cuda:0 \
      --B_word "$BWORD" --B_seed "$BSEED" \
      --seed_start "$SS" --seed_end "$SE" \
      --dim "$DIM" --T "$T" --B "$B" --n0 "$N0" \
      --v "$V" --S "$S" --lam "$LAM" --alpha "$ALPHA" \
      --save_img_every "$SAVE_EVERY" \
      --partial_id "$PID" --tag cmts --out_root "$OUT" \
      >> "$OUT/launch_T1000_g${G}_p${PID}.log" 2>&1
}
export -f run_block

echo "=== launch discrete-followup continuous tests; STAMP=$STAMP ==="

setsid bash -lc '
  cd /home/linyuliu/jxmount/diffusion_custom
  run_block 0 8 1.0 0 2 90
  run_block 0 8 0.5 0 2 91
' > "outputs/launch_followup3_${STAMP}_gpu0.log" 2>&1 </dev/null &
echo "GPU0 queue pid=$!"

setsid bash -lc '
  cd /home/linyuliu/jxmount/diffusion_custom
  run_block 1 8 1.0 2 5 92
  run_block 1 10 0.5 0 1 93
' > "outputs/launch_followup3_${STAMP}_gpu1.log" 2>&1 </dev/null &
echo "GPU1 queue pid=$!"

setsid bash -lc '
  cd /home/linyuliu/jxmount/diffusion_custom
  run_block 2 10 0.5 1 5 94
' > "outputs/launch_followup3_${STAMP}_gpu2.log" 2>&1 </dev/null &
echo "GPU2 queue pid=$!"

setsid bash -lc '
  cd /home/linyuliu/jxmount/diffusion_custom
  run_block 3 8 0.5 2 5 95
' > "outputs/launch_followup3_${STAMP}_gpu3.log" 2>&1 </dev/null &
echo "GPU3 queue pid=$!"

echo "=== four detached GPU queues launched ==="
echo "Outputs:"
echo "  outputs/cmts_a8_v1_lam100_bbright_d16_B8_T${T}_${STAMP}"
echo "  outputs/cmts_a10_v0.5_lam100_bbright_d16_B8_T${T}_${STAMP}"
echo "  outputs/cmts_a8_v0.5_lam100_bbright_d16_B8_T${T}_${STAMP}"
