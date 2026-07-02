#!/usr/bin/env bash
# Extend alpha=10, v=1, lambda=100 good3 (sim001/sim002/sim003)
# from T=1200 to T=1500.
#
# This script first waits for the alpha=8,v=1,lambda=100 T=1500 extension
# queue to finish, because that queue already owns the current free-GPU
# scheduling. After that, it waits for GPUs and launches the three good sims.
set -u
cd /home/linyuliu/jxmount/diffusion_custom

OUT=outputs/cmts_a10_v1_lam100_bbright_d16_B8_T1000_0625_2312
A8_OUT=outputs/cmts_a8_v1_lam100_bbright_d16_B8_T1000_0629_0842
MODEL=models/stabilityai/stable-diffusion-3.5-large
LOG="$OUT/extend_T1500_good3_wait.log"
MEM_LIMIT_MIB=5000

echo "=== queue alpha=10 v=1 lambda=100 good3 to T=1500 ===" >> "$LOG"
date >> "$LOG"

all_a8_done () {
  python - <<'PY'
from pathlib import Path
import pandas as pd
D=Path("outputs/cmts_a8_v1_lam100_bbright_d16_B8_T1000_0629_0842")
ok=True
for s in range(5):
    tr=D/f"sim{s:03d}"/"trajectory.csv"
    if not tr.exists():
        ok=False; break
    df=pd.read_csv(tr, usecols=["t","phase"])
    m=df[df.phase=="main"]
    if len(m)==0 or int(m.t.max()) < 1499:
        ok=False; break
print("yes" if ok else "no")
PY
}

echo "Waiting for alpha=8,v=1,lambda=100 T=1500 extension to finish..." >> "$LOG"
while true; do
  status=$(all_a8_done)
  echo "$(date '+%F %T') a8_T1500_done=$status" >> "$LOG"
  if [ "$status" = "yes" ]; then
    break
  fi
  sleep 300
done

wait_gpu () {  # gpu
  local G=$1
  while true; do
    local used util
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$G" | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$G" | tr -d ' ')
    echo "$(date '+%F %T') GPU${G} used=${used}MiB util=${util}%" >> "$LOG"
    if [ "${used:-999999}" -lt "$MEM_LIMIT_MIB" ]; then
      break
    fi
    sleep 300
  done
}

run_one () {  # gpu seed partial
  local G=$1 S=$2 PID=$3
  local RUNLOG="$OUT/extend_T1500_good3_sim${S}_g${G}.log"
  echo "Waiting GPU${G} for sim${S}" >> "$LOG"
  wait_gpu "$G"
  echo "Launching sim${S} on GPU${G}" >> "$LOG"
  date >> "$LOG"
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$G" \
    conda run -n diverse --no-capture-output \
    python scripts/73_cmts_dreamsim.py \
      --model_path "$MODEL" --device cuda:0 \
      --B_word bright --B_seed 18 \
      --seed_start "$S" --seed_end "$((S+1))" \
      --dim 16 --T 1500 --B 8 --n0 24 \
      --v 1.0 --S 8.0 --lam 100 --alpha 10 \
      --save_img_every 50 \
      --partial_id "$PID" --tag cmts --out_root "$OUT" \
      >> "$RUNLOG" 2>&1
  echo "Finished sim${S} on GPU${G}" >> "$LOG"
  date >> "$LOG"
}
export -f wait_gpu run_one
export OUT MODEL LOG MEM_LIMIT_MIB

setsid bash -lc 'cd /home/linyuliu/jxmount/diffusion_custom; run_one 0 1 110' >/tmp/a10_v1_T1500_sim1_launcher.log 2>&1 </dev/null &
echo "sim001 waiter pid=$!" >> "$LOG"
setsid bash -lc 'cd /home/linyuliu/jxmount/diffusion_custom; run_one 1 2 111' >/tmp/a10_v1_T1500_sim2_launcher.log 2>&1 </dev/null &
echo "sim002 waiter pid=$!" >> "$LOG"
setsid bash -lc 'cd /home/linyuliu/jxmount/diffusion_custom; run_one 2 3 112' >/tmp/a10_v1_T1500_sim3_launcher.log 2>&1 </dev/null &
echo "sim003 waiter pid=$!" >> "$LOG"

echo "=== launched good3 GPU waiters ===" >> "$LOG"
date >> "$LOG"
