#!/usr/bin/env bash
# Wait for GPUs to become free, then extend alpha=8, v=1, lambda=100
# from T=1000 to T=1500.
#
# Existing run:
#   outputs/cmts_a8_v1_lam100_bbright_d16_B8_T1000_0629_0842
#
# Split:
#   GPU0 -> sim000/sim001
#   GPU1 -> sim002
#   GPU2 -> sim003
#   GPU3 -> sim004
set -u
cd /home/linyuliu/jxmount/diffusion_custom

OUT=outputs/cmts_a8_v1_lam100_bbright_d16_B8_T1000_0629_0842
MODEL=models/stabilityai/stable-diffusion-3.5-large
MEM_LIMIT_MIB=5000

wait_and_run () {  # gpu seed_start seed_end partial_id
  local G=$1 SS=$2 SE=$3 PID=$4
  local LOG="$OUT/extend_T1500_g${G}_p${PID}.log"
  echo "=== wait then extend alpha=8 v=1 lambda=100 sims=[$SS,$SE) to T=1500 on GPU${G} ===" >> "$LOG"
  date >> "$LOG"

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

  echo "GPU${G} is free enough; launching extension" >> "$LOG"
  date >> "$LOG"
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$G" \
    conda run -n diverse --no-capture-output \
    python scripts/73_cmts_dreamsim.py \
      --model_path "$MODEL" --device cuda:0 \
      --B_word bright --B_seed 18 \
      --seed_start "$SS" --seed_end "$SE" \
      --dim 16 --T 1500 --B 8 --n0 24 \
      --v 1.0 --S 8.0 --lam 100 --alpha 8 \
      --save_img_every 50 \
      --partial_id "$PID" --tag cmts --out_root "$OUT" \
      >> "$LOG" 2>&1
  echo "=== extension command finished ===" >> "$LOG"
  date >> "$LOG"
}
export -f wait_and_run
export OUT MODEL MEM_LIMIT_MIB

setsid bash -lc 'cd /home/linyuliu/jxmount/diffusion_custom; wait_and_run 0 0 2 100' >/tmp/a8_v1_T1500_wait_g0.log 2>&1 </dev/null &
echo "GPU0 waiter pid=$!"
setsid bash -lc 'cd /home/linyuliu/jxmount/diffusion_custom; wait_and_run 1 2 3 101' >/tmp/a8_v1_T1500_wait_g1.log 2>&1 </dev/null &
echo "GPU1 waiter pid=$!"
setsid bash -lc 'cd /home/linyuliu/jxmount/diffusion_custom; wait_and_run 2 3 4 102' >/tmp/a8_v1_T1500_wait_g2.log 2>&1 </dev/null &
echo "GPU2 waiter pid=$!"
setsid bash -lc 'cd /home/linyuliu/jxmount/diffusion_custom; wait_and_run 3 4 5 103' >/tmp/a8_v1_T1500_wait_g3.log 2>&1 </dev/null &
echo "GPU3 waiter pid=$!"
