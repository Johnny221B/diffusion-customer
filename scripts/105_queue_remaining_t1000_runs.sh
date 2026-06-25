#!/usr/bin/env bash
# Queue the remaining T1000 runs after the currently running 73_cmts_dreamsim.py
# workers finish. Intended order:
#   1. current a20_v2_lam50 continuation, already launched separately
#   2. a30_v1_lam50
#   3. a30_v4_lam50
set -u
cd /home/linyuliu/jxmount/diffusion_custom

LOG=outputs/cmts_top4_summary_0621_0437/t1000_queue.log
mkdir -p "$(dirname "$LOG")"

wait_for_idle () {
  while pgrep -f "[7]3_cmts_dreamsim.py" >/dev/null; do
    {
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] waiting for active 73_cmts_dreamsim.py workers..."
      pgrep -af "[7]3_cmts_dreamsim.py"
    } >> "$LOG" 2>&1
    sleep 300
  done
}

run_and_wait () {
  local script=$1
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching $script" >> "$LOG"
  bash "$script" >> "$LOG" 2>&1
  sleep 60
  wait_for_idle
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] completed $script" >> "$LOG"
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] queue started" >> "$LOG"
wait_for_idle
run_and_wait scripts/103_launch_a30_v1_lam50_T1000.sh
run_and_wait scripts/104_launch_a30_v4_lam50_T1000.sh
echo "[$(date '+%Y-%m-%d %H:%M:%S')] queue finished" >> "$LOG"
