#!/usr/bin/env bash
# Four radius settings, five paired seeds each. Smoke runs are continued in place.
set -euo pipefail
cd /home/linyuliu/jxmount/diffusion_custom
SWEEP_ROOT=${1:?Usage: bash scripts/124_launch_tau_sweep.sh NEW_OUTPUT_DIRECTORY}
if [[ -e "$SWEEP_ROOT/launch_manifest.tsv" ]]; then
  echo "Refusing to duplicate an existing launch: $SWEEP_ROOT" >&2
  exit 1
fi
mkdir -p "$SWEEP_ROOT/source_snapshot/scripts" "$SWEEP_ROOT/source_snapshot/src"
cp scripts/73_cmts_dreamsim.py scripts/124_launch_tau_sweep.sh "$SWEEP_ROOT/source_snapshot/scripts/"
cp src/cmts_sim.py src/sd35_batch_generator.py src/scorer.py "$SWEEP_ROOT/source_snapshot/src/"
printf 'gpu\ttau_scale\tpid\toutput\n' > "$SWEEP_ROOT/launch_manifest.tsv"
scales=(1 1.1 1.25 1.5)
for gpu in 0 1 2 3; do
  scale=${scales[$gpu]}
  run="$SWEEP_ROOT/tau${scale}"
  mkdir -p "$run"
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$gpu" \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    MPLCONFIGDIR=/tmp/cmts_matplotlib PYTHONUNBUFFERED=1 \
    setsid bash -c '
      set -euo pipefail
      run=$1; scale=$2
      trap '\''code=$?; printf "%s\n" "$code" > "$run/exit_code.txt"'\'' EXIT
      common=(--model_path models/stabilityai/stable-diffusion-3.5-large
              --device cuda:0 --B_word bright --B_seed 18 --dim 16 --k 10
              --B 8 --n0 24 --v 0.5 --S 8 --lam 100 --alpha 8
              --ref_seed 1810772 --batch_size 8 --save_img_every 20
              --partial_id 0 --tau_scale "$scale" --out_root "$run")
      echo "SMOKE: tau_scale=$scale"
      conda run -n diverse --no-capture-output python scripts/73_cmts_dreamsim.py \
        "${common[@]}" --seed_start 0 --seed_end 1 --T 2
      echo "SMOKE PASSED; continuing five seeds to T=200"
      conda run -n diverse --no-capture-output python scripts/73_cmts_dreamsim.py \
        "${common[@]}" --seed_start 0 --seed_end 5 --T 200
      echo "COMPLETED: tau_scale=$scale"
    ' tau-worker "$run" "$scale" > "$run/worker.log" 2>&1 < /dev/null &
  pid=$!
  printf '%s\t%s\t%s\t%s\n' "$gpu" "$scale" "$pid" "$run" >> "$SWEEP_ROOT/launch_manifest.tsv"
  echo "GPU $gpu: tau_scale=$scale, PID=$pid, log=$run/worker.log"
done
