#!/usr/bin/env bash
# CPU-only fixed-image-seed discrete CM-TS screen.
# Each word is evaluated at image seed 18, matching the fixed-render-seed
# continuous setup more closely than the 40-seed oracle screen.
set -u
cd /home/linyuliu/jxmount/diffusion_custom

OUT=outputs/discrete_fixedseed_alpha_grid
mkdir -p "$OUT"

/home/linyuliu/.conda/envs/diverse/bin/python scripts/92_discrete_cmts_sweep.py \
    --fixed_image_seed 18 \
    --B_word bright --B_seed 18 \
    --alphas 5,6,8,10 \
    --vs 0.5,1,2,4 \
    --lams 5,10,20,50,100 \
    --eval_alpha 30 \
    --d 16 --N0 24 --T 200 --B 8 --n_sim 10 --S 8 \
    --plot_top 20 --roll 15 \
    --out_root "$OUT" \
    --tag fixedseed18_a_sweep
