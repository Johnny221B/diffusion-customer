#!/bin/bash

# ==============================================================================
# 并行运行四个不同偏好的 Thompson Sampling 实验
# 每个实验分配到独立的 GPU 上
# ==============================================================================

echo ">>> 正在启动并行实验任务..."

# 1. 亮绿色轻奢风 (GPU 0)
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 python -m scripts.12_run_optimized_pipeline \
    --tag "green_luxury" \
    --target_preference "a bright green light luxury style sneaker" \
    --num_epochs 200 \
    --cold_start 20 > logs_green_luxury.txt 2>&1 &
echo "[GPU 0] 已启动: Green Luxury"

# 2. 粉紫幻彩风 (GPU 1)
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=1 python -m scripts.12_run_optimized_pipeline \
    --tag "purple_pastel" \
    --target_preference "a pastel purple iridescent sneaker with semi-transparent soles" \
    --num_epochs 200 \
    --cold_start 20 > logs_purple_pastel.txt 2>&1 &
echo "[GPU 1] 已启动: Purple Pastel"

# 3. 复古焦糖风 (GPU 2)
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=2 python -m scripts.12_run_optimized_pipeline \
    --tag "retro_caramel" \
    --target_preference "a retro sneaker with caramel brown leather suede" \
    --num_epochs 200 \
    --cold_start 20 > logs_retro_caramel.txt 2>&1 &
echo "[GPU 2] 已启动: Retro Caramel"

# 4. 极简海军蓝 (GPU 3)
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=3 python -m scripts.12_run_optimized_pipeline \
    --tag "minimalist_blue" \
    --target_preference "a minimalist sneaker with navy blue accents" \
    --num_epochs 200 \
    --cold_start 20 > logs_minimalist_blue.txt 2>&1 &
echo "[GPU 3] 已启动: Minimalist Blue"

echo ">>> 所有任务已提交到后台。你可以通过 'nvidia-smi' 或查看 logs_*.txt 文件监控进度。"

# 等待所有后台任务结束（可选，如果你希望脚本在所有实验跑完前不退出）
# wait