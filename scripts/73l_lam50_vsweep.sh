#!/usr/bin/env bash
# lambda=50 FIXED, v in {4, 12} sweep, 5 seeds each. -- "广度 vs 方向" 二维探针
#
# WHY this experiment (from the ds~z diagnostic, 0612_soft10):
#   - lam16  乱撞 -> z 撒得广 -> ds~z 事后 R²=0.48 高, 但 beta 撞 clip S=8 饱和
#               -> argmax 方向退化 -> best 卡 0.42 (有数据没方向).
#   - lam100 不饱和 -> 方向真 -> best 到 0.27, 但 cov 半径太小 -> z 挤一团
#               -> ds~z R²=0.19 低 (有方向没覆盖).
#   两个 lambda 各瘸一条腿. 解法 = 去饱和(拿方向) + 补广度(拿覆盖):
#     lam=50  -> 中间值, beta 不撞 S=8, W=p(1-p) 存活 -> 方向活(像 lam100).
#     v       -> Thompson cov = v^2 * 0.5(H^-1+H^-T) 的半径旋钮, 与饱和正交.
#               v=4  = 现 baseline 半径(窄);  v=12 = 3x 半径(宽, 补回 lam16 的覆盖).
#   假设: lam50 + v=12 能同时拿到 lam100 的方向 和 lam16 的覆盖.
#
# ALPHA=30 (NOT 10): 诊断显示 alpha=10 时 alpha*std_ds=0.36 -> 标签近乎 0.5 噪声,
#   ds~z 的弱信号被 sigma 压平. alpha=30 -> alpha*std≈1.08 进入甜区, 否则 v-sweep 无结论.
#   alpha 同时驱动 soft-Bernoulli 训练标签 y~Bernoulli(sigma(alpha*(D_B-ds)))
#   和 true_p_soft 诊断列. plots/metrics 仍用 hard 指标 y_hard=1[ds<D_B].
#
# competitor 仍 black/18 (D_B=0.4704) 不变 -> 与现有 lam16/lam100 baseline 直接对照.
#
#   bash scripts/73l_lam50_vsweep.sh                  # start (new stamp)
#   bash scripts/73l_lam50_vsweep.sh STAMP=mmdd_HHMM  # resume an existing run
#
# 4 GPUs: each v split across 2 GPUs (seeds 0-2 / 3-4). DETACHED (setsid),
# RESUMABLE per-round (_ckpt.pkl), guard refuses to double-launch.
set -u
cd /home/linyuliu/jxmount/diffusion_custom

for a in "$@"; do case "$a" in STAMP=*) export "${a}";; esac; done
STAMP=${STAMP:-$(date +%m%d_%H%M)}

if pgrep -f "73_cmts_dreamsim.py" >/dev/null; then
  echo "REFUSING: 73_cmts_dreamsim.py already running. pids: $(pgrep -f 73_cmts_dreamsim.py | tr '\n' ' ')"
  exit 1
fi

LAM=50; T=200; B=8; N0=24; DIM=16; S=8.0; ALPHA=30; SAVE_EVERY=10
MODEL=models/stabilityai/stable-diffusion-3.5-large

launch () {  # $1=v  $2=gpu  $3=seed_start  $4=seed_end  $5=partial_id  $6=out_root
  local VV=$1 G=$2 SS=$3 SE=$4 PID=$5 OUT=$6
  mkdir -p "$OUT"
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=$G \
    setsid conda run -n diverse --no-capture-output python scripts/73_cmts_dreamsim.py \
      --model_path $MODEL --device cuda:0 \
      --B_word black --B_seed 18 \
      --seed_start $SS --seed_end $SE \
      --dim $DIM --T $T --B $B --n0 $N0 --v $VV --S $S --lam $LAM --alpha $ALPHA \
      --save_img_every $SAVE_EVERY \
      --partial_id $PID --tag cmts \
      --out_root "$OUT" \
      >> "$OUT/launch_g${G}.log" 2>&1 </dev/null &
  echo "    GPU$G  v=$VV  lam=$LAM  seeds[$SS,$SE)  detached pid=$!  ->  $OUT"
}

OUT4="outputs/cmts_lam50_v4.0_d16_B8_T200_${STAMP}"
OUT12="outputs/cmts_lam50_v12.0_d16_B8_T200_${STAMP}"
echo "=== [$(date +%H:%M:%S)] lam=50 fixed, v={4,12} x 5 seeds  alpha=$ALPHA S=$S T=$T  STAMP=$STAMP"
launch 4.0  0 0 3 0 "$OUT4"     # GPU0: v4  seeds 0,1,2
launch 4.0  1 3 5 1 "$OUT4"     # GPU1: v4  seeds 3,4
launch 12.0 2 0 3 2 "$OUT12"    # GPU2: v12 seeds 0,1,2
launch 12.0 3 3 5 3 "$OUT12"    # GPU3: v12 seeds 3,4
echo "=== all 4 workers detached (survive shell exit). STAMP=$STAMP ==="
echo "=== resume: bash scripts/73l_lam50_vsweep.sh STAMP=$STAMP ==="
