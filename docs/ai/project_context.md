# Project Context

## 目标

SD3.5 + Bayesian Optimization，用 pairwise 偏好学习找出能生成符合用户喜好的鞋图的词。
变量 $z$ = 单 T5 token embedding 经 PCA 降维后插入到固定 prompt 中的向量。
当前关注的核心问题：surrogate 假设 $P(y=1|z) = \sigma(\beta^\top z)$ 在我们 setting 下是否成立。

## 技术栈

- Python，conda env `diverse`（必须激活）
- PyTorch + diffusers，SD3.5-Large 本地权重
- sklearn（LogisticRegression / GaussianProcessRegressor / RandomForestClassifier / PCA / PolynomialFeatures）
- scipy（L-BFGS-B、multivariate_normal 采样）
- numpy / pandas / matplotlib
- DreamSim：models/ 下 vendored，import 已配好
- T5Tokenizer（来自 SD3.5 的 `tokenizer_3`）+ NLTK WordNet（仅 56 用）

## 首要参考文档

- **`outputs/bo_thompson_bcanvas_0512_1424/EXPERIMENT_DOC.md`** —— 当前 canonical 实验的端到端 ground truth，
  所有符号、公式、超参、复跑命令都在那里。任何实验细节先查它。
- **`EXPERIMENT_SUMMARY.md`**（仓库根） —— 历史方法学叙事（exp 46–52），讲为什么 d=8、为什么 logistic-vs-GP。
- **`docs/ai/decisions.md`** —— 关键决策与原因
- **`docs/ai/architecture.md`** —— 目录布局、术语、surrogates、多 GPU 模式
- **`docs/ai/todo.md`** —— 未决任务

## 运行命令（完整 pipeline）

详见 `EXPERIMENT_DOC.md §8`，简版：

```bash
# 1) 单词池（CPU）
python scripts/56_build_strict_pool.py --target_n 400

# 2) embedding + reference（单 GPU）
python scripts/57_collect_strict_pool.py \
    --model_path models/stabilityai/stable-diffusion-3.5-large --tag s228

# 3) 40 seed/word 图生成（4 GPU 并行 + merge）
for g in 0 1 2 3; do
  python scripts/60_collect_multiseed.py --device cuda:$g \
    --word_start $((g*57)) --word_end $(((g+1)*57)) --partial_id $g &
done; wait
python scripts/60b_merge_multiseed.py --out_dir outputs/multiseed_s228_M40_0510_0241/

# 4) BO 主实验（CPU，约 10 分钟）
python scripts/63_bo_thompson_curve.py \
    --d 8 --N0 50 --T 200 --B 8 --n_sim 100 \
    --alpha 30 --B_word canvas --B_seed 34 --tag bcanvas

# 5) 轨迹可视化
python scripts/64_render_bo_trajectory.py --run_dir outputs/bo_thompson_<tag>_<stamp>
```

多 GPU 脚本必须加这个前缀（解决 LD_LIBRARY_PATH 与 CUDA 冲突，**别去掉**）：

```bash
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0,1,2,3 python -m scripts.NN_name ...
```

## 测试命令

**没有正式 test suite / lint / CI**。`test.py` 与 `scripts/test.py` 是临时 sanity 脚本。
脚本自带 smoke 模式：缩小 `--N0/--T/--n_sim` 即可，例如 `--N0 5 --T 10 --n_sim 3`。

## 当前进展

完成：

- 56 → 64 完整 pipeline
- `EXPERIMENT_DOC.md` 完整端到端文档（含 §2.2 R/B 角色解释、§6.3 每条线的口径）
- 5-panel soft oracle 主图、15 张 trajectory 图（3 sims × 5 models）
- `diag_single_trajectory.png`、`diag_def_compare.png` 两张诊断图

未决：见 `docs/ai/todo.md`。
