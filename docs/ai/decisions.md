# Key Decisions

记录"为什么是现在这个样子"——避免新会话推翻已有判断、避免重复讨论。

## $R$ 和 $B$ 是两个不同的角色，不是同一件事的两个版本

- $R$（reference）= **距离的原点**。`ref_word=red, seed=1810772`，固定。
  整个 pipeline 所有 DreamSim 距离 $D_{w,k}$ 都是"到 $R$ 的距离"。
- $B$（competitor）= **比较的阈值**。`(canvas, 34)`，$D_B = 0.6205$。
  BO 标签 $y_{w,k} = 1 \iff D_{w,k} < D_B$。

**Why**: BO 模拟 pairwise 偏好学习 —— 用户心中有未观测的理想图 $R$、手边有当前满意图 $B$，
只能给"新图 vs $B$"二元反馈。只有 $R$ 没法二值化；只有 $B$ 没有方向。

**How to apply**: 看到 `competitor.png`（脚本 57 输出）不要以为它是 $B$——那是更早版本的 $B$ 候选，
seed 不在 multiseed 池里、当前 BO **没用**。当前 $B$ 是直接从 `dreams_matrix[canvas_idx, 34]` 查的。

详见 `EXPERIMENT_DOC.md §2.2`。

## Canonical 超参：α=30, d=8, N0=50, T=200, B=8, sims=100

- **α=30**: 标签 sigmoid 锐度，太软 ($\alpha=5$) 信号弱、太硬退化为指示函数。
- **d=8**: PCA 维度。早期实验扫过 d ∈ {16, 32, 64, 128}，d=8 在 logistic 假设 mis-specification
  和 surrogate 拟合速度间最平衡。详见 `EXPERIMENT_SUMMARY.md §2.2`。
- **prior_var=3.0** / **C=3.0**: logistic_bayesian 和 logistic_l2 共用同一损失面（仅 acquisition 不同）。
- **ε=0.1**: logistic_l2 的探索率。
- **B=canvas, s_B=34**: 这个组合下 $\bar p_{\text{oracle}} \approx 0.5$，学习信号最强。
  其它 $B$ 候选要么使 $\bar p$ 趋 0（学不到）、要么趋 1（白送）。

**How to apply**: 在 `63_bo_thompson_curve.py` 上改超参跑对比实验时，保留这套作为 baseline。

## Best-so-far 用 word-level cummax，不是 single-seed max

**Why**: 实验过用 "best single seed" 作 best-so-far，结果 5 个 surrogate 全部饱和到 0.999/1.000，
学习曲线完全失去区分度。`diag_def_compare.png` 下排展示了这个对照。

**How to apply**: 后续画图保持
`df.groupby(["sim_seed","model"])["p_soft"].cummax()` —— 每个 sim 内、按 picked 单词的 soft p_oracle 求 cummax。

## 主图用 soft oracle，不是 hard oracle

**Why**: hard oracle $\frac{1}{40}\sum 1[D<D_B]$ 是数据生成过程（Bernoulli 标签）在 $\alpha \to \infty$
极限下的退化版本；soft 才是和 §4.1 标签生成完全配套的真期望胜率。

**How to apply**: 给老师 / 报告看的图统一用 `bo_thompson_curve_per_model_soft.png` 这条线；
hard 版只在内部对比 / 诊断时用。两者 Spearman 排名相关 ≈ 0.951，结论一致。

## 步骤 57 用模板 prompt 抽 embedding，步骤 60 用空 prompt 生成

**Why**:

- 57 抽 $e_w$ 时要做"带模板"和"空模板"的**差分**才能定位 word 引入的 token 位置，没模板就没差分锚。
- 60 生成时**不要模板**，因为我们想测的就是"单 word 独自作为生成条件时 SD3.5 输出什么"，
  加模板会引入额外语义、污染 dreams 矩阵。

**How to apply**: 这种不对称是有意的，**别改成"两边都用模板"或"两边都不用模板"**。
详见 `EXPERIMENT_DOC.md §3`。

## 参考图 $R$ 一旦固定就不能动

**Why**: 改 $R$ → 整张 `dreams_matrix.npz` 的 9120 个 DreamSim 值全部要重算（4 GPU × 数小时）。

**How to apply**: 任何"换个参考图试试"的提议，先评估是不是真的需要重做整个 pipeline。
如果只是想看不同 $B$ 的效果，改 `--B_word --B_seed` 即可（dreams 不需要重算）。

## 5 个 surrogate 是固定阵容

`logistic_bayesian`（Laplace+TS）、`logistic_l2`（MAP+ε-greedy）、`poly2_logistic`、`gp_rbf`、`random_forest`。

**Why**: 这 5 个覆盖了"线性 vs 非线性"、"参数 vs 非参"、"Thompson vs greedy"几个正交维度，
作为"假设是否成立"的实证集合最有信息量。

**How to apply**: 加新 surrogate 要先想清楚它新增了哪个维度，不要为加而加。

## 单 sim_seed 内 picked == best-so-far 是正常的

**Why**: 单次 trajectory 中，`picked == best-so-far` 当且仅当这一步刷新了 cummax。
100 sims 平均后两条线会有 gap，因为不同 sim 在不同 t 刷新最大值——这不是 bug。
教师曾质疑"为什么不重合"，`diag_single_trajectory.png` 单 sim 视图证明了在单 trajectory 上确实
经常重合（红点处）。

**How to apply**: 别用"平均后两条线接近"作为"surrogate 拟合好"的判据；用 oracle ceiling 才是。
