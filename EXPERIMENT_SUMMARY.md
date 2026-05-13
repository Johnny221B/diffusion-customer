# 实验总结与未来方向

记录从 exp 46 到 exp 52 这一轮的工作：iid sample-and-train 路线的验证、模型选择、端到端 pilot，以及对当前方法的诚实定位与下一步方向。

---

## 1. 项目背景与约束

- **目标**：用 Stable Diffusion 3.5 + Bayesian Optimization 生成符合客户偏好的图片
- **可控变量**：插入到 prompt 中某位置的 token embedding（4096 维），由 PCA 映射 W 从 d 维 z-vector 投影而来
- **反馈**：人类比较"我们生成的图 vs 固定 competitor B"，只给二元标签 y ∈ {0, 1}
- **硬约束**：
  - 标签必须是二元的（不能用连续 dreamsim）
  - B 必须固定（是真实业务中的 competitor，不能 adaptive）

---

## 2. iid 验证实验的核心问题

**问题陈述**：随机采样大量词 → 取它们的 token embedding → SD3.5 生图 → 与 B 比对得到二元标签 → 训练分类器 → 能不能学到能产生好图的 z 分布？

最终结论分三层。

### 2.1 信号存在，但几何形状被早期错判

- 词 embedding 与 dreamsim 在 d=16 PCA 上的 Pearson 相关 ~0.4–0.5（exp 46），所以"embedding 里有信号"是真的。
- 信号是 **径向的 (radial)** —— 好 embedding 聚在 ref 周围一个壳内，不是沿一个线性方向。
- 这是 logistic 一直学不到东西的根因：模型设定错了（linear decision boundary）。

### 2.2 模型 × 维度的匹配关系

binary label，B 取在 dreamsim 中位数位置（B='indoor', d_B=0.4657, 50/50 pos），N=80 训练，held-out AUC：

| dim | logistic AUC | GP-RBF AUC |
|---|---|---|
| 16 | 0.64 | **0.70** |
| 32 | 0.67 | **0.72** |
| 64 | 0.64 | 0.63 |
| 128 | 0.64 | **0.54（崩塌）** |

结论：**降维不是万能的，维度 × 模型族必须匹配**。GP-RBF 在 d=16/32 享受灵活性优势；在 d≥64 因 curse of dimensionality 崩塌（RBF 核在高维里点对距离趋同）。

### 2.3 Competitor 配置比模型选择更关键

- B 取在 dreamsim 中位数位置 → 标签熵最大 → 信号最强
- 偏离中位数（例如 B 接近 ref 或接近最差词）会让正/负样本严重失衡，几乎学不到东西

### 2.4 Active 比 iid 显著更高效

exp 51 sequential bandit：
- GP-UCB 在 t=50 时 best_d=0.304
- random 需 t≈120 才能达到同等水平
- **2.4× 速度加速**

---

## 3. 实验列表（exp 46–52）

| Exp | 脚本 | 目的 | 主要结论 |
|---|---|---|---|
| 46 | `46_word_pca_corr.py` | 词 embedding ↔ dreamsim 相关性 | d=16 PCA 上 Pearson ≈ 0.4–0.5 |
| 47 | scaling 系列 | 维度扫描 d ∈ {16,32,64,128} | d=16 最稳 |
| 48 | `48_replot_from_N50.py` | 过滤 N≥50 重画曲线 | 仅可视化 |
| 49 | `49_compare_methods.py` | logistic / ridge / GP-reg 对比 | GP 强，但不公平（用了连续 oracle） |
| 50 | `50_classifier_binary.py` | logistic vs GP-clf（都用二元标签） | **公平对比，GP 胜（d=16 时 0.70 vs 0.64）** |
| 51 | `51_active_vs_iid.py` | random / UCB-log / UCB-GP 对比 | UCB-GP 2.4× 加速 |
| 52 | `52_batch_bo_pilot.py` | 端到端 SD3.5 + DreamSim pilot | 8 iter 后 batch_pos = 1.0 |

输出目录：
- `outputs/scaling_red_vs_green_d{16,32,64,128}_0416_1723/` — exp 49/50/51 的产物
- `outputs/pilot_bo_red_indoor_s0p{1,3,6}_0419_0219/` — exp 52 不同 σ 的 pilot 结果

---

## 4. GP-RBF vs Logistic 的本质区别

### 数学形式

**Logistic**：
```
P(y=1 | z) = σ(w·z + b)
```
学一个 **超平面**，决策边界是 (d-1) 维平面。

**GP-RBF 分类器**：
```
P(y=1 | z) = σ( Σᵢ αᵢ · k(zᵢ, z) ),    k(zᵢ, z) = exp(-‖zᵢ - z‖² / 2ℓ²)
```
学一个 **任意光滑决策面**。每个训练点贡献一个高斯小帽子，相加 + sigmoid。

### 几何直觉

设 ref 在 z-空间对应位置 z*。dreamsim 低 ↔ z 离 z* 近。

- **Logistic**："y=1 的点 的平均方向是什么？" —— 如果 y=1 从 z* 四面八方散开，平均方向 ≈ 0，学不到东西
- **GP-RBF**："y=1 的点 到 query 的距离是多少？" —— 天然契合径向结构

```
          +  +                       +  +
       +    *    +                +   *    +
          +  +                       +  +
```
（+ = y=1, * = ref 中心）

- Logistic 只能切一刀 → "* 左边还是右边"
- GP-RBF 能画圈 → "离 * 有多远"

### 全局 vs 局部

|  | Logistic | GP-RBF |
|---|---|---|
| 参数量 | d+1（固定） | 每个训练点贡献一项（随 N 涨） |
| 外推 | 线性外推到整个空间 | 远离训练点 → 回到先验 |
| 高维行为 | 稳定 | **崩塌**（核值趋同，分不开点） |
| 需要的 N | 少 | 多（随维度指数式涨） |

**一句话**：logistic 假设信号是"某个方向"，GP-RBF 假设信号是"某个距离"。我们的 dreamsim 是后者，所以 GP-RBF 赢；但必须先 PCA 降到 d=16 才能避免崩塌。

---

## 5. 关于端到端 pilot 的诚实更正

**之前的描述是错的**：我之前说 exp 52 是 "batch BO with GP"，**其实不是 BO**。

看 `52_batch_bo_pilot.py:64-86` 中的 `propose_batch`：

```python
# GP 只用来给 observed points 排序，选出 top-20 父代池
p = gp_predict(gp_tup, Z_known)
top_idx = np.argsort(p)[::-1][:top_m]

# 每个 batch 样本：在 top-20 里按 prob 权重挑 2 个 → 线性插值 → 加高斯噪声
a, b = rng.choice(top_idx, size=2, replace=True, p=w)
alpha = rng.uniform()
batch[k] = alpha * Z_known[a] + (1 - alpha) * Z_known[b]
batch[k] += sigma * rng.randn(d)
```

### 这其实是遗传算法式 (GA-style) 采样

- **父代池**：GP 在 observed points 上排序，取 top-20
- **Crossover**：alpha · Z_a + (1-alpha) · Z_b，alpha ~ U(0,1)
- **Mutation**：σ · randn(d)，σ=0.3

**没有真正的 BO 元素**：
- 没有对候选 z 计算 acquisition function（EI/UCB）
- 没有 argmax
- GP 只参与"选父代"，不参与"评估候选 z"
- explore-exploit 的 trade-off 完全靠 alpha 的 uniform 抽样和 σ·randn

### 这对结论的影响

pilot 中 "batch_pos=1.0 by iter 8" 这个结果是真的，但它实际证明的是：

> **当 warmup 已经包含正样本时，"在好词的 top-20 之间插值 + 加噪声" 就足以产生持续击败 B 的新 z。**

这背后的 **隐含几何假设** 是：词 embedding 空间在 top-k 区域是 convex-ish 的（两个好词的中点也是好 z）。我们 **从未单独检验过** 这个假设；它是 pilot 成立的真正原因。

GP 在这套流程里的贡献其实被高估了 —— 极端地说，把 GP 换成"random pick 2 from observed positives"可能效果差不多。

---

## 6. 未来方向（按紧迫性排序）

### 6.1 Ablation：GP 到底有没有用？（最优先）

变体对比：
- A（当前）：GP 排序 → top-20 父代 → 插值 + 噪声
- B：observed y=1 的全部点作父代池 → 插值 + 噪声（无 GP）
- C：random pick 任意 2 observed → 插值 + 噪声

如果 A ≈ B ≈ C，说明现在这套是 "posterior-weighted crossover" 的噱头，实际贡献来自词空间几何，**应该换个故事来讲方法**。

### 6.2 验证"词空间 convex-ish"这个隐含假设

随便取两个 dreamsim 低的词，插值生成图，看 dreamsim 是不是也低。这是 pilot 成立的真正原因，但从未单独检验。

### 6.3 真正的 BO 对照

- 方案 A：构造较大候选 z pool（例如 1000 个插值点），对每个用 GP 算 UCB = μ + κσ，argmax
- 方案 B：在连续空间用 scipy.optimize 做 multi-start UCB maximization

和当前 GA-style 比 batch_pos 收敛速度。

### 6.4 标签噪声鲁棒性（产品落地的 gate）

DreamSim 是确定性 oracle，但人类标签是噪声的。用 `Bernoulli(σ(k·(d_B − d)))` 模拟，sweep k（敏感度），看 GP-UCB 在 k=5/2/1 时还能不能收敛。

### 6.5 ref 从词走到真实图

现在 ref="red" 是个词，真实场景 ref 是客户的鞋图。把 ref 换成真图经 DreamSim encoder 后的 embedding，验证整个 pipeline 还稳不稳。

### 6.6 复杂 ref 的 scaling

"red" 是单属性。试 "red leather high-heel with silver buckle" 这种多属性组合，学习曲线还能不能收敛？

### 6.7 小规模 human study

5–10 人 × 50 对 (gen vs B)，验证 DreamSim 与人类偏好的 Spearman 相关。从实验室到产品的必经验证。

### 优先级建议

**6.1 → 6.2 → 6.4 → 6.5/6.6 → 6.3 → 6.7**

- 6.1, 6.2 决定方法论故事怎么讲
- 6.4 是产品落地的 gate
- 6.3 决定要不要走向严格 BO（取决于 6.1 的结果）

---

## 7. 关键 takeaways

1. **iid sample-and-train 路线在词空间内可行**，但效率低（每样本一次 SD3.5 调用很贵），active 是必须的。
2. **几何假设决定模型选择**：径向信号 → GP-RBF；线性信号 → logistic。我们的场景是前者。
3. **维度必须低**（d=16 sweet spot）。GP-RBF 在 d≥64 崩。
4. **B 必须取中位数位置**，否则标签熵不够。
5. **当前 pilot 不是真正的 BO**，是 GA-style crossover + mutation。它能 work 是因为词空间在 top-k 区域 convex-ish，**而不是因为算法本身的 BO 性质**。
6. **下一步最关键的是 ablation 6.1**：搞清楚 GP 在当前 pipeline 里到底贡献了多少。
