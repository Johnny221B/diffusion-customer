# CM-TS Canonical Continuous Setting: α=30, v=2, λ=50

> 本文记录 2026-06-22 起用于后续 continuous 实验的 canonical setting。
> 它是 `cmts_continuous.md` 的具体实验配置补充；算法总历史仍以该文档为准。
> 当前状态：同一批 5 条 trajectory 已配对续跑并完成 T=500。

## 1. TL;DR

当前推荐 baseline：

```text
alpha=30, v=2, lambda=50, d=16, k=10, S=8
n0=24, B=8, competitor=bright/18, D_B=0.4704614282
5 simulation trajectories, fixed render seed=1810772
```

选择它的原因：

- 在四组 discrete 筛选结果的 continuous 验证中，T=200 时它的 late true-p 最高：
  `0.5928 ± 0.0182 SE`。
- `predicted_p=0.6152`，late calibration gap 只有 `+0.0225`。
- 5/5 trajectory 均未发生 beta norm saturation；T=200 平均终值 `||β||=0.059`。
- 配对续跑至 T=300 后，最后 20 轮 true-p 达到 `0.6151`，最后 10 轮达到 `0.6214`。
- hard win-rate 与 mean distance 同时改善，不是只改变 competitor 口径造成的表面提升。

不要把 discrete 最佳配置 `α=20,v=0.5,λ=50` 当 continuous baseline。它在 continuous
验证中 5/5 trajectory 全部撞到 `S=8`，late true-p 只有 `0.4411`，predicted-p 则错误地贴到 1。

## 2. 固定数据与图像角色

### 2.1 Reference R

- 文件：`outputs/strict_pool_s228_0429_0119/reference.png`
- 语义来源：word=`red`, seed=`1810772`
- 角色：所有 DreamSim distance 的原点。
- 不能随意更换。更换 R 会改变所有 distance，历史实验不再可直接比较。

### 2.2 Competitor B

- word=`bright`
- seed index=`18`
- `D_B=0.4704614281654358`
- 角色：把 distance 转换为胜负概率的比较阈值。

训练标签的条件概率为：

$$
p_{t,b}^{\mathrm{true}}
= \sigma\!\left(30(D_B-D_{t,b})\right),
\qquad
y_{t,b}\sim\operatorname{Bernoulli}(p_{t,b}^{\mathrm{true}}).
$$

`bright/18` 是一个强 competitor，约在历史 image-distance 分布的 3rd percentile。
它保留了 hard win-rate 的上升空间，但也使 true-p 超过 0.6 需要生成图平均进入非常窄的低距离区域。

### 2.3 输入文件

- embeddings：`outputs/strict_pool_s228_0429_0119/embeddings.npz`
- reference：`outputs/strict_pool_s228_0429_0119/reference.png`
- competitor distance lookup：`outputs/multiseed_s228_M40_0510_0241/dreams_matrix.npz`
- SD3.5 权重：`models/stabilityai/stable-diffusion-3.5-large`
- DreamSim：仓库 `models/` 下 vendored 版本

`dreams_matrix.npz` 在 continuous 主循环中只用于取得 `D_B`。连续点生成的图片仍由 SD3.5
实时渲染并用 DreamSim 实时评分，不能从 228×40 矩阵查表。

## 3. Latent 与连续流形

1. 从 228 个 word embedding 中移除 competitor `bright`，剩余 227 个 anchor。
2. 在这 227 个 4096 维 embedding 上拟合 `PCA(n_components=16, random_state=0)`。
3. 得到 anchor matrix $Z\in\mathbb R^{227\times16}$。
4. 用同一个 PCA transform competitor embedding，得到 $z_{\mathrm{comp}}$。
5. 模型特征始终为：

   $$\phi(z)=z-z_{\mathrm{comp}}.$$

6. 连续有效域由 10-NN 几何定义：`k=10`，LOO kNN distance 的 95% quantile 为
   `tau_d=20.05482399585372`。

PCA latent 没有逐维标准化。不要在 baseline 中加入 standardization；如果要测试，必须作为独立 variant，
因为它会同时改变 beta、lambda、S、tau 和 Thompson covariance 的尺度解释。

## 4. Warm start

每条 trajectory 使用独立 `sim_seed ∈ {0,1,2,3,4}`：

1. `rng = np.random.default_rng(sim_seed * 1000 + 7)`。
2. 在 kNN 有效流形内随机生成 `n0=24` 个连续设计。
3. PCA inverse transform 回 4096 维 T5 embedding。
4. SD3.5 渲染 24 张图。
5. DreamSim 计算到 R 的 distance。
6. 按 `Bernoulli(sigmoid(30*(D_B-ds)))` 生成训练标签。
7. 用 `Phi = warm_z - z_comp` 做第一次 Laplace MAP 拟合。

不同 sim_seed 会改变 warm design、Thompson 随机数和 Bernoulli 标签，但不会改变 SD3.5 render seed。

## 5. 每轮 CM-TS 更新

每轮 $t$ 的 posterior 中心是 $\hat\beta_{t-1}$，Laplace precision 是 $H_{t-1}$。

### 5.1 Thompson batch

一次采样 `B=8` 个 beta：

$$
\tilde\beta_{t,b}\sim
\mathcal N\!\left(\hat\beta_{t-1},
v^2\frac{H^{-1}_{t-1}+H^{-\top}_{t-1}}{2}\right),
\qquad v=2.
$$

每个 draw 独立调用 `argmax_over_M`，在连续 kNN manifold 上近似最大化
$\tilde\beta^\top z$，得到 8 个连续设计。这里不是在 227 个词中做离散 argmax。

### 5.2 真图渲染与反馈

- PCA inverse transform：16 → 4096。
- `encode_batch_insert("", Z)`：空 prompt 中 sandwich-inject 单个 4096 维向量。
- 每轮真实渲染 8 张图。
- 每张图实时计算 DreamSim distance。
- 训练反馈为 soft-Bernoulli sample，不直接把 soft probability 当连续标签。

### 5.3 Laplace posterior

累计本轮 8 条 observation 后只 refit 一次：

$$
\hat\beta_t=
\arg\min_\beta\left[
-\log p(y\mid\Phi,\beta)+\frac{\lambda}{2}\|\beta\|^2
\right],\qquad \lambda=50.
$$

$$
H_t=\Phi^\top\operatorname{diag}(p(1-p))\Phi+50I.
$$

MAP 后执行 `project_norm(beta, S=8)`。在健康 trajectory 中该 clip 不应生效；当前 baseline
T=300 的最终 beta norm 为约 `0.043–0.066`。如果 beta norm 长期贴 8，应判为 saturation failure，
不能把高 predicted-p 当作成功。

## 6. SD3.5 渲染设置

主实现：`src/sd35_batch_generator.py::generate_batch`

```text
model            stabilityai/stable-diffusion-3.5-large (local fp16)
resolution       496 × 496
inference steps  20
guidance scale   5
prompt           empty string, with one injected T5 embedding
render seed      1810772 for every generated image
batch size       8
```

### 固定 render seed 的重要限制

当前 continuous setting **没有对 SD3.5 seed 做边缘化**。同一 batch 的 8 张图和所有 rounds 都使用
`ref_seed=1810772`。因此：

- 这里的 `true_p_soft` 是当前固定渲染 seed 下、连续设计产生的单图 soft probability；
- 5 个 sim_seed 不是 5 个图像生成 seed；
- discrete 的 40-seed oracle 与这里的 true-p 不是完全相同的 estimand；
- 如果以后测试随机 render seed，必须另起实验名，不能和本 baseline 混画为同一条曲线。

## 7. Canonical 超参数表

| 参数 | 值 | 含义 |
|---|---:|---|
| `d` | 16 | PCA latent dimension |
| `k` | 10 | manifold kNN |
| `tau_d` | 20.054824 | 95% LOO-kNN radius |
| `alpha` | 30 | soft-Bernoulli 标签锐度 |
| `v` | 2.0 | Thompson covariance inflation |
| `lam` | 50 | logistic prior precision / ridge |
| `S` | 8.0 | beta norm safety clip |
| `n0` | 24 | warm-start designs |
| `B` | 8 | independent Thompson draws/images per round |
| `T` | 500 completed | BO rounds per trajectory |
| sim seeds | 0–4 | five independent algorithm trajectories |
| competitor | bright/18 | `D_B=0.4704614282` |
| render seed | 1810772 | fixed SD3.5 generator seed |
| `save_img_every` | 10 | render all images, persist every tenth round |

## 8. 指标口径

每轮先对 8 张图求平均，再跨 5 条 trajectory 求 mean 和 SE。

- **hard win-rate**：$\frac18\sum_b 1[D_{t,b}<D_B]$。
- **true-p**：$\frac18\sum_b\sigma(30(D_B-D_{t,b}))$。
- **predicted-p**：$\frac18\sum_b\sigma(\hat\beta_t^\top(z_{t,b}-z_{comp}))$。
- **calibration gap**：`predicted_p - true_p`。
- **mean ds-to-R**：每轮 8 张图的平均 distance，反映分布中心。
- **best-so-far ds**：每轮先取 8 张图的最小 distance，再对时间做 cumulative minimum，反映极值探索。
- **beta norm**：posterior MAP norm；贴 8 表示 saturation failure。
- **cov eig max/min**：本轮 Thompson covariance 的谱。

主结论必须同时看 true-p、mean distance 和 calibration。hard win-rate 高不等价于真正接近 R，
best-so-far 下降也可能只是大量采样产生的极值。

## 9. 已完成结果

### 9.1 T=200 continuous top-4 验证

| 配置 | late true-p | late predicted-p | gap | saturation |
|---|---:|---:|---:|---:|
| **α30, v2, λ50** | **0.5928** | 0.6152 | +0.0225 | 0/5 |
| α20, v2, λ50 | 0.5412 | 0.5668 | +0.0256 | 0/5 |
| α20, v1, λ50 | 0.5457 | 0.7674 | +0.2217 | 2/5 |
| α20, v0.5, λ50 | 0.4411 | 1.0000 | +0.5589 | 5/5 |

这一步确立 `α30,v2,λ50` 为 continuous canonical baseline。

### 9.2 同一批 trajectory 配对续跑至 T=300

| 窗口 | hard win-rate | true-p | predicted-p | mean ds |
|---|---:|---:|---:|---:|
| rounds 0–19 | 0.6538 | 0.5032 | 0.6090 | 0.47181 |
| rounds 180–199 | 0.8613 | 0.5928 | 0.6152 | 0.45709 |
| rounds 250–299 | 0.8780 | 0.6076 | 0.6254 | 0.45471 |
| rounds 280–299 | 0.8850 | 0.6151 | 0.6271 | 0.45345 |
| rounds 290–299 | 0.8825 | **0.6214** | 0.6283 | 0.45237 |

T=200→300 的改善同时出现在 true-p 与 mean distance，说明超过 0.6 不是只由 hard threshold
或绘图平滑产生的假象。T=300 最后 10 轮 true-p 的跨-sim SE 为 `0.0308`，当前仍只有 5 条 trajectory，
报告时要保留不确定性。

主图：`outputs/cmts_top4_summary_0621_0437/a30_v2_lam50_T300_diagnostics.png`

## 10. T=500 结果

同一批 5 条 trajectory 已从 checkpoint `t=300` 续跑到 `t=500`：

```bash
bash scripts/97_extend_a30_v2_lam50_T500.sh
```

这不是新实验，也没有重跑前 300 轮。

| 窗口 | hard win-rate | true-p | predicted-p | mean ds |
|---|---:|---:|---:|---:|
| rounds 280–299 | 0.8850 | 0.6151 | 0.6271 | 0.45345 |
| rounds 450–499 | 0.9250 | 0.6517 | 0.6539 | 0.44725 |
| rounds 480–499 | 0.9213 | 0.6528 | 0.6551 | 0.44705 |
| rounds 490–499 | 0.9275 | **0.6594** | 0.6576 | 0.44582 |

T=300→500 期间 true-p 和 mean distance 仍持续改善，而且 predicted-p 与 true-p 在末段基本重合。
但 best-so-far distance 只小幅变化，说明额外 rounds 主要改善生成分布中心和稳定性，而不是发现显著更低的极值。

T=500 主图：`outputs/cmts_top4_summary_0621_0437/a30_v2_lam50_T500_diagnostics.png`

## 11. 文件与输出

### 核心代码

- worker：`scripts/73_cmts_dreamsim.py`
- continuous optimizer：`src/cmts_sim.py`
- batch renderer：`src/sd35_batch_generator.py`
- T=200 top-4 launch：`scripts/93_launch_discrete_top4_continuous.sh`
- T=300 extension：`scripts/95_extend_a30_v2_lam50_T300.sh`
- T=500 extension：`scripts/97_extend_a30_v2_lam50_T500.sh`
- top-4 summary：`scripts/94_top4_continuous_summary.py`
- T=300 diagnostic：`scripts/96_a30_v2_lam50_T300_diag.py`

### Canonical output directory

```text
outputs/cmts_a30_v2_lam50_bbright_d16_B8_T200_0621_0437/
```

目录名保留首次 T=200 launch 的名字，但内部 trajectory 已续到 T=500。
判断真实完成轮数必须读取 `sim*/summary.json` 的 `T` 或 trajectory.csv 的最大 `t`，不能相信目录名。

每条 trajectory：

```text
sim<sss>/trajectory.csv
sim<sss>/summary.json
sim<sss>/posterior.npz
sim<sss>/_ckpt.pkl
sim<sss>/images/
```

extension 会覆盖最终 `summary.json` 和 `trajectory.csv`，但 trajectory 包含从 0 开始的全部历史；
`_ckpt.pkl` 是可恢复执行的 authoritative state。

## 12. 后续实验规范

1. **baseline 必须保留本配置**：`α30,v2,λ50,d16,k10,S8,n0=24,B8,bright/18`。
2. 一次只改变一个主要因素；多因素 sweep 先用 discrete 筛选，再做 continuous 验证。
3. continuous 至少跑 5 sim seeds；需要更稳的置信区间时扩到 10，而不是只增加 rounds。
4. 参数比较优先使用相同 T、相同 sim seeds、相同 competitor 和 render seed。
5. alpha sweep 时要明确 true-p 定义随 alpha 改变；不要把不同 alpha 下的 true-p 数值直接当同一指标比较。
6. competitor sweep 会机械改变胜率；必须同时报告 raw mean distance。
7. `v` schedule、latent standardization、random render seeds 都属于新算法 variant，另起脚本/输出目录。
8. 若 final beta norm ≥7.9 的 trajectory 比例非零，单独报告 saturation fraction。
9. 多 GPU 启动保留：

   ```bash
   env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=$G \
     setsid conda run -n diverse --no-capture-output python ... </dev/null &
   ```

10. 不要并发写同一个 `simXXX`；同一输出目录可以并行，但每个 worker 的 seed range 必须互不重叠。

## 13. 已知未决问题

- 固定 render seed 下的提升是否能推广到对 SD3.5 seed 边缘化的真实用户偏好概率。
- 线性 logistic surrogate 的方向是否与 `z_red-z_comp` 对齐；尚缺 beta-direction cosine 曲线。
- T=500 后分布中心是否仍值得继续优化，以及边际 GPU 成本是否合理。
- 当前 manifold 最好的可达 distance 是否受 PCA/10-NN geometry 限制。
- 5 条 trajectory 的 SE 仍较大；若作为正式统计结论，建议追加到 10 条。
