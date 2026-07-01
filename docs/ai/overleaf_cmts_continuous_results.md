# Overleaf 写作说明：continuous CM-TS 结果与实现细节

这份文档的目的不是记录全部实验流水账，而是让后续写 Overleaf 的人可以直接知道：

1. 这部分实验在解决什么问题；
2. continuous 设置具体怎么实现；
3. 哪些结果应该放到论文/报告里；
4. 图和表应该怎么组织；
5. 哪些 caveat 必须写清楚，避免把结果讲过头。

当前最适合写入 Overleaf 的主线是：

> 我们先用 discrete proxy 做参数筛选，再把有信号的参数转到 continuous CM-TS。最终在 fixed-render-seed continuous setting 下，`alpha=8, v=1, lambda=100` 给出稳定改善，`alpha=10, v=1, lambda=100` 的成功轨迹能进一步达到更高 true probability；`alpha=10, v=0.5, lambda=100` 有非常强的单轨迹结果，但稳定性和 saturation 需要谨慎处理。

## 1. 实验目标

我们关心的是 continuous embedding space 中是否能通过 CM-TS 逐步产生更接近 reference image `R` 的图像。

核心目标不是只让 hard win-rate 变高，而是同时观察：

- `true_p_soft` 是否上升；
- `mean ds_to_R` 是否下降；
- `predicted_p` 和 `true_p_soft` 是否校准；
- posterior / theta 是否健康，没有进入 `||beta||=8` saturation。

早期问题是 hard win-rate 可以上升，但 true probability 往往不动，或者 posterior belief 饱和到 `predicted_p≈1`。后续重点转向更保守的 `lambda=100` 和较小/中等 `alpha`，希望获得更稳定的 true-p 改善。

## 2. Continuous CM-TS 设置

### 2.1 固定 reference 和 competitor

当前 continuous 实验使用：

```text
reference R:
  outputs/strict_pool_s228_0429_0119/reference.png

competitor:
  word = bright
  seed = 18
  D_B = 0.4704614281654358
```

其中 `D_B` 是 competitor image 到 reference `R` 的 DreamSim distance。每轮生成图像的 distance 记为 `D_{t,b}`。

### 2.2 Label 和 metric

训练用的是 soft-Bernoulli label：

```math
y_{t,b} \sim \mathrm{Bernoulli}\left(
\sigma\left(\alpha(D_B - D_{t,b})\right)
\right).
```

图中主要 metric：

```math
\text{hard win-rate}_t =
\frac{1}{B}\sum_{b=1}^B \mathbf{1}[D_{t,b}<D_B].
```

```math
\text{true-p}_t =
\frac{1}{B}\sum_{b=1}^B
\sigma\left(\alpha(D_B-D_{t,b})\right).
```

```math
\text{mean ds-to-R}_t =
\frac{1}{B}\sum_{b=1}^B D_{t,b}.
```

```math
\text{best-so-far}_t =
\min_{s\le t, b\le B} D_{s,b}.
```

注意：`hard win-rate` 变高不一定代表图像真正更接近 `R`。它只表示超过 competitor threshold。报告里应该优先解释 `true-p` 和 `mean ds-to-R`。

### 2.3 Continuous candidate generation

Continuous CM-TS 在 PCA latent space 中运行：

```text
d = 16
k = 10
B = 8 candidates per round
n0 = 24 warm-start designs
S = 8 beta norm clip
```

每轮流程：

1. 从 Laplace posterior 采样 Thompson direction：

   ```math
   \tilde\beta_t \sim
   \mathcal{N}(\hat\beta_{t-1}, v^2 H_{t-1}^{-1}).
   ```

2. 用采样到的 direction 在 continuous manifold 上选一个 candidate `z`。
3. 通过 SD3.5 生成图片。
4. 用 DreamSim 计算到 reference 的 distance。
5. 用 soft-Bernoulli label 更新 ridge logistic MAP。

### 2.4 重要实现 caveat：fixed render seed

当前 continuous 实验的 SD3.5 render seed 是固定的：

```text
ref_seed = 1810772
```

代码中每个 batch 都用同一个 seed：

```python
imgs = gen.generate_batch(embeds, [ref_seed] * len(chunk))
```

所以 continuous 结果的随机性来自：

- Thompson sampling；
- warm-start / sim seed；
- continuous z trajectory；

而不是来自不同 image render seed。

这点必须在 Overleaf 里写清楚，因为 discrete proxy 使用的是预生成的 40 seeds，而 continuous 当前是 fixed-render-seed setting。两者不是完全相同的 estimand。

## 3. Posterior / MAP 实现细节

模型使用 ridge logistic MAP：

```math
\hat\beta =
\arg\min_\beta
\sum_i
\left[
\log(1+\exp(\phi_i^\top \beta))
- y_i \phi_i^\top \beta
\right]
+ \frac{\lambda}{2}\|\beta\|_2^2.
```

Laplace Hessian：

```math
H =
\Phi^\top W \Phi + \lambda I,
```

其中：

```math
W_i = p_i(1-p_i).
```

之后代码会做 norm clip：

```python
beta_hat = project_norm(beta_hat, S)
```

其中 `S=8`。

### 3.1 为什么要报告 beta_norm

之前发现一些配置会进入坏分支：

- `beta_norm` 卡在 `8`；
- `predicted_p` 变成 `1.0`；
- calibration 很差；
- true-p 不一定提升。

因此 posterior 图必须包含：

- `beta_norm`;
- posterior covariance max eigenvalue;
- posterior covariance min eigenvalue;
- condition ratio `eig_max/eig_min`。

如果 `beta_norm≈8`，该 run 的 belief/predicted-p 不应被直接相信。

## 4. True-p 理论参考线

Performance 图右上角 true-p panel 需要加一条黑色虚线，表示 alpha-dependent soft success reference：

```math
\max_w
\frac{1}{40}
\sum_{k=1}^{40}
\sigma\left(\alpha(D_B-D_{w,k})\right).
```

它是 discrete 40-seed pool 的 soft oracle reference，不是 continuous manifold 的严格数学上界。

当前 relevant alpha：

```text
alpha = 8:
  discrete soft oracle ≈ 0.628
  best word = red

alpha = 10:
  discrete soft oracle ≈ 0.652
  best word = red
```

Overleaf 中推荐叫法：

> discrete soft-oracle reference

不要直接叫 strict upper bound for continuous，因为 continuous 可以产生不在 discrete word pool 里的 PCA-manifold points。

## 5. 推荐放入 Overleaf 的主结果

当前推荐主图放三组，每组两张 PNG：

```text
results/cmts_selected_alpha_lam100_fourpanels/
```

文件：

```text
a8_v1_lam100_T1000_all5_performance_4panel.png
a8_v1_lam100_T1000_all5_posterior_4panel.png

a10_v0.5_lam100_T1000_completed3_performance_4panel.png
a10_v0.5_lam100_T1000_completed3_posterior_4panel.png

a10_v1_lam100_good3_T1200_performance_4panel.png
a10_v1_lam100_good3_T1200_posterior_4panel.png
```

### 5.1 主表建议

建议在 Overleaf 中放一张 summary table：

| Setting | Sims used | Horizon | Last-window true-p | Last-window mean ds | Hard win-rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| `alpha=8, v=1, lambda=100` | 5/5 | 1000 | 0.595 | 0.4209 | 0.876 | stable, all trajectories completed |
| `alpha=10, v=1, lambda=100` good3 | 3/5 | 1200 | 0.664 | 0.4005 | 0.990 | selected successful trajectories; continued from 1000 to 1200 |
| `alpha=10, v=0.5, lambda=100` completed3 | 3/5 | 1000 | 0.659 | 0.3862 | 0.983 | strongest average among completed3, but one trajectory has saturation |

Important notes for the table:

- For `alpha=8, v=1, lambda=100`, all five trajectories completed, so this is the cleanest stable setting.
- For `alpha=10, v=1, lambda=100`, “good3” means `sim001/sim002/sim003`; it is not the average over all five trajectories.
- For `alpha=10, v=0.5, lambda=100`, only three trajectories are currently included in the plotted completed3 result. It has a very strong trajectory, but stability is still under evaluation.

### 5.2 Detailed numbers

#### `alpha=8, v=1, lambda=100`, all 5, T=1000

First 20 rounds:

```text
true_p  = 0.490
mean_ds = 0.4756
hard    = 0.591
```

Last 20 rounds:

```text
true_p  = 0.595
mean_ds = 0.4209
hard    = 0.876
pred_p  = 0.590
```

Interpretation:

- Most defensible stable result.
- True-p improves by about `+0.105`.
- Mean DreamSim distance drops by about `0.0546`.
- Predicted-p and true-p are close in the final window.

#### `alpha=10, v=1, lambda=100`, good3, T=1200

Good trajectories:

```text
sim001, sim002, sim003
```

First 20 rounds:

```text
true_p  = 0.497
mean_ds = 0.4719
hard    = 0.588
```

Last 20 at T=1000:

```text
true_p  = 0.658
mean_ds = 0.4033
hard    = 0.973
```

Last 20 at T=1200:

```text
true_p  = 0.664
mean_ds = 0.4005
hard    = 0.990
pred_p  = 0.658
```

Interpretation:

- Good trajectories keep improving slightly after 1000 rounds.
- The 1000 to 1200 extension gives only a small gain:

  ```text
  true_p: 0.658 -> 0.664
  mean_ds: 0.4033 -> 0.4005
  ```

- This suggests the setting is approaching a plateau.

#### `alpha=10, v=0.5, lambda=100`, completed3, T=1000

Completed trajectories included in current figure:

```text
sim001, sim002, sim003
```

Last 20 rounds:

```text
true_p  = 0.659
mean_ds = 0.3862
hard    = 0.983
pred_p  = 0.812
```

Important caveat:

One included trajectory has posterior saturation:

```text
sim003:
  beta_norm ≈ 8
  pred_p ≈ 1
```

Therefore, this setting should be described as high-potential but less stable. The true-p / mean-distance result is promising, but the posterior belief is not fully reliable because of saturation in one run.

## 6. Recommended figure captions

### Figure: stable continuous CM-TS result

Use:

```text
a8_v1_lam100_T1000_all5_performance_4panel.png
a8_v1_lam100_T1000_all5_posterior_4panel.png
```

Draft caption:

> Continuous CM-TS with `alpha=8`, `v=1`, and `lambda=100` over five trajectories. The method steadily improves both hard win-rate and soft success probability while reducing mean DreamSim distance to the reference. The dashed black line in the belief panel is the discrete soft-oracle reference for the same alpha. Posterior diagnostics show that the run avoids the `||beta||=8` saturation observed in unstable configurations.

### Figure: successful alpha=10 trajectories

Use:

```text
a10_v1_lam100_good3_T1200_performance_4panel.png
a10_v1_lam100_good3_T1200_posterior_4panel.png
```

Draft caption:

> Successful trajectories for continuous CM-TS with `alpha=10`, `v=1`, and `lambda=100`, extended from 1000 to 1200 rounds. The selected trajectories continue to improve slightly after 1000 rounds, reaching a final-window soft success probability around 0.664 and mean DreamSim distance around 0.400. The small gain after 1000 rounds suggests the process is nearing a plateau.

### Figure: high-potential but unstable v=0.5 setting

Use:

```text
a10_v0.5_lam100_T1000_completed3_performance_4panel.png
a10_v0.5_lam100_T1000_completed3_posterior_4panel.png
```

Draft caption:

> Continuous CM-TS with `alpha=10`, `v=0.5`, and `lambda=100` for the currently completed trajectories. This configuration produces the strongest mean DreamSim reduction among the completed trajectories, but the posterior diagnostics reveal saturation in one trajectory. We therefore treat this setting as high-potential but less stable than `alpha=8, v=1, lambda=100`.

## 7. Suggested Overleaf section structure

Recommended section outline:

```latex
\subsection{Continuous CM-TS on SD3.5 Embedding Manifold}

\paragraph{Setup.}
Describe reference image, competitor threshold, fixed render seed, and DreamSim metric.

\paragraph{Algorithm.}
Describe Thompson sampling over ridge-logistic Laplace posterior and continuous candidate optimization.

\paragraph{Metrics.}
Define hard win-rate, true-p, predicted-p, mean DreamSim distance, best-so-far distance.

\paragraph{Parameter selection from discrete proxy.}
Explain that the discrete fixed-seed sweep suggested lambda=100 and alpha in the 8--10 range.

\paragraph{Continuous results.}
Present the stable alpha=8 result, the alpha=10 good trajectories, and the high-potential v=0.5 result.

\paragraph{Limitations.}
Discuss fixed render seed, mismatch between discrete oracle and continuous manifold, and posterior saturation.
```

## 8. Paragraphs that can be adapted directly

### Setup paragraph

> We evaluate CM-TS in a continuous SD3.5 embedding manifold. A fixed reference image is selected, and the competitor threshold is defined by the DreamSim distance of the `bright/18` image to the reference, giving `D_B=0.47046`. At each round, the algorithm proposes a batch of eight continuous embedding candidates, renders them with SD3.5 using a fixed render seed, and evaluates their DreamSim distance to the reference. Preference labels are sampled from a soft Bernoulli model with probability `sigma(alpha(D_B-D))`.

### Metric paragraph

> We report both the hard win-rate, measuring the fraction of candidates that beat the competitor threshold, and the soft success probability, defined as the average of `sigma(alpha(D_B-D))` over candidates in a round. The soft success probability is the primary metric because it is sensitive to the magnitude of improvement below the threshold, whereas the hard win-rate saturates once candidates reliably beat the competitor. We also report the mean DreamSim distance to the reference, where lower is better.

### Result paragraph

> The most stable configuration is `alpha=8, v=1, lambda=100`, which completes five trajectories without posterior saturation. Its final-window soft success probability reaches approximately 0.595, while mean DreamSim distance decreases from 0.476 in the first 20 rounds to 0.421 in the last 20 rounds. The stronger `alpha=10` setting achieves higher success probability on successful trajectories, reaching approximately 0.664 after extending to 1200 rounds, but its performance is less uniform across trajectories.

### Caveat paragraph

> The discrete soft-oracle line is an alpha-dependent reference computed from the precomputed 40-seed discrete pool. It should not be interpreted as a strict upper bound for the continuous manifold, because continuous CM-TS can propose PCA-manifold points that are not one of the discrete words. Additionally, all continuous renders currently use a fixed SD3.5 seed, so these results evaluate optimization over embeddings under a fixed rendering stochasticity rather than averaging over render seeds.

## 9. What not to overclaim

Do not claim:

- “continuous has reached the global optimum”;
- “discrete oracle is a strict upper bound for continuous”;
- “`alpha=10, v=0.5` is best overall” before remaining trajectories finish and saturation is handled;
- “hard win-rate near 1 means the image is close to reference”.

Safer claims:

- `alpha=8, v=1, lambda=100` gives the cleanest stable continuous improvement.
- `alpha=10` can reach higher true-p on successful trajectories.
- `lambda=100` avoids many of the severe saturation failures observed at lower prior precision, although saturation can still happen for some low-`v` trajectories.
- More rounds beyond 1000 produce limited additional gain for `alpha=10, v=1, lambda=100`.

## 10. Current run status to remember

As of this note:

```text
alpha=8, v=1, lambda=100:
  T=1000 completed for all 5 sims.
  T=1500 extension has been queued to start when GPUs become free.

alpha=10, v=1, lambda=100:
  good3 sim001/sim002/sim003 extended to T=1200.

alpha=10, v=0.5, lambda=100:
  some trajectories completed and show strong signal;
  stability is still under evaluation.

alpha=8, v=0.5, lambda=100:
  unstable; at least one trajectory saturated at beta_norm=8.
```

If the Overleaf writeup needs final numbers for T=1500, rerun the summary after the queued extension completes.
