# TODO

按"是否还做"分组。每项标 owner / blocker / 截止条件。

## In progress

- **#17 Write combined N-sweep report**
  把 GOF k=5/10/80 三个实验合在一份报告里。当前进度：数据有，叙事框架未写。
  blocker：要等 #15 决定要不要补 k=40。如果不补，#17 可以基于现有三个 k 收尾。

## Pending（需求未确认）

- **#15 Run k=40 GOF (α=5, dims 8/16/32/64/128)**
  GOF 系列里唯一未跑的 k 值。状态：**已 outdated**，因为 GOF 系列的目标已被
  `63_bo_thompson_curve.py` 的 5-surrogate 学习曲线大致回答了。
  decision needed：跑 / 不跑（如果不跑，把 #17 报告范围收窄到 k=5/10/80 即可）。

- **#27 Write learning-curve report for B=canvas run**
  对 `61_learncurve_multiseed.py` 在 B=canvas 下的产物写一份解读。
  状态：**EXPERIMENT_DOC.md 已经覆盖了大部分核心内容**，这份报告价值在于聚焦 `61_*` 的 learning-curve
  数据本身（N-sweep at fixed surrogate），而不是 `63_*` 的 surrogate-sweep at fixed N。
  decision needed：合并到 EXPERIMENT_DOC.md 还是单独成文。

## Completed in this session

- #36 Render BO trajectory image grids（`scripts/64_render_bo_trajectory.py`，15 PNG）
- #37 Write end-to-end experiment doc（`outputs/bo_thompson_bcanvas_0512_1424/EXPERIMENT_DOC.md`）

## 可能的下一步（如果老师追问）

- 重做主图，把 oracle ceiling 标注从全局 `max` 改成 "top-5% mean"，避免 ceiling 被单个 outlier word 拉高
- 用真实人类标注替换 sigmoid 标签，做一次 sanity check（成本：~$50 MTurk + 一周）
- 把 surrogate 阵容缩成 2 个（logistic_bayesian + gp_rbf）做更密集的 d / α sweep
- 对 trajectory 图加"距 ref 收敛速度"的定量曲线（不只是看图）

这几个都是猜测，**不要主动开工**，等老师明确要求再做。
