# Architecture

## 目录布局

仓库根 `/home/linyuliu/jxmount/`，绝对路径硬编码在多个脚本默认值里，**不要随意移动**。

```
diffusion_custom/
├── CLAUDE.md                  ← Claude Code 入口
├── EXPERIMENT_SUMMARY.md      ← 历史方法学（exp 46–52）
├── docs/ai/                   ← AI 上下文文档（本目录）
│   ├── project_context.md
│   ├── decisions.md
│   ├── todo.md
│   └── architecture.md        ← 当前文件
├── scripts/                   ← 编号脚本，时间顺序
│   ├── 56_build_strict_pool.py    ← 228 单词池
│   ├── 57_collect_strict_pool.py  ← 抽 token embedding + 生成 reference
│   ├── 60_collect_multiseed.py    ← 每词 40 seed 生图（× 4 GPU）
│   ├── 60b_merge_multiseed.py     ← 合并 partial dreams matrix
│   ├── 63_bo_thompson_curve.py    ← BO 主实验（5 surrogates × 100 sims）
│   └── 64_render_bo_trajectory.py ← 轨迹图渲染
├── src/                       ← 可复用库
│   ├── sd35_batch_generator.py    ← SD35BatchEmbeddingGenerator（encode_batch_insert）
│   ├── sd35_embedding_generator.py
│   ├── sd35_controlnet_generator.py
│   ├── scorer.py                  ← DreamSimScorer, CLIPScorer
│   ├── seed_selector.py
│   ├── thompson_optimizer.py      ← LogisticThompsonOptimizer（Laplace + TS）
│   └── edge_utils.py
├── models/                    ← 本地权重（gitignored）
└── outputs/                   ← 每次 run 一个目录
    ├── strict_pool_s228_0429_0119/    ← embeddings.npz + reference.png
    ├── multiseed_s228_M40_0510_0241/  ← dreams_matrix.npz + 9120 张图
    └── bo_thompson_bcanvas_0512_1424/ ← BO 结果 + EXPERIMENT_DOC.md（首要参考）
```

## 关键术语

- **$R$（reference image）**：DreamSim 距离的原点，`ref_word=red, seed=1810772`。详见 decisions.md。
- **$B$（competitor image）**：偏好阈值，`(canvas, 34)`，$D_B = 0.6205$。
- **$D_{w,k}$**：DreamSim$(R, I_{w,k})$，存在 `dreams_matrix.npz`，形状 $228 \times 40$。
- **words_kept**：剔除 $B$ 所在 word "canvas" 后的 227 个候选词。
- **$z_w$**：word $w$ 的 PCA latent，$\mathbb{R}^d$，$d=8$。
- **Hard oracle**：$\frac{1}{40}\sum_k \mathbf{1}[D_{w,k}<D_B]$，CSV `p_oracle` 列。
- **Soft oracle**：$\frac{1}{40}\sum_k \sigma(\alpha(D_B - D_{w,k}))$，CSV `p_soft` 列。
- **picked-at-t**：第 t 步 picked 单词的 soft p_oracle，jagged。
- **best-so-far**：同 sim 内 picked p_soft 的 cummax，单调。

## 5 个 Surrogate（in `63_bo_thompson_curve.py`）

| 名字 | 函数族 | acquisition | 拟合 |
|------|--------|-------------|------|
| `logistic_bayesian` | $\sigma(\beta^\top z)$ | Thompson（Laplace 后验） | L-BFGS MAP + Hessian，每步 refit |
| `logistic_l2` | $\sigma(\beta^\top z)$ | $\varepsilon$-greedy | sklearn `LogisticRegression(C=3.0)`，每步 refit |
| `poly2_logistic` | $\sigma(\beta^\top \phi(z))$，$\phi$ = 二阶交互 | Thompson | 同 logistic_bayesian |
| `gp_rbf` | GP regression（$y\in\{0,1\}$ 当连续） | Thompson（$\mu + \sigma\xi$） | sklearn GPR，每 5 步 refit |
| `random_forest` | RF classifier | bootstrap-tree Thompson | sklearn RF（200 trees），每步 refit |

`LogisticThompsonOptimizer`（src/thompson_optimizer.py）是 logistic_bayesian / poly2_logistic 共用的引擎。

## 多 GPU 模式

两种共存：

1. **Producer/consumer queue**（脚本 22/23/24）：master 派任务、worker 各自加载 SD3.5 + DreamSim。
   master 预加载 SD3.5 后**必须 `del` + `torch.cuda.empty_cache()`** 再 spawn worker，否则首轮 OOM。
2. **Range slicing**（脚本 60）：每个 GPU 独立跑 word range，最后 merge partial。当前 pipeline 用这种。

均需 `torch.multiprocessing.set_start_method('spawn')`。

## 代码规范

- 中英文混排：保留原始注释 / print 语言风格，**不要统一翻译**。
- 脚本编号按时间顺序，新实验另起新编号，不重命名旧脚本。
- 默认 prompt 固定为
  `"Product photo of a single shoe, full shoe visible, side profile, centered on a plain white background"`，
  与 `dreams_matrix.npz` 一一对应。
- 图像分辨率：batched 生成器 496×496、单图 512×512、smoketest 1024×1024。
- `LogisticThompsonOptimizer.solve_analytical_best` 里的子空间正交化别删
  （保证 z 与已用 prompt token 正交）。
- `negative_prompt` 默认值经过调优（抑制 crop / close-up），不要随意放宽。
- 历史脚本（12–24）的低维 latent + 随机正交 $W$ 投影是旧 setup；
  新 pipeline（56–64）用 PCA of word embeddings，不要混用。

## 数据流

```
56 (CPU)  → strict_pool_v3.csv (228 words)
              ↓
57 (1 GPU) → embeddings.npz (228×4096) + reference.png + 228 张 1-seed 图
              ↓
60 (4 GPU) → 228×40 = 9120 张图 + dreams_partial_*.npz
60b (CPU) → dreams_matrix.npz (228×40 距离矩阵)
              ↓
63 (CPU)  → bo_per_step.csv (100k 行) + bo_summary.csv + oracle.npz
              ↓
(softoracle 重算) → bo_per_step_softoracle.csv (+ p_soft, p_soft_run_max 两列)
              ↓
64 (CPU)  → trajectories/sim<sss>_<model>.png × 15
```

`63` 不调用 SD3.5，所有图距离从 `dreams_matrix.npz` 查表 —— 100 sims × 5 models × 200 steps 在 CPU 上约 10 分钟跑完。
