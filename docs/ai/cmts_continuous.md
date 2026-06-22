# CM-TS Continuous Setting — 工作记录与交接

> 这份文件是 **continuous setting（CM-TS）** 这条线的端到端交接文档。新会话读完这一份，应能立刻知道
> 我们在做什么、为什么、卡在哪、下一步。符号/口径与 `decisions.md`、`architecture.md` 一致。
> **最近更新：2026-06-17**。

---

## 0. TL;DR（先读这段）

- **目标**：在**连续** $\mathbb{R}^d$（$d=16$，PCA of T5 word embedding）流形上做 Thompson-Sampling BO，
  让 SD3.5 生成的鞋图**分布真正移向参考图 $R$**，同时每轮**胜率（vs competitor $B$）呈上升趋势**。
- **和 discrete 的区别**：discrete（脚本 63）在固定 227 个词上做 **ranking**；continuous（脚本 73）在连续流形上
  **采样 + argmax_over_M**，每轮渲染 8 张真图、打软标签、Laplace 重拟合一次。
- **核心科学问题**（和 discrete 同）：surrogate 假设 $P(\text{win}\mid z)=\sigma\!\big(\beta^\top (z-z_{\text{comp}})\big)$ 在我们的 setting 下成立吗。
- **目前最大的两个杠杆**：
  1. **饱和（saturation）**——小 $\lambda$ 让 $\|\beta\|$ 撞 clip $S=8$，$\sigma$ 贴 0/1，方向学不动。**靠 $\lambda$ 去饱和**。
  2. **competitor 强度**——决定 win-rate 曲线形态。太强→封顶；太弱→钉死 1.0；只有中间窄带才有干净上升趋势。
- **最近结论（STAMP `0616_2245`，lam0.5×10seed×T300，bright/18）**：拿到了**最干净的 hard-rising demo**
  （hard win-rate 10-seed 均值 0.42→0.79, slope +0.135/100r），**但 true_p 锁死 0.52**。
  关键发现：在 3% competitor + 饱和分支下，**"rising hard trend" 与 "true_p 变大" 互斥**（详见 §6 `0616_2245` 专段）。
  另：**warm-pin 墙**（脚本 90）—— optimizer warm-start ds≈0.474，所以只有 3% competitor（$D_B$≈0.470）才给 hard 上升留空间。
- **决策点**：(1) 收 hard-趋势这条线（现成图可交）；(2) 改去饱和 sweep 顶 true_p（代价：hard 曲线被削平）。二者互斥。

---

## 1. 算法（CM-TS，引擎在 `src/cmts_sim.py`，主实验在 `scripts/73_cmts_dreamsim.py`）

每一轮 $t$：
1. **Thompson 采样** $B=8$ 个 $\tilde\beta \sim \mathcal{N}\!\big(\hat\beta,\ \mathrm{cov}\big)$，其中
   $$\mathrm{cov} = v^2 \cdot \tfrac{1}{2}\big(H^{-1}+H^{-\top}\big)$$
   （`cmts_sim.py:183-184`；$H$ 是 Laplace 后验**精度**矩阵，**只取一次逆**，对称化是数值卫生，$v^2$ 是探索半径旋钮）。
2. 每个 $\tilde\beta$ 在流形上 `argmax_over_M(β̃, Z, k=10, τ_d)` 取一个连续 $z$（k 近邻凸组合），逆 PCA 回 $\mathbb{R}^{4096}$。
3. SD3.5 渲染 8 张图，DreamSim 算到 $R$ 的距离 $\mathrm{ds}$。
4. **软-Bernoulli 标签**（训练用）：`y ~ Bernoulli(σ(α(D_B − ds)))`（`73:222`）。
5. **一次** Laplace MAP 重拟合 → 新 $\hat\beta$，再 `project_norm(β̂, S=8)` 做 norm 上限裁剪（`73:235`）。

固定量：$R$=red/seed1810772（**绝不能动**，改了要重算整张 dreams）；$d=16$；$k=10$；$\tau_d\approx 20.4\text{–}20.9$（LOO-kNN 半径，仅用于流形插值，**不缩放 $z$**）；$n_0=24$ warm-start；$T=200$；$B=8$；$S=8$。
数据矩阵：`outputs/multiseed_s228_M40_0510_0241/dreams_matrix.npz`（228×40）；embeddings：`outputs/strict_pool_s228_0429_0119/embeddings.npz`。

---

## 2. 超参语义（改之前先懂这几个）

| 旋钮 | 含义 | 关键事实 |
|---|---|---|
| **$\lambda$**（prior precision，`--lam`） | Hessian 上 $+\lambda I$ | **控饱和的就是它，不是 $S$**。小 $\lambda$→$\beta$ 撞 $S=8$ 饱和；大 $\lambda$→$\beta$ 趴 0.1 去饱和。见 §4。 |
| **$v$**（探索半径，`--v`） | 缩放 Thompson cov | 与饱和**正交**。去饱和后用 $v$ 补探索覆盖。$v=4$ 窄、$v=12$ 宽（半径 ×3）。 |
| **$S$**（norm clip，`--S`，=8） | $\|\hat\beta\|\le S$ | 对小 $\lambda$ 是天天贴着的硬约束；**对大 $\lambda$ 完全不生效**（$\beta$ 在撞到 8 前 $\sigma$ 早饱和）。 |
| **$\alpha$**（标签锐度，`--alpha`） | $\sigma(\alpha(D_B-\mathrm{ds}))$ | **校准规则 $\alpha\cdot\mathrm{std}_{ds}\approx 1$**。$\mathrm{std}_{ds}\approx0.036$ → 甜区 $\alpha\approx28\text{–}30$。$\alpha=10$ 太软（$\alpha\cdot\mathrm{std}=0.36$），标签≈0.5 噪声，方向学不动。 |
| **competitor $B$**（`--B_word --B_seed`） | 定义 $D_B$=阈值，`y=1 ⟺ ds<D_B` | **换它不用重算 dreams**，只是换查表索引，10 分钟重跑。强度决定 win-rate 形态，见 §5。 |

`z` 尺度：PCA 后**未标准化**，$\|z\|$ 中位数 ≈ **14.7**（范围 6.7–31.5），第 1 维 std≈8 主导。
推论：$|\beta^\top z|\le\|\beta\|\cdot\|z\|$，所以 $\|\beta\|=8$ 时内积 ~120 → $\sigma$ 彻底饱和；要落 $\sigma$ 软带需 $\|\beta\|\sim O(0.1)$。
> 待试 variant：把 $z$ 每维除以 std（$\|z\|\sim\sqrt{16}=4$），让 $\beta$/$S$ 的尺度更可解释。**尚未做**。

---

## 3. 四个诊断 panel（diag 脚本 78–82 通用）

每轮对 8 draw × 5 sim 平均：
- **(a) hard win-rate** $=$ 每轮 $\frac{1}{8}\sum \mathbf{1}[\mathrm{ds}<D_B]$ —— 分布**中心/利用**。注意训练标签是 soft，这条是 hard。
- **(b) belief vs truth**：
  - 实线 `true_p_soft = σ(α(D_B−ds))`（真实，与标签同源，上帝视角）
  - 虚线 `predicted_p = σ(β̂ᵀ(z−z_comp))`（模型信念）
  - 两线 gap = 校准误差。虚线≫实线 = 过度自信（饱和的症状）。
- **(c) Thompson cov 谱**：`cov_eig_max/min`（log）。小 $\lambda$→大且各向同性（乱撒）；大 $\lambda$→小且各向异性。
- **(d) best-so-far ds**（cummin）—— 分布**极值/探索**。**$\mathrm{ds}$ 与 competitor 无关**，所以**跨 run 可直接比**（diag 里画 black best=0.2803 绿线作锚）。

**关键判读规则（很重要，老师/新会话最容易搞混）**：
- **(a) 和 (d) 会脱钩**：(a) 是均值、(d) 是极小值。win-rate 高只说明"稳定打赢这个对手"，**赢 competitor ≠ 接近 $R$**（$D_B$ 这条线离 $R$ 还远）。
- **不能只看 (d) 下降就邀功**：(d) 降 **且** (b) 升 = 真学到方向；(d) 降 **但** (b) 平 = 只是 1600 次采样的**极值运气**（$\alpha$ 太软时就是这样）。
- 看 $\beta$：`trajectory.csv` 已逐轮记录 `beta_norm`（大小）、`cos_beta_prev`（方向稳定性）、`ts_cos_mean`。
  **完整 16 维 $\beta$ 向量只在 `posterior.npz` 存了终值**，但可用 `laplace_map(Phi[:n], y[:n], lam, S)` 从 `posterior.npz` 的全量 `Phi/y` **精确离线重建任意轮**（refit 确定性）。

---

## 4. 已确认的核心发现

1. **饱和是头号障碍**。小 $\lambda$ → $\|\beta\|$ 秒撞 $S=8$ → $W=p(1-p)\to0$ → $H\approx\lambda I$ → cov 退化（乱撒或塌缩）→ 方向学不动 & predicted_p 虚高。
   **$\beta$-norm 实测两极分化**（图 `outputs/vsweep_curves/theta_norm.png`）：
   - lam16 / lam5 → `‖β‖` 全程贴 **8.00**（饱和）
   - lam50 / lam100 → `‖β‖` 趴 **0.04–0.12**（去饱和，落 $\sigma$ 软带）
   - 中间 0.1–8 一片空白——当前 $\lambda$ 取值非两极即两极。
2. **$\lambda$ 控饱和，$S$ 基本不生效**（见 §2）。
3. **$\alpha$ 必须校准到甜区**（$\approx28\text{–}30$）。$\alpha=10$ 全平（标签成噪声）；$\alpha=30$ 才看到 true_p 真上升。
4. **competitor 强度决定 win-rate 形态**（§5）。
5. **win-rate（利用/均值）与 best-ds（探索/极值）脱钩**，且**强对手反而逼生成更接近 $R$**（弱对手"赢得太轻松"，best-ds 反而差）。
6. **discrete 的正则不能照搬**：discrete（脚本63）`prior_var=3.0` → 等效 $\lambda\approx 1/3\approx0.33$，比 continuous 的 $\{5,16,50,100\}$ 小 1–2 个量级。
   因为 discrete 是**离散 ranking，$\sigma$ 单调、饱和不改变 argmax 排序**，对饱和免疫；continuous 靠 Hessian→cov 做 TS，对饱和敏感，才必须上大 $\lambda$。

---

## 5. competitor 强度 → win-rate 形态（oracle 表，$\alpha=30$）

按"最好的词能赢多少比例的 seed"（`topwf`）判形态。拐点在 **18th pct**（一旦 topwf=1.0，win-rate 钉死）：

| competitor | $D_B$ | image pct | floor(随机起点) | 最好词胜率 | 形态 |
|---|---|---|---|---|---|
| **black/18** | 0.4704 | 3% | 0.03 | 0.78 | 太强 → 升但**封顶 0.75** |
| orange/39 | 0.5100 | 8% | 0.08 | 0.90 | 甜区 |
| **wine/34** | 0.5198 | 10% | 0.10 | 0.93 | **甜区（当前在用）** |
| thick/21 | 0.5283 | 12% | 0.12 | 0.95 | 甜区 |
| (18%+) | 0.549+ | ≥18% | — | **1.00** | **钉死 1.0** |
| **pigment/12** | 0.5813 | 30% | 0.30 | 1.00 | 太弱 → win-rate **开局即 1.0** |

> 注意：甜区里最好的词是 **"red"**（=$R$ 的词本身）→ 可学方向 ≈ "趋向 $R$"。是干净 demo，但跟老师汇报时要点明，免得被问"是不是因为词就是 red 才赢"。

---

## 6. 实验时间线（STAMP → 结论）

| STAMP | 配置 | 结论 |
|---|---|---|
| `0611_0515` | lam16 vs 100, α=30, black/18, v4 | lam100 win-rate 爬到 0.97 但 best-ds 卡 **0.406**（稳过线就停）；lam16 乱撒 best **0.375**。**win-rate↑≠接近R**。 |
| `0612_soft10` | lam16 vs 100, α=10, black/18, v4 | **全平**：α=10 太软，中心不动；best-ds 降到 0.27 是**采样运气**（每轮 ds 均值 0.482→0.470 没动）。 |
| `0613_0004` | lam50, v∈{4,12}, α=30, black/18 | **v4 赢趋势**（wr 0.58→0.75, slope +0.096, true_p 越过 0.5）；**v12 赢极值**（best **0.2803**）。广度 vs 方向 trade-off。 |
| `0613_0923` | lam50, v∈{4,12}, α=30, pigment/12 | **已 kill**（起了几分钟就换成 α=15 方案，未跑完，可忽略）。 |
| `0613_0931` | lam{50,5}, v4, α=15, pigment/12 | competitor 太弱 → **win-rate 钉死 1.0**，无趋势。但 **lam50 校准好**（pred 0.87≈true 0.85），**lam5 过度自信**（pred 0.98 vs true 0.84）。best-ds 0.327/0.344（弱对手→更远离 R）。 |
| `0614_2139` | lam{50,5}, v4, α=15, **wine/34** | 验证 rising-trend band（早期结果）。 |
| `0616_2245` | **lam0.5 单配置, v1, α=15, bright/18(3%), T=300, 10 seed 取平均** | **已完成。买到 hard 趋势、没买到 true_p（二者在此 setting 互斥）。** 见下方专段。 |

### `0616_2245` 详解（lam0.5 × 10 seed × T=300，当前最干净的 hard-rising demo）

- **动机**：用小 $\lambda$（饱和分支，唯一还有 rising hard trend 的分支）+ 大 $T$ + 10 seed 取平均，看趋势是否更干净。
- **结果**（diag：`outputs/vsweep_curves/lam05_bright_T300_10seed_diag_0616_2245.png`，脚本 `91`）：
  - **(a) hard win-rate**：10-seed 均值 **0.42 → 0.79**（first20→last20），线性 slope **+0.135/100r**，全程正向、不封顶不钉死、band 收窄。比单 seed（slope +0.191）缓但**干净得多** —— 这就是"平均后趋势更好看"的验证 ✅。到 T=300 还在微爬，没到 ~0.78–0.81 天花板就走平，**说明 T=300 够用，不必上 500**。
  - **(b) belief vs truth**：predicted_p 贴 ~1.0（过度自信），true_p_soft 只到 **0.52**（last20），gap 巨大。
  - **(d) best-so-far ds**：plateau 在 **0.396**，从没破 0.39 reach 墙。
  - **β-norm**：全程钉 **8.000** → 饱和分支，符合预期。
- **结论（重要）**：在 bright/18(3%) + 饱和分支下，**"rising hard trend" 与 "true_p 变大" 互斥**。饱和分支给趋势但锁死 true_p（β 撞 8、predicted_p 假性贴 1、best-ds 卡 reach 墙）；去饱和分支（lam50）给 true_p 但 hard 趋势会被 warm-pin 削平。**λ=0.5 这条线买的是 hard 趋势，不是 true_p。**

oracle ceiling 诊断（black/18, α=30）：227 词里 word-level soft 胜率 >0.5 的只有 0.4%，top5 词均值仅 0.438 →
**black/18 把 true_p 天花板压在 ~0.5**，这是后来换 competitor 的根本原因。

> warm-pin wall（脚本 90 确认）：optimizer warm-start 典型 ds≈0.474，所以任何 $D_B>0.474$（≥5% pct）的 competitor 在 warm-start 就被打赢 → hard win-rate 从 step 0 就钉住。**只有 3% competitor（bright/black, $D_B$≈0.470）才给 hard 上升留出空间**——这是 `0616_2245` 选 bright/18 的根本原因。

---

## 7. 脚本索引

- **引擎**：`src/cmts_sim.py`（`sigma`, `argmax_over_M`, `laplace_map`, `project_norm`, cov 构造在 `:183-184`）
- **主 worker**：`scripts/73_cmts_dreamsim.py`（CLI：`--dim --v --lam --S --alpha --B_word --B_seed --seed_start --seed_end --n0 --T --B --save_img_every --partial_id --out_root`）
- **launch（detached, 4-GPU, resumable）**：`73k`（lam16/100）, `73l`（lam50 v-sweep）, `73m`（pigment）, `73n`（pigment a15 lam-sweep）, `73o`（wine a15 lam-sweep）, `73u`（**lam0.5 bright T300 10seed, 当前**）
- **diag**：`78`（lam16/100）, `79`（lam50 v-sweep）, `80`（pigment）, `81`（pigment a15）, `82`（wine a15）, **`90`（true_p ceiling/reach × competitor×α 校准 + warm-pin 墙）**, **`91`（lam0.5 单配置多 seed 取平均，当前）**。图输出在 `outputs/vsweep_curves/`。
- **β-norm 速查图**：`outputs/vsweep_curves/theta_norm.png`（inline 生成，§4）。

### 多 GPU 启动铁律（**别去掉**）
```bash
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=$G setsid conda run -n diverse --no-capture-output \
    python scripts/73_cmts_dreamsim.py ... >> "$OUT/launch_g${G}.log" 2>&1 </dev/null &
```
- `conda env diverse` 必须；`--dim` 不是 `--d`；`setsid ... </dev/null &` 让 worker 脱离 shell（harness 杀监控**不**杀 worker）。
- launch 脚本里的 pgrep 守卫和 kill 都**别**把字面量 `73_cmts_dreamsim.py` 直接写进会匹配自身的命令（用 `[7]3_...` bracket 技巧）。
- 出图等 `sim*/summary.json` 凑齐 10 个（2 配置 ×5 seed）再跑对应 diag。

---

## 8. 当前状态 / 下一步

- **最近完成**：`0616_2245`（lam0.5, v1, α=15, bright/18, T=300, 10 seed）—— 见 §6 专段。
  **拿到了最干净的 hard-rising demo（0.42→0.79, slope +0.135/100r），但 true_p 锁死在 0.52。**
  复跑出图：
  ```bash
  conda run -n diverse --no-capture-output python scripts/91_lam05_bright_T300_diag.py 0616_2245 300
  ```
- **决策点（等老师/用户确认走哪条）**：
  1. **要 hard 趋势这条线** → 现在 `0616_2245` 这张图就能交，叙事 = "surrogate 在饱和分支下仍能稳定提升对强 competitor 的胜率"。
  2. **还想顶 true_p** → 必须去饱和：lam50 + α↑到 25–30 + v↓到 0.5，代价是 hard 曲线被 warm-pin 削平。要的话开新 sweep。**注意二者在此 setting 互斥（§6）。**
- **候选下一步（未开工，等确认）**：
  1. β 方向重建——画 $\hat\beta_t$ 与 "$z_{\text{red}}-z_{\text{comp}}$" 方向的 cos 随 $t$，量化"是否真朝 $R$ 走"。
  2. `z` 标准化 variant（§2），让 $\beta$/$S$ 尺度可解释。
  3. true_p 的根本天花板是 ds_reach≈0.39（manifold 现实最好），唯一基本面突破是让 manifold 真正更近 $R$。
