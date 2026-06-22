"""90 - true_p 上限标定：回答"true_p 能不能更大"这个问题。

关键区分（这是核心）：
  - CEILING   = 最好词的 word-level true_p = max_w mean_k sigma(alpha*(D_B - D_{w,k}))
                由 (alpha, competitor D_B) 决定，是"假如优化器找到最好的词"能到的上限。
  - REACHABLE = 优化器真实能逼到的 ds（流形对 R 的最近可达，实测 best_ds≈0.39）下的
                单轮 true_p = sigma(alpha*(D_B - ds_reach))。这是"完美收敛"的乐观值。
  - 实测 achieved（panel b 实线，~0.52）< REACHABLE，差距 = 探索散开把均值拉低。

三个约束彼此耦合（都来自同一个 sigma(alpha*(D_B-ds))）：
  - true_p 大  ⟸ 大 D_B(弱对手) 或 大 alpha
  - 分布铺开(不挤 0.5) ⟸ 小 alpha
  - winning-rate 起点不钉死 ⟸ 强对手(小 D_B)
所以 true_p↑ 和 (铺开 + 起点低) 直接打架。本脚本把这张 Pareto 表算出来。

不调 SD3.5，纯查 dreams_matrix.npz。
Usage: conda run -n diverse --no-capture-output python scripts/90_truep_ceiling_calib.py [ds_reach]
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs/vsweep_curves"; os.makedirs(OUT, exist_ok=True)
DS_REACH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.39   # 实测 best_ds 量级

Z = np.load("outputs/multiseed_s228_M40_0510_0241/dreams_matrix.npz", allow_pickle=True)
D = Z["dreams"]; words = list(Z["words"])           # (228,40)
all_img = D.flatten()
ds_min = float(all_img.min()); ds_med = float(np.median(all_img))
def sigma(x): return 1/(1+np.exp(-x))

# 经验锚点：典型 warm-start 设计 ds≈0.474（实测）。warm hit-rate = P(ds_warm < D_B) 的代理。
WARM_DS = 0.474

def row(DB, a):
    wp = sigma(a*(DB - D)).mean(axis=1)              # 每词 true_p (228,)
    order = np.argsort(wp)[::-1]
    ceil = float(wp[order[0]]); best_w = words[order[0]]
    top3 = float(wp[order[:3]].mean())
    nwin = int((wp > 0.5).sum())
    reach = float(sigma(a*(DB - DS_REACH)))          # 乐观可达单轮 true_p
    p_img = sigma(a*(DB - all_img))
    floor_hard = float((all_img < DB).mean())        # 随机起点 hard 胜率 = img 百分位
    warm_pin = "PIN" if WARM_DS < DB else "ok "      # warm 设计是否一定赢→hard 钉死
    return dict(DB=DB, a=a, ceil=ceil, best_w=best_w, top3=top3, nwin=nwin,
                reach=reach, std=float(p_img.std()), floor=floor_hard, warm_pin=warm_pin)

pcts = np.array([3,5,8,10,12,15,18,22,30])
DBs = np.percentile(all_img, pcts)
ALPHAS = [15, 20, 25, 30]

print(f"ds_min={ds_min:.4f}  ds_median={ds_med:.4f}  warm_design_ds≈{WARM_DS}  ds_reach(乐观)={DS_REACH}\n")
print("CEILING=最好词true_p | REACH=σ(α(D_B-0.39))乐观可达 | top3=前三词均值 | nwin=#词>0.5")
print("warm_pin=PIN 表示 warm 设计(ds≈0.474)已稳赢→hard 起点钉死(没上升空间)\n")

hdr = f"{'img%':>4} {'D_B':>6} {'warm':>4} | " + " | ".join(
    f"a={a}: ceil/reach/top3/nwin/std" for a in ALPHAS)
print(hdr)
for pct, DB in zip(pcts, DBs):
    cells = []
    pin = ""
    for a in ALPHAS:
        m = row(DB, a); pin = m["warm_pin"]
        cells.append(f"{m['ceil']:.2f}/{m['reach']:.2f}/{m['top3']:.2f}/{m['nwin']:>2d}/{m['std']:.2f}")
    print(f"{pct:>3d}% {DB:>6.4f} {pin:>4} | " + " | ".join(cells))

# ---- 图：CEILING 与 REACH 随对手强度，多 alpha ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for a in ALPHAS:
    ceils = [row(db, a)["ceil"] for db in DBs]
    reaches = [row(db, a)["reach"] for db in DBs]
    axes[0].plot(pcts, ceils, marker="o", label=f"$\\alpha$={a}")
    axes[1].plot(pcts, reaches, marker="s", label=f"$\\alpha$={a}")
for ax, ttl in zip(axes, ["CEILING = 最好词 true_p (上限)",
                          f"REACH = σ(α(D_B−{DS_REACH})) 乐观可达单轮 true_p"]):
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    # 标 warm-pin 边界：D_B>0.474 即 hard 起点钉死
    pin_pct = float(np.interp(WARM_DS, DBs, pcts))
    ax.axvspan(pin_pct, pcts.max(), color="red", alpha=0.07)
    ax.text(pin_pct+0.3, 0.46, "→hard起点钉死区\n(warm设计已稳赢)", fontsize=8, color="darkred", va="top")
    ax.set_xlabel("competitor 图片百分位 (越大=越弱)")
    ax.set_ylabel("true_p"); ax.set_title(ttl); ax.grid(alpha=0.3); ax.legend()
fig.suptitle(f"true_p 上限 vs 对手强度 × α — 红区=hard 没上升空间；要 true_p 大又要避开红区 = 挤 8–15% 窄带",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
p = os.path.join(OUT, "truep_ceiling_calib.png")
fig.savefig(p, dpi=140); plt.close(fig)
print(f"\nsaved: {p}")
