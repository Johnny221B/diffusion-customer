"""87 - 联合校准 (competitor, alpha)：找一个 competitor 使 ~10 个词能软赢它，且不过弱，
同时挑一个让每张图的 soft 概率 sigma(alpha*(D_B-ds)) 分布最"均匀"(铺开)的 alpha。

不调 SD3.5，纯查 dreams_matrix.npz。输出：
  - 控制台表：competitor 百分位 -> #词p>0.5 / 天花板 / 最好词hard胜率(钉死检查) / 分布散度
  - 图1 outputs/vsweep_curves/calib_nwin_vs_competitor.png : #词p>0.5 随对手强度，多条 alpha
  - 图2 outputs/vsweep_curves/calib_dist_<word><seed>.png   : 选定对手在多 alpha 下的 per-image 直方图

Usage: conda run -n diverse --no-capture-output python scripts/87_competitor_alpha_calib.py [target_nwin]
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs/vsweep_curves"; os.makedirs(OUT, exist_ok=True)
TARGET_NWIN = int(sys.argv[1]) if len(sys.argv) > 1 else 10

Z = np.load("outputs/multiseed_s228_M40_0510_0241/dreams_matrix.npz", allow_pickle=True)
D = Z["dreams"]; words = list(Z["words"])           # (228,40)
all_img = D.flatten()
def sigma(x): return 1/(1+np.exp(-x))

def metrics(DB, a):
    wp = sigma(a*(DB - D)).mean(axis=1)             # 每词 soft 胜率 (228,)
    bi = int(np.argmax(wp))
    best_hard = float((D[bi] < DB).mean())          # 最好词的 hard 胜率(钉死检查)
    p_img = sigma(a*(DB - all_img))                 # 每图 soft 概率 (9120,)
    return dict(nwin=int((wp > 0.5).sum()), ceil=float(wp.max()), best_word=words[bi],
                best_hard=best_hard, p_img_std=float(p_img.std()), wp_mean=float(wp.mean()))

# ---- 沿"对手图片百分位"扫 D_B；百分位 = 有多少比例图比它更近 R ----
pcts = np.array([3,5,8,10,12,15,18,22,26,30,35,40])
DBs = np.percentile(all_img, pcts)
ALPHAS = [10, 15, 20, 30]

print(f"target #词 p>0.5 ≈ {TARGET_NWIN}\n")
print(f"{'img%':>5} {'D_B':>7} | " + " ".join(f"a={a}:nwin/ceil/bestHard" for a in ALPHAS))
rows = {a: [] for a in ALPHAS}
for pct, DB in zip(pcts, DBs):
    cells = []
    for a in ALPHAS:
        m = metrics(DB, a); rows[a].append(m["nwin"])
        cells.append(f"{m['nwin']:>2d}/{m['ceil']:.2f}/{m['best_hard']:.2f}")
    print(f"{pct:>4d}% {DB:>7.4f} | " + "   ".join(cells))

# ---- 在 alpha=15 下找 #词≈TARGET 的 D_B，再定位一个真实 (word,seed) 作 competitor ----
A0 = 15
fine_pcts = np.linspace(3, 40, 200)
fine_DBs = np.percentile(all_img, fine_pcts)
nwins = np.array([metrics(db, A0)["nwin"] for db in fine_DBs])
j = int(np.argmin(np.abs(nwins - TARGET_NWIN)))
DB_star = fine_DBs[j]
# 找最接近 DB_star 的图片 (word,seed)
flat_idx = int(np.argmin(np.abs(D - DB_star)))
wi, si = divmod(flat_idx, D.shape[1])
cand_word, cand_seed, cand_DB = words[wi], si, float(D[wi, si])
mstar = metrics(cand_DB, A0)
print(f"\n>>> alpha={A0} 下 #词≈{TARGET_NWIN} 对应 D_B≈{DB_star:.4f} (img pct≈{fine_pcts[j]:.0f}%)")
print(f">>> 最接近的真实 competitor: {cand_word}/{cand_seed}  D_B={cand_DB:.4f}")
print(f"    -> #词p>0.5={mstar['nwin']}  天花板={mstar['ceil']:.3f}({mstar['best_word']})  "
      f"最好词hard胜率={mstar['best_hard']:.3f}  per-image std={mstar['p_img_std']:.3f}")

# ---- 图1：#词p>0.5 vs 对手百分位，多条 alpha ----
fig, ax = plt.subplots(figsize=(8, 5))
for a in ALPHAS:
    ax.plot(pcts, rows[a], marker="o", label=f"$\\alpha$={a}")
ax.axhline(TARGET_NWIN, color="k", ls="--", lw=1, label=f"target={TARGET_NWIN}")
ax.set_xlabel("competitor 图片百分位 (越大=对手越弱)")
ax.set_ylabel("#词 soft 胜率 >0.5 (赢面大小)")
ax.set_title("赢面 vs 对手强度 × $\\alpha$ — 找 ~10 个词能赢且不过弱的点")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); p1 = os.path.join(OUT, "calib_nwin_vs_competitor.png")
fig.savefig(p1, dpi=140); plt.close(fig)

# ---- 图2：选定 competitor 下 per-image soft 概率分布，多 alpha(看哪个最均匀) ----
fig, axes = plt.subplots(1, len(ALPHAS), figsize=(4*len(ALPHAS), 4), sharey=True)
bins = np.linspace(0, 1, 26)
for ax, a in zip(axes, ALPHAS):
    p_img = sigma(a*(cand_DB - all_img))
    ax.hist(p_img, bins=bins, color="tab:blue", alpha=0.8)
    nwin_a = metrics(cand_DB, a)["nwin"]
    ax.axvline(0.5, color="gray", ls=":", lw=1)
    ax.set_title(f"$\\alpha$={a}\n#词>0.5={nwin_a}, std={p_img.std():.3f}")
    ax.set_xlabel("per-image soft prob $\\sigma(\\alpha(D_B-ds))$")
axes[0].set_ylabel("# images (of 9120)")
fig.suptitle(f"per-image soft-prob 分布 — competitor={cand_word}/{cand_seed} (D_B={cand_DB:.4f}); "
             f"越铺开=越均匀=信息越多", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
p2 = os.path.join(OUT, f"calib_dist_{cand_word}{cand_seed}.png")
fig.savefig(p2, dpi=140); plt.close(fig)

print(f"\nsaved: {p1}")
print(f"saved: {p2}")
