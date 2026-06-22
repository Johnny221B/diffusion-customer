"""91 - single lambda=0.5, v=1.0, alpha=15, bright/18, T=300, 10-seed diagnostic.

Focus: does AVERAGING over 10 seeds make the rising hard-win-rate trend clean (vs the 1-seed scout)?
Panel (a) shows every seed as a faint line + the 10-seed mean (bold) + a +/-1 std band + linear fit/slope.
Other panels (belief, best-ds, beta-norm) plotted as 10-seed means.

Usage: conda run -n diverse --no-capture-output python scripts/91_lam05_bright_T300_diag.py [STAMP] [T] [LAM=..] [VV=..]
  e.g. ... 0616_2245 300                 # lam0.5 v1.0 (saturated branch, default)
       ... 0617_xxxx 300 LAM=50 VV=4.0   # lam50 v4.0 (de-saturated branch)
"""
import glob, os, re, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D_B = 0.4705198            # competitor bright/18
BLACK_BEST = 0.2803
S_CLIP = 8.0
WIN = 15
OUT = "outputs/vsweep_curves"; os.makedirs(OUT, exist_ok=True)

# --- parse optional KEY=VALUE overrides anywhere in argv ---
LAM = "0.5"; VV = "1.0"
pos = []
for a in sys.argv[1:]:
    if a.startswith("LAM="):   LAM = a.split("=", 1)[1]
    elif a.startswith("VV="):  VV = a.split("=", 1)[1]
    else:                      pos.append(a)
SAT = float(LAM) < 1.0                     # saturated branch if lam small
branch = "rising-trend / saturated" if SAT else "de-saturated"
tag = f"lam{LAM}_v{VV}"

T = pos[1] if len(pos) > 1 else "300"
if len(pos) > 0:
    STAMP = pos[0]
else:
    cands = sorted(glob.glob(f"outputs/cmts_lam{LAM}_v{VV}_bbright_a15_d16_B8_T{T}_*"))
    if not cands:
        sys.exit(f"no cmts_lam{LAM}_v{VV}_bbright_a15_*_T{T}_* runs found; pass STAMP explicitly")
    STAMP = re.search(rf'_T{T}_(\d{{4}}_\d{{4}})$', cands[-1]).group(1)
print(f"STAMP = {STAMP}  T = {T}")

d = f"outputs/cmts_lam{LAM}_v{VV}_bbright_a15_d16_B8_T{T}_{STAMP}"
files = sorted(glob.glob(f"{d}/sim*/trajectory.csv"))
if not files:
    sys.exit(f"no sim*/trajectory.csv under {d} yet")

wr_list, soft_list, pred_list, best_list, bn_list = [], [], [], [], []
seed_ids = []
for f in files:
    m = pd.read_csv(f).query("phase == 'main'")
    if m.empty:
        continue
    seed_ids.append(re.search(r'sim(\d+)', f).group(1))
    wr_list.append(m.groupby("t")["y_hard"].mean())
    soft_list.append(m.groupby("t")["true_p_soft"].mean())
    pred_list.append(m.groupby("t")["predicted_p"].mean())
    best_list.append(m.groupby("t")["ds_to_R"].min().cummin())
    bn_list.append(m.groupby("t")["beta_norm"].first())

n = len(wr_list)
print(f"loaded {n} seeds: {seed_ids}")
WR = pd.concat(wr_list, axis=1)          # rows=t, cols=seed
t = WR.index.values.astype(float)
wr_mean = WR.mean(axis=1); wr_std = WR.std(axis=1)
soft_mean = pd.concat(soft_list, axis=1).mean(axis=1)
pred_mean = pd.concat(pred_list, axis=1).mean(axis=1)
best_df = pd.concat(best_list, axis=1); best_mean = best_df.mean(axis=1); best_std = best_df.std(axis=1)
bn_mean = pd.concat(bn_list, axis=1).mean(axis=1)

roll = lambda s: s.rolling(WIN, min_periods=1, center=True).mean()

fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
(axWR, axGAP), (axBEST, axBN) = axes

# ---- (a) hard win-rate: per-seed faint + mean + band + lin fit ----
for c in WR.columns:
    axWR.plot(t, roll(WR[c]).values, color="tab:blue", lw=0.6, alpha=0.25)
axWR.fill_between(t, roll(wr_mean - wr_std).values, roll(wr_mean + wr_std).values,
                  color="tab:blue", alpha=0.15, label="$\\pm1$ std over seeds")
axWR.plot(t, roll(wr_mean).values, color="tab:blue", lw=2.6, label=f"mean (n={n}, roll{WIN})")
a, b = np.polyfit(t, wr_mean.values, 1)
axWR.plot(t, a * t + b, color="k", lw=1.3, ls="--", label=f"lin fit slope={a*100:+.3f}/100r")
axWR.axhline(0.5, color="gray", ls=":", lw=1)
axWR.axhline(0.775, color="darkred", ls=":", lw=1)
axWR.annotate("bright best-word hard=0.775", (t[-1], 0.775), fontsize=8, color="darkred",
              ha="right", va="bottom")
axWR.set_title("(a) hard win-rate $P(\\mathrm{ds}<D_B)$ — per-seed (faint) + 10-seed mean + band")
axWR.set_xlabel("round $t$"); axWR.set_ylabel("win-rate"); axWR.grid(alpha=0.3); axWR.legend(fontsize=8)

# ---- (b) belief vs truth ----
axGAP.plot(t, roll(soft_mean).values, color="tab:green", lw=2.2, label="true_p_soft (oracle)")
axGAP.plot(t, roll(pred_mean).values, color="tab:red", lw=1.6, ls="--", label="predicted_p (belief)")
axGAP.axhline(0.5, color="gray", ls=":", lw=1)
axGAP.set_title(f"(b) belief vs truth ($\\alpha$=15) — gap = overconfidence (lam={LAM} {branch})")
axGAP.set_xlabel("round $t$"); axGAP.set_ylabel("probability"); axGAP.grid(alpha=0.3); axGAP.legend(fontsize=8)

# ---- (c) best-so-far ds ----
bt = best_mean.index.values.astype(float)
axBEST.fill_between(bt, (best_mean - best_std).values, (best_mean + best_std).values,
                    color="tab:purple", alpha=0.15)
axBEST.plot(bt, best_mean.values, color="tab:purple", lw=2.2, label=f"mean best-so-far (n={n})")
axBEST.axhline(D_B, color="gray", ls=":", lw=1)
axBEST.annotate(f"$D_B$={D_B:.4f}", (bt[0], D_B), fontsize=8, color="gray", va="bottom")
axBEST.axhline(BLACK_BEST, color="k", ls=":", lw=1)
axBEST.annotate(f"prev best={BLACK_BEST}", (bt[0], BLACK_BEST), fontsize=8, color="k", va="bottom")
axBEST.set_title("(c) best-so-far ds-to-$R$ (lower=closer)")
axBEST.set_xlabel("round $t$"); axBEST.set_ylabel("best-so-far ds_to_R"); axBEST.grid(alpha=0.3)
axBEST.legend(fontsize=8)

# ---- (d) beta-norm ----
axBN.plot(t, bn_mean.values, color="tab:orange", lw=2.2, label=f"mean $\\|\\hat\\beta\\|$ (n={n})")
axBN.axhline(S_CLIP, color="k", ls="--", lw=1.2, label="S=8 (norm clip)")
axBN.set_title(f"(d) $\\|\\hat\\beta_t\\|$ — at S=8 => saturated (lam={LAM} {branch})")
axBN.set_xlabel("round $t$"); axBN.set_ylabel("$\\|\\hat\\beta_t\\|$"); axBN.grid(alpha=0.3); axBN.legend(fontsize=8)

fig.suptitle(f"CM-TS lam={LAM} ({branch}), v={VV}, B=bright/18 ($D_B$=0.4705, 3pct), "
             f"$\\alpha$=15, d=16, T={T}, B=8, n={n} seeds ({STAMP})", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
png = os.path.join(OUT, f"{tag}_bright_T{T}_10seed_diag_{STAMP}.png")
fig.savefig(png, dpi=140); plt.close(fig)

print(f"\nslope={a*100:+.3f}/100r  wr_f20={wr_mean.iloc[:20].mean():.3f}  "
      f"wr_l20={wr_mean.iloc[-20:].mean():.3f}  soft_l20={soft_mean.iloc[-20:].mean():.3f}  "
      f"best_ds_end={best_mean.iloc[-1]:.4f}  beta_end={bn_mean.iloc[-1]:.3f}")
print(f"saved: {png}")
