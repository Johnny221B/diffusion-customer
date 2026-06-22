"""77 — lambda-sweep learning curves: per-round average trend across the M sims.

For each lam in {16,100,1000,10000} (dirs outputs/cmts_lam*_v4.0_d16_B8_T200_0608_1146):
  - per-round win-rate y(t): for each sim, mean y over the B theta-batch at round t,
    then average across the completed sims  -> the discrete-style learning curve.
  - best-so-far distance to R: per sim cumulative min of ds_to_R, averaged across sims.

lam=16 (=d) is the SATURATED baseline (||beta|| pinned at clip S=8); larger lam
de-saturates (smaller ||beta||, sigma in the soft band). v is FIXED at 4.0 here --
this is a one-D lambda sweep, NOT a joint lambda x v grid.

Left panel  = win-rate (raw faint + rolling-mean bold + linear fit dashed).
Right panel = best-so-far ds_to_R (averaged across sims), with D_B threshold line.

CPU only. conda run -n diverse --no-capture-output python scripts/77_lamsweep_curves.py
"""
import glob, os, re
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAMP = "0608_1146"
D_B = 0.4704          # competitor (black,18) threshold; y=1 iff ds_to_R < D_B
WIN = 15              # rolling window for the smoothed win-rate
OUT = "outputs/vsweep_curves"
os.makedirs(OUT, exist_ok=True)

dirs = sorted(glob.glob(f"outputs/cmts_lam*_v4.0_d16_B8_T200_{STAMP}"),
              key=lambda p: int(re.search(r'_lam([0-9]+)_', p).group(1)))
colors = {"16": "tab:gray", "100": "tab:green", "1000": "tab:orange", "10000": "tab:red"}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6))
summary = []
for d in dirs:
    lam = re.search(r'_lam([0-9]+)_', d).group(1)
    files = sorted(glob.glob(f"{d}/sim*/trajectory.csv"))
    wr_cols, best_cols = [], []
    for f in files:
        m = pd.read_csv(f).query("phase == 'main'")
        if m.empty:
            continue
        wr_cols.append(m.groupby("t")["y"].mean())                 # round win-rate
        ds_round = m.groupby("t")["ds_to_R"].min()                 # best in this round
        best_cols.append(ds_round.cummin())                        # best-so-far this sim
    if not wr_cols:
        continue
    nsim = len(wr_cols)
    wr = pd.concat(wr_cols, axis=1).mean(axis=1)                   # avg across sims
    best = pd.concat(best_cols, axis=1).mean(axis=1)
    t = wr.index.values.astype(float)
    c = colors.get(lam, "black")
    lbl = f"$\\lambda$={lam} (n={nsim})" + (" — saturated" if lam == "16" else "")

    axL.plot(t, wr.values, color=c, alpha=0.18, lw=0.9)
    axL.plot(t, wr.rolling(WIN, min_periods=1, center=True).mean().values,
             color=c, lw=2.0, label=lbl)
    a, b = np.polyfit(t, wr.values, 1)
    axL.plot(t, a * t + b, color=c, lw=1.0, ls="--", alpha=0.8)

    axR.plot(best.index.values, best.values, color=c, lw=2.0, label=f"$\\lambda$={lam}")
    summary.append((lam, nsim, wr.iloc[:20].mean(), wr.iloc[-20:].mean(),
                    a * 100, float(best.iloc[-1])))

axL.set_xlabel("round $t$"); axL.set_ylabel("per-round win-rate (avg over sims)")
axL.set_title("Learning curve: $P(\\mathrm{ds}<D_B)$ per round (v=4.0 fixed)")
axL.grid(alpha=0.3)
axL.legend(title="prior precision $\\lambda$", fontsize=9)
axL.axhline(0.5, color="gray", ls=":", lw=1)

axR.axhline(D_B, color="gray", ls=":", lw=1)
axR.annotate(f"$D_B$={D_B}", (0, D_B), fontsize=8, color="gray", va="bottom")
axR.set_xlabel("round $t$"); axR.set_ylabel("best-so-far ds-to-$R$ (avg over sims)")
axR.set_title("Best image found so far (lower = closer to $R$)"); axR.grid(alpha=0.3)
axR.legend(title="prior precision $\\lambda$", fontsize=9)

fig.suptitle(f"CM-TS prior-precision sweep — de-saturation via $\\lambda$ "
             f"(PCA d=16, v=4.0, T=200, B=8, {STAMP})", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
png = os.path.join(OUT, "lamsweep_curves.png"); fig.savefig(png, dpi=140); plt.close(fig)

print(f"{'lam':>6} {'nSim':>5} {'wr_first20':>10} {'wr_last20':>9} {'slope/100r':>10} {'best_ds':>8}")
for s in summary:
    print(f"{s[0]:>6} {s[1]:>5} {s[2]:>10.3f} {s[3]:>9.3f} {s[4]:>+10.3f} {s[5]:>8.4f}")
print(f"\nsaved: {png}")
