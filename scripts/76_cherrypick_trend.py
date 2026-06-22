"""76 — what does cherry-picking trajectories do to the per-round win-rate trend?

For the v=4.0 pool (M=85), compute each trajectory's own per-round win-rate slope,
look at the slope distribution, then average the top-K-by-slope and plot them next
to the honest all-M average. This QUANTIFIES the selection effect: ranking by the
very quantity you then display guarantees an upward curve (selection-on-outcome),
so the top-K curve is illustrative ONLY, never a headline result.

  conda run -n diverse --no-capture-output python scripts/76_cherrypick_trend.py
"""
import glob, os, re
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "outputs/cmts_v4.0_d16_B8_T200_0603_0137"
OUT = "outputs/vsweep_curves"; os.makedirs(OUT, exist_ok=True)
WIN = 15

# per-sim per-round win-rate series
series, slopes = [], []
for f in sorted(glob.glob(f"{D}/sim*/trajectory.csv")):
    m = pd.read_csv(f).query("phase == 'main'")
    if m.empty:
        continue
    wr = m.groupby("t")["y"].mean()
    t = wr.index.values.astype(float)
    slope = np.polyfit(t, wr.values, 1)[0] * 100      # per-100-round
    series.append(wr); slopes.append(slope)
slopes = np.array(slopes)
S = pd.concat(series, axis=1)                          # rounds x M
M = S.shape[1]
t = S.index.values.astype(float)

print(f"v=4.0 pool: M={M} trajectories")
print(f"  per-traj slope/100r:  mean={slopes.mean():+.3f}  std={slopes.std():.3f}")
print(f"  positive-slope trajectories: {(slopes > 0).sum()}/{M} "
      f"({100*(slopes>0).mean():.0f}%)")
print(f"  slope quantiles  p10={np.percentile(slopes,10):+.3f}  "
      f"median={np.median(slopes):+.3f}  p90={np.percentile(slopes,90):+.3f}")

order = np.argsort(slopes)[::-1]                        # best slope first
fig, ax = plt.subplots(figsize=(8, 5))
allavg = S.mean(axis=1)
ax.plot(t, allavg.rolling(WIN, min_periods=1, center=True).mean(), color="black",
        lw=2.4, label=f"all M={M}  (slope {slopes.mean()*0:+.0f}… see below)")
a_all = np.polyfit(t, allavg.values, 1)[0] * 100
ax.plot(t, np.polyfit(t, allavg.values, 1)[1] + np.polyfit(t, allavg.values, 1)[0]*t,
        color="black", ls="--", lw=1)

rows = [("all", M, a_all, allavg.iloc[:20].mean(), allavg.iloc[-20:].mean())]
for K, c in [(20, "tab:orange"), (10, "tab:red"), (5, "tab:purple")]:
    sel = S.iloc[:, order[:K]].mean(axis=1)
    a = np.polyfit(t, sel.values, 1)[0] * 100
    ax.plot(t, sel.rolling(WIN, min_periods=1, center=True).mean(), color=c, lw=2.0,
            label=f"top-{K} by slope  (slope {a:+.3f}/100r)")
    rows.append((f"top{K}", K, a, sel.iloc[:20].mean(), sel.iloc[-20:].mean()))

ax.axhline(0.5, color="gray", ls=":", lw=1)
ax.set_xlabel("round $t$"); ax.set_ylabel("per-round win-rate (avg over selected sims)")
ax.set_title(f"Cherry-picking v=4.0 trajectories by per-round slope (M={M})\n"
             "top-K curves are SELECTION-ON-OUTCOME — illustrative only", fontsize=11)
ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3)
fig.tight_layout()
png = os.path.join(OUT, "cherrypick_trend.png"); fig.savefig(png, dpi=140); plt.close(fig)

print(f"\n{'subset':>8} {'K':>4} {'slope/100r':>11} {'wr_first20':>11} {'wr_last20':>10}")
for name, k, a, f20, l20 in rows:
    print(f"{name:>8} {k:>4} {a:>+11.3f} {f20:>11.3f} {l20:>10.3f}")
print(f"\nsaved: {png}")
