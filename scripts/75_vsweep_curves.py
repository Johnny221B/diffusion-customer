"""75 — v-sweep learning curves: per-round average trend across the M=5 trajectories.

For each v in {0.5,1.0,2.0,4.0} (dirs outputs/cmts_v*_d16_B8_T200_0603_0137):
  - per-round win-rate y(t): for each sim, mean y over the B theta-batch at round t,
    then average that across the M sims  -> the discrete-style learning curve.
  - best-so-far distance to R: per sim cumulative min of ds_to_R, averaged across sims
    (lower = closer to the reference image; the real "did we find a good word" signal).

Left panel  = win-rate (raw faint + rolling-mean bold + linear fit dashed).
Right panel = best-so-far ds_to_R (averaged across sims), with D_B threshold line.

CPU only. conda run -n diverse --no-capture-output python scripts/75_vsweep_curves.py
"""
import glob, os, re
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAMP = "0603_0137"
D_B = 0.4704          # competitor (black,18) threshold; y=1 iff ds_to_R < D_B
WIN = 15              # rolling window for the smoothed win-rate
OUT = "outputs/vsweep_curves"
os.makedirs(OUT, exist_ok=True)

dirs = sorted(glob.glob(f"outputs/cmts_v*_d16_B8_T200_{STAMP}"),
              key=lambda p: float(re.search(r'_v([0-9.]+)_', p).group(1)))
colors = {"0.5": "tab:blue", "1.0": "tab:green", "2.0": "tab:orange", "4.0": "tab:red"}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6))
summary = []
for d in dirs:
    v = re.search(r'_v([0-9.]+)_', d).group(1)
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
    wr = pd.concat(wr_cols, axis=1).mean(axis=1)                   # avg across sims
    best = pd.concat(best_cols, axis=1).mean(axis=1)
    t = wr.index.values.astype(float)
    c = colors.get(v, "gray")

    axL.plot(t, wr.values, color=c, alpha=0.18, lw=0.9)
    axL.plot(t, wr.rolling(WIN, min_periods=1, center=True).mean().values,
             color=c, lw=2.0, label=f"v={v}")
    a, b = np.polyfit(t, wr.values, 1)
    axL.plot(t, a * t + b, color=c, lw=1.0, ls="--", alpha=0.8)

    axR.plot(best.index.values, best.values, color=c, lw=2.0, label=f"v={v}")
    summary.append((v, len(wr_cols), wr.iloc[:20].mean(), wr.iloc[-20:].mean(),
                    a * 100, float(best.iloc[-1])))

axL.set_xlabel("round $t$"); axL.set_ylabel("per-round win-rate (avg over M sims)")
axL.set_title("Learning curve: $P(\\mathrm{ds}<D_B)$ per round"); axL.grid(alpha=0.3)
axL.legend(title="exploration $v$", fontsize=9)
axL.axhline(0.5, color="gray", ls=":", lw=1)

axR.axhline(D_B, color="gray", ls=":", lw=1)
axR.annotate(f"$D_B$={D_B}", (t[0], D_B), fontsize=8, color="gray", va="bottom")
axR.set_xlabel("round $t$"); axR.set_ylabel("best-so-far ds-to-$R$ (avg over M sims)")
axR.set_title("Best image found so far (lower = closer to $R$)"); axR.grid(alpha=0.3)
axR.legend(title="exploration $v$", fontsize=9)

fig.suptitle(f"CM-TS exploration sweep — M=5 averaged (PCA d=16, T=200, B=8, {STAMP})",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
png = os.path.join(OUT, "vsweep_curves.png"); fig.savefig(png, dpi=140); plt.close(fig)

print(f"{'v':>5} {'nSim':>5} {'wr_first20':>10} {'wr_last20':>9} {'slope/100r':>10} {'best_ds':>8}")
for s in summary:
    print(f"{s[0]:>5} {s[1]:>5} {s[2]:>10.3f} {s[3]:>9.3f} {s[4]:>+10.3f} {s[5]:>8.4f}")
print(f"\nsaved: {png}")
