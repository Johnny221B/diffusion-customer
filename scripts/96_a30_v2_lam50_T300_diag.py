"""Four-panel diagnostic for the paired T=200 -> T=300 extension."""

import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = "outputs/cmts_a30_v2_lam50_bbright_d16_B8_T200_0621_0437"
OUT = "outputs/cmts_top4_summary_0621_0437"
os.makedirs(OUT, exist_ok=True)
ROLL = 15
D_B = 0.4704614281654358


rounds = []
for i, path in enumerate(sorted(glob.glob(f"{ROOT}/sim*/trajectory.csv"))):
    m = pd.read_csv(path).query("phase == 'main'")
    r = m.groupby("t").agg(
        hard=("y_hard", "mean"),
        true_p=("true_p_soft", "mean"),
        pred_p=("predicted_p", "mean"),
        mean_ds=("ds_to_R", "mean"),
        round_min_ds=("ds_to_R", "min"),
        beta_norm=("beta_norm", "first"),
        cov_max=("cov_eig_max", "first"),
    ).reset_index()
    r["best_ds"] = r.round_min_ds.cummin()
    r["sim"] = i
    rounds.append(r)
r = pd.concat(rounds, ignore_index=True)


def stats(col):
    g = r.groupby("t")[col]
    mean = g.mean()
    se = g.std(ddof=1) / np.sqrt(g.count())
    return mean, se.fillna(0)


def draw(ax, col, color, label, dashed=False, band=True):
    mean, err = stats(col)
    smooth = mean.rolling(ROLL, center=True, min_periods=1).mean()
    ax.plot(mean.index, smooth, color=color, lw=2 if not dashed else 1.6,
            ls="--" if dashed else "-", label=label)
    if band:
        ax.fill_between(mean.index, mean - err, mean + err, color=color, alpha=.14)


fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), sharex=True)
(ax_hard, ax_belief), (ax_ds, ax_best) = axes
draw(ax_hard, "hard", "tab:red", "hard win-rate")
draw(ax_belief, "true_p", "tab:green", "true p")
draw(ax_belief, "pred_p", "tab:red", "predicted p", dashed=True, band=False)
draw(ax_ds, "mean_ds", "tab:purple", "mean ds-to-R")
draw(ax_best, "best_ds", "tab:blue", "best-so-far ds-to-R")

ax_hard.axhline(.5, color="0.4", ls=":")
ax_belief.axhline(.5, color="0.4", ls=":")
ax_ds.axhline(D_B, color="0.4", ls=":", label=f"$D_B$={D_B:.4f}")
ax_best.axhline(.2803, color="k", ls=":", label="previous best 0.2803")
titles = [
    "(a) hard win-rate (roll15; mean ± SE over 5 sims)",
    "(b) belief vs truth (solid=true, dashed=predicted)",
    "(c) mean ds-to-R per round (lower is better)",
    "(d) best-so-far ds-to-R (lower is better)",
]
for ax, title in zip(axes.flat, titles):
    ax.axvline(200, color="0.35", ls="--", lw=1.2, label="T=200 extension point")
    ax.set_title(title); ax.set_xlabel("round t"); ax.grid(alpha=.25)
    ax.legend(fontsize=8)
ax_hard.set_ylabel("win-rate")
ax_belief.set_ylabel("probability")
ax_ds.set_ylabel("mean ds-to-R")
ax_best.set_ylabel("best ds-to-R")
fig.suptitle(r"Continuous CM-TS: $\alpha$=30, $v$=2, $\lambda$=50, "
             r"bright/18, T=300, B=8")
fig.tight_layout(rect=(0, 0, 1, .96))
png = os.path.join(OUT, "a30_v2_lam50_T300_diagnostics.png")
fig.savefig(png, dpi=170)
plt.close(fig)


windows = []
for lo, hi, label in [(0, 19, "first20"), (180, 199, "T200_last20"),
                      (250, 299, "T300_last50"), (280, 299, "T300_last20"),
                      (290, 299, "T300_last10")]:
    x = r[r.t.between(lo, hi)]
    per_sim = x.groupby("sim")[["hard", "true_p", "pred_p", "mean_ds"]].mean()
    row = {"window": label, "t_lo": lo, "t_hi": hi}
    for col in per_sim.columns:
        row[col] = per_sim[col].mean()
        row[f"{col}_se"] = per_sim[col].std(ddof=1) / np.sqrt(len(per_sim))
    windows.append(row)
summary = pd.DataFrame(windows)
summary.to_csv(os.path.join(OUT, "a30_v2_lam50_T300_windows.csv"), index=False)
print(summary.to_string(index=False))
print(f"\nsaved: {png}")

