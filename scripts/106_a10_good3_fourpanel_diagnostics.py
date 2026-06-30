#!/usr/bin/env python3
"""Four-panel diagnostics for the successful alpha=10, v=1, lambda=100 sims.

Uses only sim001/sim002/sim003 from:
  outputs/cmts_a10_v1_lam100_bbright_d16_B8_T1000_0625_2312

Produces two figures:
  1. learning/performance four-panel
  2. posterior/theta spectrum four-panel
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("outputs/cmts_a10_v1_lam100_bbright_d16_B8_T1000_0625_2312")
OUT = Path("outputs/cmts_top4_summary_0621_0437")
OUT.mkdir(parents=True, exist_ok=True)

GOOD = ["sim001", "sim002", "sim003"]
D_B = 0.4704614281654358
PREV_BEST_DS = 0.2803


def load_rounds() -> pd.DataFrame:
    parts = []
    for sim in GOOD:
        path = ROOT / sim / "trajectory.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        m = df[df["phase"] == "main"].copy()
        g = (
            m.groupby("t")
            .agg(
                hard=("y_hard", "mean"),
                true_p=("true_p_soft", "mean"),
                pred_p=("predicted_p", "mean"),
                mean_ds=("ds_to_R", "mean"),
                round_min_ds=("ds_to_R", "min"),
                beta_norm=("beta_norm", "first"),
                cov_eig_max=("cov_eig_max", "first"),
                cov_eig_min=("cov_eig_min", "first"),
            )
            .reset_index()
        )
        g["best_ds"] = g["round_min_ds"].cummin()
        g["eig_ratio"] = g["cov_eig_max"] / g["cov_eig_min"].replace(0, np.nan)
        g["sim"] = sim
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def mean_se(df: pd.DataFrame, col: str):
    g = df.groupby("t")[col]
    mean = g.mean()
    se = (g.std(ddof=1) / np.sqrt(g.count())).fillna(0)
    return mean, se


def draw(ax, df, col, color, label, ls="-", band=True, smooth=15):
    mean, se = mean_se(df, col)
    y = mean.rolling(smooth, center=True, min_periods=1).mean()
    ax.plot(mean.index, y, color=color, lw=2.0 if ls == "-" else 1.6, ls=ls, label=label)
    if band:
        ax.fill_between(mean.index, mean - se, mean + se, color=color, alpha=0.14)


def plot_perf(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), sharex=True)
    (ax_hard, ax_belief), (ax_ds, ax_best) = axes

    draw(ax_hard, df, "hard", "tab:red", "hard win-rate")
    ax_hard.axhline(0.5, color="0.4", ls=":", lw=1)
    ax_hard.axhline(1.0, color="0.2", ls=":", lw=1, label="ceiling=1")

    draw(ax_belief, df, "true_p", "tab:green", "true p")
    draw(ax_belief, df, "pred_p", "tab:red", "predicted p", ls="--", band=False)
    ax_belief.axhline(0.5, color="0.4", ls=":", lw=1)
    ax_belief.axhline(1.0, color="0.2", ls=":", lw=1, label="ceiling=1")

    draw(ax_ds, df, "mean_ds", "tab:purple", "mean ds-to-R")
    ax_ds.axhline(D_B, color="0.4", ls=":", lw=1, label=f"$D_B$={D_B:.4f}")

    draw(ax_best, df, "best_ds", "tab:blue", "best-so-far ds-to-R")
    ax_best.axhline(PREV_BEST_DS, color="k", ls=":", lw=1, label=f"previous best {PREV_BEST_DS:.4f}")

    titles = [
        "(a) hard win-rate (roll15; good 3 sims)",
        "(b) belief vs truth (solid=true, dashed=predicted)",
        "(c) mean ds-to-R per round (lower is better)",
        "(d) best-so-far ds-to-R (lower is better)",
    ]
    for ax, title in zip(axes.flat, titles):
        ax.axvline(500, color="0.35", ls=":", lw=1.0, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("round t")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    ax_hard.set_ylabel("win-rate")
    ax_belief.set_ylabel("probability")
    ax_ds.set_ylabel("mean ds-to-R")
    ax_best.set_ylabel("best ds-to-R")

    fig.suptitle(r"Continuous CM-TS good trajectories: $\alpha$=10, $v$=1, $\lambda$=100"
                 "\nonly sim001/sim002/sim003, T=1000", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUT / "a10_v1_lam100_good3_T1000_performance_4panel.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_posterior(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), sharex=True)
    axes = axes.ravel()
    panels = [
        ("beta_norm", r"$\|\theta_t\|_2$ / beta_norm", "tab:blue", "linear"),
        ("cov_eig_max", "posterior cov max eigenvalue", "tab:orange", "log"),
        ("cov_eig_min", "posterior cov min eigenvalue", "tab:green", "log"),
        ("eig_ratio", "condition ratio max/min", "tab:purple", "log"),
    ]
    for ax, (col, ylabel, color, scale) in zip(axes, panels):
        for sim, s in df.groupby("sim"):
            ax.plot(s["t"], s[col], lw=0.8, alpha=0.22, label=None)
        draw(ax, df, col, color, "mean over good 3", band=True, smooth=15)
        ax.set_yscale(scale)
        ax.axvline(500, color="0.35", ls=":", lw=1.0, alpha=0.7)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("round t")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(r"Posterior diagnostics: $\alpha$=10, $v$=1, $\lambda$=100"
                 "\nonly sim001/sim002/sim003, T=1000", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUT / "a10_v1_lam100_good3_T1000_posterior_4panel.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def save_windows(df: pd.DataFrame):
    rows = []
    specs = [
        (0, 19, "first20"),
        (450, 499, "T500_450_499"),
        (900, 949, "T1000_900_949"),
        (950, 999, "T1000_950_999"),
        (980, 999, "last20"),
        (990, 999, "last10"),
    ]
    cols = ["hard", "true_p", "pred_p", "mean_ds", "beta_norm", "best_ds"]
    for lo, hi, label in specs:
        sub = df[df["t"].between(lo, hi)]
        ps = sub.groupby("sim")[cols].mean()
        row = {"window": label, "lo": lo, "hi": hi, "n_sims": len(ps)}
        for col in cols:
            row[col] = ps[col].mean()
            row[f"{col}_se"] = ps[col].std(ddof=1) / np.sqrt(len(ps)) if len(ps) > 1 else 0.0
        rows.append(row)
    out = OUT / "a10_v1_lam100_good3_T1000_windows.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out, pd.DataFrame(rows)


def main():
    df = load_rounds()
    p1 = plot_perf(df)
    p2 = plot_posterior(df)
    csv, windows = save_windows(df)
    print(f"saved: {p1}")
    print(f"saved: {p2}")
    print(f"saved: {csv}")
    print(windows.to_string(index=False))


if __name__ == "__main__":
    main()
