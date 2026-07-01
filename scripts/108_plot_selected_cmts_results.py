#!/usr/bin/env python3
"""Plot selected CM-TS result pairs for the results folder.

Each setting gets:
  - performance four-panel, with alpha-dependent discrete soft-oracle line in
    the top-right true-p panel
  - posterior four-panel

Only PNG files are written/copied to results.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


D_B = 0.4704614281654358
PREV_BEST_DS = 0.2803
DREAMS_NPZ = Path("outputs/multiseed_s228_M40_0510_0241/dreams_matrix.npz")
RESULTS = Path("results/cmts_selected_alpha_lam100_fourpanels")
RESULTS.mkdir(parents=True, exist_ok=True)


SETTINGS = [
    dict(
        key="a8_v1_lam100_T1000_all5",
        title=r"$\alpha$=8, $v$=1, $\lambda$=100, all 5 sims, T=1000",
        root=Path("outputs/cmts_a8_v1_lam100_bbright_d16_B8_T1000_0629_0842"),
        sims=["sim000", "sim001", "sim002", "sim003", "sim004"],
        alpha=8.0,
        xmax=1000,
    ),
    dict(
        key="a10_v0.5_lam100_T1000_completed3",
        title=r"$\alpha$=10, $v$=0.5, $\lambda$=100, completed 3 sims, T=1000",
        root=Path("outputs/cmts_a10_v0.5_lam100_bbright_d16_B8_T1000_0629_0842"),
        sims=["sim001", "sim002", "sim003"],
        alpha=10.0,
        xmax=1000,
    ),
    dict(
        key="a10_v1_lam100_good3_T1200",
        title=r"$\alpha$=10, $v$=1, $\lambda$=100, good 3 sims, T=1200",
        root=Path("outputs/cmts_a10_v1_lam100_bbright_d16_B8_T1000_0625_2312"),
        sims=["sim001", "sim002", "sim003"],
        alpha=10.0,
        xmax=1200,
    ),
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def soft_oracle(alpha: float):
    z = np.load(DREAMS_NPZ, allow_pickle=True)
    dreams = z["dreams"].astype(float)
    words = z["words"].astype(str)
    p_by_word = sigmoid(alpha * (D_B - dreams)).mean(axis=1)
    i = int(np.argmax(p_by_word))
    return float(p_by_word[i]), str(words[i])


def load_rounds(root: Path, sims: list[str]) -> pd.DataFrame:
    parts = []
    for sim in sims:
        path = root / sim / "trajectory.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        m = df[df["phase"] == "main"].copy()
        if m.empty:
            raise RuntimeError(f"No main rows in {path}")
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


def perf_plot(df: pd.DataFrame, cfg: dict, ceiling: float, ceiling_word: str):
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), sharex=True)
    (ax_hard, ax_belief), (ax_ds, ax_best) = axes

    draw(ax_hard, df, "hard", "tab:red", "hard win-rate")
    ax_hard.axhline(0.5, color="0.4", ls=":", lw=1)
    ax_hard.axhline(1.0, color="0.2", ls=":", lw=1, label="ceiling=1")

    draw(ax_belief, df, "true_p", "tab:green", "true p")
    draw(ax_belief, df, "pred_p", "tab:red", "predicted p", ls="--", band=False)
    ax_belief.axhline(0.5, color="0.4", ls=":", lw=1)
    ax_belief.axhline(1.0, color="0.2", ls=":", lw=1, label="ceiling=1")
    ax_belief.axhline(
        ceiling,
        color="k",
        ls="--",
        lw=1.3,
        label=f"discrete soft oracle={ceiling:.3f} ({ceiling_word})",
    )

    draw(ax_ds, df, "mean_ds", "tab:purple", "mean ds-to-R")
    ax_ds.axhline(D_B, color="0.4", ls=":", lw=1, label=f"$D_B$={D_B:.4f}")

    draw(ax_best, df, "best_ds", "tab:blue", "best-so-far ds-to-R")
    ax_best.axhline(PREV_BEST_DS, color="k", ls=":", lw=1, label=f"previous best {PREV_BEST_DS:.4f}")

    titles = [
        "(a) hard win-rate (roll15)",
        "(b) belief vs truth (solid=true, dashed=predicted)",
        "(c) mean ds-to-R per round (lower is better)",
        "(d) best-so-far ds-to-R (lower is better)",
    ]
    for ax, title in zip(axes.flat, titles):
        ax.set_xlim(0, cfg["xmax"])
        ax.set_title(title)
        ax.set_xlabel("round t")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    ax_hard.set_ylabel("win-rate")
    ax_belief.set_ylabel("probability")
    ax_ds.set_ylabel("mean ds-to-R")
    ax_best.set_ylabel("best ds-to-R")

    fig.suptitle("Continuous CM-TS: " + cfg["title"], fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = RESULTS / f"{cfg['key']}_performance_4panel.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def posterior_plot(df: pd.DataFrame, cfg: dict):
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), sharex=True)
    axes = axes.ravel()
    panels = [
        ("beta_norm", r"$\|\theta_t\|_2$ / beta_norm", "tab:blue", "linear"),
        ("cov_eig_max", "posterior cov max eigenvalue", "tab:orange", "log"),
        ("cov_eig_min", "posterior cov min eigenvalue", "tab:green", "log"),
        ("eig_ratio", "condition ratio max/min", "tab:purple", "log"),
    ]
    for ax, (col, ylabel, color, scale) in zip(axes, panels):
        for _, s in df.groupby("sim"):
            ax.plot(s["t"], s[col], lw=0.8, alpha=0.22)
        draw(ax, df, col, color, "mean", band=True, smooth=15)
        ax.set_xlim(0, cfg["xmax"])
        ax.set_yscale(scale)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("round t")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Posterior diagnostics: " + cfg["title"], fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = RESULTS / f"{cfg['key']}_posterior_4panel.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def main():
    for cfg in SETTINGS:
        df = load_rounds(cfg["root"], cfg["sims"])
        ceiling, word = soft_oracle(cfg["alpha"])
        p1 = perf_plot(df, cfg, ceiling, word)
        p2 = posterior_plot(df, cfg)
        print(f"{cfg['key']}:")
        print(f"  {p1}")
        print(f"  {p2}")
        tail = df.groupby("sim").tail(20).groupby("sim")[["true_p", "mean_ds", "hard"]].mean()
        print(tail.mean().to_string())


if __name__ == "__main__":
    main()
