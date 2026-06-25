#!/usr/bin/env python3
"""Plot theta norm and posterior covariance eigenvalues over CM-TS rounds.

This script reads the canonical continuous run:
  outputs/cmts_a30_v2_lam50_bbright_d16_B8_T200_0621_0437

The per-round trajectory file stores beta_norm (theta norm) plus max/min
posterior covariance eigenvalues. It does not store the full theta vector
history, only the final beta_hat in posterior.npz.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("outputs/cmts_a30_v2_lam50_bbright_d16_B8_T200_0621_0437")
OUT_DIR = Path("outputs/cmts_top4_summary_0621_0437")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "a30_v2_lam50_theta_eigs_T1000.png"
OUT_CSV = OUT_DIR / "a30_v2_lam50_theta_eigs_T1000.csv"


def load_per_round() -> pd.DataFrame:
    rows = []
    cols = ["t", "phase", "beta_norm", "cov_eig_max", "cov_eig_min"]
    for sim_dir in sorted(BASE.glob("sim[0-9][0-9][0-9]")):
        path = sim_dir / "trajectory.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=cols)
        df = df[df["t"] >= 0].copy()
        # Metrics are repeated across candidates within a round; average is safe
        # and also handles any minor per-candidate logging differences.
        g = (
            df.groupby("t", as_index=False)
            .agg(
                beta_norm=("beta_norm", "mean"),
                cov_eig_max=("cov_eig_max", "mean"),
                cov_eig_min=("cov_eig_min", "mean"),
            )
            .assign(sim=sim_dir.name)
        )
        rows.append(g)
    if not rows:
        raise SystemExit(f"No trajectory.csv files found under {BASE}")
    out = pd.concat(rows, ignore_index=True)
    out["eig_ratio"] = out["cov_eig_max"] / out["cov_eig_min"].replace(0, np.nan)
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["beta_norm", "cov_eig_max", "cov_eig_min", "eig_ratio"]
    agg = df.groupby("t").agg(n_sims=("sim", "nunique"))
    for m in metrics:
        q = df.groupby("t")[m].quantile([0.1, 0.9]).unstack()
        agg[f"{m}_mean"] = df.groupby("t")[m].mean()
        agg[f"{m}_q10"] = q[0.1]
        agg[f"{m}_q90"] = q[0.9]
    return agg.reset_index()


def plot(df: pd.DataFrame, agg: pd.DataFrame) -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    axes = axes.ravel()

    panels = [
        ("beta_norm", r"$\|\theta_t\|_2$ / beta_norm", "linear"),
        ("cov_eig_max", "posterior cov max eigenvalue", "log"),
        ("cov_eig_min", "posterior cov min eigenvalue", "log"),
        ("eig_ratio", "condition ratio max/min", "log"),
    ]

    for ax, (m, ylabel, yscale) in zip(axes, panels):
        for sim, s in df.groupby("sim"):
            ax.plot(s["t"], s[m], lw=0.8, alpha=0.22)
        ax.plot(agg["t"], agg[f"{m}_mean"], color="black", lw=2.0, label="mean over available sims")
        ax.fill_between(
            agg["t"].to_numpy(),
            agg[f"{m}_q10"].to_numpy(),
            agg[f"{m}_q90"].to_numpy(),
            color="black",
            alpha=0.13,
            label="10-90% across sims",
        )
        ax.set_ylabel(ylabel)
        ax.set_yscale(yscale)
        ax.axvline(200, color="tab:blue", ls="--", lw=1, alpha=0.45)
        ax.axvline(500, color="tab:green", ls="--", lw=1, alpha=0.45)
        ax.legend(loc="best", fontsize=9)

    axes[2].set_xlabel("round t")
    axes[3].set_xlabel("round t")

    ax2 = axes[0].twinx()
    ax2.plot(agg["t"], agg["n_sims"], color="tab:red", lw=1.2, alpha=0.8, label="n sims")
    ax2.set_ylabel("available sims", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(0, max(5.5, agg["n_sims"].max() + 0.5))

    max_t_by_sim = df.groupby("sim")["t"].max().to_dict()
    title = "CM-TS α=30, v=2, λ=50: theta norm and covariance spectrum"
    subtitle = "current data, max t by sim: " + ", ".join(f"{k}={v}" for k, v in max_t_by_sim.items())
    fig.suptitle(title + "\n" + subtitle, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)


def main() -> None:
    df = load_per_round()
    agg = summarize(df)
    agg.to_csv(OUT_CSV, index=False)
    plot(df, agg)
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_CSV}")
    print("max t by sim:")
    print(df.groupby("sim")["t"].max().to_string())
    for t in [0, 200, 500, 900, 999]:
        near = agg[agg["t"] == t]
        if len(near):
            r = near.iloc[0]
            print(
                f"t={t}: n={int(r.n_sims)} theta_norm={r.beta_norm_mean:.6g} "
                f"eig_max={r.cov_eig_max_mean:.6g} eig_min={r.cov_eig_min_mean:.6g} "
                f"ratio={r.eig_ratio_mean:.6g}"
            )


if __name__ == "__main__":
    main()
