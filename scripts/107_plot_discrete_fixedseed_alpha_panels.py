#!/usr/bin/env python3
"""Per-alpha comparison plots for fixed-image-seed discrete CM-TS sweeps.

For each training alpha, write two figures:
  1. four-panel diagnostics analogous to continuous T1000 diagnostics;
  2. theta norm / covariance spectrum analogous to theta_eigs diagnostics.

Each line is one (v, lambda) configuration.
"""

from pathlib import Path

import argparse
import json
import re

import numpy as np
import pandas as pd


def _latest_run(root: Path) -> Path:
    runs = sorted(root.glob("discrete_cmts_fixedseed18_a_sweep_*"))
    if not runs:
        raise SystemExit(f"No discrete sweep run directories found under {root}")
    return runs[-1]


def _label(v: float, lam: float) -> str:
    return f"v={v:g}, lambda={lam:g}"


def _safe_alpha(alpha: float) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "p", f"{alpha:g}")


def _mean_se(df: pd.DataFrame, value: str) -> pd.DataFrame:
    g = df.groupby(["alpha_train", "v", "lam", "t", "sim_seed"], as_index=False)[value].mean()
    out = (
        g.groupby(["alpha_train", "v", "lam", "t"])[value]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    out["se"] = out["std"].fillna(0) / np.sqrt(out["count"].clip(lower=1))
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["alpha_train", "v", "lam", "sim_seed", "t"]).copy()
    if "round_min_ds_oracle" not in df.columns:
        df["round_min_ds_oracle"] = df["mean_ds_oracle"]
    df["best_ds_oracle"] = (
        df.groupby(["alpha_train", "v", "lam", "sim_seed"])["round_min_ds_oracle"].cummin()
    )
    return df


def plot_diagnostics(df: pd.DataFrame, alpha: float, out_dir: Path, config: dict, roll: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df[df["alpha_train"] == alpha].copy()
    metrics = {
        "hard_winrate": _mean_se(sub, "hard_winrate"),
        "true_p": _mean_se(sub, "true_p_soft_oracle"),
        "pred_p": _mean_se(sub, "predicted_p"),
        "mean_ds": _mean_se(sub, "mean_ds_oracle"),
        "best_ds": _mean_se(sub, "best_ds_oracle"),
    }

    combos = (
        sub[["v", "lam"]].drop_duplicates()
        .sort_values(["v", "lam"])
        .itertuples(index=False, name=None)
    )
    combos = list(combos)
    cmap = plt.get_cmap("tab20")
    colors = {combo: cmap(i % 20) for i, combo in enumerate(combos)}

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
    ax_hard, ax_prob, ax_ds, ax_best = axes.ravel()

    for v, lam in combos:
        color = colors[(v, lam)]
        label = _label(v, lam)
        for ax, key in [(ax_hard, "hard_winrate"), (ax_ds, "mean_ds"), (ax_best, "best_ds")]:
            m = metrics[key]
            s = m[(m["v"] == v) & (m["lam"] == lam)].sort_values("t")
            y = s["mean"].rolling(roll, min_periods=1, center=True).mean()
            ax.plot(s["t"], y, lw=1.5, color=color, label=label)
        m_true = metrics["true_p"]
        s = m_true[(m_true["v"] == v) & (m_true["lam"] == lam)].sort_values("t")
        y = s["mean"].rolling(roll, min_periods=1, center=True).mean()
        ax_prob.plot(s["t"], y, lw=1.5, color=color, label=label)

        m_pred = metrics["pred_p"]
        sp = m_pred[(m_pred["v"] == v) & (m_pred["lam"] == lam)].sort_values("t")
        yp = sp["mean"].rolling(roll, min_periods=1, center=True).mean()
        ax_prob.plot(sp["t"], yp, lw=1.1, ls="--", color=color, alpha=0.9)

    D_B = config.get("D_B")
    if D_B is not None:
        ax_ds.axhline(D_B, color="0.35", lw=1.0, ls=":", label=f"D_B={D_B:.4f}")
    ax_hard.axhline(1.0, color="0.35", lw=1.0, ls=":")
    ax_prob.axhline(0.5, color="0.35", lw=1.0, ls=":")

    ax_hard.set_title("(a) hard win-rate")
    ax_prob.set_title("(b) belief vs truth (solid=true, dashed=predicted)")
    ax_ds.set_title("(c) mean ds-to-R per round (lower is better)")
    ax_best.set_title("(d) best-so-far ds-to-R (lower is better)")
    ax_hard.set_ylabel("win-rate")
    ax_prob.set_ylabel("probability")
    ax_ds.set_ylabel("mean ds-to-R")
    ax_best.set_ylabel("best ds-to-R")
    for ax in axes.ravel():
        ax.set_xlabel("round t")
        ax.grid(alpha=0.25)
    ax_hard.legend(fontsize=7, ncol=2)
    ax_prob.legend(fontsize=7, ncol=2)
    fig.suptitle(
        f"Discrete fixed-seed CM-TS diagnostics: alpha={alpha:g}, "
        f"B={config.get('B_word')}/{config.get('B_seed')}, fixed seed={config.get('fixed_image_seed_resolved')}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / f"diagnostics_alpha{_safe_alpha(alpha)}.png", dpi=170)
    plt.close(fig)


def plot_theta(df: pd.DataFrame, alpha: float, out_dir: Path, roll: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df[df["alpha_train"] == alpha].copy()
    sub["eig_ratio"] = sub["cov_eig_max"] / sub["cov_eig_min"].replace(0, np.nan)
    metrics = {
        "beta_norm": _mean_se(sub, "beta_norm"),
        "cov_eig_max": _mean_se(sub, "cov_eig_max"),
        "cov_eig_min": _mean_se(sub, "cov_eig_min"),
        "eig_ratio": _mean_se(sub, "eig_ratio"),
    }
    combos = list(
        sub[["v", "lam"]].drop_duplicates().sort_values(["v", "lam"]).itertuples(index=False, name=None)
    )
    cmap = plt.get_cmap("tab20")
    colors = {combo: cmap(i % 20) for i, combo in enumerate(combos)}

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
    panels = [
        ("beta_norm", r"$\|\theta_t\|_2$ / beta_norm", "linear"),
        ("cov_eig_max", "posterior cov max eigenvalue", "log"),
        ("cov_eig_min", "posterior cov min eigenvalue", "log"),
        ("eig_ratio", "condition ratio max/min", "log"),
    ]
    for ax, (key, ylabel, scale) in zip(axes.ravel(), panels):
        for v, lam in combos:
            s = metrics[key]
            s = s[(s["v"] == v) & (s["lam"] == lam)].sort_values("t")
            y = s["mean"].rolling(roll, min_periods=1, center=True).mean()
            ax.plot(s["t"], y, lw=1.5, color=colors[(v, lam)], label=_label(v, lam))
        ax.set_ylabel(ylabel)
        ax.set_yscale(scale)
        ax.set_xlabel("round t")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Discrete fixed-seed CM-TS theta/cov spectrum: alpha={alpha:g}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / f"theta_eigs_alpha{_safe_alpha(alpha)}.png", dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--root", default="outputs/discrete_fixedseed_alpha_grid")
    ap.add_argument("--roll", type=int, default=15)
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else _latest_run(Path(args.root))
    metrics_path = run_dir / "round_metrics.csv"
    config_path = run_dir / "config.json"
    if not metrics_path.exists():
        raise SystemExit(f"Missing {metrics_path}")
    df = prepare(pd.read_csv(metrics_path))
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    out_dir = run_dir / "alpha_panels"
    out_dir.mkdir(exist_ok=True)

    for alpha in sorted(df["alpha_train"].unique()):
        plot_diagnostics(df, float(alpha), out_dir, config, args.roll)
        plot_theta(df, float(alpha), out_dir, args.roll)
    print(f"Saved per-alpha panels to {out_dir}")


if __name__ == "__main__":
    main()
