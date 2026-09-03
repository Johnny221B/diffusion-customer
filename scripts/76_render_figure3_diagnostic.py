"""Render the paper's two-panel embedding/image-change diagnostic (Figure 3)."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


COLORS = {"fire_ocean": "#D55E00", "leather_neon": "#0072B2"}


def fit_line(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 200)
    return xx, slope * xx + intercept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random_csv", required=True)
    ap.add_argument("--fire_ocean_csv", required=True)
    ap.add_argument("--leather_neon_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    random_df = pd.read_csv(args.random_csv)
    fire = pd.read_csv(args.fire_ocean_csv)
    leather = pd.read_csv(args.leather_neon_csv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    r_random, p_random = pearsonr(random_df["z_dist"], random_df["dreamsim_dist"])
    r_fire, p_fire = pearsonr(fire["emb_dist_to_w1"], fire["dreamsim_to_w1"])
    r_leather, p_leather = pearsonr(leather["emb_dist_to_w1"], leather["dreamsim_to_w1"])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.05), constrained_layout=True)

    ax = axes[0]
    ax.scatter(random_df["z_dist"], random_df["dreamsim_dist"],
               s=22, color="#777777", alpha=0.62, edgecolors="white", linewidths=0.3)
    xx, yy = fit_line(random_df["z_dist"], random_df["dreamsim_dist"])
    ax.plot(xx, yy, color="#222222", lw=1.5)
    ax.text(0.04, 0.95, rf"$r={r_random:.3f}$", transform=ax.transAxes,
            ha="left", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=.9))
    ax.set_title("Random directions")
    ax.set_xlabel(r"Embedding displacement, $\|z-z_0\|_2$")
    ax.set_ylabel("Image change (DreamSim distance)")

    ax = axes[1]
    paths = [
        (fire, "fire_ocean", r"fire $\rightarrow$ ocean", r_fire),
        (leather, "leather_neon", r"leather $\rightarrow$ neon", r_leather),
    ]
    for df, key, label, rval in paths:
        x, y = df["emb_dist_to_w1"], df["dreamsim_to_w1"]
        ax.scatter(x, y, s=30, color=COLORS[key], alpha=.78,
                   edgecolors="white", linewidths=.35,
                   label=rf"{label}  ($r={rval:.3f}$)")
        xx, yy = fit_line(x, y)
        ax.plot(xx, yy, color=COLORS[key], lw=1.7)
    ax.set_title("Word-anchored directions")
    ax.set_xlabel(r"Embedding displacement, $\|z-z_0\|_2$")
    ax.set_ylabel("Image change (DreamSim distance)")
    ax.legend(frameon=False, loc="upper left", handletextpad=.4)

    for label, ax in zip(("a", "b"), axes):
        ax.text(-0.16, 1.08, label, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")
        ax.grid(True, color="#D9D9D9", lw=.6, alpha=.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)

    stem = output / "figure3_embedding_image_diagnostic"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    stats = pd.DataFrame([
        {"condition": "random directions", "n": len(random_df), "pearson_r": r_random, "p_value": p_random},
        {"condition": "fire to ocean", "n": len(fire), "pearson_r": r_fire, "p_value": p_fire},
        {"condition": "leather to neon", "n": len(leather), "pearson_r": r_leather, "p_value": p_leather},
    ])
    stats.to_csv(stem.with_name(stem.name + "_statistics.csv"), index=False)
    print(stats.to_string(index=False))
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
