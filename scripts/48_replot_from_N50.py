"""Re-plot scaling curves from saved scaling.csv, starting at N=50."""
import os
import glob
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_one(csv_path, n_min=50):
    out_dir = os.path.dirname(csv_path)
    df = pd.read_csv(csv_path)
    df = df[df["N"] >= n_min].reset_index(drop=True)
    if df.empty:
        print(f"[skip] {csv_path}: no rows with N>={n_min}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.errorbar(df["N"], df["cos_mean"], yerr=df["cos_std"], marker="o", capsize=3)
    ax.set_xlabel("Number of iid samples (N)")
    ax.set_ylabel("Cos(learned_dir, true_dir)")
    ax.set_title("Direction Alignment")
    ax.axhline(0, color="k", alpha=0.3)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.errorbar(df["N"], df["rpred_mean"], yerr=df["rpred_std"], marker="o", capsize=3, color="green")
    ax.set_xlabel("Number of iid samples (N)")
    ax.set_ylabel("Pearson(pred_reward, -dreamsim)")
    ax.set_title("Reward Prediction Correlation")
    ax.axhline(0, color="k", alpha=0.3)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.errorbar(df["N"], df["topk_mean"], yerr=df["topk_std"], marker="o", capsize=3, color="red")
    ax.set_xlabel("Number of iid samples (N)")
    ax.set_ylabel("Top-K Precision")
    ax.set_title("Top-K Precision (held-out)")
    ax.grid(True, alpha=0.3)

    dim_tag = os.path.basename(out_dir)
    fig.suptitle(f"{dim_tag}  (N>={n_min})", fontsize=13)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"scaling_from_N{n_min}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", type=str,
                        default="outputs/scaling_red_vs_green_d*_0416_1723/scaling.csv")
    parser.add_argument("--n_min", type=int, default=50)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.glob))
    print(f"Found {len(paths)} csv files")
    for p in paths:
        plot_one(p, n_min=args.n_min)


if __name__ == "__main__":
    main()
