#!/usr/bin/env python3
"""Plot an individual saved evaluation, without cross-run uncertainty bands."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cmts_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path,
                        default=PROJECT / "outputs/continuous_diagnostics/continuous_mse_regret_v05")
    parser.add_argument("--sim", type=int, default=1)
    args = parser.parse_args()
    metadata = json.loads((args.evaluation / "evaluation.json").read_text())
    cfg = metadata["config"]
    data = pd.read_csv(args.evaluation / "metrics.csv")
    data = data.loc[data.sim == args.sim].sort_values("round").copy()
    if data.empty:
        raise ValueError(f"No metrics for sim{args.sim:03d}")
    np.testing.assert_array_equal(data["round"], np.arange(1, len(data)+1))
    source = Path(metadata["run"])
    if not source.is_absolute():
        source = PROJECT / source
    trajectory = pd.read_csv(source / f"sim{args.sim:03d}" / "trajectory.csv")
    main_rows = trajectory.loc[trajectory.phase == "main"]
    losses = (metadata["oracle"]["probability"]-main_rows.true_p_soft).groupby(main_rows.t).sum()
    np.testing.assert_allclose(data.cumulative_regret, losses.cumsum(), atol=1e-8)

    out = args.evaluation / f"sim{args.sim:03d}"
    out.mkdir(parents=True, exist_ok=True)
    data.to_csv(out / "metrics.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, column, title in zip(axes, ["norm_mse", "cumulative_regret"],
                                ["Normalized MSE", "Approximate Cumulative Regret"]):
        ax.plot(data["round"], data[column], color="#0072B2", lw=1.8)
        ax.set(title=title, ylabel=title, xlabel=f"Round ({cfg['B']} designs per round)")
        ax.grid(alpha=.25)
    fig.suptitle(rf"Continuous CM-TS: sim{args.sim:03d} "
                 rf"($\alpha={cfg['alpha']:g}$, $v={cfg['v']:g}$, $\lambda={cfg['lam']:g}$)",
                 y=.97, fontsize=14)
    fig.text(.5, .025,
             f"Single trajectory; fixed test set (N={metadata['test_count']}); "
             f"approximate reference probability = {metadata['oracle']['probability']:.4f}.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.085, right=.985, bottom=.17, top=.8, wspace=.27)
    stem = f"continuous_sim{args.sim:03d}_normalized_mse_regret"
    for extension in ["png", "pdf"]:
        fig.savefig(out / f"{stem}.{extension}", dpi=250)
    plt.close(fig)
    (out / "README.md").write_text(
        f"# Continuous sim{args.sim:03d}\n\n"
        f"Source evaluation: {args.evaluation}\n\n"
        f"All {len(data)} rounds shown without smoothing or cross-run error bands. "
        "Normalized MSE uses the fixed independent 120-point test set. "
        "Approximate regret sums signed probability gaps over all 8 designs per round "
        "against the fixed retrospective observed-search reference "
        f"p={metadata['oracle']['probability']:.10f}. "
        "The sum was checked directly against trajectory.csv.\n\n"
        f"Historical out-of-domain selections (per parent evaluation tolerance): "
        f"{int(data.invalid_count.sum())}. These remain in the plotted historical results. "
        "This trajectory was selected after inspecting multiple runs and does not "
        "represent an across-seed performance estimate.\n")
    print(out / f"{stem}.png")


if __name__ == "__main__":
    main()
